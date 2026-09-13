"""Regression test for generate_heatmap in app.py.

Run from the repo root with no test framework:

    python3 tests/test_gradcam.py

It execs the Grad-CAM block straight out of app.py against a real (untrained)
ResNet34, so it needs no model.pth, no AWS and no API keys.

Check 6 is the one that matters most. The pre-fix implementation kept the
captured activation and gradient in module-level globals and multiplied into the
activation in place; run against 8 concurrent threads it returned the wrong
heatmap for all 8 inputs, and could abort the interpreter with "double free or
corruption". That comparison is deliberately not reproduced here, because
running the broken version can take the test process down with it.
"""
import threading, warnings
import numpy as np, torch
from torchvision import models
warnings.filterwarnings("ignore")

def build_model():
    torch.manual_seed(0)
    m = models.resnet34(weights=None)
    m.fc = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(m.fc.in_features, 5))
    m.eval()
    return m

def load(src, start, end, model):
    """exec the real source of the CAM block from app.py into a namespace."""
    body = src[src.index(start):src.index(end)]
    ns = {"torch": torch, "np": np, "threading": threading, "model": model}
    exec(compile(body, "<app.py excerpt>", "exec"), ns)
    return ns

NEW = open("app.py").read()

model = build_model()
new = load(NEW, "TARGET_LAYER = model.layer4[2].conv2", "def overlay_heatmap", model)
gen = new["generate_heatmap"]
LAYER = new["TARGET_LAYER"]

def inp(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, 224, 224, generator=g)

def naive(x, idx):
    """Independent implementation in the original per-channel-loop style."""
    cap = {}
    h1 = LAYER.register_forward_hook(lambda m, i, o: cap.__setitem__("a", o.detach().clone()))
    h2 = LAYER.register_full_backward_hook(lambda m, gi, go: cap.__setitem__("g", go[0].detach().clone()))
    model.zero_grad(set_to_none=True)
    model(x)[0, idx].backward()
    h1.remove(); h2.remove()
    a, g = cap["a"], cap["g"]
    pooled = g.mean(dim=(0, 2, 3))
    acc = torch.zeros_like(a)
    for i in range(a.shape[1]):
        acc[:, i] = a[:, i] * pooled[i]
    cam = torch.relu(acc.mean(dim=1)).squeeze(0)
    return (cam / cam.max()).numpy()

fails = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  -- " + detail) if detail and not ok else ""))
    if not ok: fails.append(name)

print("1. shape, dtype, range")
h = gen(inp(1), 2)
check("returns 2-D array", h.ndim == 2, str(h.shape))
check("matches layer4 spatial size 7x7", h.shape == (7, 7), str(h.shape))
check("float32", h.dtype == np.float32, str(h.dtype))
check("no NaN / inf", np.isfinite(h).all())
check("within [0, 1]", h.min() >= 0.0 and h.max() <= 1.0 + 1e-6, f"min={h.min():.4f} max={h.max():.4f}")
check("peak normalized to 1", abs(h.max() - 1.0) < 1e-5, f"max={h.max():.6f}")

print("\n2. output unchanged vs an independent naive implementation")
for seed, idx in [(1, 2), (7, 0), (13, 4)]:
    a, b = gen(inp(seed), idx), naive(inp(seed), idx)
    check(f"seed={seed} class={idx} identical", np.allclose(a, b, atol=1e-5),
          f"max abs diff {np.abs(a-b).max():.2e}")

print("\n3. all-negative CAM returns zeros, not NaN (old code divided by zero)")
class ZeroGrad(torch.nn.Module):
    """Output depends on x but with zero gradient, so every channel weight is 0
    and the whole CAM collapses to <= 0 -- the peak<=0 branch."""
    def forward(self, x): return (x.sum() * 0.0).reshape(1, 1).expand(1, 5)
saved_fc = model.fc
try:
    model.fc = ZeroGrad()
    z = gen(inp(3), 0)
    check("finite", np.isfinite(z).all())
    check("all zeros", np.all(z == 0), f"max={z.max()}")
finally:
    model.fc = saved_fc

print("\n4. hooks are removed after every call")
before = len(LAYER._forward_hooks) + len(LAYER._backward_hooks) + len(getattr(LAYER, "_full_backward_hooks", {}))
for i in range(5): gen(inp(i), i % 5)
after = len(LAYER._forward_hooks) + len(LAYER._backward_hooks) + len(getattr(LAYER, "_full_backward_hooks", {}))
check("no hook accumulation", before == after == 0, f"before={before} after={after}")

print("\n5. parameter .grad buffers left clean")
gen(inp(9), 1)
dirty = [n for n, p in model.named_parameters() if p.grad is not None]
check("no leftover gradients", not dirty, f"{len(dirty)} params still hold .grad")

print("\n6. CONCURRENCY: 8 threads, each result must equal its single-threaded value")
cases = [(20 + i, i % 5) for i in range(8)]
baseline = {s: gen(inp(s), c) for s, c in cases}

def run(fn):
    got, errs = {}, {}
    def work(s, c):
        try: got[s] = fn(inp(s), c)
        except Exception as e: errs[s] = f"{type(e).__name__}: {e}"
    ts = [threading.Thread(target=work, args=(s, c)) for s, c in cases]
    for t in ts: t.start()
    for t in ts: t.join()
    return got, errs

got, errs = run(gen)
bad = [s for s, _ in cases if s in got and not np.allclose(got[s], baseline[s], atol=1e-5)]
check("no thread raised", not errs, str(errs)[:160])
check("all 8 results correct under load", not bad and len(got) == 8,
      f"mismatched seeds={bad} returned={len(got)}")

print("\n7. heatmap colours are not inverted")
import cv2
from PIL import Image as _PILImage
_ov_src = NEW[NEW.index("def overlay_heatmap("):NEW.index("def merge_images_side_by_side(")]
_ns = {"cv2": cv2, "np": np}
exec(compile(_ov_src, "<app.py excerpt>", "exec"), _ns)
overlay_heatmap = _ns["overlay_heatmap"]

hm = np.zeros((8, 8), dtype=np.float32)
hm[:, :4] = 1.0                                  # hot left half, cold right half
rgb = overlay_heatmap(hm, _PILImage.new("RGB", (8, 8), (0, 0, 0)))
hot, cold = rgb[4, 1].astype(int), rgb[4, 6].astype(int)
check("high activation renders red, not blue", hot[0] > hot[2], f"R={hot[0]} B={hot[2]}")
check("low activation renders blue, not red", cold[2] > cold[0], f"R={cold[0]} B={cold[2]}")
check("output is uint8", rgb.dtype == np.uint8, str(rgb.dtype))
check("output keeps the image size", rgb.shape == (8, 8, 3), str(rgb.shape))

# a mid-grey photo must still be visible through the overlay, not blown to white
grey = overlay_heatmap(hm, _PILImage.new("RGB", (8, 8), (128, 128, 128)))
check("photo survives the blend (not saturated to white)", grey.max() < 255,
      f"max channel {grey.max()}")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
