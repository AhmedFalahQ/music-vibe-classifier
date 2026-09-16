"""Regression test for utils/images.load_image.

    python3 tests/test_image_loading.py

Covers the failure that reached production: an iPhone HEIC upload. Pillow has
no native HEIF support, so Image.open raised UnidentifiedImageError, the old
imageio fallback decoded the file to a (1, 1, 512, 3) strip of garbage, and
Image.fromarray rejected it with "Cannot handle this data type".
"""
import io, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")

from utils.images import load_image

fails = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ((" -- " + detail) if detail and not ok else ""))
    if not ok: fails.append(name)

def encode(fmt, size=(640, 480), **kw):
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 60, 110)).save(buf, format=fmt, **kw)
    return buf.getvalue()

print("1. ordinary formats still work")
for fmt, kw in [("JPEG", {"quality": 90}), ("PNG", {}), ("WEBP", {}), ("BMP", {})]:
    try:
        im = load_image(encode(fmt, **kw))
        check(f"{fmt} decodes to RGB {im.size}", im.mode == "RGB" and im.size == (640, 480))
    except Exception as e:
        check(f"{fmt} decodes", False, f"{type(e).__name__}: {e}")

print("\n2. HEIC -- the format that broke in production")
try:
    import pillow_heif
    buf = io.BytesIO()
    pillow_heif.from_pillow(Image.new("RGB", (800, 600), (90, 30, 60))).save(
        buf, format="HEIF", quality=80)
    heic = buf.getvalue()
    check("wrote a real HEIC to test against", heic[4:8] == b"ftyp", repr(heic[:16]))
    im = load_image(heic)
    check("HEIC decodes through load_image", im.mode == "RGB" and im.size == (800, 600),
          f"{im.mode} {im.size}")
except ImportError:
    check("pillow-heif installed", False, "pip install pillow-heif")

print("\n3. grayscale and RGBA normalise to RGB")
for mode in ("L", "RGBA", "P"):
    buf = io.BytesIO()
    Image.new(mode, (64, 48)).save(buf, format="PNG")
    im = load_image(buf.getvalue())
    check(f"{mode} -> RGB", im.mode == "RGB", im.mode)

print("\n4. undecodable input raises ValueError, not TypeError")
for name, payload in [("random bytes", os.urandom(4096)),
                      ("empty", b""),
                      ("truncated JPEG", encode("JPEG")[:120])]:
    try:
        load_image(payload)
        check(f"{name} rejected", False, "no exception raised")
    except ValueError as e:
        check(f"{name} -> ValueError", True)
    except Exception as e:
        check(f"{name} -> ValueError", False, f"got {type(e).__name__}: {e}")

print("\n5. the exact production shape is reported, not crashed on")
import utils.images as ui
real = ui._imread
try:
    ui._imread = lambda *a, **k: np.zeros((1, 1, 512, 3), dtype=np.uint8)
    try:
        load_image(os.urandom(64))
        check("degenerate decode rejected", False, "no exception")
    except ValueError as e:
        check("degenerate decode -> ValueError naming the shape",
              "shape" in str(e), str(e))
    except Exception as e:
        check("degenerate decode -> ValueError", False, f"got {type(e).__name__}")
finally:
    ui._imread = real

print("\n6. a genuine frame stack is unwrapped rather than rejected")
try:
    ui._imread = lambda *a, **k: np.zeros((1, 480, 640, 3), dtype=np.uint8)
    im = load_image(os.urandom(64))
    check("(1, 480, 640, 3) -> single frame", im.size == (640, 480), str(im.size))
finally:
    ui._imread = real

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
