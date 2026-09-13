"""Browser test for what templates/index.html actually shows.

Run from the repo root:

    pip install playwright jinja2 pillow && playwright install chromium
    python3 tests/test_ui_visibility.py

Set PLAYWRIGHT_CHROMIUM to a chrome binary to use one already on disk. Skips
with exit code 0 if playwright or a browser is unavailable, so it never blocks
a checkout that does not have them.

This exists because of a real shipped bug: the `hidden` attribute is only
display:none in the UA stylesheet, so `.analyzing { display: grid }` beat it and
a fixed, full-viewport overlay covered the app on every page load. Jinja
rendering could not catch that -- only a browser computing real styles could.
"""
import os, shutil, sys, tempfile, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jinja2 import Environment, FileSystemLoader, StrictUndefined
import translations as tr

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("SKIP: playwright is not installed (pip install playwright)")
    raise SystemExit(0)
warnings.filterwarnings("ignore")

env = Environment(loader=FileSystemLoader("templates"), undefined=StrictUndefined, autoescape=True)
env.globals["url_for"] = lambda ep, **kw: kw["filename"] if ep == "static" else "index.html"
tpl = env.get_template("index.html")

def ctx(lang="en", **extra):
    labels = tr.GENRE_LABELS[lang]
    c = dict(lang=lang, direction=tr.DIRECTION[lang], other_lang=tr.other(lang), t=tr.UI[lang],
             accent=tr.FALLBACK_ACCENT, genre_accents=tr.GENRE_ACCENTS,
             genre_chips=[(k, labels[k]) for k in tr.GENRE_ACCENTS], prediction=None, error=None)
    c.update(extra); return c

TRACK = dict(title="Neon Arcadia", artist="Nightform", url="https://youtube.com/watch?v=a",
             video_id="a", thumbnail="")
def results(lang="en"):
    return ctx(lang, prediction="electronic", genre_label=tr.GENRE_LABELS[lang]["electronic"],
        prediction_info=tr.GENRE_DESCRIPTIONS[lang]["electronic"],
        accent=tr.GENRE_ACCENTS["electronic"],
        distribution=[{"key":"electronic","label":"E","pct":68,"is_top":True}],
        grad_text="g", original_text="o", image_data="", gradcam_data="",
        original_tracks=[TRACK]*3, khaleeji_tracks=[TRACK]*3, track_total=6, error=None)

work = tempfile.mkdtemp()
shutil.copy("static/styles.css", work)
shutil.copy("static/app.js", work)

fails = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ((" -- " + detail) if detail and not ok else ""))
    if not ok: fails.append(name)

with sync_playwright() as p:
    launch = {"args": ["--no-sandbox", "--disable-dev-shm-usage"]}
    if os.environ.get("PLAYWRIGHT_CHROMIUM"):
        launch["executable_path"] = os.environ["PLAYWRIGHT_CHROMIUM"]
    try:
        browser = p.chromium.launch(**launch)
    except Exception as exc:
        print(f"SKIP: no chromium available ({str(exc).splitlines()[0]})")
        print("      run `playwright install chromium`, or set PLAYWRIGHT_CHROMIUM")
        shutil.rmtree(work, ignore_errors=True)
        raise SystemExit(0)
    page = browser.new_page(viewport={"width": 1440, "height": 900})

    for label, context in [("landing", ctx()), ("landing (ar)", ctx("ar")), ("results", results())]:
        path = os.path.join(work, "index.html")
        open(path, "w", encoding="utf-8").write(tpl.render(**context))
        page.goto("file://" + path)
        page.wait_for_timeout(250)

        print(f"\n{label}:")
        overlay = page.locator("#analyzing")
        check(f"[{label}] analyzing overlay is NOT shown", not overlay.is_visible())

        # would the overlay be covering the page?
        box = overlay.bounding_box()
        covering = bool(box and box["width"] > 1000 and box["height"] > 600)
        check(f"[{label}] overlay is not covering the viewport", not covering,
              f"occupies {box['width']:.0f}x{box['height']:.0f}" if box else "")

        if context["prediction"] is None:
            check("[landing] submit button hidden until a file is chosen",
                  not page.locator("#submit-btn").is_visible())
            check("[landing] file input itself stays hidden",
                  not page.locator("#image-input").is_visible())
            # the thing a user must be able to click
            check("[landing] dropzone is reachable (nothing on top of it)",
                  page.locator("#dropzone").is_visible())

    # --- drive the real upload interaction on the landing page ---
    path = os.path.join(work, "index.html")
    open(path, "w", encoding="utf-8").write(tpl.render(**ctx()))
    page.goto("file://" + path)
    page.wait_for_timeout(250)

    print("\nupload interaction:")
    sample = os.path.join(work, "sample.jpg")
    from PIL import Image as _Im
    _Im.new("RGB", (80, 60), (30, 40, 80)).save(sample)

    page.set_input_files("#image-input", sample)
    page.wait_for_timeout(200)
    check("choosing a file reveals the submit button", page.locator("#submit-btn").is_visible())
    check("dropzone shows the chosen filename",
          "sample.jpg" in page.locator(".drop__title").inner_text(),
          page.locator(".drop__title").inner_text())
    check("overlay still hidden before submit", not page.locator("#analyzing").is_visible())

    # stop the navigation so we can observe what submit does
    page.evaluate("document.getElementById('upload-form')"
                  ".addEventListener('submit', e => e.preventDefault())")
    page.click("#submit-btn")
    page.wait_for_timeout(300)
    check("submitting reveals the analyzing overlay", page.locator("#analyzing").is_visible())
    check("overlay previews the chosen image",
          bool(page.get_attribute("#analyzing-image", "src")))
    check("a step is marked active", page.locator(".step.is-active").count() >= 1)
    page.wait_for_timeout(3400)
    check("later steps advance over time", page.locator(".step.is-done").count() >= 1,
          f'{page.locator(".step.is-done").count()} done')

    # --- nothing may be wider than the viewport (any state, any language) ---
    print("\nhorizontal overflow:")
    for label, context, vw, vh in [("results  390", results(), 390, 844),
                                   ("results ar 390", results("ar"), 390, 844),
                                   ("results 1440", results(), 1440, 900),
                                   ("landing  390", ctx(), 390, 844)]:
        pg = browser.new_page(viewport={"width": vw, "height": vh})
        open(path, "w", encoding="utf-8").write(tpl.render(**context))
        pg.goto("file://" + path)
        pg.wait_for_timeout(300)
        # .rail__strip and .frame__meta scroll sideways on purpose; everything
        # else must fit. A grid/flex item with min-width:auto is the usual cause.
        wide = pg.evaluate("""(vw) => {
            const skip = ['rail__strip', 'frame__meta'];
            const out = [];
            document.querySelectorAll('body *').forEach(el => {
                const c = (el.className && el.className.baseVal === undefined ? el.className : '').toString();
                if (skip.some(s => c.includes(s))) return;
                if (el.getBoundingClientRect().width > vw + 2)
                    out.push(el.tagName.toLowerCase() + '.' + c.split(' ')[0] +
                             '=' + Math.round(el.getBoundingClientRect().width));
            });
            return out.slice(0, 6);
        }""", vw)
        doc = pg.evaluate("document.documentElement.scrollWidth")
        check(f"[{label}] no element exceeds the viewport", not wide, ", ".join(wide))
        check(f"[{label}] page does not scroll sideways", doc <= vw + 2, f"scrollWidth={doc}")
        pg.close()

    browser.close()

shutil.rmtree(work, ignore_errors=True)
print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED"))
sys.exit(1 if fails else 0)
