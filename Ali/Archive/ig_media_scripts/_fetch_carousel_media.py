"""
Capture the full per-slide media JSON for the 4 video-carousel posts by loading
each post page in the logged-in Playwright profile and intercepting the JSON the
page itself fetches (which carries carousel_media[].clips_metadata = music, and
image-slide display URLs).

Raw JSON is dumped to _api_dump/<sc>.json for offline parsing.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = Path(__file__).parent
USER_DATA_DIR = str((BASE / "playwright_ig_profile").resolve())
DUMP = BASE / "_api_dump"
DUMP.mkdir(exist_ok=True)

SHORTCODES = ["Cu6xsKttA4T", "DVqnRDaiP2_", "BpCOgjTi6r_", "BocKi58Cgol"]


def _clear_lock():
    for name in ("SingletonLock", "lockfile"):
        lp = Path(USER_DATA_DIR) / name
        try:
            if lp.exists(): lp.unlink()
        except Exception: pass


def find_media_node(obj, code):
    """Find a dict with carousel_media whose code/shortcode EXACTLY matches.
    No fallback — a wrong node is worse than none."""
    def walk(o):
        if isinstance(o, dict):
            if (isinstance(o.get("carousel_media"), list)
                    and (o.get("code") == code or o.get("shortcode") == code)):
                return o
            for v in o.values():
                r = walk(v)
                if r: return r
        elif isinstance(o, list):
            for v in o:
                r = walk(v)
                if r: return r
        return None
    return walk(obj)


_clear_lock()
with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        USER_DATA_DIR, headless=True,
        viewport={"width": 1280, "height": 900},
        args=["--disable-blink-features=AutomationControlled"])
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://www.instagram.com/", wait_until="domcontentloaded")

    for sc in SHORTCODES:
        captured = {"node": None}

        def on_response(resp):
            ct = resp.headers.get("content-type", "")
            if "application/json" not in ct:
                return
            try:
                body = resp.text()
            except Exception:
                return
            if "carousel_media" not in body:
                return
            try:
                data = json.loads(body)
            except Exception:
                return
            node = find_media_node(data, sc)
            if node and (captured["node"] is None or
                         len(node["carousel_media"]) > len(captured["node"]["carousel_media"])):
                captured["node"] = node

        page.on("response", on_response)
        try:
            page.goto(f"https://www.instagram.com/p/{sc}/",
                      wait_until="networkidle", timeout=40000)
            page.wait_for_timeout(3000)
            # Nudge: some posts only fetch full media JSON after interaction
            try:
                page.click("article", timeout=2000)
            except Exception:
                pass
            page.wait_for_timeout(2000)

            # Also scan embedded <script type="application/json"> blobs — modern IG
            # ships the post data inside the HTML document, not always as XHR.
            if captured["node"] is None:
                scripts = page.evaluate(
                    "() => [...document.querySelectorAll('script[type=\"application/json\"]')]"
                    ".map(s => s.textContent)")
                for txt in scripts:
                    if not txt or "carousel_media" not in txt:
                        continue
                    try:
                        data = json.loads(txt)
                    except Exception:
                        continue
                    node = find_media_node(data, sc)
                    if node:
                        captured["node"] = node
                        break
        finally:
            page.remove_listener("response", on_response)

        node = captured["node"]
        if node:
            (DUMP / f"{sc}.json").write_text(json.dumps(node, ensure_ascii=False), encoding="utf-8")
            n = len(node["carousel_media"])
            print(f"{sc}: captured media JSON with {n} slides -> _api_dump/{sc}.json")
        else:
            print(f"{sc}: NO media JSON captured")
        page.wait_for_timeout(4000)

    ctx.close()
print("DONE")
