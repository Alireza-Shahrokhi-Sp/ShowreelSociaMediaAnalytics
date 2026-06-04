"""Download the missing slide 7 of DVqnRDaiP2_ via the logged-in Playwright
session (fresh CDN url + authed referer). Saves as DVqnRDaiP2__07.mp4 so the
reorg step renumbers it into sequence."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = Path(__file__).parent
USER_DATA_DIR = str((BASE / "playwright_ig_profile").resolve())
SC = "DVqnRDaiP2_"
DEST = BASE / "playwright_downloads" / "carousel" / SC

def _clear_lock():
    for name in ("SingletonLock", "lockfile"):
        lp = Path(USER_DATA_DIR) / name
        try:
            if lp.exists(): lp.unlink()
        except Exception: pass

def find_node(obj, code):
    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("carousel_media"), list) and o.get("code") == code:
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
        args=["--disable-blink-features=AutomationControlled"])
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://www.instagram.com/", wait_until="domcontentloaded")

    node = {"n": None}
    def on_resp(resp):
        if "application/json" not in resp.headers.get("content-type", ""):
            return
        try:
            t = resp.text()
        except Exception:
            return
        if "carousel_media" not in t:
            return
        try:
            d = json.loads(t)
        except Exception:
            return
        nn = find_node(d, SC)
        if nn:
            node["n"] = nn

    page.on("response", on_resp)
    page.goto(f"https://www.instagram.com/p/{SC}/", wait_until="networkidle", timeout=40000)
    page.wait_for_timeout(3000)
    if node["n"] is None:
        scripts = page.evaluate(
            "() => [...document.querySelectorAll('script[type=\"application/json\"]')].map(s=>s.textContent)")
        for txt in scripts:
            if txt and "carousel_media" in txt:
                try:
                    nn = find_node(json.loads(txt), SC)
                    if nn:
                        node["n"] = nn; break
                except Exception:
                    pass

    n = node["n"]
    if not n:
        print("FAILED: no media node"); ctx.close(); raise SystemExit

    slide7 = n["carousel_media"][6]
    vv = sorted(slide7.get("video_versions") or [],
                key=lambda v: v.get("width",0)*v.get("height",0), reverse=True)
    if not vv:
        print("FAILED: slide 7 has no video_versions"); ctx.close(); raise SystemExit
    url = vv[0]["url"]
    r = ctx.request.get(url, headers={"Referer": f"https://www.instagram.com/p/{SC}/"})
    if r.ok:
        out = DEST / f"{SC}_07.mp4"
        out.write_bytes(r.body())
        print(f"OK: saved slide 7 -> {out.name} ({len(r.body())//1024} KB)")
    else:
        print(f"FAILED: download status {r.status}")
    ctx.close()
