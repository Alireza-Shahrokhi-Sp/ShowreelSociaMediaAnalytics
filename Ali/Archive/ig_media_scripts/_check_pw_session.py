"""Check if the Playwright persistent profile is still logged into Instagram,
and if so, fetch one post's private-API JSON to confirm per-slide music is reachable."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = Path(__file__).parent
USER_DATA_DIR = str((BASE / "playwright_ig_profile").resolve())

SHORTCODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
def sc_to_pk(sc):
    pk = 0
    for ch in sc:
        pk = pk * 64 + SHORTCODE_ALPHABET.index(ch)
    return pk

def _clear_lock():
    for name in ("SingletonLock", "lockfile"):
        lp = Path(USER_DATA_DIR) / name
        try:
            if lp.exists(): lp.unlink()
        except Exception: pass

_clear_lock()
with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        USER_DATA_DIR, headless=True,
        args=["--disable-blink-features=AutomationControlled"])
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://www.instagram.com/", wait_until="domcontentloaded")
    logged_in = any(c["name"] == "sessionid" for c in ctx.cookies("https://www.instagram.com"))
    print("LOGGED_IN:", logged_in)

    if logged_in:
        pk = sc_to_pk("BpCOgjTi6r_")
        # Fetch the private API via in-page fetch (carries auth + headers automatically)
        result = page.evaluate("""async (pk) => {
            const r = await fetch(`/api/v1/media/${pk}/info/`, {
                headers: {'X-IG-App-ID': '936619743392459', 'X-Requested-With': 'XMLHttpRequest'}
            });
            const txt = await r.text();
            return {status: r.status, head: txt.slice(0, 60), ok: txt.startsWith('{')};
        }""", pk)
        print("API status:", result["status"], "| json:", result["ok"], "| head:", result["head"])
    ctx.close()
