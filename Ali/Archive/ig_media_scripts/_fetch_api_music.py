"""
Hit the private media-info API using the LOGGED-IN Playwright profile's fresh
session (via ctx.request.get, which shares the browser cookies but is not a page
navigation, so it returns real JSON instead of the SPA shell).

Saves raw API item per post to _api_dump/<sc>_api.json — this DOES carry
clips_metadata.music_info per slide.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = Path(__file__).parent
USER_DATA_DIR = str((BASE / "playwright_ig_profile").resolve())
DUMP = BASE / "_api_dump"
DUMP.mkdir(exist_ok=True)

SHORTCODES = ["Cu6xsKttA4T", "DVqnRDaiP2_", "BpCOgjTi6r_", "BocKi58Cgol"]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

def sc_to_pk(sc):
    pk = 0
    for ch in sc:
        pk = pk * 64 + ALPHABET.index(ch)
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

    csrf = next((c["value"] for c in ctx.cookies("https://www.instagram.com")
                 if c["name"] == "csrftoken"), "")

    for sc in SHORTCODES:
        pk = sc_to_pk(sc)
        url = f"https://www.instagram.com/api/v1/media/{pk}/info/"
        r = ctx.request.get(url, headers={
            "X-IG-App-ID": "936619743392459",
            "X-CSRFToken": csrf,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://www.instagram.com/p/{sc}/",
            "Accept": "*/*",
        })
        body = r.body()
        if body[:1] == b"{":
            data = json.loads(body)
            item = (data.get("items") or [None])[0]
            if item:
                (DUMP / f"{sc}_api.json").write_text(
                    json.dumps(item, ensure_ascii=False), encoding="utf-8")
                n = len(item.get("carousel_media") or [])
                print(f"{sc}: API OK, {n} slides -> _api_dump/{sc}_api.json")
            else:
                print(f"{sc}: API JSON but no items (status={data.get('status')})")
        else:
            print(f"{sc}: non-JSON (status={r.status}, head={body[:60]})")
        page.wait_for_timeout(4000)

    ctx.close()
print("DONE")
