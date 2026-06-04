"""
Targeted audio recovery for the 3 manually-verified reels.
Scrapes ONLY these shortcodes into audio_recovery/<sc>/ (separate from
playwright_downloads/video, so the verified videos are never touched).
Captures every audio/video CDN stream so the mux step can pick the audio
matching the verified duration.
"""
import json
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = Path(__file__).parent
USER_DATA_DIR = str((BASE / "playwright_ig_profile").resolve())
OUT = BASE / "audio_recovery"
REFERER = {"referer": "https://www.instagram.com/"}
SHORTCODES = ["CRBdst5I1Go", "CX_Z-pwhDet", "DVeTxJEDnKP"]
LAUNCH = dict(headless=False, viewport={"width": 1280, "height": 900},
              args=["--disable-blink-features=AutomationControlled"])


def is_logged_in(ctx):
    return any(c["name"] == "sessionid" for c in ctx.cookies("https://www.instagram.com"))


def save_url(ctx, url, dest):
    r = ctx.request.get(url, headers=REFERER)
    if r.ok:
        dest.write_bytes(r.body())
        return True
    return False


def main():
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(USER_DATA_DIR, **LAUNCH)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://www.instagram.com/", wait_until="domcontentloaded")
        if not is_logged_in(ctx):
            print("NOT_LOGGED_IN")
            ctx.close()
            return

        for sc in SHORTCODES:
            target = OUT / sc
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)

            urls = []

            def on_resp(resp):
                u = resp.url
                if "cdninstagram" in u or "fbcdn" in u:
                    ct = resp.headers.get("content-type", "")
                    if ".mp4" in u or ct.startswith("video") or ct.startswith("audio"):
                        urls.append(u.split("&bytestart")[0])

            page.on("response", on_resp)
            td = None
            try:
                page.goto(f"https://www.instagram.com/p/{sc}/", wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(2500)
                try:
                    page.click("article video", timeout=2000)
                    page.wait_for_timeout(2500)
                except Exception:
                    pass
                try:
                    page.wait_for_function(
                        "() => { const v = document.querySelector('article video');"
                        " return v && isFinite(v.duration) && v.duration > 0; }",
                        timeout=8000,
                    )
                    td = page.evaluate("() => document.querySelector('article video').duration")
                except Exception:
                    pass
                (target / "_target.json").write_text(json.dumps({"target_duration": td}), encoding="utf-8")
            finally:
                page.remove_listener("response", on_resp)

            saved = 0
            for i, u in enumerate(dict.fromkeys(urls)):
                if save_url(ctx, u, target / f"{sc}_{i}.mp4"):
                    saved += 1
            print(f"{sc}: saved {saved} streams, target_duration={td}")
            page.wait_for_timeout(5000)

        ctx.close()
        print("DONE")


if __name__ == "__main__":
    main()