"""
Standalone Playwright Instagram scraper (run as a separate process).

Why a separate script? On Windows, Playwright needs the Proactor event loop to
launch the browser, but Jupyter's kernel uses a Selector loop -> NotImplementedError.
Running this as its own process sidesteps that entirely.

Usage (from the notebook or a terminal):
    python playwright_scrape.py login    # open browser, log in by hand (one time)
    python playwright_scrape.py scrape    # download the posts in to_download/posts_to_download.csv
"""
import sys
import time
import random
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = Path(__file__).parent
USER_DATA_DIR = str((BASE / "playwright_ig_profile").resolve())  # persistent login
OUT = BASE / "playwright_downloads"
CSV = BASE / "to_download" / "posts_to_download.csv"
REFERER = {"referer": "https://www.instagram.com/"}
LAUNCH_ARGS = dict(
    headless=False,
    viewport={"width": 1280, "height": 900},
    args=["--disable-blink-features=AutomationControlled"],
)


def _clear_lock():
    for name in ("SingletonLock", "lockfile"):
        lp = Path(USER_DATA_DIR) / name
        try:
            if lp.exists():
                lp.unlink()
        except Exception:
            pass


def is_logged_in(ctx):
    return any(c["name"] == "sessionid" for c in ctx.cookies("https://www.instagram.com"))


def do_login():
    _clear_lock()
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(USER_DATA_DIR, **LAUNCH_ARGS)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://www.instagram.com/", wait_until="domcontentloaded")
        if is_logged_in(ctx):
            print("Already logged in (persistent profile). Nothing to do.")
            ctx.close()
            return
        print("Log in to Instagram in the opened window (you have up to 5 minutes)...")
        for _ in range(300):
            if is_logged_in(ctx):
                print("Login detected and saved to the profile. You can run 'scrape' now.")
                break
            time.sleep(1)
        else:
            print("Timed out waiting for login. Re-run 'login' and try again.")
        ctx.close()


def save_url(ctx, url, dest):
    r = ctx.request.get(url, headers=REFERER)
    if r.ok:
        dest.write_bytes(r.body())
        return True
    return False


def scrape_post(ctx, page, shortcode, media_type):
    import json as _json
    import shutil as _shutil

    sub = {"VIDEO": "video", "CAROUSEL_ALBUM": "carousel"}.get(media_type, "image")
    target = OUT / sub / shortcode
    # Start clean so stale captures from a previous run don't linger
    _shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    video_urls = []

    def on_response(resp):
        u = resp.url
        if ("cdninstagram" in u or "fbcdn" in u):
            ct = resp.headers.get("content-type", "")
            if ".mp4" in u or ct.startswith("video"):
                video_urls.append(u.split("&bytestart")[0])

    # Only attach the network listener for videos; images use DOM queries instead.
    if media_type == "VIDEO":
        page.on("response", on_response)

    try:
        page.goto(f"https://www.instagram.com/p/{shortcode}/", wait_until="networkidle", timeout=30000)
        # Give the post content time to render; article may appear after networkidle
        try:
            page.wait_for_selector("article", timeout=8000)
        except Exception:
            pass  # continue anyway; JS query will simply return [] if nothing found
        page.wait_for_timeout(2000)

        if media_type == "VIDEO":
            try:
                page.click("article video", timeout=2000)
                page.wait_for_timeout(1500)
            except Exception:
                pass
            # Record the focused post's real duration so reconcile picks the right DASH stream.
            target_dur = None
            try:
                page.wait_for_function(
                    "() => { const v = document.querySelector('article video');"
                    " return v && isFinite(v.duration) && v.duration > 0; }",
                    timeout=8000,
                )
                target_dur = page.evaluate(
                    "() => { const v = document.querySelector('article video');"
                    " return v ? v.duration : null; }"
                )
            except Exception:
                pass
            (target / "_target.json").write_text(
                _json.dumps({"target_duration": target_dur}), encoding="utf-8"
            )
    finally:
        if media_type == "VIDEO":
            page.remove_listener("response", on_response)

    # --- VIDEO: save every captured CDN stream (mux step picks the right one later) ---
    saved = 0
    if media_type == "VIDEO":
        for i, u in enumerate(dict.fromkeys(video_urls)):
            if save_url(ctx, u, target / f"{shortcode}_{i}.mp4"):
                saved += 1
        return saved

    # --- IMAGE: use og:image meta tag — always points to the post's own image ---
    JS_OG = "() => { const m = document.querySelector('meta[property=\"og:image\"]'); return m ? m.content : null; }"

    if media_type == "IMAGE":
        og = page.evaluate(JS_OG)
        if og and ("cdninstagram" in og or "fbcdn" in og):
            if save_url(ctx, og, target / f"{shortcode}_0.jpg"):
                saved += 1
        return saved

    # --- CAROUSEL: og:image for slide 1, then filtered network capture per slide ---
    # We only collect images that arrive AFTER each "Next" click, so feed images
    # (which loaded before the first click) are never included.
    slide_urls = []
    pending = []

    def on_img(resp):
        u = resp.url
        if ("cdninstagram" in u or "fbcdn" in u):
            ct = resp.headers.get("content-type", "")
            if ct.startswith("image") and "s150x150" not in u and "s320x320" not in u:
                pending.append(u)

    # Slide 1: use og:image (already loaded, network is quiet)
    og = page.evaluate(JS_OG)
    if og:
        slide_urls.append(og)

    page.on("response", on_img)
    try:
        for _ in range(20):  # Instagram carousel max ~20 slides
            nxt = page.query_selector('button[aria-label="Next"]')
            if not nxt:
                break
            pending.clear()
            try:
                nxt.click(timeout=2000)
                page.wait_for_timeout(1500)  # let the new slide image load
            except Exception:
                break
            # The first large image that arrived after this click is the new slide
            if pending:
                slide_urls.append(pending[0])
    finally:
        page.remove_listener("response", on_img)

    for i, u in enumerate(slide_urls):
        if save_url(ctx, u, target / f"{shortcode}_{i}.jpg"):
            saved += 1
    return saved


def scrape():
    import pandas as pd

    if not CSV.exists():
        print(f"List not found: {CSV}")
        return
    todo = pd.read_csv(CSV)
    print(f"Posts to scrape: {len(todo)}")

    succeeded, failed = [], []
    _clear_lock()
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(USER_DATA_DIR, **LAUNCH_ARGS)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://www.instagram.com/", wait_until="domcontentloaded")
        if not is_logged_in(ctx):
            print("Not logged in. Run:  python playwright_scrape.py login")
            ctx.close()
            return

        for n, row in enumerate(todo.itertuples(index=False), 1):
            sc = str(row.shortcode)
            mt = str(row.media_type)
            sub = {"VIDEO": "video", "CAROUSEL_ALBUM": "carousel"}.get(mt, "image")
            dest = OUT / sub / sc
            # Skip if already has media (videos were manually verified, don't redo them)
            exts = {"*.mp4"} if mt == "VIDEO" else {"*.jpg"}
            if dest.exists() and any(f for e in exts for f in dest.glob(e)):
                print(f"[{n}/{len(todo)}] {sc} ({mt}) ... already done, skipping")
                succeeded.append(sc)
                continue
            print(f"[{n}/{len(todo)}] {sc} ({mt}) ...", end=" ", flush=True)
            try:
                got = scrape_post(ctx, page, sc, mt)
                if got:
                    print(f"saved {got} file(s)")
                    succeeded.append(sc)
                else:
                    print("no media captured")
                    failed.append(sc)
            except Exception as e:
                print(f"error: {e}")
                failed.append(sc)
            page.wait_for_timeout(int(random.uniform(4000, 9000)))  # gentle pacing

        ctx.close()

    print(f"\nDone. saved={len(succeeded)}  failed={len(failed)}")
    if failed:
        print("Failed shortcodes:", failed)
        (OUT / "still_failed.txt").parent.mkdir(parents=True, exist_ok=True)
        (OUT / "still_failed.txt").write_text("\n".join(failed), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scrape"
    if cmd == "login":
        do_login()
    else:
        scrape()
