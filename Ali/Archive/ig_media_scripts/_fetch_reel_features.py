"""
Fetch enriched features (music, tagged accounts, coauthors, audio, commercial)
for the 3 manually-verified reels by capturing their web-embed media JSON in the
logged-in Playwright profile. Single reels carry top-level clips_metadata (unlike
carousels, where the container strips it).

Dumps raw node to _api_dump/<sc>_reel.json and prints the parsed features.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = Path(__file__).parent
USER_DATA_DIR = str((BASE / "playwright_ig_profile").resolve())
DUMP = BASE / "_api_dump"; DUMP.mkdir(exist_ok=True)
REELS = ["CRBdst5I1Go", "CX_Z-pwhDet", "DVeTxJEDnKP"]


def _clear_lock():
    for name in ("SingletonLock", "lockfile"):
        lp = Path(USER_DATA_DIR) / name
        try:
            if lp.exists(): lp.unlink()
        except Exception: pass


def find_media(obj, code):
    """Find the media node whose code==code and that has video_versions/clips_metadata."""
    def walk(o):
        if isinstance(o, dict):
            if o.get("code") == code and ("video_versions" in o or "clips_metadata" in o
                                          or "image_versions2" in o):
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

    for sc in REELS:
        cap = {"n": None}
        def on_resp(resp):
            if "application/json" not in resp.headers.get("content-type", ""):
                return
            try:
                t = resp.text()
            except Exception:
                return
            if sc not in t and "clips_metadata" not in t:
                return
            try:
                d = json.loads(t)
            except Exception:
                return
            nn = find_media(d, sc)
            if nn:
                cap["n"] = nn
        page.on("response", on_resp)
        try:
            page.goto(f"https://www.instagram.com/reel/{sc}/", wait_until="networkidle", timeout=40000)
            page.wait_for_timeout(3000)
            if cap["n"] is None:
                scripts = page.evaluate(
                    "() => [...document.querySelectorAll('script[type=\"application/json\"]')].map(s=>s.textContent)")
                for txt in scripts:
                    if txt and sc in txt:
                        try:
                            nn = find_media(json.loads(txt), sc)
                            if nn: cap["n"] = nn; break
                        except Exception:
                            pass
        finally:
            page.remove_listener("response", on_resp)

        n = cap["n"]
        if not n:
            print(f"{sc}: NO node captured"); page.wait_for_timeout(4000); continue
        (DUMP / f"{sc}_reel.json").write_text(json.dumps(n, ensure_ascii=False), encoding="utf-8")

        clips = n.get("clips_metadata") or {}
        mi = (clips.get("music_info") or {}).get("music_asset_info") or {}
        osi = clips.get("original_sound_info") or {}
        if mi:
            msrc, title, artist = "licensed", mi.get("title"), mi.get("display_artist")
        elif osi:
            msrc, title, artist = "original", osi.get("original_audio_title"), (osi.get("ig_artist") or {}).get("username")
        else:
            msrc, title, artist = "none", None, None
        tagged = [ (u.get("user") or {}).get("username")
                   for u in ((n.get("usertags") or {}).get("in")) or [] ]
        coauth = [ c.get("username") for c in (n.get("coauthor_producers") or []) ]
        print(f"{sc}: clips_metadata={'yes' if clips else 'NO'} music={msrc} "
              f"title={title!r} artist={artist!r} tagged={tagged} coauthors={coauth} "
              f"has_audio={n.get('has_audio')} dur={n.get('video_duration')}")
        page.wait_for_timeout(4000)

    ctx.close()
print("DONE")
