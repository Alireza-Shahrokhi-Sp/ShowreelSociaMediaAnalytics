"""
Download full-resolution Instagram images (and video-carousel slides) via
the private media-info API.

Uses the same cookie file as ig_refetch.py (instagram_cookies.txt).
Saves to playwright_downloads/{image,carousel}/<shortcode>/ — same layout
as the reconcile cell / fix_media_dataset.py.

Carousel slides that are videos are saved as .mp4; image slides as .jpg.

Usage:
    python ig_api_download.py
"""
import json
import logging
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ig_refetch import INFO_URL, UA, shortcode_to_pk, load_cookies, build_headers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("ig_api_dl")

BASE   = Path(__file__).parent
OUT    = BASE / "playwright_downloads"
COOKIE = BASE / "instagram_cookies.txt"

SLEEP_BASE   = 7.0
SLEEP_JITTER = 5.0

# Video-carousel posts confirmed available by the user
TARGETS = [
    ("Cu6xsKttA4T", "CAROUSEL_ALBUM"),
    ("DVqnRDaiP2_", "CAROUSEL_ALBUM"),
    ("BpCOgjTi6r_", "CAROUSEL_ALBUM"),
    ("BocKi58Cgol", "CAROUSEL_ALBUM"),
]


def fetch_item(shortcode: str, headers: dict) -> dict:
    pk  = shortcode_to_pk(shortcode)
    url = INFO_URL.format(pk=pk)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    items = payload.get("items") or []
    if not items:
        raise RuntimeError(f"empty items list (status={payload.get('status')})")
    return items[0]


def best_image_url(item: dict) -> str | None:
    candidates = (item.get("image_versions2") or {}).get("candidates") or []
    for c in sorted(candidates, key=lambda x: x.get("width", 0), reverse=True):
        if c.get("width", 0) > 200:
            return c["url"]
    return None


def best_video_url(item: dict) -> str | None:
    """Return the highest-resolution video URL from video_versions."""
    versions = item.get("video_versions") or []
    for v in sorted(versions, key=lambda x: x.get("width", 0) * x.get("height", 0), reverse=True):
        if v.get("url"):
            return v["url"]
    return None


def slide_files(slide: dict, sc: str, idx: int) -> list[tuple[str, str]]:
    """Return [(url, filename)] for one carousel slide (image or video)."""
    # media_type 2 = video, 1 = image/photo
    if slide.get("media_type") == 2:
        url = best_video_url(slide)
        return [(url, f"{sc}_{idx}.mp4")] if url else []
    else:
        url = best_image_url(slide)
        return [(url, f"{sc}_{idx}.jpg")] if url else []


def download_bytes(url: str) -> bytes | None:
    cdn_headers = {"User-Agent": UA, "Referer": "https://www.instagram.com/"}
    try:
        req = urllib.request.Request(url, headers=cdn_headers)
        with urllib.request.urlopen(req, timeout=40) as resp:
            return resp.read()
    except Exception as e:
        LOGGER.warning("CDN download failed: %s", e)
        return None


def main():
    if not COOKIE.exists():
        print(f"Cookie file not found: {COOKIE}")
        sys.exit(1)

    cookies = load_cookies(str(COOKIE))
    headers = build_headers(cookies)

    succeeded, failed = [], []

    for n, (sc, mt) in enumerate(TARGETS, 1):
        sub     = "carousel" if mt == "CAROUSEL_ALBUM" else "image"
        dest    = OUT / sub / sc
        label   = f"[{n}/{len(TARGETS)}] {sc} ({mt})"

        print(f"{label} ...", end=" ", flush=True)

        # Fetch media info from private API
        try:
            item = fetch_item(sc, headers)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print(f"\n401 — session expired. Re-export instagram_cookies.txt and retry.")
                sys.exit(1)
            elif e.code == 404:
                print(f"404 (deleted/private)")
                failed.append(sc)
            else:
                print(f"HTTP {e.code}")
                failed.append(sc)
            time.sleep(SLEEP_BASE + random.uniform(0, SLEEP_JITTER))
            continue
        except Exception as e:
            print(f"fetch error: {e}")
            failed.append(sc)
            time.sleep(SLEEP_BASE + random.uniform(0, SLEEP_JITTER))
            continue

        # Build list of (url, dest_filename) to download
        to_dl = []
        if mt == "IMAGE":
            url = best_image_url(item)
            if url:
                to_dl.append((url, f"{sc}_0.jpg"))
        else:  # CAROUSEL_ALBUM — slides may be images OR videos
            for i, slide in enumerate(item.get("carousel_media") or []):
                to_dl.extend(slide_files(slide, sc, i))

        if not to_dl:
            print("no media URLs in API response")
            failed.append(sc)
            time.sleep(SLEEP_BASE + random.uniform(0, SLEEP_JITTER))
            continue

        slide_summary = ", ".join(
            ("vid" if f.endswith(".mp4") else "img") for _, f in to_dl
        )
        print(f"[{slide_summary}] ...", end=" ", flush=True)

        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)

        saved = 0
        for url, fname in to_dl:
            data = download_bytes(url)
            if data and len(data) > 5000:
                (dest / fname).write_bytes(data)
                saved += 1

        if saved == len(to_dl):
            print(f"saved {saved} file(s)")
            succeeded.append(sc)
        elif saved > 0:
            print(f"partial: saved {saved}/{len(to_dl)}")
            succeeded.append(sc)
        else:
            print("nothing saved (CDN blocked?)")
            failed.append(sc)

        time.sleep(SLEEP_BASE + random.uniform(0, SLEEP_JITTER))

    print(f"\nDone.  succeeded={len(succeeded)}  failed={len(failed)}")
    if failed:
        print("Failed:", failed)
        (OUT / "api_failed.txt").write_text("\n".join(failed), encoding="utf-8")


if __name__ == "__main__":
    main()
