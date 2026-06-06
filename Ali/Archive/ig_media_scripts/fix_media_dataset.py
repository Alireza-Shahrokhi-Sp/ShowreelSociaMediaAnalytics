"""
Replace stale/wrong image+carousel files in multimodal_dataset_fixed with the
correct full-resolution files already downloaded by ig_api_download.py.

What it does per shortcode:
  1. Reads the post timestamp from ig_posts_multimodal_enriched.parquet
  2. Removes ALL .jpg and .json files from multimodal_dataset_fixed/{image,carousel}/{sc}/
  3. Copies from playwright_downloads/{image,carousel}/{sc}/ with proper UTC naming
  4. Writes a fresh .json sidecar (same format as the reconcile cell)

Run once — idempotent (safe to re-run).
"""
import json
import shutil
from pathlib import Path

import pandas as pd

BASE  = Path(__file__).parent
PW    = BASE / "playwright_downloads"
FIXED = BASE / "multimodal_dataset_fixed"
PARQ  = BASE / "Output" / "ig_posts_multimodal_enriched.parquet"

df = pd.read_parquet(PARQ, columns=["shortcode", "media_type", "timestamp",
                                     "permalink", "media_id", "caption",
                                     "like_count", "comments_count"])
meta = {r["shortcode"]: r for _, r in df.iterrows()}

TARGETS = [
    ("Cu6xsKttA4T", "CAROUSEL_ALBUM"),
    ("DVqnRDaiP2_", "CAROUSEL_ALBUM"),
    ("BpCOgjTi6r_", "CAROUSEL_ALBUM"),
    ("BocKi58Cgol", "CAROUSEL_ALBUM"),
]


def date_str(sc: str) -> str:
    r = meta.get(sc)
    if r is None:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    return pd.to_datetime(r["timestamp"], utc=True).strftime("%Y-%m-%d_%H-%M-%S_UTC")


def meta_dict(sc: str) -> dict:
    r = meta.get(sc)
    permalink = f"https://www.instagram.com/p/{sc}/"
    d = {"id": sc, "shortcode": sc, "webpage_url": permalink,
         "original_url": permalink, "url": permalink, "source": "playwright"}
    if r is not None:
        for k in ("media_id", "caption", "media_type", "timestamp",
                  "like_count", "comments_count"):
            v = r.get(k)
            if v is not None and pd.notna(v):
                d[k] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return d


ok = err = 0

for sc, mt in TARGETS:
    sub = "carousel" if mt == "CAROUSEL_ALBUM" else "image"
    src_dir  = PW    / sub / sc
    dest_dir = FIXED / sub / sc

    # Source must exist and have media files
    jpgs = sorted(src_dir.glob("*.jpg")) if src_dir.exists() else []
    vids = sorted(src_dir.glob("*.mp4")) if src_dir.exists() else []
    if not jpgs and not vids:
        print(f"  SKIP  {sc} — no media in playwright_downloads/{sub}/{sc}/")
        err += 1
        continue

    # Wipe old jpg + json in destination (keep anything else like frames/)
    if dest_dir.exists():
        for f in dest_dir.iterdir():
            if f.suffix in (".jpg", ".json") and f.is_file():
                f.unlink()
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)

    ds = date_str(sc)

    if mt == "IMAGE":
        # Pick the largest jpg (only one expected, but be safe)
        best = max(jpgs, key=lambda p: p.stat().st_size)
        shutil.copy(best, dest_dir / f"{ds}.jpg")
        (dest_dir / f"{ds}.json").write_text(
            json.dumps(meta_dict(sc), ensure_ascii=False, indent=2), encoding="utf-8")
        from PIL import Image
        with Image.open(dest_dir / f"{ds}.jpg") as im:
            w, h = im.size
        print(f"  OK  image    {sc}  {w}x{h}  -> image/{sc}/{ds}.jpg")

    else:  # CAROUSEL_ALBUM — slides may be .jpg or .mp4
        all_slides = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.mp4"))
        # sort by numeric index in filename (sc_0.jpg, sc_1.mp4, ...)
        def slide_idx(p):
            try:
                return int(p.stem.rsplit("_", 1)[-1])
            except ValueError:
                return 999
        all_slides = sorted(all_slides, key=slide_idx)

        slide_info = []
        for i, src in enumerate(all_slides, 1):
            ext = src.suffix  # .jpg or .mp4
            dest_name = f"{ds}_{i}{ext}"
            shutil.copy(src, dest_dir / dest_name)
            if ext == ".jpg":
                from PIL import Image
                with Image.open(dest_dir / dest_name) as im:
                    slide_info.append(f"{im.width}x{im.height}")
            else:
                slide_info.append(f"video({src.stat().st_size // 1024}KB)")

        (dest_dir / f"{ds}.json").write_text(
            json.dumps(meta_dict(sc), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  OK  carousel {sc}  {len(all_slides)} slides -> carousel/{sc}/")
        print(f"        {' '.join(slide_info)}")

    ok += 1

print(f"\nDone. fixed={ok}  skipped/err={err}")
