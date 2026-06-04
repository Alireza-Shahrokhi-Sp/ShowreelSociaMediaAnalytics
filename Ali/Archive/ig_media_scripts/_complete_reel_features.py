"""
Build full enriched-feature rows for the 3 manually-verified reels from their
captured web-embed JSON (_api_dump/<sc>_reel.json), reusing ig_refetch.parse_item,
and append them to Output/ig_posts_multimodal_enriched.parquet so they become
first-class rows (no longer 'not_fetched').

Base metadata (media_id, engagement, timestamp, derived date parts) comes from
the cleaned parquet — the canonical Graph-API engagement, never the API snapshot.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ig_refetch import parse_item

BASE = Path(__file__).parent
DUMP = BASE / "_api_dump"
ENR  = BASE / "Output" / "ig_posts_multimodal_enriched.parquet"
CLEAN = BASE / "../Data/ig_posts_cleaned.parquet"
FIXED = BASE / "multimodal_dataset_fixed"
REELS = ["CRBdst5I1Go", "CX_Z-pwhDet", "DVeTxJEDnKP"]

# parse_item keys that are NOT in the enriched schema (unreliable API engagement)
DROP = {"ig_code", "like_count", "comment_count", "play_count", "view_count",
        "like_and_view_counts_disabled", "reshare_count", "share_count_disabled"}
RENAME = {"media_type": "media_type_api", "product_type": "product_type_api"}


def ffprobe_dur(mp4):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "csv=p=0", str(mp4)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


enr = pd.read_parquet(ENR)
cl = pd.read_parquet(CLEAN)
cl["shortcode"] = cl["permalink"].astype(str).str.rstrip("/").str.split("/").str[-1]
cl_base_cols = [c for c in enr.columns if c in cl.columns]  # base cols shared w/ cleaned

new_rows = []
for sc in REELS:
    node = json.loads((DUMP / f"{sc}_reel.json").read_text(encoding="utf-8"))
    feats = parse_item(node)
    # rename + drop to match enriched schema
    feats = {RENAME.get(k, k): v for k, v in feats.items() if k not in DROP}
    # duration fallback from the local mp4
    if not feats.get("video_duration"):
        feats["video_duration"] = ffprobe_dur(FIXED / "reel" / sc / f"{sc}.mp4")

    base = cl[cl["shortcode"] == sc].iloc[0]
    row = {c: base[c] for c in cl_base_cols}
    row["shortcode"] = sc
    row.update(feats)
    new_rows.append(row)
    print(f"{sc}: music={feats.get('music_source')} '{feats.get('song_title')}' by "
          f"{feats.get('artist')} | dur={feats.get('video_duration')} | "
          f"tagged={feats.get('n_tagged')} coauth={feats.get('n_coauthors')}")

add = pd.DataFrame(new_rows)
# align columns to enriched, fill any missing with NA
for c in enr.columns:
    if c not in add.columns:
        add[c] = pd.NA
add = add[enr.columns]

# drop any pre-existing rows for these reels, then append
enr2 = pd.concat([enr[~enr["shortcode"].isin(REELS)], add], ignore_index=True)
enr2.to_parquet(ENR, index=False)
print(f"\nEnriched: {len(enr)} -> {len(enr2)} rows (+{len(enr2)-len(enr)})")
