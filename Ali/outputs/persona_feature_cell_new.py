import numpy as np
from datetime import timedelta

# 1. Binary flags from pre-computed counts
ig_comments["has_emoji"]    = (ig_comments["emoji_count"] > 0).astype(int)
ig_comments["has_question"] = (ig_comments["question_count"] > 0).astype(int)
ig_comments["has_exclaim"]  = (ig_comments["exclamation_count"] > 0).astype(int)
ig_comments["is_reply"]     = ig_comments["reply_to_comment_id"].notna().astype(int)
print("Binary flags derived.")

# 2. Temporal features vs. post time (uses Instagram post metadata when available)
ig_comments["timestamp"] = pd.to_datetime(ig_comments["timestamp"], errors="coerce", utc=True)
if ATTACH_MEDIA and ig_media is not None and "timestamp" in ig_media.columns:
    pm = ig_media[["media_id", "timestamp"]].rename(columns={"timestamp": "post_timestamp"})
    ig_comments = ig_comments.merge(pm, on="media_id", how="left")
    ig_comments["post_timestamp"]  = pd.to_datetime(ig_comments["post_timestamp"], errors="coerce", utc=True)
    ig_comments["hours_to_comment"] = (
        (ig_comments["timestamp"] - ig_comments["post_timestamp"]).dt.total_seconds() / 3600
    ).clip(lower=0)
else:
    ig_comments["hours_to_comment"] = np.nan
    print("   (no post metadata -> hours_to_comment = NaN)")


# 3. Room-vibe enrichment — load post-level sentiment summary from sentiment pipeline output
import logging, os

_log = logging.getLogger("persona_pipeline")

def load_post_vibes(path: str) -> "pd.DataFrame | None":
    """
    Load post-level room-vibe metrics written by sentiment_pipeline.ipynb.
    Expected columns: media_id, room_vibe, room_consensus, room_sponsorship_alignment.
    Returns None (and logs a warning) if the file is absent — pipeline continues
    without room-vibe features so it does not block pre-sentiment runs.
    """
    if not os.path.exists(path):
        _log.warning(
            "POST_VIBES_PATH not found (%s) — room-vibe features will be absent. "
            "Run sentiment_pipeline retrieve step first.", path
        )
        return None
    df = pd.read_parquet(path, columns=["media_id", "room_vibe", "room_consensus", "room_sponsorship_alignment"])
    df["media_id"] = df["media_id"].astype(str)
    _log.info("post_vibes loaded: %d posts", len(df))
    return df

_post_vibes = load_post_vibes(POST_VIBES_PATH)


# 4. User-level aggregation
def build_user_feature_matrix(df: pd.DataFrame, post_vibes: "pd.DataFrame | None" = None) -> pd.DataFrame:
    grp = df.groupby("author_id")
    agg = {
        "total_comments":          grp.size(),
        "unique_posts_commented":  grp["media_id"].nunique(),
        "total_replies_made":      grp["is_reply"].sum(),
        "reply_ratio":             grp["is_reply"].mean(),
        "mean_hours_to_comment":   grp["hours_to_comment"].mean(),
        "median_hours_to_comment": grp["hours_to_comment"].median(),
        "pct_comments_under_1h":   grp["hours_to_comment"].apply(lambda x: (x < 1).mean()),
        "pct_comments_under_24h":  grp["hours_to_comment"].apply(lambda x: (x < 24).mean()),
        "activity_span_days":      grp["timestamp"].apply(lambda x: (x.max() - x.min()).days if x.notna().any() else 0),
        "mean_word_count":         grp["word_count"].mean(),
        "mean_mention_count":      grp["mention_count"].mean(),
        "emoji_usage_rate":        grp["has_emoji"].mean(),
        "question_rate":           grp["has_question"].mean(),
        "exclamation_rate":        grp["has_exclaim"].mean(),
    }
    feat = pd.DataFrame(agg).reset_index()
    feat["post_concentration_ratio"] = (
        feat["unique_posts_commented"] / feat["total_comments"]
    ).clip(upper=1.0)
    # Fill timing features that are NaN when post metadata is absent (non-IG platforms).
    for c in ["mean_hours_to_comment", "median_hours_to_comment",
              "pct_comments_under_1h", "pct_comments_under_24h"]:
        feat[c] = feat[c].fillna(0)

    # ── Room-vibe enrichment (IG only; skipped gracefully when post_vibes is None) ──
    if post_vibes is not None:
        # Join comments to per-post vibe metrics, then aggregate per user.
        enriched = df[["author_id", "media_id"]].merge(
            post_vibes, on="media_id", how="left"
        )
        vibe_grp = enriched.groupby("author_id")

        # mean consensus and sponsorship tolerance across posts the user engaged with
        feat = feat.merge(
            vibe_grp["room_consensus"].mean().rename("mean_engaged_consensus"),
            on="author_id", how="left"
        )
        feat = feat.merge(
            vibe_grp["room_sponsorship_alignment"].mean().rename("mean_sponsorship_tolerance"),
            on="author_id", how="left"
        )
        # dominant vibe: the single most-frequent room_vibe across that user's engaged posts
        dominant_vibe = (
            vibe_grp["room_vibe"]
            .agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "neutral")
            .rename("dominant_room_vibe")
        )
        feat = feat.merge(dominant_vibe, on="author_id", how="left")
        feat["dominant_room_vibe"] = feat["dominant_room_vibe"].fillna("neutral").astype(str)

        n_enriched = feat["mean_engaged_consensus"].notna().sum()
        _log.info("room-vibe features attached for %d / %d users", n_enriched, len(feat))
        print(f"  room-vibe enrichment: {n_enriched:,}/{len(feat):,} users have vibe data")
    else:
        # Sentinel columns so downstream cells don't KeyError when vibes are unavailable.
        feat["mean_engaged_consensus"]    = np.nan
        feat["mean_sponsorship_tolerance"] = np.nan
        feat["dominant_room_vibe"]         = "neutral"
        print("  room-vibe enrichment skipped (post_vibes not loaded)")

    return feat


user_features = build_user_feature_matrix(ig_comments, post_vibes=_post_vibes)
print(f"User feature matrix built: {len(user_features):,} users.")

# 5. Attach representative comment text from comments_llm (filter to PLATFORM, join by comment_id)
print(f"Loading comment text from {COMMENTS_LLM_PATH} ...")
_text_chunks = []
for _chunk in pd.read_json(COMMENTS_LLM_PATH, lines=True, chunksize=200_000):
    if "platform" in _chunk.columns:
        _chunk = _chunk[_chunk["platform"] == PLATFORM]
    if len(_chunk):
        _text_chunks.append(_chunk[["comment_id", "text"]])
llm_text = (pd.concat(_text_chunks, ignore_index=True)
            if _text_chunks else pd.DataFrame(columns=["comment_id", "text"]))

txt = ig_comments[["comment_id", "author_id"]].merge(llm_text, on="comment_id", how="inner")
top_comments = (
    txt.groupby("author_id").head(5)
       .groupby("author_id")["text"]
       .apply(lambda ts: " ||| ".join(ts.astype(str).tolist()))
       .reset_index()
       .rename(columns={"text": "top_comments_sample"})
)
user_features = user_features.merge(top_comments, on="author_id", how="left")
print(f"Representative comment history attached for {top_comments.shape[0]:,} users.")
