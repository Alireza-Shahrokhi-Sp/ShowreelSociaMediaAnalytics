import pandas as pd

def _comments_loaded() -> bool:
    try:
        df = globals()["comments"]
    except KeyError:
        return False
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False
    if "platform" in df.columns and df["platform"].iloc[0] != PLATFORM:
        return False
    if "text" not in df.columns:
        return False
    mi = globals().get("media_index", "__missing__")
    if ATTACH_MEDIA and mi is None:
        return False
    if not ATTACH_MEDIA and mi != "__missing__" and mi is not None:
        return False
    return True


if _comments_loaded():
    print(
        f"[data] already loaded for platform={PLATFORM!r} — skipping GCS reads.\n"
        f"  comments: {len(comments):,} | authors: {comments['author_id'].nunique():,} "
        f"| posts: {comments['media_id'].nunique():,}\n"
        f"  media_index: {'loaded' if media_index is not None else 'OFF'}"
    )
else:
    print(f"[data] loading prepared comments for platform={PLATFORM!r} ...")

    # 1) Comment feature matrix — predicate-pushdown filter to PLATFORM (skips other platforms).
    comments = pd.read_parquet(COMMENTS_ML_PATH, filters=[("platform", "==", PLATFORM)])
    comments["author_id"] = comments["author_id"].astype(str)
    comments["media_id"]  = comments["media_id"].astype(str)
    print(f"  comments_ml[{PLATFORM}]: {len(comments):,} comments | "
          f"{comments['author_id'].nunique():,} authors | {comments['media_id'].nunique():,} posts")

    # 2) Raw text for EVERY comment (comments_llm has no author_id -> join by comment_id).
    print(f"  loading text from {COMMENTS_LLM_PATH} ...")
    _chunks = []
    for _c in pd.read_json(COMMENTS_LLM_PATH, lines=True, chunksize=200_000):
        if "platform" in _c.columns:
            _c = _c[_c["platform"] == PLATFORM]
        if len(_c):
            _chunks.append(_c[["comment_id", "text"]])
    llm_text = pd.concat(_chunks, ignore_index=True) if _chunks else pd.DataFrame(columns=["comment_id", "text"])
    comments = comments.merge(llm_text, on="comment_id", how="left")
    comments["text"] = comments["text"].fillna("").astype(str)
    print(f"  text attached to {(comments['text'].str.len() > 0).sum():,}/{len(comments):,} comments")

    # 3) Instagram-only multimodal context: post metadata + media index (bare-numeric media_id).
    if ATTACH_MEDIA:
        media_index = pd.read_parquet(MEDIA_INDEX_PATH)
        media_index = media_index[media_index["media_id"].notna()].copy()
        media_index["media_id"] = media_index["media_id"].astype(str)
        print(f"  media_index: {len(media_index):,} posts | {media_index['shortcode'].nunique():,} shortcodes")
    else:
        media_index = None
        print(f"  media context OFF for platform={PLATFORM} -> text-only run")

    # 4) Keep only comments that actually have text to analyse.
    _before = len(comments)
    comments = comments[comments["text"].str.strip() != ""].reset_index(drop=True)
    print(f"\n[data] ready: {len(comments):,}/{_before:,} comments with text")
