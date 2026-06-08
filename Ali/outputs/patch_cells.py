import json

with open('sentiment_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── CONFIG CELL ────────────────────────────────────────────────────────────────
CONFIG = """\
import os

# ============================ PLATFORM TOGGLE ============================
PLATFORM = "instagram"        # instagram | youtube | tiktok | facebook
# ========================================================================

# GCP / Vertex AI
GCP_PROJECT_ID = "gen-lang-client-0792749758"
GCP_LOCATION   = "us-central1"
GCS_BUCKET     = "afb_showreel"

# Prepared comments (ALL platforms; filtered to PLATFORM at load)
COMMENTS_ML_PATH  = f"gs://{GCS_BUCKET}/Preped_Comments/comments_ml.parquet"
COMMENTS_LLM_PATH = f"gs://{GCS_BUCKET}/Preped_Comments/comments_llm.jsonl"

# Multimodal context (Instagram only)
MEDIA_INDEX_PATH = f"gs://{GCS_BUCKET}/ig_multimodal_final.parquet"
MEDIA_GCS_PREFIX = "multimodal_dataset_fixed/"
ATTACH_MEDIA     = (PLATFORM == "instagram")

# ── MODEL TIERS ──────────────────────────────────────────────────────────────
# Flash -> high-volume zero-shot comment classification (THIS pipeline).
# Pro   -> reserved for persona consolidation in persona_pipeline.ipynb (stage 1).
MODEL_SENTIMENT = "gemini-2.5-flash"   # DO NOT switch to Pro here; volume is prohibitive.

# Generation config
TEMPERATURE       = 0.0
TOP_P             = 1.0
MAX_OUTPUT_TOKENS = 8192
THINKING_BUDGET   = 512   # light reasoning aids sarcasm/irony detection; 0 to disable

# Batching
COMMENTS_PER_REQUEST   = 50
MAX_COMMENTS           = None   # None = ALL
PRIORITIZE_MEDIA_POSTS = True
SAMPLE_SEED            = 42

# Media attachment caps (cost control)
MAX_IMAGES_PER_POST  = 3
MAX_FRAMES_PER_VIDEO = 11
INCLUDE_TRANSCRIPT   = True
MAX_TRANSCRIPT_CHARS = 1500
MAX_TEXT_CHARS       = 500

# GCS batch I/O
BATCH_INPUT_PREFIX    = "sentiment_batch/input/"
BATCH_OUTPUT_PREFIX   = "sentiment_batch/output/"
POLL_INTERVAL_SECONDS = 60

# Local artifacts
LOCAL_DIR     = "outputs"
MANIFEST_PATH = f"{LOCAL_DIR}/sentiment_manifest.json"
RESULTS_PATH  = f"{LOCAL_DIR}/sentiment_{PLATFORM}.parquet"
ROOM_LLM_PATH = f"{LOCAL_DIR}/room_llm_{PLATFORM}.parquet"
ROOM_PATH     = f"{LOCAL_DIR}/room_vibe_{PLATFORM}.parquet"

# |score| <= NEUTRAL_BAND -> "neutral"  (used in retrieve_sentiment + room vibe cells)
NEUTRAL_BAND = 0.15

os.makedirs(LOCAL_DIR, exist_ok=True)

print("Configuration loaded.")
print(f"  Platform : {PLATFORM}  (media: {'ON' if ATTACH_MEDIA else 'OFF'})")
print(f"  Model    : {MODEL_SENTIMENT}  (Pro reserved for persona_pipeline.ipynb)")
print(f"  Cap      : {MAX_COMMENTS} comments | {COMMENTS_PER_REQUEST}/request")
print(f"  Neutral  : +/-{NEUTRAL_BAND}")
"""

# ── REQUESTS CELL ─────────────────────────────────────────────────────────────
REQUESTS = """\
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SENTIMENT_SCHEMA
#   emotion: Plutchik's 8 core emotions + neutral.
#   Replaces 19-class GoEmotions to prevent class sparsity in downstream
#   Random Forest and UMAP clustering.
# ---------------------------------------------------------------------------
SENTIMENT_SCHEMA = (
    '{\"i\": int, '
    '\"sentiment\": \"positive|negative|neutral|mixed\", '
    '\"score\": float (-1.0 very negative .. 1.0 very positive), '
    '\"emotion\": \"joy|trust|fear|surprise|sadness|disgust|anger|anticipation|neutral\", '
    '\"intensity\": \"low|medium|high\", '
    '\"sarcasm\": true|false, '
    '\"toxicity\": \"none|mild|severe\", '
    '\"intent\": \"praise|affection|support|criticism|question|suggestion|tag_share|spam_promo|joke|other\", '
    '\"target\": \"creator|appearance|content_work|product|other_user|off_topic|none\", '
    '\"lang\": \"ISO-639-1 code, e.g. it,en\"}'
)

# ---------------------------------------------------------------------------
# ROOM_SCHEMA
#   sponsorship_alignment: audience receptiveness to the commercial integration
#     (meaningful only when post metadata signals an ad; return 0.5 otherwise).
#   split_axis: strictly <= 5 words for clean downstream categorical grouping.
# ---------------------------------------------------------------------------
ROOM_SCHEMA = (
    '{\"vibe\": \"celebratory|supportive|appreciative|amused|mixed|debated|divided|critical|hostile|spam_heavy|neutral\", '
    '\"consensus\": float 0..1 (1 = commenters strongly agree with each other), '
    '\"alignment\": float -1..1 (overall stance TOWARD the creator/post: 1 all support, -1 all against, 0 split), '
    '\"controversy\": float 0..1 (1 = strong opposing camps arguing), '
    '\"sponsorship_alignment\": float 0..1 (audience receptiveness to ad/commercial content; 0.5 if not sponsored), '
    '\"split_axis\": \"<=5 words naming what divides the room, or null if united\", '
    '\"dominant_stance\": \"the majority position in one short phrase\", '
    '\"summary\": \"1-2 sentences capturing the vibe of the room\"}'
)


def sentiment_system_prompt(platform: str) -> str:
    return (
        f"You are an expert multilingual social-media sentiment analyst. The comments below were "
        f"posted on {platform.upper()} under a single post by an Italian influencer. Many comments "
        f"are in Italian; some mix languages, slang, abbreviations and emojis.\\n"
        f"When POST MEDIA CONTEXT is given (images / video frames + transcript + metadata), USE IT: "
        f"the same words can be sincere or sarcastic depending on what the post shows, and 'target' "
        f"must reflect what the comment reacts to.\\n"
        f"Judge sentiment from the COMMENTER's standpoint toward the post/creator. Emojis carry "
        f"sentiment. Detect irony/sarcasm explicitly (a glowing phrase on a fail/joke video is often "
        f"sarcastic). 'score' is a float: -1.0 very negative .. +1.0 very positive.\\n"
        f"'emotion' must be exactly one of Plutchik's 8 core emotions or 'neutral': "
        f"joy, trust, fear, surprise, sadness, disgust, anger, anticipation, neutral.\\n"
        f"Then read the WHOLE comment set together to judge the 'vibe of the room'.\\n"
        f"'sponsorship_alignment': if post metadata marks this as a paid partnership or ad, rate "
        f"audience receptiveness to the commercial content (0=hostile, 1=enthusiastic). "
        f"Return 0.5 if the post is not sponsored.\\n"
        f"'split_axis': use AT MOST 5 words, or null if the room is united.\\n"
        f"Return ONLY a JSON object (no markdown, no preamble): "
        f'{{\"req\": str, \"items\": [<item>, ...], \"room\": <room>}}  with EXACTLY one item per comment.\\n'
        f"item schema: {SENTIMENT_SCHEMA}\\n"
        f"room schema: {ROOM_SCHEMA}"
    )


_FEATS = [("word_count", "w"), ("emoji_count", "e"), ("exclamation_count", "!"),
          ("question_count", "?"), ("mention_count", "@"), ("hashtag_count", "#")]

def _feat_tag(row) -> str:
    bits = []
    for col, sym in _FEATS:
        v = row.get(col)
        if pd.notna(v) and int(v) != 0:
            bits.append(f"{sym}{int(v)}")
    return ("[" + " ".join(bits) + "] ") if bits else ""

def build_sentiment_line(media_id, group_df, req_id: str):
    parts = [{"text": sentiment_system_prompt(PLATFORM)}]
    mparts = build_post_media_parts(media_id) if ATTACH_MEDIA else []
    parts.extend(mparts)
    has_media = bool(mparts)
    listing = [f"\\n--- COMMENTS TO ANALYSE  (req={req_id} | platform={PLATFORM} | "
               f"media_context={'yes' if has_media else 'no'}) ---"]
    order = []
    for i, (_, row) in enumerate(group_df.iterrows()):
        order.append(row["comment_id"])
        listing.append(f"[{i}] {_feat_tag(row)}{str(row['text'])[:MAX_TEXT_CHARS]}".rstrip())
    parts.append({"text": "\\n".join(listing)})
    parts.append({"text": f"\\nReturn the JSON object: 'req'='{req_id}', one item per comment "
                          f"(i = 0..{len(group_df) - 1}), and one 'room' read over all of them."})
    line = {"request": {"contents": [{"role": "user", "parts": parts}],
                        "generationConfig": gen_config_dict(MAX_OUTPUT_TOKENS, THINKING_BUDGET)}}
    man = {"req": req_id, "media_id": str(media_id), "has_media": has_media, "comment_ids": order}
    return line, man

def select_comments() -> pd.DataFrame:
    df = comments
    if MAX_COMMENTS and len(df) > MAX_COMMENTS:
        if PRIORITIZE_MEDIA_POSTS and ATTACH_MEDIA and posts_with_media:
            mask = df["media_id"].map(_norm_id).isin(posts_with_media)
            df = pd.concat([
                df[mask].sample(frac=1, random_state=SAMPLE_SEED),
                df[~mask].sample(frac=1, random_state=SAMPLE_SEED),
            ]).head(MAX_COMMENTS)
        else:
            df = df.sample(n=MAX_COMMENTS, random_state=SAMPLE_SEED)
    return df.reset_index(drop=True)

def build_requests(df: pd.DataFrame):
    lines, manifest, rid = [], {}, 0
    for media_id, g in df.groupby("media_id", sort=False):
        for s in range(0, len(g), COMMENTS_PER_REQUEST):
            chunk = g.iloc[s:s + COMMENTS_PER_REQUEST]
            req_id = f"R{rid}"
            line, man = build_sentiment_line(media_id, chunk, req_id)
            lines.append(line)
            manifest[req_id] = man
            rid += 1
    return lines, manifest

def submit_sentiment():
    df = select_comments()
    n_media = df["media_id"].map(_norm_id).isin(posts_with_media).sum() if (ATTACH_MEDIA and posts_with_media) else 0
    print(f"\\n{'='*60}\\nSENTIMENT submit -- {PLATFORM}\\n"
          f"  {len(df):,} comments | {df['media_id'].nunique():,} posts | "
          f"{n_media:,} with media context | model: {MODEL_SENTIMENT}\\n{'='*60}")
    lines, manifest = build_requests(df)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"[manifest] {len(manifest):,} requests -> {MANIFEST_PATH}")
    in_uri  = upload_to_gcs(write_jsonl(lines, f"{LOCAL_DIR}/sentiment_input.jsonl"),
                            GCS_BUCKET, BATCH_INPUT_PREFIX + "sentiment_input.jsonl")
    out_uri = f"gs://{GCS_BUCKET}/{BATCH_OUTPUT_PREFIX}"
    job = submit_batch_job(in_uri, out_uri, MODEL_SENTIMENT)
    record_batch_job("sentiment", job, out_uri)
    print(f"[submit] {len(lines):,} requests queued. Run retrieve_sentiment() when done.")
    return job

def _as_bool(v):
    return v is True or str(v).strip().lower() in ("true", "1", "yes")

def retrieve_sentiment(save: bool = True) -> pd.DataFrame:
    job = get_recorded_job("sentiment")
    if job.state != JobState.JOB_STATE_SUCCEEDED:
        print(f"Not ready (state={job.state}). Re-run later.")
        return None
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    rows, room_rows, bad = [], [], 0
    for t in retrieve_response_texts(job, GCS_BUCKET):
        if not t:
            bad += 1; continue
        try:
            obj = json.loads(strip_fences(t))
        except Exception:
            bad += 1; continue
        man = manifest.get(obj.get("req"))
        if not man:
            bad += 1; continue
        cids = man["comment_ids"]
        for it in (obj.get("items") or []):
            i = it.get("i")
            if not isinstance(i, int) or i < 0 or i >= len(cids):
                continue
            rows.append({
                "comment_id":      cids[i],
                "media_id":        man["media_id"],
                "media_context":   man["has_media"],
                "sentiment":       it.get("sentiment"),
                "sentiment_score": it.get("score"),
                "emotion":         it.get("emotion"),
                "intensity":       it.get("intensity"),
                "sarcasm":         _as_bool(it.get("sarcasm")),
                "toxicity":        it.get("toxicity"),
                "intent":          it.get("intent"),
                "target":          it.get("target"),
                "lang":            it.get("lang"),
            })
        rm = obj.get("room") or {}
        if rm:
            room_rows.append({
                "req":                       man["req"],
                "media_id":                  man["media_id"],
                "media_context":             man["has_media"],
                "n_comments_chunk":          len(cids),
                "llm_vibe":                  rm.get("vibe"),
                "llm_consensus":             pd.to_numeric(rm.get("consensus"),              errors="coerce"),
                "llm_alignment":             pd.to_numeric(rm.get("alignment"),              errors="coerce"),
                "llm_controversy":           pd.to_numeric(rm.get("controversy"),            errors="coerce"),
                "llm_sponsorship_alignment": pd.to_numeric(rm.get("sponsorship_alignment"),  errors="coerce"),
                "split_axis":                rm.get("split_axis"),
                "dominant_stance":           rm.get("dominant_stance"),
                "room_summary":              rm.get("summary"),
            })
    res = pd.DataFrame(rows).drop_duplicates("comment_id")
    res["sentiment_score"] = pd.to_numeric(res["sentiment_score"], errors="coerce")
    _band = NEUTRAL_BAND  # defined in config cell
    def _to_cat(row):
        s = str(row["sentiment"]).lower().strip() if pd.notna(row["sentiment"]) else ""
        if s == "positive": return "positive"
        if s == "negative": return "negative"
        if s == "neutral":  return "neutral"
        sc = row["sentiment_score"]
        if pd.isna(sc): return "neutral"
        return "positive" if sc > _band else "negative" if sc < -_band else "neutral"
    res["sentiment_cat"] = res.apply(_to_cat, axis=1)
    res["sarcasm_label"] = res["sarcasm"].map({True: "sarcastic", False: "not sarcastic"})
    print(f"[retrieve] {len(res):,} comments  |  {bad} failed/unparsed")
    res = res.merge(comments.drop(columns=["media_id"]), on="comment_id", how="left")
    room_llm = pd.DataFrame(room_rows)
    if save:
        res.to_parquet(RESULTS_PATH, index=False)
        print(f"[save] {RESULTS_PATH}")
        if len(room_llm):
            room_llm.to_parquet(ROOM_LLM_PATH, index=False)
            print(f"[save] {len(room_llm):,} room reads -> {ROOM_LLM_PATH}")
    return res

print("Sentiment functions ready:  submit_sentiment()  /  retrieve_sentiment()")
"""

patched = 0
for cell in nb['cells']:
    cid = cell.get('id')
    if cid == 'config':
        cell['source'] = CONFIG
        cell['outputs'] = []
        patched += 1
        print("Patched: config")
    elif cid == 'requests':
        cell['source'] = REQUESTS
        cell['outputs'] = []
        patched += 1
        print("Patched: requests")

with open('sentiment_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write('\n')

print(f"Done. {patched}/2 cells patched.")
