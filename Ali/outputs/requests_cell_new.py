import json
import logging
import numpy as np
from tqdm import tqdm

log = logging.getLogger("sentiment_pipeline")


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
# Design principles:
#   - Zero manual JSON stringification: field names are described in prose.
#   - Reasoning anchor: the model MUST emit the "reasoning" field first.
#     Because the schema enforces field order, all label tokens are generated
#     AFTER the scratchpad is committed — preventing label bleeding.
#   - Raw-valence extraction: the LLM labels true emotional valence of text
#     regardless of target. Brand-equity isolation is handled downstream in
#     calculate_brand_equity_scores() via vectorized Pandas, not prompt rules.
#   - Sarcasm-first gate: irony resolution happens before any label is
#     written, not as an afterthought.
# ---------------------------------------------------------------------------

def sentiment_system_prompt(platform: str) -> str:
    return (
        # ── Role ──────────────────────────────────────────────────────────────
        f"You are an expert multilingual social-media sentiment analyst "
        f"specialising in Italian influencer content on {platform.upper()}. "
        f"Comments you receive were posted under a single post. Many are in "
        f"Italian; expect slang, abbreviations, mixed languages, and emoji.\n\n"

        # ── Media context ─────────────────────────────────────────────────────
        f"MEDIA CONTEXT: When POST MEDIA CONTEXT is provided (images, video "
        f"frames, transcript, metadata), you MUST use it. The same phrase can "
        f"be sincere praise on a glamour post or cutting sarcasm on a fail "
        f"video. Let the media anchor your interpretation of ambiguous comments.\n\n"

        # ── Reasoning process ─────────────────────────────────────────────────
        f"REASONING PROCESS — write your reasoning field FIRST. Work through "
        f"these five steps in order before committing to any label:\n\n"

        f"  1. SARCASM CHECK\n"
        f"     Is the surface meaning the opposite of intent? Cues: hyperbole, "
        f"     ironic praise on a fail moment, emoji/word tone mismatch, "
        f"     deliberate understatement. If sarcastic, flip the inferred "
        f"     sentiment before labeling. Set sarcasm=true.\n\n"

        f"  2. TARGET CHECK\n"
        f"     Who or what is the comment actually directed at?\n"
        f"     — 'creator'       : the person posting\n"
        f"     — 'appearance'    : how the creator looks\n"
        f"     — 'content_work'  : the post itself, video, or brand deal work\n"
        f"     — 'product'       : a featured item or sponsor\n"
        f"     — 'other_user'    : a reply or attack directed at another commenter\n"
        f"     — 'off_topic'     : unrelated to the post or creator\n"
        f"     — 'none'          : no specific target\n\n"

        f"  3. SENTIMENT LABEL\n"
        f"     Evaluate the raw, genuine sentiment of the text exactly as it is "
        f"     written, regardless of who the target is. If a user is furiously "
        f"     attacking another commenter, label the sentiment as 'negative' and "
        f"     the target as 'other_user'. Do NOT artificially force it to neutral. "
        f"     Emojis carry semantic weight — treat them as words. "
        f"     Assign score ∈ [-1.0, +1.0] and the matching categorical label: "
        f"     'positive' (score > 0.15), 'negative' (score < -0.15), "
        f"     'neutral' (|score| <= 0.15).\n\n"

        f"  4. EMOTION & INTENSITY\n"
        f"     Pick the single best-fit Plutchik core emotion: joy, trust, fear, "
        f"     surprise, sadness, disgust, anger, anticipation. Use 'neutral' "
        f"     only when no clear emotion is detectable. Assign intensity: "
        f"     low (mild expression), medium (clear emotion), high (strong/intense).\n\n"

        f"  5. ROOM READ\n"
        f"     After scoring all individual comments, read them holistically. "
        f"     Fill the room object: choose a vibe label, estimate consensus "
        f"     [0,1], alignment [-1,1], controversy [0,1], and "
        f"     sponsorship_alignment [0,1] (set 0.5 if post is not sponsored). "
        f"     split_axis: at most 5 words naming what divides the room, or null "
        f"     if the room is united. dominant_stance: one short phrase. "
        f"     summary: 1-2 sentences.\n\n"

        # ── Output ────────────────────────────────────────────────────────────
        f"OUTPUT: Return a single JSON object with no markdown, no preamble, "
        f"no explanation. Structure:\n"
        f'  {{"req": "<req_id>", '
        f'"items": [<one SentimentItem per comment>], '
        f'"room": <one RoomRead>}}\n\n'
        f"Each SentimentItem must have fields in this exact order: "
        f"i, reasoning, sentiment, score, emotion, intensity, sarcasm, "
        f"toxicity, intent, target, lang.\n"
        f"i matches the [index] shown before each comment in the prompt."
    )


# ---------------------------------------------------------------------------
# Feature-tag helper — compact numeric annotation prepended to each comment
# ---------------------------------------------------------------------------

_FEATS = [
    ("word_count",        "w"),
    ("emoji_count",       "e"),
    ("exclamation_count", "!"),
    ("question_count",    "?"),
    ("mention_count",     "@"),
    ("hashtag_count",     "#"),
]

def _feat_tag(row) -> str:
    bits = [f"{sym}{int(row[col])}" for col, sym in _FEATS
            if pd.notna(row.get(col)) and int(row.get(col, 0)) != 0]
    return ("[" + " ".join(bits) + "] ") if bits else ""


# ---------------------------------------------------------------------------
# Step 4: JSONL Line Builder
# ---------------------------------------------------------------------------
# Returns a raw JSON *string* (not a dict) so callers write directly:
#     f.write(line + "\n")
#
# generationConfig is retrieved once per line from get_batch_generation_config_dict()
# which embeds the fully-flattened, enum-typed responseSchema.  The schema
# is the hardware-level logit mask; the system prompt is the reasoning guide.
# Both must be consistent — changing one without the other causes drift.
# ---------------------------------------------------------------------------

def build_batch_jsonl_line(
    req_id:         str,
    comments_chunk: "pd.DataFrame",
    platform:       str,
    media_id=None,
) -> tuple[str, dict]:
    """
    Build one Vertex AI Batch JSONL request line.

    Returns:
        line_str      -- raw JSON string ready for `f.write(line_str + "\\n")`
        manifest_entry -- dict for the manifest keyed by req_id
    """
    parts: list[dict] = [{"text": sentiment_system_prompt(platform)}]

    # Attach post media parts when available (IG multimodal path)
    mparts    = build_post_media_parts(media_id) if (ATTACH_MEDIA and media_id is not None) else []
    parts.extend(mparts)
    has_media = bool(mparts)

    # Comment listing — indices [0]..[n-1] match the 'i' field in SentimentItem
    header = (
        f"\n--- COMMENTS TO ANALYSE "
        f"(req={req_id} | platform={platform.upper()} | "
        f"media_context={'yes' if has_media else 'no'}) ---"
    )
    lines_buf = [header]
    order: list[str] = []

    for i, (_, row) in enumerate(comments_chunk.iterrows()):
        order.append(row["comment_id"])
        text = str(row["text"])[:MAX_TEXT_CHARS].rstrip()
        lines_buf.append(f"[{i}] {_feat_tag(row)}{text}")

    parts.append({"text": "\n".join(lines_buf)})
    parts.append({
        "text": (
            f"\nReturn the JSON object with req='{req_id}', "
            f"one item per comment (i = 0..{len(comments_chunk) - 1}), "
            f"and one room read across all of them."
        )
    })

    gen_cfg = get_batch_generation_config_dict()

    payload = {
        "request": {
            "contents":       [{"role": "user", "parts": parts}],
            "generationConfig": gen_cfg,
        }
    }

    # Serializability guard: raises TypeError here rather than at GCS upload time
    try:
        line_str = json.dumps(payload, ensure_ascii=False)
    except TypeError as exc:
        raise TypeError(
            f"[build_batch_jsonl_line] req={req_id} is not JSON-serializable: {exc}"
        ) from exc

    manifest_entry = {
        "req":         req_id,
        "media_id":    str(media_id) if media_id is not None else None,
        "has_media":   has_media,
        "comment_ids": order,
    }

    log.debug("built req=%s  comments=%d  media=%s", req_id, len(order), has_media)
    return line_str, manifest_entry


# ---------------------------------------------------------------------------
# Comment selection & request batching
# ---------------------------------------------------------------------------

def select_comments() -> "pd.DataFrame":
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


def build_requests(df: "pd.DataFrame") -> tuple[list[str], dict]:
    """Returns list of raw JSONL line strings + manifest dict."""
    line_strings, manifest, rid = [], {}, 0
    for media_id, g in df.groupby("media_id", sort=False):
        for s in range(0, len(g), COMMENTS_PER_REQUEST):
            chunk  = g.iloc[s : s + COMMENTS_PER_REQUEST]
            req_id = f"R{rid}"
            line_str, man = build_batch_jsonl_line(req_id, chunk, PLATFORM, media_id)
            line_strings.append(line_str)
            manifest[req_id] = man
            rid += 1
    log.info("built %d JSONL lines for %d posts", len(line_strings), df["media_id"].nunique())
    return line_strings, manifest


# ---------------------------------------------------------------------------
# Submit / retrieve
# ---------------------------------------------------------------------------

def write_jsonl_strings(lines: list[str], path: str) -> str:
    """Write pre-serialized JSONL strings to disk; return path."""
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    log.info("wrote %d JSONL lines -> %s", len(lines), path)
    return path


def submit_sentiment():
    df = select_comments()
    n_media = (
        df["media_id"].map(_norm_id).isin(posts_with_media).sum()
        if (ATTACH_MEDIA and posts_with_media) else 0
    )
    log.info(
        "SENTIMENT submit -- %s | %d comments | %d posts | %d with media | model: %s",
        PLATFORM, len(df), df["media_id"].nunique(), n_media, MODEL_SENTIMENT,
    )
    line_strings, manifest = build_requests(df)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    log.info("manifest: %d requests -> %s", len(manifest), MANIFEST_PATH)

    local_path = write_jsonl_strings(line_strings, f"{LOCAL_DIR}/sentiment_input.jsonl")
    in_uri  = upload_to_gcs(local_path, GCS_BUCKET, BATCH_INPUT_PREFIX + "sentiment_input.jsonl")
    out_uri = f"gs://{GCS_BUCKET}/{BATCH_OUTPUT_PREFIX}"
    job     = submit_batch_job(in_uri, out_uri, MODEL_SENTIMENT)
    record_batch_job("sentiment", job, out_uri)
    log.info("%d requests queued. Run retrieve_sentiment() when done.", len(line_strings))
    return job


def _as_bool(v):
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


def retrieve_sentiment(save: bool = True) -> "pd.DataFrame":
    job = get_recorded_job("sentiment")
    if job.state != JobState.JOB_STATE_SUCCEEDED:
        log.warning("Not ready (state=%s). Re-run later.", job.state)
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
                "reasoning":       it.get("reasoning"),
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
                "llm_consensus":             pd.to_numeric(rm.get("consensus"),             errors="coerce"),
                "llm_alignment":             pd.to_numeric(rm.get("alignment"),             errors="coerce"),
                "llm_controversy":           pd.to_numeric(rm.get("controversy"),           errors="coerce"),
                "llm_sponsorship_alignment": pd.to_numeric(rm.get("sponsorship_alignment"), errors="coerce"),
                "split_axis":                rm.get("split_axis"),
                "dominant_stance":           rm.get("dominant_stance"),
                "room_summary":              rm.get("summary"),
            })
    res = pd.DataFrame(rows).drop_duplicates("comment_id")
    res["sentiment_score"] = pd.to_numeric(res["sentiment_score"], errors="coerce")
    _band = NEUTRAL_BAND
    def _to_cat(row):
        s = str(row["sentiment"]).lower().strip() if pd.notna(row["sentiment"]) else ""
        if s in ("positive", "negative", "neutral"):
            return s
        sc = row["sentiment_score"]
        if pd.isna(sc):
            return "neutral"
        return "positive" if sc > _band else "negative" if sc < -_band else "neutral"
    res["sentiment_cat"] = res.apply(_to_cat, axis=1)
    res["sarcasm_label"] = res["sarcasm"].map({True: "sarcastic", False: "not sarcastic"})
    log.info("parsed %d comment sentiments | %d failed/unparsed", len(res), bad)
    res = res.merge(comments.drop(columns=["media_id"]), on="comment_id", how="left")
    room_llm = pd.DataFrame(room_rows)
    if save:
        res.to_parquet(RESULTS_PATH, index=False)
        log.info("saved per-comment sentiment -> %s", RESULTS_PATH)
        if len(room_llm):
            room_llm.to_parquet(ROOM_LLM_PATH, index=False)
            log.info("saved %d per-chunk room reads -> %s", len(room_llm), ROOM_LLM_PATH)
    return res


# ---------------------------------------------------------------------------
# Post-processing: Brand Equity Score Derivation
# ---------------------------------------------------------------------------
# Decouples LLM extraction from business logic.
#
# The LLM captures raw emotional valence for ALL comments — including
# inter-user attacks (other_user) and noise (off_topic / none).  Those
# signals are real text-level data but are irrelevant to brand/creator health.
#
# This function applies the isolation rule as a vectorized Pandas transform
# AFTER retrieval, keeping the LLM prompt clean and the business rule
# auditable, testable, and changeable without resubmitting a batch job.
#
# Input columns required:  sentiment, score (float), target (str)
# Output columns added:    brand_sentiment (str), brand_score (float)
# ---------------------------------------------------------------------------

#: Targets whose sentiment directly reflects audience opinion of the brand/creator.
_BRAND_TARGETS: frozenset[str] = frozenset(
    {"creator", "appearance", "content_work", "product"}
)

#: Targets whose sentiment is noise relative to brand health; zeroed out.
_NOISE_TARGETS: frozenset[str] = frozenset(
    {"other_user", "off_topic", "none"}
)


def calculate_brand_equity_scores(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Derive brand_sentiment and brand_score from raw LLM outputs.

    For comments directed at the brand/creator (target in _BRAND_TARGETS),
    the raw LLM sentiment and score pass through unchanged.

    For noise comments (target in _NOISE_TARGETS), brand_sentiment is set to
    'neutral' and brand_score to 0.0 — the comment carries no signal for
    brand health regardless of its actual emotional charge.

    Args:
        df: DataFrame produced by retrieve_sentiment(); must contain
            'sentiment', 'score', and 'target' columns.

    Returns:
        df with two new columns appended in-place: brand_sentiment, brand_score.
        The original sentiment / score columns are preserved for analysis.
    """
    if not {"sentiment", "score", "target"}.issubset(df.columns):
        raise ValueError(
            "calculate_brand_equity_scores requires columns: sentiment, score, target. "
            f"Got: {list(df.columns)}"
        )

    score  = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
    target = df["target"].astype(str).to_numpy()

    is_brand = np.isin(target, list(_BRAND_TARGETS))

    df["brand_sentiment"] = np.where(is_brand, df["sentiment"].astype(str), "neutral")
    df["brand_score"]     = np.where(is_brand, score, 0.0)

    n_brand = int(is_brand.sum())
    n_noise = len(df) - n_brand
    log.info(
        "brand equity scores computed: %d brand-directed | %d noise-zeroed",
        n_brand, n_noise,
    )
    return df


log.info("Sentiment functions ready:  submit_sentiment()  /  retrieve_sentiment()  /  calculate_brand_equity_scores()")
