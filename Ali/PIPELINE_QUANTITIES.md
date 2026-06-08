# Sentiment & Persona Pipeline: Quantities Reference

This document lists all quantities (fields, metrics, computed values) calculated across the sentiment and persona pipelines.

---

## Sentiment Pipeline

### Per-Comment Level (Comment-Level Table)

Direct outputs from LLM sentiment analysis per individual comment:

| Quantity | Type | Range | Description |
|----------|------|-------|-------------|
| `comment_id` | string | — | Unique comment identifier |
| `media_id` | integer | — | Instagram post/reel ID |
| `sentiment` | categorical | positive, negative, neutral, mixed | LLM's categorical sentiment judgment |
| `sentiment_score` | float | -1.0 to +1.0 | Numeric sentiment: -1 very negative, 0 neutral, +1 very positive |
| `emotion` | categorical | (varies) | Emotion detected: joy, sadness, anger, fear, surprise, disgust, trust, anticipation, etc. |
| `intensity` | float | 0.0 to 1.0 | Strength of the detected emotion |
| `sarcasm` | boolean | True / False | Whether comment contains sarcasm |
| `toxicity` | float | 0.0 to 1.0 | Toxicity/abusiveness score |
| `intent` | categorical | (varies) | Commenter's intent: praise, criticism, question, joke, support, etc. |
| `target` | categorical | content_work, creator, other_user, appearance, product, off_topic, other, none | Who/what is the comment directed at |
| `lang` | string | language code | Detected language |
| `platform` | string | instagram, tiktok, etc. | Source platform |

### Per-Comment Derived (Categorization)

Derived from the LLM output and sentiment_score:

| Quantity | Type | Definition |
|----------|------|-----------|
| `sentiment_cat` | categorical: positive, negative, neutral | Collapsed 3-class sentiment using NEUTRAL_BAND threshold (±0.15): score > +0.15 → positive; score in [-0.15, +0.15] → neutral; score < -0.15 → negative |
| `sarcasm_label` | categorical | "sarcastic" or "not sarcastic" (from boolean `sarcasm`) |
| `toxicity_label` | categorical | "high" (≥0.7), "moderate" (0.3–0.7), "low" (<0.3), or "unknown" (if NA) |

### Text Feature Engineering (Per-Comment)

Quantitative features derived from comment text:

| Quantity | Type | Description |
|----------|------|-------------|
| `text_length` | integer | Character count of comment |
| `word_count` | integer | Number of words |
| `emoji_count` | integer | Total emoji count |
| `unique_emoji_count` | integer | Count of distinct emojis used |
| `emoji_entropy` | float | Shannon entropy of emoji distribution (how varied) |
| `emoji_variety_ratio` | float | unique_emoji_count / emoji_count |
| `emoji_per_word_ratio` | float | emoji_count / word_count |
| `url_count` | integer | Number of URLs/links in comment |
| `mention_count` | integer | Number of @mentions |
| `hashtag_count` | integer | Number of hashtags |
| `exclamation_count` | integer | Number of exclamation marks |
| `question_count` | integer | Number of question marks |
| `avg_word_length` | float | Mean characters per word |
| `has_numbers` | boolean | Contains numeric digits |
| `has_links` | boolean | Contains any links |

**Output table**: `sentiment_instagram.parquet` (one row per comment)

---

### Post-Level Room Vibe Analysis

**Two complementary approaches** to quantify "vibe of the room" under each post:

#### A. LLM Narrative Room Read (Per-Chunk)

Direct LLM judgment across all comments in a chunk, returned as structured JSON:

| Quantity | Type | Range | Description |
|----------|------|-------|-------------|
| `llm_vibe` | categorical | celebratory, supportive, appreciative, amused, mixed, debated, divided, critical, hostile, spam_heavy, neutral | LLM's qualitative label for the room atmosphere |
| `llm_consensus` | float | 0.0 to 1.0 | How strongly commenters agree with each other (1 = unanimous) |
| `llm_alignment` | float | -1.0 to +1.0 | Overall stance toward creator/post: +1 = all support, -1 = all against, 0 = split |
| `llm_controversy` | float | 0.0 to 1.0 | Strength of opposing viewpoint camps arguing (1 = strong debate) |
| `split_axis` | string or null | — | What divides the room (if controversial), e.g. "whether the cat was truly stolen" |
| `dominant_stance` | string | — | The majority position in one short phrase |
| `room_summary` | string | — | 1–2 sentence narrative capture of the room vibe |

**Note**: Merged at post level using the largest comment chunk per post as most representative.

#### B. Computed Room Metrics (Per-Post Aggregation)

Quantitative metrics derived by aggregating comment-level sentiment scores:

| Quantity | Type | Calculation |
|----------|------|-----------|
| `vibe_score` | float, -1.0 to +1.0 | Mean of all `sentiment_score` values for the post |
| `dispersion` | float, 0.0+ | Standard deviation of sentiment_score (0 = all same, high = divergent opinions) |
| `pos_frac` | float, 0.0 to 1.0 | Fraction of comments with sentiment_cat = "positive" |
| `neg_frac` | float, 0.0 to 1.0 | Fraction of comments with sentiment_cat = "negative" |
| `neu_frac` | float, 0.0 to 1.0 | Fraction of comments with sentiment_cat = "neutral" |
| `stance_alignment` | float, -1.0 to +1.0 | Mean of sign(sentiment_score) per comment: +1 = all positive, -1 = all negative, 0 = split |
| `polarization` | float, 0.0 to 1.0 | Measure of opinion spread (high = strong opposing camps) |
| `consensus` | float, 0.0 to 1.0 | Agreement level among commenters |
| `sarcasm_rate` | float, 0.0 to 1.0 | Fraction of sarcastic comments |
| `toxicity_mean` | float, 0.0 to 1.0 | Mean toxicity score across comments |
| `room_state` | categorical | "united" (high consensus), "mixed" (neutral), or "divided" (high polarization/controversy) |
| `media_context` | boolean | Whether post contained media (image, video, carousel) |
| `n_comments` | integer | Total comments analyzed for this post |

**Output table**: `room_vibe_instagram.parquet` (one row per post, merged with LLM reads)

---

### Post-Level Promotion Analysis

For sponsored/promotional posts only:

| Quantity | Type | Description |
|----------|------|-------------|
| `is_promo` | boolean | Marked as promotional content |
| `sponsor` | string | Brand/sponsor name |
| `promo_type` | categorical | "paid_partnership", "caption_disclosure", "ad", etc. |
| `vibe_score` | float | Mean sentiment for promo post |
| `pos_frac`, `neg_frac`, `sarcasm_rate` | float | Composition and sarcasm rate |
| `reception` | categorical | "well_received" (score ≥ 0.4, neg_frac < 0.15, sarcasm_rate < 0.15), "poorly_received" (score < 0.1 OR neg_frac ≥ 0.30 OR sarcasm_rate ≥ 0.25), or "mixed" |

**Output table**: `promo_reception_instagram.parquet`

---

## Persona Pipeline

### Stage 1: Candidate Persona Discovery

LLM identifies audience archetypes from aggregated commenter behavior:

| Quantity | Type | Description |
|----------|------|-------------|
| `persona_codename` | string (UPPER_SNAKE_CASE) | Short identifier, e.g., "SUPERFAN", "CRITICAL_ANALYST" |
| `persona_label` | string (Title Case) | Human-readable label, e.g., "Super Fan" |
| `description` | string | Narrative description of this persona archetype |
| `signal_markers` | list of strings | Behavioral/linguistic signals that identify this persona, e.g., ["excessive praise emojis", "asks for merch"] |
| `example_comments` | list of strings | 2–3 representative comments from this archetype |
| `frequency_estimate` | categorical or float | Rough prevalence: "common", "moderate", "rare", or percentage |

**Output**: `taxonomy_instagram.json` — consolidated MECE taxonomy of ≤ 12 personas

---

### Stage 2: Per-User Classification

Each commenter assigned to exactly one persona:

| Quantity | Type | Range | Description |
|----------|------|-------|-------------|
| `author_id` | integer | — | Instagram user ID |
| `persona_codename` | string | — | Assigned persona (from final taxonomy) |
| `persona_label` | string | — | Human-readable persona name |
| `confidence` | float | 0.0 to 1.0 | LLM confidence in assignment (0.4 = threshold for "insufficient data") |
| `justification` | string | — | Brief explanation for the assignment |
| `media_ids` | list of integers | — | Posts this user commented on (used for classification) |
| `comment_count` | integer | — | Total comments by this user in dataset |
| `avg_sentiment_score` | float | -1.0 to +1.0 | Mean sentiment of their comments |
| `dominant_emotion` | categorical | — | Most frequent emotion in their comments |
| `sarcasm_rate` | float | 0.0 to 1.0 | Fraction of their comments that are sarcastic |
| `target_preference` | categorical | — | What they most frequently comment about (content_work, creator, product, etc.) |

**Output table**: `user_personas.parquet` (one row per unique commenter)

---

## Configuration Parameters

These control behavior across pipelines:

| Parameter | Value | Use |
|-----------|-------|-----|
| `NEUTRAL_BAND` | 0.15 | Threshold for 3-class sentiment categorization: scores in [-0.15, +0.15] are "neutral" |
| `MODEL_SENTIMENT` | gemini-2.5-pro | LLM for per-comment sentiment analysis |
| `MODEL_PERSONA` | gemini-2.5-pro | LLM for persona discovery and classification |
| `BATCH_SIZE` | (varies) | Comments per batch request to LLM |
| `target_personas` | 12 | Maximum personas in final MECE taxonomy |

---

## Cross-Pipeline Integration

### Joining Sentiment → Persona

```
sentiment_instagram.parquet
  ↓ (group by author_id, aggregate sentiment features)
  ↓ (classify via LLM into persona)
user_personas.parquet
```

**Derivable analysis**: sentiment patterns per persona, which personas engage with which content, persona evolution over time.

---

## Key Thresholds & Defaults

- **Neutral band**: ±0.15 sentiment score
- **Toxicity "high"**: ≥ 0.7
- **Toxicity "moderate"**: 0.3–0.7
- **Well-received promo**: vibe_score ≥ 0.4 AND neg_frac < 0.15 AND sarcasm_rate < 0.15
- **Poorly-received promo**: vibe_score < 0.1 OR neg_frac ≥ 0.30 OR sarcasm_rate ≥ 0.25
- **Confidence threshold (persona)**: ≤ 0.4 = insufficient data
