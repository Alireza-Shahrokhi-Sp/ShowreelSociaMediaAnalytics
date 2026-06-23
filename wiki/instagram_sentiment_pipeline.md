---
title: Instagram sentiment pipeline
type: source-summary
sources:
  - Ali/sentiment_pipeline.ipynb
  - Ali/Sentiment_EDA/sentiment_eda.ipynb
  - Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb
related:
  - "[[instagram_findings]]"
  - "[[sentiment-pipeline]]"
created: 2026-06-14
updated: 2026-06-14
confidence: high
---

# Instagram sentiment pipeline

## Overview

Three-stage pipeline producing comment-level sentiment labels and post-level room vibes.

**Stage 1**: LLM-based comment sentiment classification (Vertex AI Batch)
**Stage 2a**: Post-level vibe aggregation (summary statistics + secondary LLM)
**Stage 2b**: Aggregated room narrative and dominant stance extraction

---

## Stage 1: Comment-level sentiment (Vertex AI Batch)

**Notebook**: `Ali/sentiment_pipeline.ipynb`

**Input**: Full comment text + post caption (optional context)

**LLM task**: Assign per-comment labels using Claude via Vertex AI Batch API.

### Output schema: sentiment_instagram.parquet

**File**: `Ali/outputs/stage2_sentiment/sentiment_instagram.parquet`

**Size**: 487,604 rows (filtered to media_context = True)

**Key**: comment_id (unique)

| Column | Type | Values |
|---|---|---|
| comment_id | string | Unique identifier |
| media_id | string | FK to post |
| sentiment | string | **positive** / **neutral** / **negative** |
| sentiment_score | float64 | [0, 1] confidence / strength |
| emotion | string | **joy**, trust, sadness, anger, disgust, anticipation, fear, neutral, surprise |
| intensity | string | **low** / **medium** / **high** |
| sarcasm | bool | True / False |
| toxicity | string | **none** / **mild** / **severe** (ordinal, not boolean) |
| intent | string | **praise**, support, question, criticism, spam_promo, other |
| target | string | **creator**, content_work, appearance, other_user, product, off_topic, none |
| lang | string | Language code ("it" for Italian) |
| author_id | string | Commenting user ID |
| platform | string | "instagram" |
| timestamp | datetime | Comment publication time |
| text | string | Full comment body |
| text_length | int64 | Character count (inherited from comments_ml) |
| word_count | int64 | Word count (inherited) |
| emoji_count | int64 | Emoji count (inherited) |
| ... | ... | All 15 linguistic features from comments_ml |

**Total columns**: 33

### Batch processing details

**Model**: Claude (Vertex AI Batch)

**Prompt structure**:
```
Analyze this Instagram comment for sentiment, emotion, intent, and potential toxicity.

Context: [post caption, if available]
Comment: [full comment text]

Respond in JSON with keys: sentiment, sentiment_score, emotion, intensity, sarcasm, toxicity, intent, target
```

**Response parsing**: Vertex Batch returns JSON; fields are extracted and validated.

**Cost**: ~$X per M tokens (Batch pricing tier, ~3–10x cheaper than online API)

**Processing time**: ~4–6 hours for 487k comments in large batches

---

## Ordinal toxicity column

**Critical note**: `toxicity` is ordinal (none / mild / severe), not boolean.

- **none**: 484,523 comments (99.37%)
- **mild**: 2,752 comments (0.56%)
- **severe**: 329 comments (0.07%)
- **is_toxic flag** (mild + severe): 3,081 comments (0.63%)

Analyses distinguish:
- Binary "is_toxic" for moderation thresholds
- Ordinal level for nuance (severe comments require escalation; mild may be context-dependent)
- Toxicity rate = (mild + severe) / total

---

## Stage 2a: Post-level room vibe aggregation

**Notebook**: `Ali/sentiment_pipeline.ipynb` (Stage 2 section)

**Input**: All comments on a post + comment-level sentiment

**Processing**:
1. Aggregate comment-level sentiment to post level
2. Compute dispersion (spread of sentiment across post)
3. Call secondary LLM to assign narrative vibe label and controversy score

### Output: room_vibe_instagram.parquet

**File**: `Ali/outputs/stage2_sentiment/room_vibe_instagram.parquet`

**Size**: 1,501 rows (one per post)

**Key**: media_id

| Column | Type | Description |
|---|---|---|
| media_id | string | Post ID |
| n_comments | int64 | Number of comments analysed for this post |
| media_context | bool | Post is active (not deleted) |
| vibe_score | float64 | [0, 1] aggregate positivity |
| dispersion | float64 | [0, 1] spread of sentiment (0 = unanimous, 1 = fully polarised) |
| pos_frac | float64 | [0, 1] share of positive comments |
| neg_frac | float64 | [0, 1] share of negative comments |
| neu_frac | float64 | [0, 1] share of neutral comments |
| stance_alignment | float64 | [0, 1] degree audience coalesces around single stance |
| polarization | float64 | [0, 1] strength of disagreement in comment room |
| sarcasm_rate | float64 | [0, 1] share of comments flagged sarcasm |
| toxicity_rate | float64 | [0, 1] share of toxic (mild + severe) comments |
| room_state | string | **united** / **mixed** / **fragmented** |
| llm_vibe | string | **appreciative** / **critical** / **mixed** / **celebratory** / **supportive** |
| llm_consensus | float64 | [0, 1] LLM confidence in vibe assignment |
| dominant_stance | string | Narrative summary (LLM-generated) |
| split_axis | string | If mixed: the dimension of disagreement (e.g. "artist quality vs engagement") |
| llm_alignment | float64 | Agreement between comment sentiment and room vibe |
| llm_controversy | float64 | [0, 1] controversy score (conflict between stances) |
| room_summary | string | Narrative description of comment room |

---

## Stage 2b: Narrative stance extraction

**Notebook**: `Ali/sentiment_pipeline.ipynb`

**Input**: room_vibe aggregates + sample of comments per post

**Task**: Secondary LLM pass to extract narrative dominant stance and summary

### Output: room_llm_instagram.parquet

**File**: `Ali/outputs/stage2_sentiment/room_llm_instagram.parquet`

**Size**: 1,501 rows

| Column | Type | Description |
|---|---|---|
| media_id | string | Key |
| dominant_stance | string | Natural language sentence summarising the audience's primary position |
| split_axis | string | If room_state="mixed": what is the disagreement about? |
| room_summary | string | Multi-sentence narrative of comment room tone and themes |

**Example values**:
- dominant_stance: "Overwhelmingly positive, with strong support for the creator's artistic direction"
- split_axis: "Photoshoot quality vs engagement and relatability"
- room_summary: "The comments express enthusiasm for the visual aesthetic but some users discuss the brand partnership and commercial aspects..."

---

## Sentiment EDA and analysis

### sentiment_eda.ipynb

**Location**: `Ali/Sentiment_EDA/sentiment_eda.ipynb`

Comprehensive exploratory analysis of comment-level sentiment with joins to persona, RFM, post metadata.

**Sections**:
1. Data loading and merge architecture (comment → persona → post → RFM)
2. Sentiment overview (distribution, emotion, intent)
3. Negative sentiment deep-dive (emotion, intent, by hour/day, user profiles)
4. Toxicity analysis (by sentiment, persona, media type, sponsored)
5. Intent × sentiment heatmap (composition per intent)
6. Emotion × intent heatmap (co-occurrence)
7. Topic → sentiment (media type, sponsored, hashtag density, caption keywords)
8. Persona and RFM lens
9. Linguistic features vs negativity and toxicity

**Key outputs**:
- `01_sentiment_overview.png` — Distribution bars and toxicity pie
- `02_emotion_sentiment_heatmap.png` — Row-normalised emotion vs sentiment %
- `03_sentiment_monthly_trend.png` — Monthly % mix time series
- `04_negative_deepdive.png` — 6-panel negative analysis (emotions, intents, intensity, hourly, daily, by post)
- `05_neg_vs_likes.png` — Scatter: negative rate vs likes with regression
- `06_toxicity_analysis.png` — Toxicity by sentiment, persona, media type, sponsorship
- `07_toxicity_sent_intensity.png` — Toxicity by sentiment × intensity heatmap
- `08_intent_sentiment_heatmap.png` — Intent vs sentiment (% within intent)
- `09_emotion_intent_heatmap.png` — Emotion vs intent (% within emotion)
- `10_toxicity_by_intent.png` — Toxicity rate ranked by intent
- `11_toxicity_by_emotion.png` — Toxicity rate ranked by emotion
- `12_sentiment_by_mediatype.png` — Sentiment mix: FEED vs REELS stacked bar
- `13_emotion_by_mediatype.png` — Emotion heatmap: FEED vs REELS
- `14_sponsored_sentiment.png` — Sentiment mix and rates: sponsored vs organic
- `15_hashtag_q_sentiment.png` — Negativity by hashtag quartile
- `16_vibe_sentiment.png` — Comment sentiment by post LLM vibe (% row)
- `17_caption_keywords.png` — Caption keywords → negative/toxic rates (top 30)
- `18_persona_sentiment.png` — Sentiment mix by persona (stacked bar)
- `19_persona_metrics.png` — 3-panel: neg rate, tox rate, comment volume per persona
- `20_persona_emotion_heatmap.png` — Emotion profile by persona (% row)
- `21_rfm_sentiment.png` — Negative and toxicity by RFM dimension quartiles
- `22_persona_rfm_negative.png` — Persona × recency negative rate heatmap
- `22b_rfm_named_cluster_sentiment.png` — Sentiment metrics by Mickey's k=4 clusters
- `23_feature_correlations.png` — Point-biserial r of all 15 linguistic features
- `24_feature_distributions.png` — KDE overlays of linguistic features (negative vs non-negative)
- `25_feature_distributions_toxic.png` — KDE overlays of linguistic features (toxic vs non-toxic)

---

### persona_sentiment_rfm.ipynb

**Location**: `Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb`

Focused analysis crossing persona × sentiment × window-level RFM cluster, including sentiment targets.

**Sections**:
- A: Persona × sentiment mix and toxicity lift
- B: Window-level cluster sentiment rates and time series
- C: Persona × cluster crossing (bivariate heatmaps)
- D: Cluster transitions and sentiment shift
- E: Sentiment target profiles and persona × target conditional rates

**Key outputs**:
- `A1_persona_sentiment_mix.png` — Persona sentiment stacked bar
- `A2_persona_toxicity_lift.png` — Toxicity lift per persona vs baseline
- `A3_persona_emotion_heatmap.png` — Emotion signature per persona (% row)
- `B1_cluster_sentiment_mix.png` — Window-cluster sentiment stacked bar
- `B2_cluster_neg_tox_rate.png` — Negative and toxicity rates by cluster
- `B3_cluster_negrate_timeseries.png` — Negative rate trend per cluster over 40 months
- `C1_persona_x_cluster_negrate.png` — Persona × cluster negative rate heatmap
- `C2_cluster_persona_composition.png` — Persona composition within each cluster (% stacked)
- `D1_transition_sentiment_shift.png` — Sentiment change: stayers vs switchers, transition-specific
- `E1_target_profile.png` — 3-panel: target share, neg rate, tox rate
- `E2_target_intent_emotion.png` — Intent and emotion fingerprints per target
- `E3_persona_target_negrate.png` — Persona × target conditional negative rate heatmap (with sample sizes)
- `E3b_product_neg_contribution.png` — Volume-weighted contribution to product negativity

---

## Batch parameters and cost

**Vertex AI Batch API**:
- Model: claude-opus-4 (or later)
- Input batch size: 5,000–10,000 comments per batch job
- Temperature: 0.3–0.5 (low randomness)
- Max tokens: 500 (per-response limit)
- Retry strategy: Automatic 3× on transient failures

**Cost estimate**: ~$150–300 for full 487k comment analysis (Batch tier pricing, ~1/10th of online API)

**Processing time**: 4–8 hours (depending on batch queue)

**Quality control**: Sample audit of ~100 randomly selected output comments to verify JSON parsing and schema compliance

---

## Known issues and limitations

1. **Language assumption**: Assumes Italian comments (model trained on multilingual data but optimised for Italian). Non-Italian comments may receive less accurate labels.

2. **Context loss**: Brief comments with minimal context (emoji-only, single word) may receive uncertain labels. Confidence scores help identify these.

3. **Sarcasm detection**: Sarcasm flag is rare and has lower confidence than sentiment. Not recommended for hard filtering.

4. **Toxicity threshold**: "Mild" and "severe" distinction is LLM-based and may not match platform moderation standards. Use toxicity labels as signals, not absolute verdicts.

5. **Temporal coverage**: Dataset spans 2016–2026; older comments may have different language patterns than recent ones. No temporal bias correction applied.

6. **Target classification**: "target" field is inferred from comment text only (not explicit user input). Comments without clear target are marked "none" or "off_topic" conservatively.

---

## Integration into analyses

All three analysis notebooks (`sentiment_eda.ipynb`, `persona_sentiment_rfm.ipynb`, and the main findings aggregations) merge the sentiment table with:
- comments_ml (linguistic features)
- user_personas_combined (persona labels)
- ig_multimodal_final (post metadata)
- room_vibe_instagram (post-level vibes)
- RFM rolling windows (user lifecycle features)

**Coverage notes**:
- Sentiment: 487,604 comments (all with media_context = True)
- Persona: 19.3% of comments (40,019 labeled users)
- RFM: 13.3% of comments (window-level cluster assignments)
- Room vibe: All 1,501 posts have vibe labels
