# AFB_Lab — Data Cleaning & Feature Engineering Report

**Generated:** 2026-06-13  
**Scope:** All notebooks and Python scripts across Ali/, Reza/, Mickey/ and root patch scripts

---

## Table of Contents

1. Pipeline Overview
2. Ali: Comment Preprocessing — `Ali/Data_Preparation_Pipeline.ipynb`
3. Ali: Multimodal Final Dataset Builder — `Ali/build_final_dataset.py`
4. Ali: Persona Feature Engineering — `Ali/persona/features.py`
5. Ali: Persona Data Loader — `Ali/persona/data.py`
6. Ali: Persona Clustering Prep — `Ali/persona/clustering.py`
7. Ali: Persona Pipeline Orchestration — `Ali/persona_pipeline.ipynb`
8. Ali: Sentiment EDA / Room Vibe Aggregation — `Ali/Sentiment_EDA.ipynb`
9. Ali: Sentiment Pipeline — `Ali/sentiment_pipeline.ipynb`
10. Ali: Initial EDA / Data Quality — `Ali/EDA.ipynb`
11. Reza: NER Extraction — `reza/ner/extract_entities.py`
12. Reza: Transcript Deep-Clean — `reza/ner/reclean_and_rebuild_jobs.py`
13. Reza: Transcript Merge — `reza/pipeline_tools/add_transcripts.py`
14. Reza: Transcript Chunking and Job Prep — `reza/pipeline_tools/prepare_transcript_jobs.py`
15. Reza: Thumbnail CV Features — `reza/thumbnail_cv/thumbnail_features.py`
16. Reza: Thumbnail Feature Merge — `reza/thumbnail_cv/merge_thumbnail_features.py`
17. Reza: YouTube EDA — `reza/eda/youtube_EDA.ipynb`
18. Mickey: IG Rolling Quadrimesters RFM — `RFM_IG_rolling_quadrimesters/rolling_quadrimesters_new_definitions.ipynb`
19. Mickey: TK Rolling Quadrimesters RFM — `RFM_TK_rolling_quadrimesters/tk_rolling_quadrimesters.ipynb`
20. Mickey: IG Non-Overlapping 3-Year Periods — `RFM_IG_3_years/non_overlapping_3y_periods_new_definitions.ipynb`
21. Mickey: EDA — RFM IG vs TK Quadrimesters — `EDA_RFM_IG_vs_TK_rolling_quadrimesters.ipynb`
22. Mickey: EDA — Cluster Matrix IG vs TK — `EDA_cluster_matrix_IG_vs_TK.ipynb`
23. Mickey: User Cluster Journey Analysis — `Mickey/User_Cluster_Journey_Analysis.ipynb`
24. Root: Patch Scripts
25. End-to-End Data Flow Summary
26. Critical Assessment: Does It Make Sense?

---

## 1. Pipeline Overview

The AFB_Lab codebase processes social media data from three platforms — **Instagram (IG)**, **TikTok (TK)**, and **YouTube (YT)** — through a multi-stage ML pipeline:

```
Raw social data
    │
    ▼
[Data_Preparation_Pipeline] ── comment-level cleaning & text features
    │
    ├── comments_ml_{platform}.parquet     ← ML numeric features
    ├── comments_llm_{platform}.jsonl      ← raw text for LLM batch jobs
    └── comments_gml_{platform}.parquet    ← graph edges
         │
         ├── [sentiment_pipeline] ── per-comment LLM sentiment labels
         │        └── sentiment_{platform}.parquet
         │
         └── [persona/features.py] ── user-level behavioral aggregation
                  └── user_features_{platform}.parquet
                           │
                           └── [clustering.py] ── UMAP+HDBSCAN+LLM persona discovery
                                    └── final_taxonomy.json + user_micro_personas.parquet

[build_final_dataset.py] ── IG multimodal audit & metadata join
    └── ig_multimodal_final.parquet

[reza/thumbnail_cv] ── visual feature extraction
    └── thumbnail_features.parquet

[reza/ner] ── named entity extraction from transcripts & comments
    └── entity_resonance.parquet

[Mickey] ── RFM temporal segmentation + cluster journey tracking
```

---

## 2. Ali — Comment Preprocessing (`Ali/Data_Preparation_Pipeline.ipynb`)

### Purpose
Unified ingestion and cleaning of raw Instagram, Facebook, and TikTok comment exports into three downstream formats.

### Input Data
| Platform | Raw Row Count |
|---|---|
| Instagram | 573,377 |
| Facebook | 394,084 |
| TikTok | 19,654 |

### Schema Unification (Data Cleaning Step 1)
Each platform has different field names for the same concepts. A `RawComment` dataclass normalises them:

| Platform | author_id source | media_id source |
|---|---|---|
| Instagram | `from_id` | `media_id` |
| Facebook | `from_id` | `media_id` |
| TikTok | `uid` | `aweme_id` |

### Data Cleaning Steps

| Step | Operation | Columns Affected |
|---|---|---|
| Strip whitespace | `text.strip()` | `comment_text` |
| Null filtering | Drop rows where `author_id` or `media_id` is missing | `author_id`, `media_id` |
| Type casting | Convert IDs to `str` | `author_id`, `media_id` |
| Timestamp normalisation | `pd.to_datetime(..., utc=True)` | `timestamp` |
| Emoji extraction | Unicode category scan | `emoji_list` (new) |
| Deduplication | Synthetic ID `{platform}_{author_id}_{idx}` | `comment_id` (new) |
| Text normalisation | Remove emojis for word-level analysis | `text_no_emoji` (new) |

### Feature Engineering — Text-Level Features (14 columns per comment)

All features are computed per comment row and stored in `comments_ml_{platform}.parquet`.

| Feature | Formula / Logic |
|---|---|
| `text_length` | `len(comment_text)` |
| `word_count` | `len(comment_text.split())` |
| `emoji_count` | Count of emoji characters in `emoji_list` |
| `unique_emoji_count` | `len(set(emoji_list))` |
| `emoji_entropy` | Shannon entropy: `-sum(p * log2(p))` over emoji frequencies |
| `emoji_variety_ratio` | `unique_emoji_count / emoji_count` (0 if no emojis) |
| `emoji_per_word_ratio` | `emoji_count / word_count` (0 if no words) |
| `url_count` | Regex match count for `https?://\S+` |
| `mention_count` | Regex match count for `@\w+` |
| `hashtag_count` | Regex match count for `#\w+` |
| `exclamation_count` | `comment_text.count('!')` |
| `question_count` | `comment_text.count('?')` |
| `avg_word_length` | Mean character length of whitespace-split tokens |
| `has_numbers` | Binary: `1` if `re.search(r'\d', text)` else `0` |
| `has_links` | Binary: `1` if `url_count > 0` else `0` |

### Outputs

| File | Format | Content |
|---|---|---|
| `comments_ml_{platform}.parquet` | Parquet | All 14 numeric text features + IDs |
| `comments_llm_{platform}.jsonl` | JSONL | `comment_id, text, author_id, platform, timestamp` |
| `comments_gml_{platform}.parquet` | Parquet | `comment_id, author_id, media_id, reply_to_comment_id, platform, timestamp` |

---

## 3. Ali — Multimodal Final Dataset Builder (`Ali/build_final_dataset.py`)

### Purpose
Join folder-based media audit results with enriched IG post metadata to create one canonical row per Instagram post.

### Inputs
| File | Role |
|---|---|
| `multimodal_dataset_fixed/` folder | Ground-truth audit of downloaded media files |
| `Output/ig_posts_multimodal_enriched.parquet` | Enriched IG post features (from ig_refetch.py) |
| `Output/ig_media_features.parquet` | Fetch status per post |
| `../Data/ig_posts_cleaned.parquet` | Fallback metadata for missing values |

### Data Cleaning Steps

| Step | Operation |
|---|---|
| Shortcode extraction | `permalink.str.split('/').str[-1]` → `shortcode` (string) |
| Deduplication | `drop_duplicates('shortcode')` on base metadata |
| Left join | Merge audit onto enriched parquet, backfill from `ig_posts_cleaned` |
| Status flagging | `features_status` = `ok` / `fetch_error` / `not_fetched` |

### Feature Engineering — Folder Audit Features

| Feature | Logic |
|---|---|
| `content_form` | Folder name: `reel`, `feed`, `carousel`, `carousel_video`, `image` |
| `folder` | Absolute path to media folder |
| `n_jpg` | Count of `*.jpg` files in folder |
| `n_mp4` | Count of `*.mp4` files in folder |
| `n_slides` | Carousel slide count (form-specific: only meaningful for carousel/carousel_video) |
| `has_frames` | Boolean: frame extraction completed |
| `n_frames` | Count of extracted frames |
| `has_transcript` | Boolean: transcript file exists |
| `transcript_nonempty` | Boolean: transcript file is non-empty |

### Sanity Checks
- Hard assert: `len(merged) == len(audit)` (no row loss)
- Print breakdown by `features_status`

### Output
- `Output/ig_multimodal_final.parquet` — 1501 rows, one per shortcode
- `Output/ig_multimodal_final.csv` — human-inspectable copy

---

## 4. Ali — Persona Feature Engineering (`Ali/persona/features.py`)

### Purpose
Aggregate comment-level features up to the user level to create behavioral profiles for clustering.

### Inputs
| File | Content |
|---|---|
| `comments_ml_{platform}.parquet` | 14 text features per comment |
| `ig_media/{platform}.parquet` | Post publication timestamps (optional) |
| `post_vibes_path` | Room-vibe sentiment data per post (optional) |
| `comments_llm_path` | Raw comment text for representative samples |

### Data Cleaning Steps

| Step | Operation | Column(s) |
|---|---|---|
| Type casting | `author_id`, `media_id` → `str` | `author_id`, `media_id` |
| Timestamp normalisation | `pd.to_datetime(..., utc=True)` | `timestamp` |
| Temporal fillna | Fill `hours_to_comment` with `0` if media timestamps missing | `hours_to_comment` |
| Deduplication | Keep first occurrence per `author_id` on media join | — |
| Binary flag derivation | Derive `has_emoji`, `has_question`, `has_exclaim`, `is_reply` in-place | see below |

**Derived binary flags added to comment rows:**

| Flag | Condition |
|---|---|
| `has_emoji` | `emoji_count > 0` |
| `has_question` | `question_count > 0` |
| `has_exclaim` | `exclamation_count > 0` |
| `is_reply` | `reply_to_comment_id.notna()` |

### Feature Engineering — User-Level Aggregations (18+ columns)

Computed via `groupby('author_id')`:

| Feature | Formula |
|---|---|
| `total_comments` | `count()` |
| `unique_posts_commented` | `media_id.nunique()` |
| `total_replies_made` | `is_reply.sum()` |
| `reply_ratio` | `is_reply.mean()` |
| `mean_hours_to_comment` | `hours_to_comment.mean()` |
| `median_hours_to_comment` | `hours_to_comment.median()` |
| `pct_comments_under_1h` | `(hours_to_comment < 1).mean()` |
| `pct_comments_under_24h` | `(hours_to_comment < 24).mean()` |
| `activity_span_days` | `(max_timestamp - min_timestamp).days` |
| `mean_word_count` | `word_count.mean()` |
| `mean_mention_count` | `mention_count.mean()` |
| `emoji_usage_rate` | `has_emoji.mean()` |
| `question_rate` | `has_question.mean()` |
| `exclamation_rate` | `has_exclaim.mean()` |
| `post_concentration_ratio` | `unique_posts_commented / total_comments`, clipped to `[0, 1]` |

**Optional room-vibe features (if `post_vibes` loaded):**

| Feature | Formula |
|---|---|
| `mean_engaged_consensus` | Mean `room_consensus` across user's commented posts |
| `mean_sponsorship_tolerance` | Mean `room_sponsorship_alignment` across user's posts |
| `dominant_room_vibe` | Modal `room_vibe` value (categorical) |

**Representative comment samples:**
- Top 5 comments per user fetched from `comments_llm_path`
- Aggregated as `top_comments_sample` (pipe-separated string)

### Caching
- Checks for existing `user_features_cache_path` parquet before recomputing.

### Output
- `user_features_{platform}.parquet` — one row per `author_id`, 18+ columns

---

## 5. Ali — Persona Data Loader (`Ali/persona/data.py`)

### Purpose
Standardised data loading with validation and optional reply-edge enrichment.

### Data Cleaning Steps

| Step | Operation |
|---|---|
| Type casting | `author_id`, `media_id` → `str` |
| Null validation | Assert `author_id.notna() & (author_id != "")` |
| Reply deduplication | `edges_replies.drop_duplicates('comment_id')`, keep first |
| Missing ID filtering | Drop rows with null `author_id` or `media_id` |

### Feature Engineering
- Adds `reply_to_comment_id` column via left-merge with `edges_replies_path` (null if no reply edge found).

### Output
- Tuple `(ig_comments DataFrame, media_index DataFrame or None)`

---

## 6. Ali — Persona Clustering Prep (`Ali/persona/clustering.py` — `PathwayBClusterer`)

### Purpose
Dimensionality reduction and unsupervised clustering for persona discovery, followed by LLM-based cluster labelling.

### Inputs
- User features from `FeatureEngineer` output
- Stage-1 stratified user sample DataFrame

### Data Cleaning Steps

| Step | Operation |
|---|---|
| Type casting | `author_id` → `str` |
| Numeric null imputation | `fillna(median)` on all numeric feature columns |

### Feature Engineering — Matrix Preparation

**Candidate numeric features (15 total):**
`total_comments`, `unique_posts_commented`, `reply_ratio`, `mean_hours_to_comment`, `median_hours_to_comment`, `pct_comments_under_1h`, `pct_comments_under_24h`, `activity_span_days`, `mean_word_count`, `mean_mention_count`, `emoji_usage_rate`, `question_rate`, `exclamation_rate`, `post_concentration_ratio`, `total_replies_made`

| Step | Tool |
|---|---|
| Numeric normalisation | `StandardScaler` |
| Categorical encoding | `OneHotEncoder` on `dominant_room_vibe` (if available) |
| Matrix stacking | `hstack([x_num, x_vibe])` or numeric-only |
| Dimensionality reduction | UMAP → `n_components=2` |
| Clustering | HDBSCAN (auto-tuned toward target cluster count) |

**Per-cluster summary features (for LLM labelling):**

| Feature | Content |
|---|---|
| `cluster_id` | Integer cluster label |
| `n_users` | User count |
| `pct_audience` | Share of total audience |
| `mean_behavioral_stats` | Mean of 9 behavioral feature values |
| `dominant_room_vibe_mix` | Value counts (%) of `room_vibe` within cluster |
| `sample_comments` | Up to 30 comment fragments (max 2 per user) |

### LLM Labelling (Gemini)
Fixed 5-key output schema per cluster:

| Key | Format |
|---|---|
| `codename` | `UPPER_SNAKE_CASE` |
| `label` | Title Case |
| `description` | 1–2 sentences |
| `quantitative_signals` | 3–5 distinguishing behaviors |
| `example_comments` | 2–3 verbatim fragments |

### Output
- `final_taxonomy.json` — array of cluster descriptors
- `user_micro_personas.parquet` — per-user cluster assignment

---

## 7. Ali — Persona Pipeline Stage 0/1/2 Orchestration (`Ali/persona_pipeline.ipynb`)

### Purpose
Top-level orchestration notebook that runs the full persona discovery and classification pipeline using Vertex AI batch inference.

### Key Configuration Decisions (Data Shaping)

| Config | Value | Effect |
|---|---|---|
| `PLATFORM` | `"instagram"` | Filters `comments_ml.parquet` to IG only |
| `SAMPLE_N_USERS` | 20,000 | Stage 1 stratified sample size |
| `SAMPLE_SEED` | 42 | Reproducibility |
| `STAGE1_USERS_PER_REQUEST` | 3 | LLM prompt packing (taxonomy discovery) |
| `STAGE2_USERS_PER_REQUEST` | 3 | LLM prompt packing (classification) |
| `MAX_MEDIA_POSTS_PER_USER` | 2 | Multimodal cost cap: max posts per user |
| `MAX_IMAGES_PER_POST` | 2 | Max images attached per post |
| `MAX_TRANSCRIPT_CHARS` | 1,500 | Transcript truncation threshold |
| `STRATIFY_BY_SENTIMENT` | `True` | Stratifies Stage 1 sample by sentiment distribution |

### Stage 0 — User Feature Engineering

Reads from:

- `outputs/comments_ml.parquet` (filtered to `PLATFORM`)
- `outputs/edges_replies_to.parquet` (reply edges)
- `outputs/ig_multimodal_final.parquet` (media index, IG only)
- `outputs/stage2_sentiment/sentiment_{PLATFORM}.parquet` (optional room-vibe enrichment)

Calls `persona/features.py` `FeatureEngineer` — produces `user_features_{platform}.parquet` (18+ behavioral aggregates, detailed in Section 4).

### Stage 1 — Stratified Sampling

**Stratification by sentiment** (when `STRATIFY_BY_SENTIMENT=True`):

- Joins user features with post-level sentiment (from `post_vibes_instagram.parquet`)
- Computes `dominant_room_vibe` per user
- Draws a stratified sample of `SAMPLE_N_USERS` users preserving the sentiment-vibe distribution

**Without sentiment**: simple random sample of 20,000 users.

### Stage 2 — Batch Classification

- Full user cohort classified against `final_taxonomy.json` via Vertex batch
- Output: `outputs/stage2_persona/user_personas.parquet` + `persona_map_{platform}.parquet`

### Outputs
| File | Content |
|---|---|
| `outputs/stage1_persona/taxonomy.json` | Discovered persona taxonomy (Stage 1) |
| `outputs/stage2_persona/user_personas.parquet` | Per-user persona assignment |
| `outputs/stage2_persona/persona_map_{platform}.parquet` | `author_id → persona_codename` map |
| `outputs/pathway_b_assignments_{platform}.parquet` | Pathway B (UMAP/HDBSCAN) cluster assignments |

---

## 8. Ali — Sentiment EDA / Room Vibe Aggregation (`Ali/Sentiment_EDA.ipynb`)

### Purpose
Computes comment-level sentiment QA and derives post-level **room vibe** features consumed by the persona pipeline.

### Input

- `outputs/sentiment_instagram.parquet` (497,861 comments)

### Data Cleaning

| Step | Operation |
|---|---|
| Numeric coercion | `pd.to_numeric(sentiment_score, errors='coerce')` |
| Null fill | `sentiment.fillna("neutral")` — missing labels treated as neutral |

### Feature Engineering — Post-Level Room Vibe (written to `post_vibes_instagram.parquet`)

Grouped by `media_id` (1,501 posts):

| Feature | Formula |
|---|---|
| `vibe_score` (→ renamed `room_vibe`) | `mean(sentiment_score)` over all comments on the post |
| `vibe_std` | `std(sentiment_score, ddof=0)` |
| `pos_frac` | `mean(sentiment == 'positive')` |
| `neg_frac` | `mean(sentiment == 'negative')` |
| `neu_frac` | `max(0, 1 - pos_frac - neg_frac)` |
| `stance_alignment` | `mean(sign(sentiment_score))` |
| `polarization` | `2 * min(pos_frac, neg_frac)` |
| `controversy` | `4 * pos_frac * neg_frac` |
| `consensus` (→ renamed `room_consensus`) | `1 - normalised_entropy(pos, neg, neu)` where entropy normalised by `log2(3)` |
| `sarcasm_rate` | `mean(sarcasm.fillna(False))` |
| `toxicity_rate` | `mean(toxicity != 'none')` |

**Room state classification** (derived from `controversy` and `consensus`):

| Condition | `room_state` |
|---|---|
| `controversy >= 0.3` | `"divided"` |
| `consensus >= 0.6` | `"united"` |
| Otherwise | `"mixed"` |

### Cross-validation Check

- Derives a score-based category using `|score| > 0.15` band thresholds
- Compares categorical LLM label vs. score-derived label → agreement matrix
- Validates that the two label sources are consistent

### Output Schema — `post_vibes_instagram.parquet`

| Column | Description |
|---|---|
| `media_id` | Post ID |
| `room_vibe` | Mean sentiment score across all comments |
| `room_consensus` | 1 − normalised entropy of sentiment label mix |
| `stance_alignment` | Mean sign of sentiment scores |

---

## 9. Ali — Sentiment Pipeline (`Ali/sentiment_pipeline.ipynb`)

### Purpose
Vertex AI Batch inference for per-comment multimodal sentiment classification.

### Inputs
- `comments_llm_{platform}.jsonl` (text + metadata)
- Optional: media files via `gs://` URIs (images/frames/transcripts)

### Processing
- Posts grouped so media is shared across all comments on the same post
- Multimodal context: images sent as `photo+meta`, video as `frames+transcript`
- Pydantic v2 schema (`SentimentItem` / `RoomRead` / `SentimentResponse`) → flattened OpenAPI `responseSchema`
- `_sanitize_schema` removes `$defs`/`$ref` for Vertex Batch compatibility

### Output Schema — `sentiment_{platform}.parquet`

| Column | Description |
|---|---|
| `comment_id` | Unique comment identifier |
| `author_id` | Commenter ID |
| `media_id` | Post ID |
| `sentiment_cat` | `negative` / `neutral` / `positive` |
| `sentiment_score` | Float in `[-1, 1]` |
| `toxicity` | `none` / `mild` / `severe` |
| `emotion` | Emotion label (anger, joy, sadness, etc.) |
| `intent` | Behavioural intent label |
| `target` | Who/what the comment is directed at |
| `sarcasm_label` | Sarcasm category |
| `sarcasm` | Boolean |

---

## 10. Ali — Initial EDA / Data Quality (`Ali/EDA.ipynb`)

### Purpose
First-pass quality audit of raw social media datasets before any cleaning pipeline runs. Reads raw parquet/CSV exports from Facebook, Instagram, TikTok, and YouTube.

### Data Cleaning Checks Performed (per platform, per table)

| Check | Operation |
|---|---|
| Missing values | `isnull().sum()` per column, % missing computed |
| Duplicates | `duplicated().sum()` on full rows for posts and comments tables |
| Numeric distributions | `describe()` + histogram per numeric column (skew, kurtosis annotated) |
| Categorical cardinality | `value_counts()` for first 3 object columns |
| Correlations | Pearson correlation heatmap, flags abs(r) >= 0.7 pairs |
| Missingness heatmap | Visual `seaborn.heatmap(df.isnull())` per table |

### Key Findings Recorded

Duplicate counts and missing-value profiles are printed per run for: Facebook posts/comments, Instagram posts/comments, TikTok posts/comments, YouTube channels/videos/comments. The notebook does not hard-code findings — results are displayed inline and used to inform downstream cleaning decisions.

This notebook does not write any output files — it is a diagnostic-only notebook whose findings informed the cleaning decisions in `Data_Preparation_Pipeline.ipynb`.

---

## 11. Reza — NER Extraction (`reza/ner/extract_entities.py`)

### Purpose
Named-entity recognition from YouTube video transcripts and comments using spaCy.

### Inputs
| File | Content |
|---|---|
| `yt_videos_with_local_transcripts.parquet` | Video metadata + cleaned transcripts |
| `yt_comments_*_cleaned.parquet` (4 files) | Comment text per video |

### Data Cleaning

| Step | Operation |
|---|---|
| Null filtering | Drop rows where `local_transcript.isna()` |
| Entity type filtering | Keep only: `PER`, `ORG`, `LOC`, `GPE`, `MISC`, `PROD` |
| Length filtering | Remove entities shorter than 2 characters |

### Feature Engineering — Entity Resonance Table

**Per-video entity extraction** (spaCy `it_core_news_lg`):
- Extracts entities from transcript text → `{text, label, count}` sorted descending by count
- Same extraction applied to comment text

**Entity resonance output (one row per `(video_id, entity_text, entity_label)`):**

| Column | Content |
|---|---|
| `video_id` | YouTube video ID |
| `entity_text` | Entity surface form |
| `entity_label` | spaCy entity type |
| `transcript_count` | Occurrence count in transcript |
| `comment_count` | Occurrence count in comments |
| `in_transcript` | Boolean |
| `in_comments` | Boolean |

### Outputs
- `entities_transcripts.parquet`
- `entities_comments.parquet`
- `entity_resonance.parquet`

---

## 12. Reza — Transcript Deep-Clean (`reza/ner/reclean_and_rebuild_jobs.py`)

### Purpose
Multi-pass text cleaning of raw VTT/SRT transcripts before NER inference.

### Input
- `yt_videos_with_local_transcripts.parquet` (raw `local_transcript` column)

### Data Cleaning Passes (Applied Sequentially to Transcript Text)

| Pass | Operation |
|---|---|
| 1 | `html.unescape()` — decode HTML entities |
| 2 | Regex strip `Kind:`, `Language:` VTT header tokens |
| 3 | Regex strip VTT/SRT timestamp lines (`HH:MM:SS.mmm --> HH:MM:SS.mmm`) |
| 4 | Regex strip cue settings: `align:`, `position:`, `line:`, `size:` |
| 5 | Remove empty angle brackets `<>` |
| 6 | Sliding-window consecutive duplicate deduplication (min_len=4 words) |
| 7 | Multi-space collapse → single space |

### JSONL Rebuild Configs

| Config | `max_tokens` | `overlap` | `dedupe` |
|---|---|---|---|
| `production` | 1800 | 150 | False |
| `test-v1` | 400 | 50 | True |
| `test-v2` | 400 | 50 | True |

### Output
- Cleaned transcript column written back to parquet
- Three JSONL job files

---

## 13. Reza — Transcript Merge (`reza/pipeline_tools/add_transcripts.py`)

### Purpose
Discover and attach locally downloaded VTT/SRT/TXT transcript files to the video metadata parquet.

### Inputs
- `yt_videos_cleaned.parquet` (base video data)
- `transcripts/` folder tree (VTT/SRT/TXT files, named by video ID)

### Data Cleaning (`strip_timestamps_and_tags` function)

| Pass | Operation |
|---|---|
| 1 | `html.unescape()` |
| 2 | Remove `WEBVTT:` header line |
| 3 | Strip all VTT/SRT timing formats |
| 4 | Remove HTML tags `<...>` |
| 5 | Strip cue settings: `align:`, `position:`, `line:`, `size:` |
| 6 | Remove SRT index numbers |
| 7 | Sliding-window word overlap deduplication |
| 8 | Multi-space to single space |

### Feature Engineering
- Fuzzy file discovery: strips language suffixes and prefixes to match `videoId` → transcript file
- New column: `local_transcript` (cleaned full transcript text)

### Output
- `yt_videos_with_local_transcripts.parquet`

---

## 14. Reza — Transcript Chunking & Job Prep (`reza/pipeline_tools/prepare_transcript_jobs.py`)

### Purpose
Prepare GCP-ready inference job bundles from cleaned transcripts.

### Input
- `yt_videos_with_local_transcripts.parquet`

### Data Cleaning
- Re-applies `clean_transcript()` (same regex passes as above)
- Language detection via `detect_langs()` per transcript
- Content-hash deduplication: `stable_hash(normalized_text)` → skip already-seen transcripts

### Feature Engineering — Token-Aware Chunking

| Feature | Method |
|---|---|
| `token_count` | tiktoken `cl100k_base` encoder (fallback: `len(text) / 4`) |
| Chunking | `chunk_text(text, max_tokens, overlap_tokens, encoder)` — overlapping windows |

**JSONL job record fields:**

| Field | Content |
|---|---|
| `video_id` | YouTube video ID |
| `title` | Video title |
| `chunk_index` | 0-indexed chunk position |
| `chunk_text` | Cleaned, token-bounded text chunk |
| `token_count` | Estimated token count |
| `language` | Detected language code |
| `hash` | Content hash for deduplication |

### Outputs
- `transcript_jobs.jsonl` (production: 1800 tokens, 150 overlap)
- `test_transcript_jobs_v1.jsonl` (400 tokens, 50 overlap)
- `test_transcript_jobs_v2.jsonl` (400 tokens, 50 overlap)
- Manifest and reject files

---

## 15. Reza — Thumbnail CV Features (`reza/thumbnail_cv/thumbnail_features.py`)

### Purpose
Extract computer-vision and CLIP features from YouTube thumbnail images via GCP Batch.

### Inputs
- `gs://{bucket}/thumbnails/{video_id}.jpg`
- `metadata/yt_videos_metadata.parquet`

### Feature Engineering — 15+ Visual Features Per Thumbnail

**Colour features (HSV):**

| Feature | Formula |
|---|---|
| `brightness_mean` | Mean V channel (HSV), normalised to `[0, 1]` |
| `saturation_mean` | Mean S channel (HSV), normalised to `[0, 1]` |

**Face detection (MediaPipe):**

| Feature | Description |
|---|---|
| `face_count` | Number of detected faces |
| `face_present` | Boolean |
| `face_area_ratio` | Bounding box area / image area |
| `face_position_x` | Normalised x-coordinate of face centre |
| `face_position_y` | Normalised y-coordinate of face centre |
| `face_center_distance` | Euclidean distance from image centre |

**Face pose (FaceMesh + solvePnP):**

| Feature | Description |
|---|---|
| `face_yaw_deg` | Horizontal head rotation (degrees) |
| `face_pitch_deg` | Vertical head rotation (degrees) |
| `face_roll_deg` | Tilt rotation (degrees) |
| `face_is_frontal` | Boolean: `|yaw| < 25` and `|pitch| < 20` |

**Face expression (FaceMesh landmarks):**

| Feature | Description |
|---|---|
| `mouth_openness` | Vertical lip gap / mouth width |
| `eye_openness_mean` | Mean of left + right eye gaps |
| `gaze_off_camera` | Boolean: iris displacement > 15% eye width |

**Text recognition (EasyOCR, Italian + English):**

| Feature | Description |
|---|---|
| `text_present` | Boolean |
| `text_area_ratio` | Total text bounding box area / image area |
| `text_word_count` | Word count from OCR output |
| `text_has_number` | Boolean |

**CLIP alignment (ViT-L/14):**

| Feature | Description |
|---|---|
| `clip_title_align` | Cosine similarity between image embedding and video title text embedding |

### Output
- GCS shards: `part_{N:04d}.parquet`
- Merged locally: `thumbnail_features.parquet` (~25k rows, 15+ columns)

---

## 16. Reza — Thumbnail Feature Merge (`reza/thumbnail_cv/merge_thumbnail_features.py`)

### Purpose
Consolidate GCS shards into a single local parquet file.

### Data Cleaning
- `pd.concat(shards, ignore_index=True)` — no deduplication applied at this stage.

### Output
- `thumbnail_features.parquet`

---

## 17. Reza — YouTube EDA (`reza/eda/youtube_EDA.ipynb`)

### Purpose
Exploratory analysis of YouTube video metadata and comment corpora.

### Inputs
- `yt_videos_with_local_transcripts.parquet`
- `yt_comments_*_cleaned.parquet` (4 files)

### Data Cleaning

| Step | Operation |
|---|---|
| Timestamp parsing | `pd.to_datetime(..., utc=True)` then `.dt.tz_convert(None)` |
| Numeric coercion | `pd.to_numeric(..., errors='coerce')` on view/like/comment counts |
| Deduplication | `drop_duplicates()` on `(videoId, publishedAt)` |
| Date filters | `> 2022-01-01` threshold for recency analysis |

### Feature Engineering

| Feature | Formula |
|---|---|
| `word_count` | `len(local_transcript.split())` |
| `duration_minutes` | `duration_seconds / 60` |
| `month` | `publishedAt.dt.to_period('M').dt.to_timestamp()` |
| `is_short` | Boolean split for Shorts vs. long-form content |
| Late comment flag | Comments published ≥ 2022 on videos published before 2022 |

**Aggregations:**
- Monthly publication counts by `channel_title` × `is_short`
- Per-channel engagement KDE (minimum 3 videos threshold)
- Correlation matrix: `views`, `likes`, `comments`, `duration_seconds`

---

## 18. Mickey — IG Rolling Quadrimesters RFM (`RFM_IG_rolling_quadrimesters/rolling_quadrimesters_new_definitions.ipynb`)

### Purpose
Build per-user rolling 4-month RFM metrics for Instagram, classify each user-window into Low/Mid/High per dimension, then run KMeans (k=4) to assign cluster labels across the full timeline.

### Inputs

- `ig_comments_clean.csv` — raw IG comments
- `ig_posts_clean.csv` — raw IG post timestamps

### Data Cleaning Steps

| Step | Operation |
|---|---|
| Type casting | All IDs cast via `COMMENT_DTYPES` dict to `"string"` before CSV read |
| Timestamp parsing | `pd.to_datetime(timestamp, errors='coerce')` on both comments and posts |
| Drop nulls | `dropna(subset=["comment_id","media_id","timestamp"])` |
| String strip | `from_username`, `from_id`, `parent_id` filled with `""` then `.str.strip()` |
| Creator removal | Rows where `is_creator==True` OR `from_username.casefold() == "camihawke"` are dropped |
| Empty-comment filter | `word_count > 0` filter applied using `count_words_with_emoji()` |
| Reply exclusion | `(is_reply==True) OR parent_id != ""` rows dropped from base set |
| Min-activity filter | Users with only 1 comment across the full period are dropped (`comment_count > 1`) |
| Date window filter | Only posts/comments from `2022-09-01` to `2026-04-01` retained |
| Midnight backoff | Post timestamps at exactly `00:00:00` are back-filled to `first_comment_timestamp - 2min` to fix data-entry artefact where posts were logged at midnight |

### Feature Engineering — Rolling Window Metrics (per user × window)

**Window definition:** 4-month sliding windows, monthly step, 40 windows total (Sep 2022 → Mar 2026).

| Feature | Formula |
|---|---|
| `recency` | `(window_end - last_comment_timestamp).days / window_total_days`, bounded [0,1]; 0 = commented at end (recent), 1 = commented at start (stale) |
| `coverage` | `unique_posts_commented / posts_in_window`; share of window posts the user touched |
| `engagement` | Mean likes/engagement received per comment (raw count) |
| `delay` | Mean hours between post publication and the user's comment, clipped `>= 0`; uses `delay_post_timestamp` (midnight-corrected) |
| `tenure` | Days between user's first-ever comment (across full history) and window start; 0 for new users |
| `gini` | Gini coefficient of comment distribution across posts; 0 when user left ≤1 comment (all comments on one post) |

**Threshold classification (Low / Mid / High per feature):**

| Feature | Low threshold | High threshold | Source |
|---|---|---|---|
| `recency` | 30th percentile | 70th percentile | Data-driven |
| `coverage` | 60th percentile | 90th percentile | Data-driven |
| `engagement` | 60th percentile | 90th percentile | Data-driven |
| `delay` | 68.6th percentile | 87.5th percentile | Data-driven |
| `tenure` | 30th percentile | 70th percentile | Data-driven |
| `gini` | 0.20 | 0.60 | Fixed |

**Coverage is window-specific:** threshold = `1 / posts_in_window` (user must comment on at least 1 post to be "High").

**Composite profile string:** `"R=Low | C=High | E=Mid | D=Low | T=High | G=Low"` (RCEDTG).

### KMeans Clustering (k=4)

- Input: per-user profile level counts (how many windows each user was Low/Mid/High per dimension)
- Normalisation: `StandardScaler` on count columns
- `KMeans(n_clusters=4, n_init=50, random_state=42)`
- Silhouette score computed; cluster labels mapped to interpretable names

### Outputs

| File | Content |
|---|---|
| `ig_user_rolling_quadrimesters_new_definitions.csv` | All metrics per (user, window) |
| `Ig_RCEDTG_classification.csv` | Level labels per (user, window) |
| `Ig_RCEDTG_kmeans_k4_user_clusters.csv` | User → cluster assignment |
| `Ig_RCEDTG_k4_monthly_cluster_matrix.csv` | Wide matrix: user × month → cluster label |
| Transition matrix PNGs | Per 4-month block, Jan/May/Sep checkpoints |

---

## 19. Mickey — TK Rolling Quadrimesters RFM (`RFM_TK_rolling_quadrimesters/tk_rolling_quadrimesters.ipynb`)

### Purpose
Same rolling-window RFM pipeline as IG but adapted for TikTok's schema and data characteristics.

### Key Differences from IG Pipeline

| Aspect | Instagram | TikTok |
|---|---|---|
| Comment ID column | `comment_id` | `cid` |
| Author ID column | `from_id` | `uid` |
| Timestamp column | `timestamp` (string ISO) | `create_time` (Unix epoch seconds → `pd.to_datetime(..., unit='s')`) |
| Reply detection | `is_reply==True OR parent_id != ""` | `reply_id` and `reply_to_reply_id` both zero/empty |
| Creator filter | `is_creator==True OR from_username == "camihawke"` | `uid == CREATOR_UID` OR `nickname == "camihawke"` |
| Tenure | Continuous days | **Binary**: 1 if user has any comment before window start, 0 otherwise |
| Features | R, C, E, D, T, G | R, C, E, G, tenure\_binary (no Delay) |
| Final k | 4 | 3 |

### Data Cleaning — TK-Specific Steps

| Step | Operation |
|---|---|
| Epoch timestamp | `pd.to_numeric(create_time, errors='coerce')` then `pd.to_datetime(..., unit='s')` |
| Reply filter | Keep only rows where both `reply_id` and `reply_to_reply_id` are zero or empty string (`is_zeroish()`) |
| No min-activity filter | All users retained (unlike IG which requires `> 1` comment) |

### Feature Engineering — Window Metrics (TK)

Same `recency`, `coverage`, `engagement`, `gini` formulas as IG. `tenure_binary` replaces the continuous `tenure` and `delay` is omitted.

### KMeans Clustering (k=3)

- Profile counts: how many windows at each level per user per dimension
- `StandardScaler` + `KMeans(n_clusters=3, n_init=50, random_state=42)`
- Clusters named: C1=Brand Loyalists, C2=Established Supporters, C3=Peripheral Audience

### Outputs

| File | Content |
|---|---|
| `tk_user_rolling_quadrimesters.csv` | Metrics per (user, window) |
| `tk_RCEGT_classification.csv` | Level labels per (user, window) |
| `tk_RCEGT_kmeans_k3_user_clusters.csv` | User → cluster |
| `tk_RCEGT_k3_monthly_cluster_matrix.csv` | Wide matrix: user × month → cluster |

---

## 20. Mickey — IG Non-Overlapping 3-Year Periods (`RFM_IG_3_years/non_overlapping_3y_periods_new_definitions.ipynb`)

### Purpose
Same RCEDTG feature set as the rolling notebook but computed over three fixed non-overlapping multi-year periods (2017–2019, 2020–2022, 2023–2026) to study long-run cohort behaviour.

### Data Cleaning — Additional Steps vs Rolling Notebook

| Step | Operation |
|---|---|
| Wider date range | All comments from `2017-01-01` retained (vs `2022-09-01` for rolling) |
| Same midnight backoff | Post timestamps at `00:00:00` corrected to `first_comment_ts - 2min` |
| Same creator/reply/empty filter | Identical logic |

### Feature Engineering — Period Metrics

Same features as rolling (R, C, E, D, T, G) but `period_*` columns instead of `window_*`. Thresholds recomputed per the full population of each period rather than global.

### Outputs

- `ig_user_nonoverlapping_3y_periods_new_definitions.csv`
- `Ig_RCEDTG_classification_nonoverlapping_3y.csv`

---

## 21. Mickey — EDA: RFM IG vs TK Quadrimesters (`EDA_RFM_IG_vs_TK_rolling_quadrimesters.ipynb`)

### Purpose
Cross-platform exploratory analysis of the RCEDTG (IG) and RCEGT (TK) rolling-quadrimester feature tables.

### Inputs

- `RFM_IG_rolling_quadrimesters/ig_user_rolling_quadrimesters_new_definitions.csv` — 174,287 rows (18,752 users × 40 windows)
- `RFM_TK_rolling_quadrimesters/tk_user_rolling_quadrimesters.csv` — 56,300 rows (13,249 users × 40 windows)

### Data Cleaning

| Step | Operation |
|---|---|
| Quality report | `dtype`, `n_missing`, `pct_missing`, `n_zero`, `pct_zero`, `skew` per feature |
| No nulls found | Both tables have 0 missing values across all features |
| Log transformation | `engagement`, `delay`, `tenure` transformed via `log1p` for KDE and correlation |

### Feature Engineering / Analytical Derivations

| Derived object | Logic |
|---|---|
| Distribution-shift tests | KS test + Mann-Whitney U between IG and TK for each common feature |
| Within-platform Pearson + Spearman correlation matrices | Computed for all feature pairs; strongest: `coverage` ↔ `gini` (ρ=0.78 on IG) |
| VIF (Variance Inflation Factor) | Computed per feature to detect multicollinearity; `coverage` has highest VIF on both platforms (IG: 4.8, TK: 3.1) |
| KMeans k=4 evaluation | `StandardScaler` + KMeans on IG and TK separately; silhouette, similarity/dissimilarity coefficients per cluster |
| Window-level aggregates | Per-window mean of each feature → 40-row time series for each platform |
| Cross-platform window correlation | Spearman correlation of IG window-means vs TK window-means for shared features |
| Detrended cross-platform check | First-difference (`diff()`) of window means before correlating → removes shared growth trend |

### Key Quantitative Findings Documented

| Finding | Value |
|---|---|
| `coverage` ↔ `gini` Spearman (IG) | ρ = 0.78 |
| `gini` ↔ `tenure_binary` Spearman (TK) | ρ = 0.497 |
| IG delay skew | 21.03 (extreme right tail) |
| TK gini zeros | 95.3% of rows (single-comment users dominate) |
| Detrended coverage co-movement (IG vs TK) | ρ = 0.33, p = 0.039 (weakly significant) |
| Detrended engagement co-movement | ρ = 0.22, p = 0.18 (not significant) |

---

## 22. Mickey — EDA: Cluster Matrix IG vs TK (`EDA_cluster_matrix_IG_vs_TK.ipynb`)

### Purpose
Lifecycle and cross-platform analysis of the wide user × month cluster assignment matrices.

### Inputs

- `RFM_TK_rolling_quadrimesters/tk_RCEGT_k3_monthly_cluster_matrix.csv` — 13,249 users × 40 months
- `RFM_IG_rolling_quadrimesters/Ig_RCEDTG_k4_monthly_cluster_matrix.csv` — 18,752 users × 39 months

### Data Cleaning

| Step | Operation |
|---|---|
| Wide → long melt | `melt(id_vars=[user_key, user_id, username], var_name='month', value_name='state')` |
| Null fill | `fillna("Not yet active")` for months before user's first activity |
| Timestamp parsing | Month column strings parsed via `pd.to_datetime(month, format="%Y-%m")` |
| State canonicalisation | `state_class = "Active"` if state starts with "C", else state verbatim |
| Shared window filter | Analysis window clipped to `2023-01 → 2026-03` for cross-platform comparison |

### Feature Engineering — Lifecycle Metrics

| Feature | Formula |
|---|---|
| `active_months` | Count of months where `state_class == "Active"` after debut |
| `inactive_months` | Count of months where `state_class == "Inactive"` after debut |
| `observed_span_months` | Total months from debut to end of data |
| `pct_active` | `active_months / observed_span_months` |
| `re_entry` | Boolean: user went Inactive then Active at least once |
| `entropy` | Shannon entropy (bits) of cluster assignment sequence across active months |
| `n_unique_clusters` | Number of distinct cluster labels the user ever occupied |
| `cluster_switch` | Boolean per consecutive-active-month pair: cluster changed |
| Survival span | Months from debut to last active month; `event=1` if not active in final month |

### Key Findings Documented

| Metric | TikTok | Instagram |
|---|---|---|
| Mean active months | 4.25 | 9.02 |
| Re-entry rate | 6.4% | 63.0% |
| Dominant state | Inactive (49.4%) | Inactive (61.3%) |
| C3 share of active | 88.7% | 69.0% |
| Overall cluster-switch rate | Computed per run | Computed per run |
| Detrended coverage co-movement | ρ = 0.33 (p=0.04) | — |

### Cross-Platform Cluster Share Correlation

- C1 share (IG) vs C1 share (TK): Spearman ρ = 0.341, p = 0.033 (significant in raw shares)
- First-difference (detrended): all cross-cluster pairs non-significant (p > 0.10)
- Conclusion: raw correlation driven by shared growth trend; genuine shock co-movement not detected

---

## 23. Mickey — User Cluster Journey Analysis (`Mickey/User_Cluster_Journey_Analysis.ipynb`)

### Purpose
Track how individual users move between persona clusters (C1/C2/C3/C4) across monthly periods and compute state transition matrices at quarterly checkpoints.

### Inputs

- `Ig_RCEDTG_k4_monthly_cluster_matrix.csv` or TK equivalent (wide matrix)
- `tk_RCEGT_k3_monthly_cluster_matrix.csv`

### Data Cleaning

| Step | Operation |
|---|---|
| Wide → long | Same `melt` + `fillna("Not yet active")` as cluster matrix EDA |
| Sort | `sort_values(["user_key","window_index"])` before any sequential operations |
| Spell detection | `shift(1)` on cluster_id per user to detect state changes; `cumsum` on change flag = spell number |

### Feature Engineering — Spells and Transitions

| Feature | Formula |
|---|---|
| `spell_number` | Cumulative count of state changes per user |
| `spell_start/end` | Min/max `window_index` within a spell |
| `duration_updates` | Length of spell in months |
| `next_state_id` | `shift(-1)` on `state_id` per user |
| `is_censored` | Boolean: `next_state_id.isna()` (user's last spell) |
| Discrete hazard | `P(exit at month t | survived to t)` per state, per duration-in-state |

**Transition matrices:** Row-normalised count tables `P(state_to | state_from)` computed at Jan/May/Sep × year checkpoint blocks.

**Sankey diagram:** User flows across 10 quarterly checkpoints (Jan 2023 → Jan 2026), showing volume of users moving between C1/C2/C3/Inactive/Not-yet-active.

---

## 24. Root — Patch Scripts

These scripts apply targeted corrections to previously computed outputs. They are data corrections, not primary feature engineering.

| Script | Purpose |
|---|---|
| `patch_full_user_comments.py` | Re-attach full comment text that was truncated |
| `patch_noise_toxicity.py` | Correct toxicity label assignments |
| `patch_persona_desc_before_viz.py` | Update persona descriptions before visualisation |
| `patch_persona_viz.py` | Patch persona visualisation metadata |
| `patch_remove_rationale.py` | Strip rationale fields from outputs |
| `patch_taxonomy_source.py` | Add taxonomy source tracking |
| `patch_token_limit.py` | Enforce token limits on text fields |
| `fix_col_rename.py` | Rename columns for schema consistency |
| `fix_fstring.py` | Fix f-string formatting issues |
| `fix_pb_label.py` | Correct Pathway B label assignments |
| `fix_sentiment_features.py` | Fix sentiment feature column errors |
| `fix_sent_fallback.py` | Add fallback logic for missing sentiment values |

These scripts apply targeted corrections to previously computed outputs. They are data corrections, not primary feature engineering.

| Script | Purpose |
|---|---|
| `patch_full_user_comments.py` | Re-attach full comment text that was truncated |
| `patch_noise_toxicity.py` | Correct toxicity label assignments |
| `patch_persona_desc_before_viz.py` | Update persona descriptions before visualisation |
| `patch_persona_viz.py` | Patch persona visualisation metadata |
| `patch_remove_rationale.py` | Strip rationale fields from outputs |
| `patch_taxonomy_source.py` | Add taxonomy source tracking |
| `patch_token_limit.py` | Enforce token limits on text fields |
| `fix_col_rename.py` | Rename columns for schema consistency |
| `fix_fstring.py` | Fix f-string formatting issues |
| `fix_pb_label.py` | Correct Pathway B label assignments |
| `fix_sentiment_features.py` | Fix sentiment feature column errors |
| `fix_sent_fallback.py` | Add fallback logic for missing sentiment values |

---

## 25. End-to-End Data Flow Summary

### Critical Intermediate Files

| File | Produced By | Consumed By |
|---|---|---|
| `comments_ml_{platform}.parquet` | `Data_Preparation_Pipeline` | `persona/features.py`, `clustering.py` |
| `comments_llm_{platform}.jsonl` | `Data_Preparation_Pipeline` | `sentiment_pipeline.ipynb`, `persona/features.py` |
| `comments_gml_{platform}.parquet` | `Data_Preparation_Pipeline` | HeteroGraph construction |
| `sentiment_{platform}.parquet` | `sentiment_pipeline.ipynb` | `sentiment_analysis.py`, `persona/features.py` (optional) |
| `user_features_{platform}.parquet` | `persona/features.py` | `clustering.py` |
| `ig_multimodal_final.parquet` | `build_final_dataset.py` | `persona_pipeline.ipynb` (media context) |
| `final_taxonomy.json` | `clustering.py` | Stage-2 classification prompt |
| `user_micro_personas.parquet` | `clustering.py` | `User_Cluster_Journey_Analysis`, RFM notebooks |
| `yt_videos_with_local_transcripts.parquet` | `add_transcripts.py` | `reclean_and_rebuild_jobs.py`, `prepare_transcript_jobs.py`, `extract_entities.py` |
| `thumbnail_features.parquet` | `thumbnail_features.py` | YouTube EDA, downstream models |
| `entity_resonance.parquet` | `extract_entities.py` | YouTube EDA, content analysis |

### Data Volume Estimates

| Stage | Platform | Approx. Rows |
|---|---|---|
| Raw comments | IG | 573,377 |
| Raw comments | FB | 394,084 |
| Raw comments | TK | 19,654 |
| IG multimodal final | IG | 1,501 posts |
| Thumbnail features | YT | ~25,000 thumbnails |

---

## 26. Critical Assessment: Does It Make Sense?

### What Works Well

**Comment-level text features (Section 2)**
The 14 features are all derivable from raw text without external data and map to interpretable user behaviors: expressiveness (emoji entropy, variety ratio), urgency (exclamation, question), promotional activity (url_count, hashtag_count). The emoji Shannon entropy is a notably good choice — it distinguishes "spam one emoji repeatedly" from "genuinely expressive". The `avg_word_length` proxy for vocabulary formality is a reasonable heuristic. Overall this feature set is solid.

**User-level behavioral aggregations (Section 4)**
The aggregation logic is clean and the features collectively capture five orthogonal behavioral axes: **volume** (`total_comments`), **breadth** (`unique_posts_commented`), **responsiveness** (`reply_ratio`, `hours_to_comment` percentiles), **text style** (`mean_word_count`, `emoji_usage_rate`), and **engagement concentration** (`post_concentration_ratio`). Clustering on these will yield meaningfully different archetypes.

**Transcript cleaning pipeline (Sections 10–12)**
The multi-pass VTT cleaning is thorough and correctly ordered (HTML unescape first, then structural removal, then deduplication). The sliding-window consecutive-duplicate deduplication is necessary for VTT files where the same caption line appears in overlapping cue windows — this is a known VTT artefact and is handled correctly.

**Thumbnail CV pipeline (Section 13)**
Using CLIP cosine similarity between the image and the video title to measure `clip_title_align` is a smart feature for YouTube content strategy analysis. Face pose via solvePnP from FaceMesh landmarks is sophisticated and appropriate for the thumbnail click-bait analysis use case.

---

### Potential Issues and Gaps

#### 1. `post_concentration_ratio` clipping is backwards
```python
post_concentration_ratio = unique_posts_commented / total_comments  # clipped to [0, 1]
```
This is always `≤ 1` by construction (you cannot comment on more unique posts than you have total comments), so the clip is redundant. More importantly, this is actually a **diversity ratio** — higher values mean the user spreads comments across many posts, lower means they concentrate on few. The name is misleading: "concentration" implies high = concentrated, but the formula gives high = dispersed. **Recommend renaming to `post_diversity_ratio` or inverting the formula.**

#### 2. `hours_to_comment` filled with 0 when media timestamps are missing
```python
hours_to_comment.fillna(0)
```
Filling with `0` implies "user commented immediately after posting" — which conflates a missing value with the most extreme engagement behavior. This will bias `pct_comments_under_1h` upward for users whose post timestamps are unavailable. **Recommend filling with the median or dropping when used for percentile-based features.**

#### 3. `emoji_entropy` undefined for single-emoji comments
Shannon entropy of a distribution with one event is `0` (not undefined), which is correct mathematically. However it means a comment with 10 identical emojis gets the same entropy as a comment with 1 emoji. The `emoji_variety_ratio` (unique/total) partially compensates, but the two features are correlated in this edge case. Not a blocking issue, but worth noting.

#### 4. Thumbnail merge has no deduplication
`merge_thumbnail_features.py` concatenates GCS shards with `ignore_index=True` but applies no deduplication. If a video's thumbnail was processed in multiple GCS batch jobs (e.g. after a retry), it will appear twice in `thumbnail_features.parquet`. **Recommend a `drop_duplicates('video_id')` after concat.**

#### 5. `activity_span_days` is 0 for single-comment users
Users with exactly one comment get `(max - min) = 0` days, which is accurate but groups all single-comment users together regardless of *when* they commented. This matters for churn analysis — a user who commented once in 2021 looks identical to one who commented once last week. **Recommend adding `days_since_last_comment` as a complementary recency feature.**

#### 6. Sentiment pipeline schema: `sarcasm_label` and `sarcasm` are redundant
The output has both a `sarcasm` boolean and a `sarcasm_label` categorical. If `sarcasm_label` is just a string form of the boolean (e.g. "sarcastic" / "not sarcastic"), one of them is redundant. If `sarcasm_label` has more categories (e.g. "irony", "hyperbole"), it should be documented. As-is, the schema is ambiguous.

#### 7. No cross-platform normalisation of text features
The 14 comment-level features are computed independently per platform. TikTok comments may be structurally shorter (due to platform norms), which means raw `text_length` and `word_count` are not comparable across platforms. If any downstream model trains on multi-platform data, these features should be z-score normalised within platform before joining.

#### 8. RFM "Monetary" proxy is underspecified
The RFM notebooks define **Monetary** as "engagement value (likes received, sentiment weights, etc.)" but the exact formula varies by notebook and is not standardised. A commenter's received likes is a post-level metric (likes on the post, not on the comment itself) unless comment-level likes are available. If comment-level likes are not available for IG, this is a proxy that should be documented clearly.

#### 9. Entity resonance table has no normalisation
`entity_resonance.parquet` stores raw `transcript_count` and `comment_count`. These are not normalised by transcript length or total comment volume, so a longer transcript will produce higher counts simply by being longer. **Recommend adding `transcript_count_per_1k_tokens` and `comment_count_per_1k_comments` normalised variants.**

---

### Overall Verdict

**Yes, the data cleaning and feature engineering make sense** for the project's goals (social media audience persona discovery + YouTube content analysis). The architecture is well-structured, the feature choices map to interpretable audience behaviors, and the cleaning passes handle real-world artefacts (VTT duplicate cues, HTML entities, platform schema divergence) correctly.

The issues identified above are fixable without redesigning the pipeline:
- **High priority:** `hours_to_comment` fillna(0) bias, thumbnail deduplication gap
- **Medium priority:** `post_concentration_ratio` naming inversion, entity count normalisation
- **Low priority:** `activity_span_days` for single-comment users, sarcasm schema clarification, cross-platform normalisation

None of the issues invalidate the feature set; they are calibration and documentation gaps rather than fundamental design flaws.
