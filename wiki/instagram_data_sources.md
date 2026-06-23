---
title: Instagram data sources
type: source-summary
sources:
  - Data/ig_posts_cleaned.parquet
  - Data/ig_comments_cleaned.parquet
  - Data/ig_posts_media_synced.parquet
related:
  - "[[instagram_findings]]"
  - "[[sentiment-pipeline]]"
  - "[[persona-pipeline]]"
created: 2026-06-14
updated: 2026-06-14
confidence: high
---

# Instagram data sources

## Raw cleaned datasets

These are the canonical source tables from the data ingestion pipeline. All downstream analyses depend on these.

### ig_posts_cleaned.parquet

| Column | Type | Description |
|---|---|---|
| media_id | string | Instagram post ID (unique key) |
| username | string | Creator username |
| caption | string | Post caption (may be null) |
| post_url | string | Instagram post URL |
| timestamp | datetime | Post publication timestamp |
| like_count | int64 | Number of likes at collection time |
| comments_count | int64 | Number of comments at collection time |
| media_type | string | IMAGE / VIDEO / CAROUSEL_ALBUM |
| video_duration | float64 | Duration in seconds (null for images) |
| has_audio | bool | Whether video has audio track |
| is_paid_partnership | bool | Sponsored post flag |
| n_hashtags | int64 | Count of hashtags in caption |
| n_mentions | int64 | Count of @mentions in caption |

**Size**: 1,493 posts | **Key**: media_id (unique)

**Coverage**: Posts with collected comments (not all posts in creator's feed).

### ig_comments_cleaned.parquet

| Column | Type | Description |
|---|---|---|
| comment_id | string | Unique comment identifier |
| media_id | string | FK to posts |
| author_id | string | Commenting user ID |
| username | string | Commenter username |
| text | string | Comment body |
| timestamp | datetime | Comment publication timestamp |
| likes | int64 | Comment-level likes |
| reply_count | int64 | Replies to this comment |
| media_context | bool | Whether comment is on the target post vs deleted/archived post |

**Size**: ~3.66M comments | **Key**: comment_id (unique)

**Note**: The sentiment pipeline filters this to focus on comments where `media_context = True` (comments on intact posts), yielding the 487,604 analysed comments.

### ig_posts_media_synced.parquet

**Purpose**: Media file synchronisation metadata.

| Column | Type | Description |
|---|---|---|
| media_id | string | Post ID |
| local_filepath | string | Path to downloaded media file |
| url_source | string | Original Instagram CDN URL |
| file_hash | string | Hash of downloaded file (for dedup) |
| download_status | string | success / failed / retry |

**Size**: 1,493 rows

---

## Intermediate pipeline outputs

### Multimodal enrichment

**Location**: `Ali/Output/`

- `ig_multimodal_final.parquet` (1,493 rows) — Posts with extracted audio transcripts, frame embeddings, music metadata
- `ig_posts_multimodal_enriched.parquet` (1,493 rows) — Same posts with feature-engineered multimodal signals
- `ig_media_features.parquet` (1,493 rows) — Extracted audio/visual/music features per post

**Schema**: All inherit media_id as key. Additional columns include:
- video_transcript (whisper-extracted from audio)
- music_name, music_artist (shazam/metadata lookup)
- visual_embedding (frame-level vision features)
- carousel_video (boolean flag for carousel with video slides)
- media_product_type (FEED vs REELS distinction)

---

## Comment-level features

### comments_ml.parquet

Linguistic and structural features extracted from every comment text.

| Column | Type | Description |
|---|---|---|
| comment_id | string | Key |
| author_id | string | FK to user |
| media_id | string | FK to post |
| platform | string | "instagram" |
| text_length | int64 | Character count |
| word_count | int64 | Word count |
| emoji_count | int64 | Count of emoji characters |
| unique_emoji_count | float64 | Distinct emoji types |
| emoji_entropy | float64 | Shannon entropy of emoji distribution |
| emoji_variety_ratio | float64 | Diversity of emoji types relative to count |
| emoji_per_word_ratio | float64 | Emoji density |
| url_count | int64 | Number of URLs |
| mention_count | int64 | @mention count |
| hashtag_count | int64 | #hashtag count |
| exclamation_count | int64 | ! count |
| question_count | int64 | ? count |
| avg_word_length | float64 | Mean characters per word |
| has_numbers | bool | Contains numeric digits |
| has_links | bool | Contains URLs |
| timestamp | datetime | Comment timestamp |

**Size**: 3,661,495 rows | **Key**: comment_id

**Coverage**: All comments (not filtered to media_context = True). Joins to sentiment on comment_id.

---

## Analysis-ready master tables

### stage2_sentiment/sentiment_instagram.parquet

**Source**: Vertex AI Batch processing of comment body through Claude LLM.

| Column | Type | Values |
|---|---|---|
| comment_id | string | Key |
| sentiment | string | positive / neutral / negative |
| sentiment_score | float64 | [0, 1] confidence |
| emotion | string | joy / trust / sadness / anger / disgust / anticipation / fear / surprise / neutral |
| intensity | string | low / medium / high |
| sarcasm | bool | True / False |
| toxicity | string | **ordinal**: none / mild / severe |
| intent | string | praise / support / question / criticism / spam_promo / other |
| target | string | creator / content_work / appearance / other_user / product / off_topic / none |
| lang | string | Language code (mostly "it" for Italian) |

Plus inherited columns from comments_ml (text_length, word_count, emoji_count, etc.)

**Size**: 487,604 rows (filtered to media_context = True, active posts)

**Key**: comment_id

---

### stage2_sentiment/room_vibe_instagram.parquet

Post-level vibe aggregation from comment-level sentiment.

| Column | Type | Description |
|---|---|---|
| media_id | string | Key |
| n_comments | int64 | Comment count on post |
| media_context | bool | Post is active (not deleted) |
| vibe_score | float64 | [0, 1] post-level positivity |
| dispersion | float64 | Sentiment spread (variance) |
| pos_frac / neg_frac / neu_frac | float64 | Sentiment composition |
| stance_alignment | float64 | Audience cohesion |
| polarization | float64 | Sentiment conflict |
| sarcasm_rate | float64 | Share of comments with sarcasm flag |
| toxicity_rate | float64 | Share of toxic comments |
| room_state | string | united / mixed / fragmented |
| llm_vibe | string | appreciative / critical / mixed / celebratory / supportive |
| llm_consensus | float64 | [0, 1] LLM confidence in vibe assessment |
| llm_alignment | float64 | Agreement between comment sentiment and room vibe |
| llm_controversy | float64 | [0, 1] controversy score |

**Size**: 1,501 rows | **Key**: media_id

---

### stage2_sentiment/room_llm_instagram.parquet

Secondary LLM pass extracting dominant stance and narrative summaries.

| Column | Type | Description |
|---|---|---|
| media_id | string | Key |
| dominant_stance | string | Natural language summary of audience stance |
| split_axis | string | If mixed: the dimension of disagreement |
| room_summary | string | Narrative summary of comment room character |

**Size**: 1,501 rows | **Key**: media_id

---

## HeteroGraph representation

**Location**: `Ali/Output/Prepared Comments/HeteroGraph/`

Structured knowledge graph of comments, users, and relationships.

### Nodes

- `nodes_author.parquet` — User nodes (author_id, username, account properties)
- `nodes_comment.parquet` — Comment nodes (comment_id, text, sentiment, timestamp)
- `nodes_media.parquet` — Post nodes (media_id, caption, likes, timestamp)
- `nodes_author_ig_tags.parquet` — User tags (hashtags in bio, creator tags)

### Edges

- `edges_posted.parquet` — author → media (user posted the media)
- `edges_replies_to.parquet` — comment → comment (reply structure)
- `edges_tagged.parquet` — comment → user (mentioned in comment)
- `edges_belongs_to.parquet` — comment ∈ media (comment on post)
- `edges_coauthored.parquet` — user ↔ user (users who comment on same posts)
- `edges_derived_from.parquet` — repost / shared media links
- `edges_similar_to.parquet` — comment ~ comment (cosine similarity on embeddings)

**Purpose**: Multi-view community analysis, network-based anomaly detection, influence propagation.

---

## Summary table: source → processed pipeline

| Raw source | Processing | Output |
|---|---|---|
| ig_posts_cleaned | multimodal extraction (yt-dlp, whisper, music lookup) | ig_multimodal_final |
| ig_comments_cleaned | linguistic feature engineering | comments_ml |
| ig_comments_cleaned + comments_ml | Vertex Batch LLM | sentiment_instagram |
| ig_posts_cleaned + sentiment (aggregated) | room-level LLM vibe | room_vibe_instagram |
| sentiment + room_vibe | narrative LLM pass | room_llm_instagram |
| all of above | graph construction | HeteroGraph edges/nodes |
