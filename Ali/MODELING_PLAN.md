# Instagram Modeling Plan (for Sonnet)

**You are Sonnet. Read this file fully before doing anything. Follow it literally.**
This plan was written by an Opus agent that already inspected every dataset below. The
schemas, row counts, dtypes, and join keys stated here are VERIFIED FACTS as of
2026-06-14 — trust them, but re-print `df.dtypes`/`df.shape` once after each load to
guard against silent drift. Do not go exploring beyond the files listed in §1.

---

## 0. Scope & guardrails (READ FIRST)

- **Instagram ONLY.** Every dataset must be filtered to `platform == "instagram"`
  (where the column exists) before use. Never let YouTube/TikTok/Facebook rows leak in.
  `comments_ml.parquet` is 3.66M rows but only **499,752** are Instagram.
- **Goal:** build a *modeling* layer on top of the already-computed sentiment, persona,
  and post features. You are NOT re-running any LLM/Vertex batch job. All LLM outputs
  already exist as parquet. You consume them.
- **No new cloud calls. No Vertex. No GCS.** Everything is local under `Ali/outputs/`
  and `Data/`. If a path you need isn't in §1, STOP and ask — do not invent one.
- **Work in a new notebook** `Ali/modeling_pipeline.ipynb` (create it). Do not edit
  `persona_pipeline.ipynb` or `sentiment_pipeline.ipynb` — those are upstream producers.
- Save all model artifacts/figures under `Ali/outputs/modeling/` (create the dir).

### Files you are STRICTLY FORBIDDEN to read (irrelevant — will waste tokens & mislead)
- Anything under `Mickey/`, `reza/`, or any non-Ali contributor folder.
- Any YouTube / TikTok / Facebook parquet in `Data/` (`YTcomments_*`, `tk_*`, `fb_*`).
- `Ali/Backup/`, `Ali/Archive/`, `Ali/batch_results/` — these are stale duplicates.
- `Ali/IG_Download.ipynb`, `ig_refetch.py`, any download/scraping code.
- The `persona_pipeline.ipynb` / `sentiment_pipeline.ipynb` internals (Vertex/batch
  code). You only need their OUTPUT parquets, already described in §1. Read those two
  notebooks ONLY IF a column meaning is genuinely unclear after §1 — and then read only
  the markdown cells, not the batch-submission code.

---

## 1. Canonical data sources (these and ONLY these)

| # | Path | Rows | Grain | Use |
|---|------|------|-------|-----|
| A | `Ali/outputs/comments_ml.parquet` | 3.66M (499,752 IG) | one comment | base behavioral/text features per comment. **Filter `platform=="instagram"`.** |
| B | `Ali/outputs/stage2_sentiment/sentiment_instagram.parquet` | 487,604 | one comment | LLM sentiment/affect labels + comment features + `text`. Already IG-only. |
| C | `Ali/outputs/stage2_persona_combined/user_personas_combined.parquet` | 40,019 | one user (`author_id`) | persona label + behavioral aggregates per user. |
| D | `Ali/outputs/ig_multimodal_final.parquet` | 1,493 | one post (`media_id`/`shortcode`) | post-level multimodal metadata (music, tags, paid-partnership, etc.). |
| E | `Data/ig_posts_cleaned.parquet` | 1,493 | one post (`media_id`) | post engagement metrics (`like_count`, `reach`, `views`, `saved`, `total_interactions`, `comments_count`). Use D as primary; pull engagement cols from E if missing in D. |
| F (optional) | `Ali/outputs/promo_reception_instagram.parquet` | ~small | one promo post | precomputed promo reception labels — context only, optional. |

**Do NOT use** `user_features_instagram_cache.parquet` or
`pathway_b_assignments_instagram.parquet` unless a task below explicitly references them
(they don't). Ignore them.

### Verified schemas (columns you will actually use)

**A — comments_ml** (IG slice): `comment_id`, `author_id`, `media_id`, `platform`,
`text_length`, `word_count`, `emoji_count`, `unique_emoji_count`, `emoji_entropy`,
`emoji_variety_ratio`, `emoji_per_word_ratio`, `url_count`, `mention_count`,
`hashtag_count`, `exclamation_count`, `question_count`, `avg_word_length`,
`has_numbers`, `has_links`, `timestamp`.

**B — sentiment_instagram**: `comment_id`, `media_id`, `media_context`, `sentiment`
(positive/negative/neutral), `sentiment_score` (-1..1), `emotion` (8-way Plutchik +
neutral), `intensity`, `sarcasm` (bool), `toxicity`, `intent` (praise/joke/support/
affection/tag_share/other/question/suggestion/...), `target` (content_work/creator/
other_user/appearance/product/none/off_topic), `lang`, `sentiment_cat`, `sarcasm_label`,
`author_id`, `platform`, + all comment features from A, + `text`.
→ **B already contains A's features + the LLM labels + text.** For per-comment modeling,
B is your richest single table. Use A only if you need the ~12k IG comments that are in
A but not in B (B is a sampled subset: 487k of 499k).

**C — user_personas_combined**: `author_id`, `total_comments`, `activity_span_days`,
`mean_hours_to_comment`, `pct_comments_under_1h`, `reply_ratio`, `mean_word_count`,
`persona_codename` (10 classes, see §3), `confidence`, `justification`.

**D — ig_multimodal_final**: `media_id`, `shortcode`, `caption`, `like_count`, `reach`,
`views`, `saved`, `total_interactions`, `comments_count`, `media_type`,
`media_product_type`, `content_form`, `n_hashtags`, `n_mentions`, `music_source`,
`audio_type`, `song_title`, `artist`, `is_explicit`, `n_tagged`, `n_coauthors`,
`n_product_tags`, `is_paid_partnership`, `sponsor_usernames`, `timestamp`, `year`,
`month`, `dayofweek`, `hour`, `has_audio`, `video_duration`, `location_name`, etc.

### VERIFIED join keys (all string dtype — keep them string, never cast to int)
- comment ↔ sentiment: `comment_id` (e.g. `ig_comment_17845...`).
- comment/sentiment ↔ user: `author_id` (bare numeric string, e.g. `1445738050663508`).
  **98.2%** of persona users appear in B. ~2% won't — left-join, expect NaN persona.
- comment/sentiment ↔ post: `media_id` (bare numeric string, e.g. `17878142638896365`).
  **99.5%** of B's media_ids are in D. ~0.5% won't — left-join, expect NaN post meta.
- post D ↔ post E: `media_id`. post ↔ shortcode: `shortcode` (only if you need media tree; you don't).

> **id-space landmine (cost a real prior investigation):** `comments_ml`/sentiment/
> multimodal all use **bare-numeric** `media_id` & `author_id`. Some OTHER files in the
> repo (HeteroGraph `nodes_comment`) use **prefixed** `ig_media_<n>`/`ig_author_<n>`.
> You are NOT using those files, so you won't hit this — but it's WHY you must not pull
> in any HeteroGraph/nodes file. Stick to §1.

---

## 2. Build the modeling tables (do this first, before any model)

Create three clean, reusable frames and save each to `Ali/outputs/modeling/`:

1. **`comments_model.parquet`** — per-comment. Start from **B** (it has features + labels
   + text). Left-join **C**'s `persona_codename` (+ `confidence`) on `author_id`.
   Left-join selected **D** post columns on `media_id`: `content_form`, `media_type`,
   `is_paid_partnership`, `music_source`, `audio_type`, `n_hashtags`, `like_count`,
   `reach`, `views`, `total_interactions`, post `timestamp` (rename to `post_timestamp`).
   Result grain = one comment, enriched with its author's persona and its post's metadata.

2. **`users_model.parquet`** — per-user. Start from **C**. Aggregate B up to `author_id`:
   mean/std `sentiment_score`, `%positive/%negative/%neutral` (`sentiment_cat`),
   `sarcasm_rate`, mean `toxicity`, dominant `emotion`, dominant `intent`,
   dominant `target`, mean `intensity`, n_distinct `media_id` (post diversity).
   Join those onto C. This is the table for user-level persona modeling.

3. **`posts_model.parquet`** — per-post. Start from **D** (engagement + metadata).
   Aggregate B up to `media_id`: comment count, mean `sentiment_score`, %pos/%neg,
   `controversy = 4·%pos·%neg`, sarcasm_rate, toxicity_rate, dominant emotion/intent.
   Join onto D. This is the table for post-reception / engagement modeling.

Print `.shape`, null counts on join keys, and `value_counts` of each label column after
building each. Sanity-check: per-comment frame ≈ 487k rows; users ≈ 40k; posts ≈ 1.49k.

---

## 3. Class-imbalance & data-shape warnings (PAY EXTRA ATTENTION)

These are the things most likely to make a naive model look great but be useless:

- **Sentiment is massively positive-skewed:** positive 394k / neutral 69k / negative 25k
  (≈ 81% / 14% / 5%). A classifier predicting "always positive" scores 81% accuracy and
  is worthless. **Always report macro-F1 / balanced accuracy / per-class recall**, use
  `class_weight="balanced"` (or resampling), and stratify splits.
- **Persona classes are imbalanced** (THE_TAGGER 11.6k … THE_HATER 164, THE_SPAMMER 218).
  Same rule: macro metrics, stratified split, class weights. Consider grouping the rare
  tail or reporting them separately.
- **emotion / intent / target are multi-class & skewed** (joy & praise dominate). Same
  treatment.
- **B is a SAMPLE** (487k of 499k IG comments) and the sentiment run *prioritized
  media-bearing posts*. So B is NOT a uniform random sample of comments — do not present
  B-derived rates as unbiased population estimates of IG sentiment. State this caveat.
- **paid-partnership is rare** (~47 paid + ~130 promo-ish of 1,493 posts). Any
  promo-vs-organic model is small-N; use it descriptively, not as a heavy classifier.
- **posts_model has only 1,493 rows.** Engagement regression on ~1.5k rows with many
  metadata features overfits fast. Keep models simple (regularized linear / shallow
  tree), use CV, and prefer interpretation over raw accuracy.
- **`timestamp` columns:** comment `timestamp` and post `post_timestamp` are separate.
  Confirm dtype (likely datetime or epoch) on load; if epoch, convert with
  `pd.to_datetime(..., unit=...)` — check the magnitude before assuming s vs ms.
- **`sentiment_score` range:** already validated to [-1,1] upstream, but re-check
  `min/max`; clip if any stray values.
- **Leakage:** when modeling `sentiment` from features, `sentiment_score`/`sentiment_cat`/
  `emotion`/`intent` are all LLM-derived from the SAME pass — they are not independent
  predictors, they're near-duplicates of the target. Don't predict `sentiment` from
  `sentiment_score`. Predict labels from the *behavioral/text features* (A's columns) and
  the *post metadata* (D), not from sibling LLM labels.

---

## 4. Modeling tasks (in priority order)

Confirm the exact target with the user before heavy work if ambiguous, but the default
deliverables are:

### Task 1 — Persona classification (user-level, supervised)
Target `persona_codename` from `users_model` behavioral features
(`total_comments`, `activity_span_days`, `mean_hours_to_comment`,
`pct_comments_under_1h`, `reply_ratio`, `mean_word_count`, + the aggregated affect
features you built). Train a gradient-boosted tree (or logistic) with stratified CV,
`class_weight`/`scale_pos_weight` for imbalance. Report macro-F1, per-class recall,
confusion matrix, feature importances. **Interpretation is the product** — which behaviors
define each persona. (Note: persona labels were themselves LLM-assigned, so this is
"can behavior recover the LLM's persona" — frame it that way, not as ground truth.)

### Task 2 — Comment sentiment model (comment-level, supervised, leakage-careful)
Target `sentiment_cat` (3-class) from **non-LLM** features only: A's text/behavioral
features + post metadata from D + the author's `persona_codename`. Stratified split,
balanced weights, macro-F1 + per-class recall. Goal: how much of sentiment is predictable
from structure alone vs needing the text. Report which features matter.

### Task 3 — Post reception / engagement (post-level, regression, small-N)
On `posts_model` (1.49k rows). Two sub-questions: (a) does post metadata
(`content_form`, `music_source`, `is_paid_partnership`, `n_hashtags`, posting `hour`/
`dayofweek`, etc.) predict the *audience vibe* (mean `sentiment_score` / controversy)?
(b) does it predict *engagement* (`total_interactions`/`reach`)? Use regularized models +
CV; emphasize coefficients/SHAP over R². Flag small-N overfitting explicitly.

### Task 4 (optional, if time) — Persona × sentiment / persona × post-type analysis
Descriptive cross-tabs and a simple test: do personas differ in sentiment distribution?
Do certain content_forms attract certain personas? This is EDA-with-stats, not a model,
and is often the most useful business output.

### For every task
- Stratified train/test (and CV). Set `random_state`.
- Report **macro** metrics, never bare accuracy.
- Save: fitted model (joblib), metrics json, and figures (confusion matrix, feature
  importance) to `Ali/outputs/modeling/`.
- Write one short markdown cell per task interpreting the result in plain business terms.

---

## 5. Output deliverables
- `Ali/modeling_pipeline.ipynb` — the notebook, runs top-to-bottom headless.
- `Ali/outputs/modeling/{comments_model,users_model,posts_model}.parquet`
- Per-task model artifacts + metrics json + PNG figures under `Ali/outputs/modeling/`.
- A final markdown summary cell: key findings + the imbalance/sampling caveats from §3.

---

## 6. Definition of done
- Notebook runs clean top-to-bottom on the local machine (no Vertex/GCS).
- Every model reports macro-F1 / balanced metrics + per-class breakdown.
- §3 caveats are stated in the notebook, not just assumed.
- No forbidden file (§0) was read; only §1 sources were used.
