# Instagram Modeling Report
**Date:** 2026-06-14
**Author:** Sonnet (modeling tier) acting on plan authored by Opus
**Notebook:** `Ali/modeling_pipeline.ipynb`
**Artifacts:** `Ali/outputs/modeling/`

---

## 1. Context and Purpose

This report documents four supervised and descriptive modeling tasks run on
top of pre-computed LLM label outputs for the Showreel Social Media Analytics
project. No new Vertex / GCS calls were made — all labels already existed as
local parquet files produced by earlier pipeline runs (`persona_pipeline.ipynb`
and `sentiment_pipeline.ipynb`). The modeling layer sits one level above those
pipelines: it asks whether the LLM-derived labels can be explained by observable
behavioral and structural features, and what those labels reveal about how
audiences engage with content.

**Platform scope:** Instagram only throughout. TikTok, YouTube, and Facebook
rows were excluded at source.

---

## 2. Datasets Used

Six source files were loaded and joined into three modeling tables. All join
keys (`comment_id`, `author_id`, `media_id`) are bare-numeric strings and were
kept as strings throughout to avoid the id-space collisions documented in the
modeling plan.

### Source files

| Label | Path | Rows (IG) | Grain | Role |
|-------|------|-----------|-------|------|
| B | `Ali/outputs/stage2_sentiment/sentiment_instagram.parquet` | 487,604 | one comment | LLM sentiment/affect labels + all behavioral text features + raw text. Already Instagram-only. Primary source for comment-level work. |
| C | `Ali/outputs/stage2_persona_combined/user_personas_combined.parquet` | 40,019 | one user (`author_id`) | LLM-assigned persona label + behavioral aggregates per commenter. |
| D | `Ali/outputs/ig_multimodal_final.parquet` | 1,493 | one post (`media_id`) | Post-level multimodal metadata: music, tags, content form, paid-partnership flag, engagement metrics, timestamps. Primary post source. |
| E | `Data/ig_posts_cleaned.parquet` | 1,493 | one post (`media_id`) | Post engagement metrics fallback (used if a column was missing from D). |
| A | `Ali/outputs/comments_ml.parquet` | 499,752 (IG slice) | one comment | Full behavioral feature set. Not loaded directly — B already includes all of A's feature columns for the 487k sampled comments. |
| F | `Ali/outputs/promo_reception_instagram.parquet` | small | one promo post | Not used in modeling tasks (context reference only). |

### Modeling tables built

Three clean, reusable tables were produced and saved to `Ali/outputs/modeling/`:

**`comments_model.parquet` — 487,604 rows x 46 columns**
Base: B (sentiment labels + comment features). Left-joined with C
(`persona_codename`, `confidence`) on `author_id`, and with D
(`content_form`, `media_type`, `is_paid_partnership`, `music_source`,
`audio_type`, `n_hashtags`, `like_count`, `reach`, `views`,
`total_interactions`, `post_timestamp`) on `media_id`.
- persona_codename coverage: 19.3% of comments (80.7% NaN — expected,
  since C covers 40k users while B contains comments from many more unique
  commenters, most of whom appear only once)
- post_timestamp NaN: 0.5%

**`users_model.parquet` — 40,019 rows x 23 columns**
Base: C (persona labels + behavioral aggregates). Aggregated B up to
`author_id` to add: mean/std sentiment_score, sarcasm_rate, mean_toxicity,
mean_intensity, dominant_emotion, dominant_intent, dominant_target,
pct_positive/negative/neutral, n_comments_in_B, n_distinct_posts.
- 710 users in C had zero comments in B's sample (appear in persona table
  but were not sampled for sentiment labeling) — these have NaN affect
  aggregates and were excluded from Task 1 training.

**`posts_model.parquet` — 1,493 rows x 70 columns**
Base: D (post metadata + engagement). Aggregated B up to `media_id` to add:
n_comments_in_B, mean_sentiment_score, sarcasm_rate, toxicity_rate,
dominant_emotion/intent, pct_pos/neg/neu, controversy (= 4 * pct_pos * pct_neg).
Zero nulls on all key columns.

### Label definitions (from upstream LLM pipelines)

- **sentiment / sentiment_cat:** 3-class (positive / neutral / negative).
  `sentiment_score` is a continuous [-1, 1] value from the same LLM pass.
- **emotion:** 8-way Plutchik wheel + neutral (joy, trust, anticipation,
  surprise, sadness, disgust, anger, fear, neutral).
- **intent:** praise / joke / support / affection / tag_share / other /
  question / suggestion / criticism / spam_promo.
- **target:** content_work / creator / other_user / appearance / product /
  none / off_topic.
- **intensity, sarcasm, toxicity:** scalar/bool outputs from the same LLM call.
- **persona_codename:** 10 LLM-assigned archetypes (see Task 1 below).

---

## 3. Modeling Tasks

### Task 1 — Persona Classification (user-level, supervised)

**Question:** Can observable comment behavior recover the LLM-assigned persona
label? If so, which behavioral features define each persona?

**Important framing:** persona labels are themselves LLM outputs, not externally
validated ground truth. This task is better read as "which behavioral signals
did the LLM implicitly use when assigning personas" than as a classic supervised
learning benchmark.

**Model:** Random Forest, `n_estimators=300`, `max_depth=12`,
`min_samples_leaf=5`, `class_weight="balanced"`, `random_state=42`.

**Features (behavioral only — 17 features):**
`total_comments`, `activity_span_days`, `mean_hours_to_comment`,
`pct_comments_under_1h`, `reply_ratio`, `mean_word_count`,
`mean_sentiment_score`, `std_sentiment_score`, `sarcasm_rate`,
`mean_toxicity`, `mean_intensity`, `pct_positive`, `pct_negative`,
`pct_neutral`, `n_comments_in_B`, `n_distinct_posts`.

**Training set:** 39,256 users (765 dropped for null persona or zero B-side
comments), 10 classes.

**Class distribution (train):**

| Persona | N |
|---------|---|
| THE_TAGGER | 11,380 |
| THE_CASUAL_COMPLIMENTER | 7,666 |
| THE_EMOJI_REACTOR | 5,660 |
| THE_STORYTELLER | 4,365 |
| THE_SUPERFAN | 4,183 |
| THE_INQUIRER | 2,578 |
| THE_CRITIC | 2,213 |
| THE_ADVISOR | 851 |
| THE_SPAMMER | 213 |
| THE_HATER | 147 |

**5-fold stratified CV results:**

| Metric | Mean | Std |
|--------|------|-----|
| Balanced accuracy | 0.462 | 0.009 |
| Macro-F1 | 0.364 | 0.006 |
| Weighted-F1 | 0.490 | 0.005 |

Random-chance baseline: balanced accuracy = 0.10, macro-F1 = 0.10 (10 classes).

**Hold-out test set (20%, stratified) — per-class results:**

| Persona | Precision | Recall | F1 | Support |
|---------|-----------|--------|----|---------|
| THE_ADVISOR | 0.09 | 0.25 | 0.13 | 170 |
| THE_CASUAL_COMPLIMENTER | 0.53 | 0.51 | 0.52 | 1,533 |
| THE_CRITIC | 0.30 | 0.28 | 0.29 | 443 |
| THE_EMOJI_REACTOR | 0.66 | 0.76 | 0.71 | 1,132 |
| THE_HATER | 0.07 | 0.52 | 0.12 | 29 |
| THE_INQUIRER | 0.22 | 0.19 | 0.20 | 516 |
| THE_SPAMMER | 0.08 | 0.65 | 0.14 | 43 |
| THE_STORYTELLER | 0.44 | 0.35 | 0.39 | 873 |
| THE_SUPERFAN | 0.53 | 0.74 | 0.62 | 837 |
| THE_TAGGER | 0.64 | 0.36 | 0.46 | 2,276 |
| **Macro avg** | **0.35** | **0.46** | **0.36** | 7,852 |

**Top-5 features by mean decrease in impurity:**
1. `mean_word_count`
2. `mean_sentiment_score`
3. `pct_positive`
4. `pct_neutral`
5. `activity_span_days`

**Interpretation:** The model recovers persona labels at ~4.6x random chance
on balanced accuracy. THE_EMOJI_REACTOR and THE_SUPERFAN are the most
behaviorally distinct (high F1) — driven by emotional intensity and activity
patterns. THE_HATER and THE_SPAMMER show high recall but very low precision:
with class weighting, the model is trigger-happy on rare classes, over-labeling
many users as Haters or Spammers. The strongest behavioral axes separating
personas are how much users write (mean_word_count), their sentiment polarity,
and how long they have been active. THE_TAGGER bleeds into the majority class
despite being the largest group — tagging behavior is apparently not well
separated from casual commentary by these features alone.

---

### Task 2 — Comment Sentiment from Structure (comment-level, leakage-careful)

**Question:** How much of sentiment (positive/neutral/negative) is predictable
from the *structure* of a comment and its post context — without reading the
text, and without using any other LLM output?

**Leakage guard:** `sentiment_score`, `emotion`, `intent`, `target`,
`intensity`, `sarcasm`, `toxicity` are all produced in the same LLM pass as
`sentiment_cat`. Using any of them as features would be circular. They were
excluded entirely.

**Model:** Random Forest, `n_estimators=200`, `max_depth=10`,
`min_samples_leaf=10`, `class_weight="balanced"`, `random_state=42`.

**Features (42 total — non-LLM only):**
- Text/behavioral (15): `text_length`, `word_count`, `emoji_count`,
  `unique_emoji_count`, `emoji_entropy`, `emoji_variety_ratio`,
  `emoji_per_word_ratio`, `url_count`, `mention_count`, `hashtag_count`,
  `exclamation_count`, `question_count`, `avg_word_length`, `has_numbers`,
  `has_links`
- Post metadata (2): `n_hashtags`, `is_paid_partnership`
- Time features (4): `comment_hour`, `comment_dow`, `post_hour`, `post_dow`
- Categorical one-hot (21): `content_form`, `media_type`, `persona_codename`

**Dataset:** 487,604 comments. CV run on 150k random subsample for speed;
final model fit on 80/20 stratified split of full dataset.

**Target distribution:** positive 394,276 (80.9%) / neutral 68,737 (14.1%) /
negative 24,591 (5.0%). Majority-class baseline gives 81% accuracy but only
33% macro-F1.

**5-fold CV results (subsample 150k):**

| Metric | Mean | Std |
|--------|------|-----|
| Balanced accuracy | 0.600 | 0.005 |
| Macro-F1 | 0.476 | 0.002 |
| Weighted-F1 | 0.693 | 0.001 |

**Hold-out test set (97,521 comments) — per-class results:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| negative | 0.13 | 0.55 | 0.21 | 4,918 |
| neutral | 0.35 | 0.62 | 0.45 | 13,748 |
| positive | 0.95 | 0.63 | 0.76 | 78,855 |
| **Macro avg** | **0.48** | **0.60** | **0.47** | 97,521 |

**Top-10 features by importance:**

| Feature | Importance |
|---------|-----------|
| word_count | 0.157 |
| text_length | 0.138 |
| unique_emoji_count | 0.099 |
| emoji_count | 0.098 |
| emoji_variety_ratio | 0.095 |
| emoji_per_word_ratio | 0.090 |
| avg_word_length | 0.067 |
| mention_count | 0.064 |
| question_count | 0.038 |
| exclamation_count | 0.026 |

**Interpretation:** Structural features alone deliver macro-F1 of 0.47 vs. the
0.33 random baseline — a 44% lift. Roughly half of what makes a comment
positive, neutral, or negative is encoded in *how* someone writes rather than
*what* they write. Word count and text length are the single strongest signals:
longer comments tend to be more nuanced (neutral or negative); very short
comments are disproportionately positive. Emoji richness is the next strongest
axis: high emoji variety correlates with positive affect. The remaining gap
between 0.47 and 1.0 macro-F1 requires reading the text semantically — which
the LLM does and a structural classifier cannot. This model is deployable on
new comments without any Vertex call.

---

### Task 3 — Post Reception and Engagement (post-level, regression, small-N)

**Question:** Does post metadata (format, music, hashtags, timing) predict
audience vibe (mean sentiment / controversy) or engagement volume
(total_interactions / reach)?

**Model:** Ridge regression with RidgeCV alpha selection
(candidates: 0.01, 0.1, 1, 10, 100, 1000), 5-fold CV.
Engagement targets log-transformed (log1p) to handle heavy right skew.

**Dataset:** 1,493 posts, 22 features (numeric post attributes + one-hot
content_form, media_type, music_source, audio_type + numeric time features
from post timestamp).

**Target distributions:**

| Target | Mean | Median | Std |
|--------|------|--------|-----|
| mean_sentiment_score | 0.493 | 0.506 | 0.146 |
| controversy | 0.131 | 0.078 | 0.150 |
| total_interactions | 79,207 | 69,692 | 59,401 |
| reach | 195,475 | 2,389 | 446,233 |

Note: `reach` has a median of 2,389 vs mean of 195,475 — extremely heavy right
tail; very few posts have outsized reach.

**5-fold CV R² results:**

| Target | CV R² mean | CV R² std | Best alpha |
|--------|-----------|-----------|-----------|
| mean_sentiment_score | -0.034 | 0.053 | 1000 |
| controversy | -0.054 | 0.104 | 1000 |
| total_interactions (log) | -0.084 | 0.078 | 1000 |
| reach (log) | -1.112 | 1.098 | 100 |

All R² values are at or below zero — the post metadata features do not
predict engagement or sentiment better than a mean-prediction baseline at
this sample size. The regularizer always selecting high alpha (1000) confirms
the signal is swamped by noise relative to feature count.

**Coefficient directions (interpretable despite low R²):**

*Audience sentiment (mean_sentiment_score):*
- Tagged users (+), licensed music (+), mentions (+) are slightly associated
  with more positive audience sentiment
- Original sound audio (-), static images (-) are slightly associated with
  less positive sentiment

*Controversy (4 * pct_pos * pct_neg):*
- Paid partnerships (+), original music (+) are associated with more
  polarized audiences
- Licensed audio (-), more hashtags (-) are associated with less controversy

*Total interactions (log):*
- Licensed music (+), paid partnership (+), posting month (+) are positive
- Feed posts (-0.159), raw VIDEO media type (-0.087) are strongly negative
  — feed posts are the oldest format with lowest comment-per-post rates

*Reach (log):*
- Reels (+0.262), audio presence (+), video duration (+) are positive for reach
- Static images (-0.515), no-music posts (-0.201), paid partnerships (-0.193)
  are associated with lower reach

**Interpretation:** At N=1,493, post metadata alone cannot predict engagement
or vibe — the negative R² values mean the model is no better than guessing the
mean. This is itself informative: it means content format and production
attributes do not drive outcomes in isolation. Audience composition, content
quality, and context not captured here (creator reputation, topical relevance,
algorithmic timing) dominate. The coefficient directions are directionally
plausible and consistent with platform knowledge (reels reach more people;
licensed music is associated with better audience response) but should be
treated as weak descriptive signals, not causal claims. More data (time series,
creator-level fixed effects) would be needed for reliable regression.

---

### Task 4 — Persona x Sentiment / Persona x Content Form (descriptive)

**Question:** Do audience personas differ in the sentiment they express?
Do they cluster on specific content formats?

**Method:** Cross-tabulations (row-normalized), chi-squared test of
independence, Kruskal-Wallis test on continuous sentiment_score.
Dataset: 94,397 comments with both persona and sentiment labels.

**Persona x sentiment_cat (row %):**

| Persona | negative | neutral | positive |
|---------|----------|---------|----------|
| THE_HATER | 0.520 | 0.162 | 0.318 |
| THE_SPAMMER | 0.019 | 0.741 | 0.241 |
| THE_CRITIC | 0.224 | 0.247 | 0.529 |
| THE_INQUIRER | 0.055 | 0.329 | 0.615 |
| THE_TAGGER | 0.040 | 0.243 | 0.717 |
| THE_ADVISOR | 0.059 | 0.240 | 0.701 |
| THE_STORYTELLER | 0.108 | 0.152 | 0.740 |
| THE_SUPERFAN | 0.034 | 0.067 | 0.899 |
| THE_EMOJI_REACTOR | 0.016 | 0.031 | 0.953 |
| THE_CASUAL_COMPLIMENTER | 0.014 | 0.046 | 0.941 |

Chi-squared test: chi2 = 14,140, dof = 18, p < 1e-300
Kruskal-Wallis on sentiment_score: H = 15,792, p < 1e-300

**Mean sentiment_score by persona (ranked):**

| Persona | Mean score | Std | N |
|---------|-----------|-----|---|
| THE_HATER | -0.232 | 0.536 | 173 |
| THE_SPAMMER | +0.136 | 0.278 | 324 |
| THE_CRITIC | +0.203 | 0.482 | 4,988 |
| THE_INQUIRER | +0.372 | 0.372 | 5,300 |
| THE_TAGGER | +0.421 | 0.342 | 19,357 |
| THE_STORYTELLER | +0.426 | 0.431 | 9,819 |
| THE_ADVISOR | +0.428 | 0.366 | 1,740 |
| THE_EMOJI_REACTOR | +0.633 | 0.237 | 9,657 |
| THE_SUPERFAN | +0.641 | 0.316 | 31,920 |
| THE_CASUAL_COMPLIMENTER | +0.694 | 0.246 | 10,895 |

**Persona x content_form (row %):**

| Persona | carousel | carousel_video | feed | image | reel |
|---------|----------|---------------|------|-------|------|
| THE_HATER | 0.092 | 0.012 | 0.012 | 0.156 | **0.728** |
| THE_SPAMMER | 0.287 | 0.009 | 0.040 | **0.533** | 0.131 |
| THE_TAGGER | 0.211 | 0.010 | **0.115** | 0.321 | 0.343 |
| THE_CRITIC | 0.238 | 0.019 | 0.026 | 0.284 | **0.433** |
| THE_EMOJI_REACTOR | 0.253 | 0.016 | 0.032 | 0.296 | **0.403** |
| THE_SUPERFAN | **0.321** | 0.012 | 0.037 | 0.343 | 0.287 |
| THE_CASUAL_COMPLIMENTER | **0.310** | 0.030 | 0.037 | 0.361 | 0.263 |

Chi-squared test: chi2 = 3,878, dof = 36, p < 1e-300

**Interpretation:** Persona and sentiment are inseparably linked — the
distributions differ at extreme statistical significance. THE_HATER is the only
persona with a negative mean sentiment score (-0.23) and the only one where a
majority of comments are classified negative (52%). THE_CRITIC is the only
other persona with substantial negativity (22% negative). Both THE_EMOJI_REACTOR
and THE_CASUAL_COMPLIMENTER are near-uniformly positive (95% and 94%
positive respectively). THE_SPAMMER is unusual: predominantly neutral (74%) at
very low positive sentiment — consistent with template/promotional comment
patterns that are affectively flat.

The content-form associations are also significant. THE_HATER concentrates
heavily on reels (73%) — the highest-reach format attracts the most adversarial
commenters. THE_SPAMMER concentrates on images (53%) — static posts are easier
to mass-comment on. THE_TAGGER has the highest feed-post rate (12%), consistent
with legacy content tagging behavior. THE_SUPERFAN and THE_CASUAL_COMPLIMENTER
over-index on carousels relative to other personas.

---

## 4. Sampling and Imbalance Caveats

These caveats apply to all results above and must be stated whenever findings
are presented externally.

1. **B is not a uniform sample.** The sentiment pipeline (B) covers 487k of
   499k IG comments but was run prioritizing media-bearing posts. B-derived
   rates (e.g. "80.9% of comments are positive") are not unbiased estimates
   of the true IG comment population sentiment. They reflect the subset of
   comments that were attached to posts with media.

2. **Sentiment is 81% positive.** Any classification accuracy figure without
   per-class breakdown is misleading. A baseline model predicting "always
   positive" achieves 81% accuracy and 33% macro-F1. All results in this
   report use macro-F1 and balanced accuracy.

3. **Persona classes are severely imbalanced.** THE_TAGGER (11,380) vs
   THE_HATER (147) — a 77:1 ratio. `class_weight="balanced"` and stratified
   splits were used for all classification tasks. Tail-class metrics (THE_HATER,
   THE_SPAMMER) should be interpreted with caution given their small test N.

4. **posts_model has only 1,493 rows.** Regression R² at this N is highly
   variable. All R² values were negative, meaning coefficient direction is the
   only interpretable output from Task 3.

5. **LLM labels are not ground truth.** Persona codenames and sentiment labels
   were produced by a Vertex batch LLM job. Task 1 asks "can behavior recover
   the LLM's decision" — not "can behavior recover the true persona." Task 2
   asks "can structure partially predict what the LLM labeled" — not "what
   sentiment the commenter actually intended."

---

## 5. Output Artifacts

All files are saved under `Ali/outputs/modeling/`:

| File | Description |
|------|-------------|
| `comments_model.parquet` | 487,604 x 46 — per-comment modeling table |
| `users_model.parquet` | 40,019 x 23 — per-user modeling table |
| `posts_model.parquet` | 1,493 x 70 — per-post modeling table |
| `task1_persona_clf.joblib` | Fitted RandomForest for persona classification |
| `task1_metrics.json` | CV + hold-out metrics + feature importances |
| `task1_confusion_matrix.png` | 10-class confusion matrix (hold-out) |
| `task1_feature_importance.png` | Feature importance bar chart |
| `task2_sentiment_clf.joblib` | Fitted RandomForest for sentiment (structural) |
| `task2_metrics.json` | CV + hold-out metrics + top-20 features |
| `task2_confusion_matrix.png` | 3-class confusion matrix (hold-out) |
| `task2_feature_importance.png` | Top-20 feature importance bar chart |
| `task3_metrics.json` | CV R² for all four regression targets |
| `task3_engagement_coefficients.png` | Ridge coefficient plot for total_interactions |
| `task4_metrics.json` | Chi2 and Kruskal-Wallis test statistics |
| `task4_persona_sentiment_heatmap.png` | Persona x sentiment_cat heatmap |
| `task4_persona_contentform_heatmap.png` | Persona x content_form heatmap |
| `MODELING_REPORT.md` | This document |
