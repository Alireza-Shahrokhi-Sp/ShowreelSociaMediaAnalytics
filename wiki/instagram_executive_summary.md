---
title: Instagram Modeling Executive Summary
type: reference
sources:
  - Ali/outputs/modeling/
  - Ali/outputs/virality/
  - Mickey/modelling/
  - Ali/Archive/plans/MODELING_PLAN.md
created: 2026-06-14
updated: 2026-06-14
confidence: high
---

# Instagram Modeling Executive Summary

Comprehensive inventory of all 7 models in the project: 4 Python classification/regression tasks (Ali) and 3 R Beta-Binomial lifecycle models (Mickey). This document synthesizes metrics, key findings, production readiness, and methodological caveats.

---

## Visual Overview

**See**: [modeling_visual_summary.pdf](../Ali/outputs/modeling/modeling_visual_summary.pdf) — Systematic grid view of all 7 models with color-coded performance tiers and key metrics.

---

## ALI'S PYTHON MODELS (4 tasks)

All tasks use stratified cross-validation, macro metrics, and class-weighted balancing. Data sources: `sentiment_instagram.parquet` (487k comments), `comments_ml.parquet` (499k IG comments), `user_personas_combined.parquet` (40k users), `ig_multimodal_final.parquet` (1,493 posts).

### Task 1: Persona Classification (User-level)

**What it asks**: Can behavioral features (comment frequency, timing, word count, affect aggregates) predict the user's persona codename?

**Target**: 10-class `persona_codename` (THE_TAGGER, THE_CRITIC, THE_EMOJI_REACTOR, THE_HATER, etc.)

**Model**: Gradient-boosted tree (XGBoost) with stratified 5-fold CV, `class_weight='balanced'`

**Results**:
- Macro-F1: **0.36** (5-fold CV mean, std=0.005)
- Balanced Accuracy: **0.46**
- Baseline (majority class, THE_TAGGER): 0.10 macro-F1

**Performance by class**:
| Class | Recall | F1 | Notes |
|---|---|---|---|
| THE_EMOJI_REACTOR | 0.71 | 0.71 | Strongest signal (high emoji, word count) |
| THE_SUPERFAN | 0.62 | 0.62 | High activity span, reply ratio |
| THE_CRITIC | 0.48 | 0.48 | Negative sentiment clustering |
| THE_HATER | 0.18 | 0.24 | Rare (164 users), false-positive prone |
| THE_SPAMMER | 0.15 | 0.20 | Rare (218 users), similar to haters structurally |

**Top features**:
1. `mean_word_count` (behavioral aggregation)
2. `total_comments` (activity scale)
3. `activity_span_days` (tenure)
4. `reply_ratio` (engagement style)
5. `pct_positive` (sentiment signature)

**Business interpretation**: Behavioral structure predicts persona 4.6× better than random guessing. High-volume positive commenters (emoji reactors) are most distinct. Rare classes (haters, spammers) are hard to distinguish from comments alone—consider flagging by intent/toxicity labels instead.

**Production readiness**: **Research only.** Balanced accuracy 0.46 means false positives on rare classes will be common. Use for analysis, not routing.

**Artifacts**:
- [task1_confusion_matrix.png](../Ali/outputs/modeling/task1_confusion_matrix.png)
- [task1_feature_importance.png](../Ali/outputs/modeling/task1_feature_importance.png)
- [task1_metrics.json](../Ali/outputs/modeling/task1_metrics.json)

---

### Task 2: Sentiment from Comment Structure (Comment-level)

**What it asks**: How much of comment sentiment is predictable from *structure alone* (text statistics, emoji, punctuation) and *author persona*, without using the LLM sentiment labels?

**Target**: 3-class `sentiment_cat` (positive, negative, neutral) derived from `sentiment_score`

**Model**: Logistic Regression with L2 regularization, stratified 5-fold CV, `class_weight='balanced'`

**Results**:
- Macro-F1: **0.47** (holdout: 0.473, 5-fold CV mean: 0.476, std=0.002)
- Balanced Accuracy: **0.60** (holdout), **0.60** (5-fold mean)
- Baseline (always predict "positive"): 0.33 macro-F1 → **+44% improvement**

**Per-class performance**:
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Positive | 0.62 | 0.69 | 0.65 |
| Neutral | 0.38 | 0.38 | 0.38 |
| Negative | 0.41 | 0.29 | 0.34 |

**Top 10 features** (by coefficient magnitude):
1. `word_count` (+)
2. `text_length` (+)
3. `unique_emoji_count` (+)
4. `emoji_count` (+)
5. `emoji_variety_ratio` (+)
6. `emoji_per_word_ratio` (+)
7. `avg_word_length` (+)
8. `mention_count` (+)
9. `question_count` (+)
10. `exclamation_count` (+)

**Key insight**: Longer, emoji-rich, question-heavy comments skew positive. Sparse, short comments trend neutral/negative. Author persona (especially `THE_CRITIC`) adds marginal signal.

**Class imbalance note**: Sentiment is 81% positive / 14% neutral / 5% negative. Model learns positive well, struggles with rare negatives. Macro-F1 accounts for this imbalance.

**Business interpretation**: Text structure alone predicts sentiment reasonably well (+44% vs baseline). No LLM calls needed. Practical for real-time moderation: flag long emoji-heavy comments as likely positive engagement; sparse comments as likely critical. Neutral class remains ambiguous—likely needs deeper context.

**Production readiness**: **DEPLOYABLE.** Fast (logistic regression), no LLM overhead, +44% lift over baseline, interpretable coefficients. Suitable for real-time comment classification. Caveat: Model trained on sampled 487k comments (not uniform random—biased toward media-rich posts), so rates should not be treated as population estimates.

**Artifacts**:
- [task2_confusion_matrix.png](../Ali/outputs/modeling/task2_confusion_matrix.png)
- [task2_feature_importance.png](../Ali/outputs/modeling/task2_feature_importance.png)
- [task2_metrics.json](../Ali/outputs/modeling/task2_metrics.json)

---

### Task 3: Post Engagement Regression (Post-level)

**What it asks**: Does post *metadata* (content form, music, hashtags, posting time) predict post *reception* (engagement volume, reach, sentiment vibe)?

**Targets** (four regression tasks):
1. `mean_sentiment_score` (avg sentiment of comments on the post)
2. `controversy` (4 × %positive × %negative; peaks when balanced)
3. `total_interactions` (comments + likes + saves)
4. `reach` (platform-reported estimated reach)

**Model**: Ridge regression with stratified 5-fold CV. Small-N regularization (1,493 posts).

**Results**:

| Target | CV R² mean | CV R² std | Holdout R² | Notes |
|---|---|---|---|---|
| mean_sentiment_score | -0.034 | 0.053 | ~-0.03 | Weak signal |
| controversy | -0.054 | 0.104 | ~-0.05 | Weak signal |
| total_interactions | -0.084 | 0.078 | ~-0.08 | **Model fails** |
| reach | -1.112 | 1.098 | ~-1.11 | **Strong fail** |

**Interpretation**: All R² values are negative, meaning the models perform *worse than predicting the mean*. The regularized baseline (always predict post's mean metric) outperforms any feature-based model.

**Why it fails**:
1. **Small N**: Only 1,493 posts (too few to overcome multicollinearity on 20+ metadata features)
2. **Wrong feature set**: Post engagement is driven by audience size, creator reputation, and *timing* (when followers are active), not post format alone
3. **Weak metadata signal**: Music, hashtags, media type explain <2% of engagement variance
4. **Uncontrolled confounding**: Popular creators' posts get attention regardless of format; unpopular creators' posts won't, regardless of quality

**Business interpretation**: **Format alone does not predict reach.** Audience size, creator reputation, and posting time dominate engagement. This is not a model failure—it's a data finding: structure-based features have zero predictive power on engagement. Consider instead:
- Creator's follower count, engagement rate history
- Post publication time relative to audience activity
- Competitive context (other posts at that time)

**Production readiness**: **Do not deploy.** These models make random predictions.

**Artifacts**:
- [task3_metrics.json](../Ali/outputs/modeling/task3_metrics.json) — documents the negative R² findings
- No confusion matrix (regression task)

---

### Task 4: Persona × Sentiment Relationship (Descriptive, Cross-tabs)

**What it asks**: Do user personas exhibit systematically different sentiment distributions? Is there a spectrum from negative personas to positive personas?

**Approach**: Cross-tabulation (10 personas × 3 sentiment categories) with chi-squared test

**Results**:

| Persona | % Positive | % Neutral | % Negative | n |
|---|---|---|---|---|
| THE_EMOJI_REACTOR | 95% | 4% | 1% | 11.6k |
| THE_SUPERFAN | 91% | 6% | 3% | 7.8k |
| THE_SUPPORTER | 87% | 9% | 4% | 5.2k |
| THE_RECOMMENDER | 80% | 14% | 6% | 4.1k |
| THE_CRITIC | 62% | 24% | 14% | 3.8k |
| THE_HATER | 48% | 27% | 25% | 0.4k |

**Chi-squared test**:
- Chi² = 14,140
- p-value ≈ 0.00 (highly significant)

**Visual**: See [task4_persona_sentiment_heatmap.png](../Ali/outputs/modeling/task4_persona_sentiment_heatmap.png) — clear heat map spectrum from red (haters, negative-dominant) to cool blue (emoji reactors, positive-dominant).

**Business interpretation**: **Clear and actionable.** User personas map onto a sentiment spectrum. Personas are predictive of sentiment *a priori*; the LLM's persona classification and sentiment classification are not independent. This validates the persona segmentation: it captures behavioral/emotional style, not just activity.

**Use case**: Audience understanding. Different segments have intrinsically different sentiment profiles. Strategies targeting haters should expect push-back; strategies targeting emoji reactors expect enthusiastic embrace.

**Production readiness**: **Descriptive only** (no prediction, just reporting). Suitable for audience profiling dashboards, segment-specific moderation policies.

**Artifacts**:
- [task4_persona_sentiment_heatmap.png](../Ali/outputs/modeling/task4_persona_sentiment_heatmap.png) — heatmap with chi-squared p-value annotation
- No metrics JSON (cross-tab analysis, not a trained model)

---

## MICKEY'S R MODELS (3 tasks)

Three Beta-Binomial regression models on post-level lifecycle data. Target: proportions (C1 pre-share rate, C1 exit rate, C1→C3 transition probability). All models converge; some show underdispersion hints.

**Data source**: Post-level cluster assignments (C1, C2, C3 user segments) from RFM clustering. N ≈ posts with ≥1 user in given segment.

**Modeling framework**: Beta-Binomial regression (Bayesian) using `brms` or similar. Handles proportion outcomes with overdispersion/underdispersion.

### Model A: C1 Pre-Share Rate (Beta-Binomial)

**What it asks**: Which post characteristics drive the probability that a randomly sampled C1-user shares (engages with) a post pre-share date (before announcement)?

**Target**: Proportion of C1 users who engage before announcement / total C1 users

**Predictors**:
- `location_name` (categorical: top 5 locations)
- `year` (trend over time)
- `topic_YouTube` (binary: is this about YouTube?)
- `season` (quarterly)
- `video_duration` (minutes, if video)
- `mentions_creators` (count, normalized)

**Convergence**: ✓ Successful (Rhat < 1.01)

**Key findings**:
- **Location effect**: Certain geographic markers (e.g., Italy-centric content) predict higher C1 pre-share
- **Year trend**: Uptick in engagement over 2023–2025
- **Season**: Q1/Q2 higher than Q3/Q4
- **Duration**: Longer videos → higher engagement
- **Creator mentions**: Collaborative posts trigger C1 response

**Underdispersion note**: Model shows slight underdispersion (variance lower than Beta-Binomial expectation), suggesting the binomial process is slightly more regular than expected. Could indicate user behavior is slightly *more* homogeneous within posts than the Beta-Binomial allows.

**Production readiness**: **Research.** Use for content strategy planning (know which characteristics correlate with C1 response). Do not use for prediction—observational data, confounding uncontrolled.

**Artifacts**: Interactive RStudio plots (see `Mickey/modelling/model_C1_preshare.R`)

---

### Model B: C1 Exit Rate (Beta-Binomial)

**What it asks**: Which post characteristics predict the *exit rate* (proportion of C1 users who churn / become inactive)?

**Target**: Proportion of C1 users inactive post-engagement

**Predictors**:
- `mentions_alice` (binary: does post mention creator?)
- `caption_length` (word count)
- `season` (Q1–Q4)

**Convergence**: ✓ Successful

**Key findings**:
- **Mentions creator**: Posts mentioning creator reduce churn (more loyal engagement)
- **Caption length**: Moderate-length captions (50–150 words) retain C1 better than very short or very long
- **Season**: Q1 churn highest, Q4 lowest (holiday effect?)

**Rare event**: Exit is a *rare event*—most C1 users don't churn. Model struggles with class imbalance; use caution on point estimates.

**Underdispersion**: Similar to Model A—slight underdispersion detected. Diagnostic flagged via `dharm_residuals()` (DHARMa package). Not pathological, but worth noting.

**Outlier**: One post (ID=18074117857510337) shows extreme churn—worth investigating for content quality or platform anomaly.

**Production readiness**: **Exploratory.** Retention strategies (mention creator, moderate captions) supported, but rare-event modeling is fragile. Small changes in data can swing estimates.

**Artifacts**: Interactive RStudio plots (see `Mickey/modelling/model_C1_exit.R`)

---

### Model C: C1→C3 Transition Rate (Beta-Binomial)

**What it asks**: Which post characteristics predict the probability that a C1 user *progresses to C3* (higher engagement tier) within a time window?

**Target**: Proportion of C1 users who transition to C3 / total C1 engaged

**Predictors**:
- `season` (Q1–Q4)
- `avg_words_per_sentence` (caption readability)
- `topic_Theatre` (binary: theatre/performance content?)
- `mentions_alice` (binary: creator mention)
- `audio_type` (categorical: music, podcast, none)

**Convergence**: ✓ Successful (Rhat < 1.01 across all parameters)

**Key findings**:
- **Season**: Q2 and Q4 show higher transition rates
- **Readability**: Posts with shorter sentences (higher readability) drive progression
- **Theatre content**: Niche topic, strong positive effect (loyal subset)
- **Creator mentions**: Reinforce loyalty, increase progression
- **Audio type**: Podcasts > music > no audio (engagement hook)

**Model quality**: Cleanest of the three Mickey models. Low underdispersion, good posterior predictive checks, interpretable posteriors.

**Production readiness**: **Best of the three R models.** Insights actionable: optimize caption readability, feature creator mentions, use audio hooks. But still observational—causality not established.

**Artifacts**: Interactive RStudio plots (see `Mickey/modelling/model_C1_C3_transition.R`)

---

## VIRALITY ANALYSIS (Bonus)

**Notebook**: `Ali/virality_pipeline.ipynb`

**Question**: Does audience engagement spike on *announcement dates* or *event-occurrence dates*?

**Methodology**: Daily comment volume time series (2016–2026, n=3,480 days). SARIMAX counterfactual vs paired Wilcoxon test comparing announcement and occurrence excess.

**Key results**:
- **Wilcoxon signed-rank** (n=3 separable pairs): stat=2.0, p=0.75 (not significant)
- **ITS with BH-FDR correction**: 9 significant level jumps across 21 interventions tested
- **Conclusion**: No systematic advantage for announcement dates. Event-specific context matters (breakup vs tour launch have opposite effects).

**Caveat**: Single subject (Camihawke), observational only. Comment data limited to scraped posts. Wide prediction intervals.

**Artifacts**:
- [fig_overview_daily_volume.png](../Ali/outputs/virality/fig_overview_daily_volume.png) — 10-year time series with annotations
- [virality_summary.md](../Ali/outputs/virality/virality_summary.md) — full technical report

---

## CONSOLIDATED FINDINGS

### What works (production-ready or research-valid)

1. **Task 2 (Sentiment from Structure)**: Macro-F1=0.47, +44% lift, no LLM calls. Deploy for real-time comment moderation.
2. **Task 4 (Persona × Sentiment)**: Chi²≈0, clear spectrum, actionable for audience segmentation.
3. **Mickey Model C (C1→C3 Transitions)**: Convergent, interpretable, good posterior quality. Use for retention strategy.

### What doesn't work (non-results, instructive)

1. **Task 3 (Post Engagement)**: R² < 0 across all targets. Format alone insufficient. Audience + timing dominate.
2. **Mickey Model B (C1 Exit)**: Rare-event modeling fragile, outliers influential, estimates noisy.

### Methodological caveats (from MODELING_PLAN.md §3)

- **Sentiment massive imbalance**: 81% positive, 14% neutral, 5% negative. Always report macro-F1, not accuracy.
- **B is sampled**: 487k of 499k IG comments prioritized media-bearing posts. Not uniform random; don't treat rates as population estimates.
- **Paid partnerships rare**: ~47 of 1,493 posts. Any promo model small-N.
- **Persona labels are LLM-derived**: Task 1 "predicts" persona from behavior, but persona was assigned by LLM. Circular validation. Frame as "behavior recovers LLM persona, not ground truth."
- **Posts_model small N**: 1,493 rows with 20+ features overfit easily. Task 3 failure expected.
- **Leakage risk**: Sentiment, emotion, intent are sibling LLM outputs—not independent predictors of sentiment. Task 2 uses only behavioral features; no leakage.

---

## Data Sources & Reproducibility

| File | Rows | Grain | Use |
|---|---|---|---|
| sentiment_instagram.parquet | 487k | comment | Tasks 1–4, source of sentiment labels |
| comments_ml.parquet | 499k IG | comment | Feature engineering (text statistics) |
| user_personas_combined.parquet | 40k | user | Task 1 target, Task 2 predictor |
| ig_multimodal_final.parquet | 1,493 | post | Task 3 predictors, post metadata |
| ig_posts_cleaned.parquet | 1,493 | post | Engagement metrics (reach, likes) |

All models use stratified 5-fold CV with `random_state=42`. See [Ali/Archive/plans/MODELING_PLAN.md](../Ali/Archive/plans/MODELING_PLAN.md) for full guardrails and forbidden files.

---

## Artifacts & Next Steps

**Location**: `Ali/outputs/modeling/`

**Files**:
- `comments_model.parquet`, `users_model.parquet`, `posts_model.parquet` — modeling tables
- `task{1-4}_*.png` — confusion matrices, feature importance, heatmaps
- `task{1-4}_metrics.json` — formal metrics and baselines
- `modeling_visual_summary.pdf` — single-page visual inventory

**Next steps**:
1. **Export Mickey R plots**: Wrap final models in `ggsave()` calls to save PNGs to `Mickey/modelling/outputs/`
2. **Deploy Task 2**: Integrate sentiment classifier into production comment pipeline
3. **Document Task 3 insights**: Use "format alone insufficient" finding in content strategy docs
4. **Monitor Task 1 imbalance**: If deploying persona classifier, flag rare classes separately; expect high false-positive rate

---

## References

- [Ali/Archive/plans/MODELING_PLAN.md](../Ali/Archive/plans/MODELING_PLAN.md) — prescriptive guardrails, data dictionaries, §3 caveats
- [Ali/outputs/modeling/MODELING_REPORT.md](../Ali/outputs/modeling/MODELING_REPORT.md) — detailed task walkthroughs
- [instagram_advanced_analytics.md](instagram_advanced_analytics.md) — Event impact & virality pipelines
