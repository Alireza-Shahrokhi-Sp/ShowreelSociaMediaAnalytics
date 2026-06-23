---
title: Instagram advanced analytics
type: reference
sources:
  - Ali/modeling_pipeline.ipynb
  - Ali/event_impact_pipeline.ipynb
  - Ali/virality_pipeline.ipynb
  - Ali/outputs/modeling/
  - Ali/outputs/event_impact/
  - Ali/outputs/virality/
related:
  - "[[instagram_findings]]"
  - "[[instagram_sentiment_pipeline]]"
created: 2026-06-14
updated: 2026-06-14
confidence: medium
---

# Instagram advanced analytics

## Overview

Three specialized analytical pipelines beyond core sentiment/persona/RFM analyses:

1. **Modeling pipeline** — ML models on comment and user data
2. **Event impact pipeline** — Causal inference of specific events on engagement
3. **Virality pipeline** — Announcement vs event-occurrence timing analysis

---

## Modeling pipeline

**Notebook**: `Ali/modeling_pipeline.ipynb`

**Purpose**: Build predictive models for comment sentiment, user engagement, and post virality.

### Input data

- `sentiment_instagram.parquet` — Comment-level labels (sentiment, emotion, intent, toxicity)
- `comments_ml.parquet` — Linguistic features
- `ig_multimodal_final.parquet` — Post metadata
- `user_personas_combined.parquet` — User segmentation

### Model types

| Model | Target | Features | Output |
|---|---|---|---|
| Comment sentiment classifier | sentiment (pos/neg/neutral) | text embeddings + linguistic features | logits or probabilities |
| Comment toxicity ordinal regressor | toxicity (none/mild/severe) | text + features | ordinal predictions |
| User engagement predictor | likes per comment (regression) | user history + post context | continuous prediction |
| Post virality classifier | viral yes/no (top 20% by engagement) | caption embeddings + media type + persona dist | binary classification |

### Output files

**Location**: `Ali/outputs/modeling/`

- `comments_model.parquet` — Per-comment predictions and residuals
- `users_model.parquet` — Per-user engagement predictions
- `posts_model.parquet` — Per-post virality predictions

**Schema (typical)**:

| Column | Type | Description |
|---|---|---|
| comment_id / author_id / media_id | string | ID (key) |
| actual | float/string | Ground truth label |
| pred | float/string | Model prediction |
| pred_proba | float | [0, 1] prediction confidence |
| residual | float | actual - pred (regression models) |
| fold | int | Cross-validation fold (if CV) |
| feature_importance | dict | Top-5 influential features (if tree-based) |

### Model evaluation

Standard metrics:
- Classification: ROC-AUC, F1 (macro), precision/recall per class
- Regression: MAE, RMSE, R²
- Cross-validation: 5-fold stratified

---

## Event impact pipeline

**Notebook**: `Ali/event_impact_pipeline.ipynb`

**Purpose**: Quantify causal impact of specific events (launches, announcements, milestones) on comment volume and sentiment.

### Methodology

**Interrupted Time Series (ITS)** with segmented regression:

```
log(comment_count) = β₀ + β₁*t + β₂*D + β₃*t*D + ε
```

Where:
- `t` = days since series start (trend)
- `D` = indicator for post-intervention (0 before, 1 after)
- `β₂` = level jump (immediate effect)
- `β₃` = slope change (trend shift after intervention)

**Robustness checks**:
- Seasonal-naive baseline (same-weekday median from past 52 weeks)
- Pre-intervention trend validation (≥90 days baseline)
- Multiple hypothesis correction (BH-FDR on p-values)

### Event timeline

Curated list of ~26 milestones for creator "Camihawke":
- First Public Announcements (9 events)
- Occurrences / Milestones (17 events)

Milestones span 2016–2026 (career evolution):
- TV shows (Pink Different, Nella MIA Cucina, The Traitors Italia)
- Publishing (novel launch)
- Podcast launches
- Theater tours (solo and collaborative)
- Festival appearances
- Awards and accolades
- Personal events (breakup, etc.)

### Output files

**Location**: `Ali/outputs/event_impact/`

- `monthly_series.parquet` — Monthly aggregated comment volume + sentiment

**Schema**:

| Column | Type | Description |
|---|---|---|
| date | date | First day of month |
| month | string | YYYY-MM |
| n_comments | int64 | Total comments in month |
| n_active_users | int64 | Unique commenters |
| n_active_posts | int64 | Posts with comments |
| mean_sentiment | float64 | [0, 1] average sentiment score |
| pct_pos / pct_neg / pct_neu | float64 | Sentiment shares |
| mean_toxicity | float64 | Average toxicity severity |

---

## Virality pipeline

**Notebook**: `Ali/virality_pipeline.ipynb`

**Purpose**: Test the hypothesis: does audience engagement spike on the announcement date or the event-occurrence date?

### Hypothesis

If audiences react primarily to news (announcements), engagement bursts cluster around announcement dates. If they react to the event itself, spikes occur at occurrence dates. Gap between announcement and occurrence allows discrimination.

### Methodology

**SARIMAX counterfactual** per intervention date:

1. Fit SARIMAX model on 365 pre-intervention days
2. Forecast 14 days post-intervention (with 95% prediction interval)
3. Compare observed vs forecast to measure excess engagement
4. Paired Wilcoxon signed-rank test (announcement vs occurrence cumulative excess)

**Robustness**:
- ITS on 90-day windows pre/post
- Seasonal-naive baseline

### Input data

- Daily comment volume: 2016-08-18 to 2026-03-20 (3,502 days)
- Intervention pairs: 26 milestones (9 announcement+occurrence pairs, 17 occurrence-only)
- Post publication dates (for new-post coincidence checks)

### Output files

**Location**: `Ali/outputs/virality/`

- `daily_series.parquet` — Daily comment volume + sentiment aggregates
- `intervention_pairs.csv` — All 26 milestones with gap analysis and coverage flags
- `virality_results.csv` — SARIMAX per-intervention results (multiple windows)
- `announce_vs_event.csv` — Paired comparison of announcement vs occurrence excess
- `robustness_results.csv` — Seasonal-naive + ITS results for all interventions
- `virality_summary.md` — Executive summary with Wilcoxon test results
- `fig_overview_daily_volume.png` — 3,500-day time series with annotation
- `fig_ann_*.png` / `fig_occ_*.png` — 21 per-intervention SARIMAX counterfactual plots

### Key findings

**Paired Wilcoxon result** (separable pairs only, n=3):
- Wilcoxon stat = 2.0, p = 0.75 (two-sided)
- No significant systematic advantage for announcement dates
- Median excess difference: +1,003.3 comments favoring announcements, but high variance

**ITS results** (after BH-FDR correction):
- 9 interventions with significant level jump (q < 0.05)
- Notable positive jumps: Nella MIA Cucina announcement, Autumn Tour re-ignition
- Notable negative jumps: TEDxRimini, Solo Tour announcement, Etna Comics

**Interpretation**:
- Single-subject, observational study (no control creator)
- Evidence inconclusive on announcement vs occurrence hypothesis
- Event-specific context (tour premières, breakup announcements) may confound
- Comment data limited to scraped posts; unscraped activity uncounted

---

## Data pipeline summary

```
Raw comments_ml → daily series → event intervention pairs → SARIMAX models → virality test
                ↓
          sentiment_instagram → daily sentiment aggregates
                ↓
            user_personas_combined → persona composition over time
                ↓
               ig_multimodal_final → post-level context (media type, sponsor)
```

---

## Notebook locations and features

| Notebook | Purpose | Dependencies | Output type |
|---|---|---|---|
| `Ali/modeling_pipeline.ipynb` | ML model pipeline | sentiment, comments_ml, personas | .parquet tables, metrics |
| `Ali/event_impact_pipeline.ipynb` | ITS analysis of events | daily series, timeline CSV | monthly_series.parquet, figures |
| `Ali/virality_pipeline.ipynb` | Announcement vs occurrence | daily series, intervention timeline | SARIMAX plots, Wilcoxon test |

---

## Cross-references

- These analyses integrate sentiment labels from [[instagram_sentiment_pipeline]]
- User segmentation from [[instagram_persona_pipeline]]
- User lifecycle features from [[instagram_rfm_clustering]]
- Raw data sourced from [[instagram_data_sources]]
- Findings synthesized in [[instagram_findings_detailed]]
