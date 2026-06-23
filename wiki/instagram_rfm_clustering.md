---
title: Instagram RFM clustering
type: source-summary
sources:
  - Mickey/RFM_IG_rolling_quadrimesters/
  - Mickey/RFM_IG_3_years/
  - Mickey/EDA_RFM_IG_vs_TK_rolling_quadrimesters.ipynb
  - Mickey/EDA_cluster_matrix_IG_vs_TK.ipynb
related:
  - "[[instagram_findings]]"
  - "[[rfm-clustering]]"
created: 2026-06-14
updated: 2026-06-14
confidence: high
---

# Instagram RFM clustering

## Overview

Two independent RFM clustering schemes on Instagram data:

1. **Rolling quadrimester** (40 monthly sliding 4-month windows, 2022-09 to 2026-03) — captures user lifecycle transitions
2. **Non-overlapping 3-year periods** (3 disjoint periods covering full history) — captures long-term evolution

Both use KMeans clustering on RCEDTG features (Recency, Coverage, Engagement, Delay, Tenure, Gini).

---

## Rolling quadrimester analysis

**Location**: `Mickey/RFM_IG_rolling_quadrimesters/`

Each row is one (user, window) observation. 40 windows sliding monthly across 4-month spans.

### Feature table

**File**: `ig_user_rolling_quadrimesters_new_definitions.csv`

| Column | Type | Description |
|---|---|---|
| window_id | string | w01–w40 |
| window_label | string | "2022-09 to 2022-12" (start-end of 4-month span) |
| window_index | int | 1–40 |
| window_start | date | First day of window |
| window_end | date | Last day of window |
| window_total_days | int | Calendar days in window (120–122) |
| posts_in_window | int | Count of posts published in window |
| user_key | int64 | User identifier |
| user_id | float64 | Numeric form of user_key |
| username | string | Creator username |
| comment_count | int64 | Comments user made in window |
| last_comment_timestamp | datetime | User's last comment in window |
| recency | float64 | [0, 1] normalised days-since-last-comment |
| coverage | float64 | comment_count / posts_in_window |
| engagement | float64 | Mean likes per comment in window |
| delay | float64 | Mean hours between post and comment |
| tenure | float64 | Days since user's first ever comment |
| gini | float64 | [0, 1] inequality of comment distribution across posts |

**Size**: 174,287 rows | **Unique users**: 18,752 | **Windows**: 40

**Key**: (window_id, user_key) composite

### Clustering outputs

**k=4 user-level assignments** (latest window per user)

**File**: `Ig_RCEDTG_kmeans_k4_user_clusters.csv`

| Column | Description |
|---|---|
| user_id | Numeric user ID |
| cluster | [0, 1, 2, 3] cluster assignment |
| cluster_name | Named cluster: "Brand advocates" / "Expressive regulars" / "Passive regulars" / "Delayed visitors" |

**Size**: 18,752 rows (one per unique user, latest window assignment)

**Silhouette score**: reported in notebook Section 3 (measure of clustering quality)

---

**Window-level monthly cluster matrix**

**File**: `Ig_RCEDTG_windowlevel_kmeans_k4_monthly_cluster_matrix.csv`

| Row | Columns | Values |
|---|---|---|
| user_id | 2022-09, 2022-10, ..., 2026-03 (40 month columns) | Cluster code C1/C2/C3/C4, "Inactive", "Not yet active" |

**Size**: 18,752 users × 40 months (long format melt available in analysis notebooks)

**Active occupancy**: 165,330 rows (users × months where state ∈ {C1, C2, C3, C4})

---

**Cluster summaries**

**File**: `Ig_RCEDTG_kmeans_k4_cluster_summary.csv`

Per-cluster aggregate statistics:

| Column | Description |
|---|---|
| cluster | [0, 1, 2, 3] |
| cluster_name | Human name |
| count | Users in cluster |
| recency_mean / coverage_mean / engagement_mean / delay_mean / tenure_mean / gini_mean | Feature means |

---

**User frequency profiles**

**File**: `Ig_RCEDTG_user_frequency_profiles.csv`

Per-user summary across all windows:

| Column | Description |
|---|---|
| user_id | |
| n_windows_active | Count of windows user was in any active cluster |
| cluster_sequence | Comma-separated cluster codes (user's lifecycle trajectory) |
| most_common_cluster | Mode cluster |
| cluster_switching_rate | % of active windows where user switched cluster |

---

**Window-level cluster summaries**

**File**: `Ig_RCEDTG_windowlevel_kmeans_k4_cluster_summary.csv`

For each window, per-cluster statistics:

| Column | Description |
|---|---|
| window_id | w01–w40 |
| cluster | C1/C2/C3/C4 |
| count | Users in cluster during this window |
| recency_mean / coverage_mean / ... | Feature means within window-cluster |

---

## Non-overlapping 3-year periods analysis

**Location**: `Mickey/RFM_IG_3_years/`

Clusters users across three disjoint ~3-year time blocks covering the full dataset span. Answers: "How does the user population structure evolve across years?"

### Feature table

**File**: `ig_user_nonoverlapping_3y_periods_new_definitions.csv`

Same columns as rolling quadrimester table, but for three periods:
- Period 1: ~2016–2018
- Period 2: ~2019–2021
- Period 3: ~2022–2024 (approx; boundaries in file)

**Size**: 62,500+ rows (users × periods, sparse — many users don't span all periods)

### Clustering outputs

**k=4 KMedoids assignments** (different algorithm than rolling analysis)

**File**: `Ig_RCEDTG_nonoverlapping_3y_kmedoids_k4_assignment.csv`

| Column | Description |
|---|---|
| user_id | |
| period | Period 1 / Period 2 / Period 3 |
| cluster | [0, 1, 2, 3] |
| cluster_name | Cluster label |

---

**Period-specific cluster summaries**

**File**: `Ig_RCEDTG_nonoverlapping_3y_kmedoids_k4_cluster_summary.csv`

Aggregate stats per period × cluster:

| Column | Description |
|---|---|
| period | 1 / 2 / 3 |
| cluster | [0, 1, 2, 3] |
| count | Users |
| recency_mean / coverage_mean / ... | Feature aggregates |
| user_count_absolute | Raw user count (for distinguishing share vs absolute) |

---

**Cluster descriptions (narrative)**

**File**: `Ig_RCEDTG_nonoverlapping_3y_kmedoids_k4_cluster_descriptions.csv`

Per-cluster characteristics (free-text):

| Column | Description |
|---|---|
| cluster | [0, 1, 2, 3] |
| description | Narrative characterization of cluster behavioural profile |

---

**Radar plot data**

**Location**: `cluster_radar_plots_nonoverlapping_3y/`

Scaled feature profiles for radial visualization.

- `ig_nonoverlapping_3y_kmedoids_k4_cluster_period_profiles_raw.csv` — Raw feature values
- `ig_nonoverlapping_3y_kmedoids_k4_cluster_period_profiles_scaled.csv` — Min-max scaled [0, 1] for radar

| Columns | Description |
|---|---|
| cluster_period | e.g., "C0_P1" (cluster 0, period 1) |
| recency / coverage / engagement / delay / tenure / gini | Scaled feature values |

---

## EDA notebooks

### `EDA_RFM_IG_vs_TK_rolling_quadrimesters.ipynb`

**Output directory**: `Mickey/eda_rfm_outputs/`

Comprehensive within-platform and cross-platform distribution, correlation, and temporal analysis.

**Sections**:
1. Data quality and feature summary
2. Univariate distributions (raw + log1p for heavy-tailed)
3. Common-feature IG vs TK comparison (KS test, Mann-Whitney)
4. Pearson and Spearman correlations within each platform
5. Feature pairplots and focused pairs (hexbin density)
6. Temporal evolution across 40 windows
7. Cross-platform window-aggregate correlations (and detrended first differences)
8. VIF (multicollinearity) analysis
9. Summary findings

**Key output files**:
- `ig_distributions.png` — Univariate histogram grid
- `ig_corr_heatmap.svg/png` — Pearson and Spearman correlation blocks
- `ig_pairplot.svg/png` — Full 6×6 pairwise scatter (sampled)
- `ig_focused_pairs.png` — Interpretable hexbin pairs
- `temporal_means.png` — Feature trends over 40 windows
- `common_feature_density.png` — IG vs TK KDE overlay
- `cross_platform_same_feature.png` — IG-TK aligned scatter (coverage, engagement, etc.)
- `cross_platform_full_heatmap.png` — All IG aggregates vs all TK aggregates (Spearman)

---

### `EDA_cluster_matrix_IG_vs_TK.ipynb`

**Output directory**: `Mickey/eda_cluster_matrix_outputs/`

Lifecycle, activity rate, cluster transitions, and cross-platform cluster dynamics.

**Sections**:
1. Data loading (wide → long transformation)
2. Activity rate over time (% active vs inactive vs not-yet-active)
3. Cluster-state composition per month
4. First-activation (debut) timing distribution
5. Active lifespan and re-entry rate
6. Cluster loyalty (Shannon entropy per user)
7. Month-over-month cluster switch rate (within active users)
8. Cross-platform overlay (shared window 2023-01 to 2026-03)
9. Cluster share time series (C1/C2/C3 monthly stacked area)
10. Detrended cluster share (first differences)
11. Survival curves (Kaplan-Meier style)

**Key output files**:
- `01_activity_rate.png` — Stacked bar: not-yet-active / inactive / active by month
- `02_active_pct_line.png` — Line: % active users over time (IG vs TK)
- `03_active_count_line.png` — Line: absolute active-user counts
- `04_tk_cluster_share.png` / `05_ig_cluster_share.png` — Cluster share stacked bars
- `06_cluster_counts.png` — Cluster user counts as lines (absolute, not %)
- `07_debut_timing.png` — Histogram of first-active month
- `08_lifecycle.png` — Active months distribution, re-entry rate pie, pct_active histogram
- `09_entropy.png` — Cluster loyalty distribution + IG vs TK overlay
- `10_switch_rate.png` — Month-over-month cluster switch % trend
- `11_shared_active_pct.png` — Shared-window activity overlap
- `12_cross_cluster_share_corr.png` — IG cluster share × TK cluster share Spearman heatmap
- `14_cluster_share_raw_series.png` / `15_cluster_share_diff_series.png` — Time series of C1/C2/C3 shares (raw and detrended)
- `16_cluster_corr_heatmaps.png` — Raw and detrended (first-difference) Spearman heatmaps
- `17_scatter_delta_same_cluster.png` — Scatter of ΔC1 (IG vs TK) aligned by time
- `13_survival.png` — Kaplan-Meier style survival curves

---

## Key statistics summary

| Metric | Rolling | 3-year |
|---|---|---|
| Total observations | 174,287 | ~62,500 |
| Unique users | 18,752 | 18,752 (subset per period) |
| Time bins | 40 windows | 3 periods |
| Algorithm | KMeans k=4 | KMedoids k=4 |
| Feature set | RCEDTG (6 features) | RCEDTG (6 features) |
| User-month activity rate | ~17% (active / total rows) | ~8–15% per period (varies) |
| Re-entry rate | 63% | Varies by period |
| Dominant cluster | C3 Passive regulars | C3 (70–75%) |
| VIF max | 4.8 (coverage) | — |
| Top correlation (Pearson, IG) | coverage × gini: ρ=0.78 | — |
