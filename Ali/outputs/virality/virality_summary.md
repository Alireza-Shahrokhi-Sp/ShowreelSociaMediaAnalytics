# Virality Analysis: Announcement vs Occurrence
Generated: 2026-06-14

## Hypothesis
The audience burst tracks the NEWS announcement date, not the event-occurrence date.
Tested via SARIMAX counterfactual daily comment volume; paired Wilcoxon signed-rank.

## Data
- Daily IG comment volume: 2016-08-18 to 2026-03-20 (3,480 distinct days, median=42, max=6,633)
- Timeline: 9 First Public Announcements + 17 Milestones/Occurrences from combined CSV

## Intervention Pairs Summary
- Nella MIA Cucina Rai2: announce=2019-09-12, occur=2019-09-16, gap=4.0d (INSEPARABLE (14d))
- TEDxRimini 2019: announce=2019-11-20, occur=2019-12-14, gap=24.0d (separable)
- Novel Pre-Order: announce=2021-03-09, occur=2021-04-20, gap=42.0d (separable)
- Solo Tour Launch: announce=2023-07-03, occur=2023-04-15, gap=-79.0d (separable) [INVERTED: announce after occurrence]
- Avanguardia Pura Tour: announce=2024-06-14, occur=2025-01-12, gap=212.0d (separable)
- Etna Comics 2025: announce=2025-04-10, occur=2025-04-10, gap=0.0d (INSEPARABLE (14d))
- The Traitors Italia S2: announce=2025-11-07, occur=2026-06-12, gap=217.0d (separable)

## Test 1 - SARIMAX Counterfactual (14-day window, 365-day lookback)
Total intervention dates tested: 21
Dates with majority of post-window days above 95% PI: 0
  No intervention date had majority post-window days above 95% PI.

## Test 2 - Announcement vs Occurrence (Paired Wilcoxon)
Separable paired milestones: 3
Inseparable (gap < 14d): 2
Wilcoxon signed-rank (two-sided): stat=2.000, p=0.7500
Announcement excess > occurrence: 2/3 separable pairs
Result: No significant systematic advantage for announcement dates (p >= 0.05).

## Test 3 - Robustness (Seasonal-Naive + ITS)
ITS level-jump BH-FDR significant interventions: 9
  - Nella MIA Cucina Rai2 (announcement): jump=1.1562, q=0.0255
  - TEDxRimini 2019 (announcement): jump=-1.8377, q=0.0255
  - Novel Pre-Order (announcement): jump=1.2460, q=0.0435
  - Solo Tour Launch (announcement): jump=-1.3253, q=0.0435
  - Solo Tour Launch (occurrence): jump=1.0689, q=0.0435
  - Etna Comics 2025 (announcement): jump=-0.9603, q=0.0435
  - Etna Comics 2025 (occurrence): jump=-0.9603, q=0.0435
  - Rai Radio2 Broadcast (occurrence): jump=-1.0190, q=0.0170
  - Autumn Tour Re-ignition (occurrence): jump=2.8146, q=0.0000

## Caveats
1. Single subject (Camihawke), observational, no control creator. Counterfactual rests on pre-trend.
2. Comment data only covers posts scraped ? activity on unscraped posts is uncounted.
3. Daily comment counts may reflect old posts re-activated, not new-post virality; cross-check n_active_posts.
4. Inseparable pairs (gap < 14d) cannot distinguish announcement from occurrence burst.
5. Heavy-tailed volume (max=6,633) means SARIMAX PIs can be wide; model on log1p to reduce heteroscedasticity.
6. With only 9 pairs (fewer separable), Wilcoxon power is limited; interpret descriptively.

## Outputs
- outputs/virality/daily_series.parquet
- outputs/virality/intervention_pairs.csv
- outputs/virality/virality_results.csv
- outputs/virality/announce_vs_event.csv
- outputs/virality/robustness_results.csv
- outputs/virality/fig_overview_daily_volume.png
- outputs/virality/fig_ann_*.png (per-announcement SARIMAX figure)
- outputs/virality/fig_occ_*.png (per-occurrence SARIMAX figure)