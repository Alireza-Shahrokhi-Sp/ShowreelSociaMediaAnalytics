# Announcement-vs-Event Virality Plan (for Sonnet)

**You are Sonnet (Engineering & Modeling tier). Read this file fully, then follow it.**
This is a `/code` task: implement the counterfactual time-series design an Opus agent
already worked out below, after inspecting the data on 2026-06-14. The schemas, join keys,
date ranges, and method choice here are VERIFIED FACTS — implement them, do not redesign.
Re-print `df.shape`/`df.dtypes` after each load to guard against drift.

---

## 0. The hypothesis (what we are actually testing)

The earlier event-impact analysis (`EVENT_IMPACT_PLAN.md`) tested whether *events* were
turning points using monthly proxies. **New, sharper hypothesis:**

> Camihawke "went viral" when the **NEWS of her involvement was first announced**
> (the press-release / cast-reveal / pre-order moment), NOT on the date the event itself
> occurred (the tour premiere, the TV airing, the festival day).

So for each milestone we now have TWO candidate intervention dates:
- **Announcement** = `Entry_Type == "First Public Announcement"` (news breaks).
- **Occurrence**   = `Entry_Type == "Milestone / Occurrence"` (the thing happens).

We test, per pair, whether the **spike in audience activity/sentiment is larger and/or
earlier around the announcement than around the occurrence.** "Viral" = a short-horizon
*excess* over a forecasted baseline (a counterfactual), measured in **daily** resolution.

This is the right framing: virality is a days-to-weeks burst, so monthly buckets (used in
the prior plan) would smear it out. **Everything here is DAILY.**

---

## 1. Guardrails (READ FIRST)

- **Instagram only.** Single creator (Camihawke).
- **No LLM/Vertex/GCS.** All data local. No new label generation.
- **NEW notebook:** `Ali/virality_pipeline.ipynb`. Do not edit any other notebook,
  and do not edit `EVENT_IMPACT_PLAN.md`'s notebook — this is a separate analysis.
- Save artifacts/figures to `Ali/outputs/virality/` (create it).
- ASCII only (no emoji), per CLAUDE.md.
- Follow CLAUDE.md "context frugality": chunked todo list (tasks+subtasks); after each
  task give the user key result pointers so they can `/compact`, then continue.

### Files you MUST read (and ONLY these)
| Role | Path | Verified facts |
|------|------|----------------|
| **Intervention dates (paired)** | `Ali/camihawke_combined_comprehensive_timeline.csv` | 26 rows: 17 `Milestone / Occurrence` + **9 `First Public Announcement`**. Cols: `Date`(YYYY-MM-DD),`Event_Title`,`Entry_Type`,`Category`,`Details`,`Context_or_Metrics`. |
| **Daily activity + sentiment** | `Ali/outputs/stage2_sentiment/sentiment_instagram.parquet` | 487,604 IG comments; `timestamp` is a **string** "YYYY-MM-DD HH:MM:SS"; has `sentiment_score`,`sentiment_cat`,`toxicity`,`media_id`,`author_id`. Range **2016-08-18 → 2026-03-20**. |
| **Daily activity (full volume)** | `Ali/outputs/comments_ml.parquet` | 499,752 IG comments (filter `platform=="instagram"`); `timestamp` string, same range; **3,480 distinct days, median 42 comments/day, max 6,633**. Use this for the volume series (more complete than the sentiment sample). |
| Post dates (to attribute spikes to posts) | `Ali/outputs/ig_multimodal_final.parquet` | 1,493 posts; `media_id`,`timestamp`,`caption`,`permalink`,`like_count`,`reach`. Optional. |

### Files STRICTLY FORBIDDEN (do not open — waste/irrelevant)
- The OLD timeline `Ali/camihawke_major_events_timeline.csv` (superseded by the combined
  one above — use ONLY the combined file).
- Anything under `Mickey/`, `reza/`, `Ali/Backup/`, `Ali/Archive/`, `Ali/batch_results/`,
  `Ali/multimodal_dataset*/`, `Ali/Output/ig_raw/`, any `*.info.json`, any download/
  scraping code, any producer notebook internals.
- YouTube/TikTok/Facebook data in `Data/`.
- The cluster matrix from the prior plan — virality is a daily-burst question and the
  monthly cluster proxy cannot resolve it. Do NOT use Proxy 1 here.

---

## 2. Build the daily series (do this first; checkpoint here)

Build a **daily** dataframe indexed by calendar date over the full IG window
(2016-08-18 → 2026-03-20), **reindexed to every calendar day** (fill missing days with 0
for counts; NaN for rate metrics):

- `n_comments` — daily IG comment count (from `comments_ml`, IG-filtered). PRIMARY virality signal.
- `n_active_users` — daily distinct `author_id`.
- `n_active_posts` — daily distinct `media_id` receiving comments (proxy for "a new post dropped / got attention").
- `mean_sentiment`, `pct_pos`, `pct_neg`, `mean_toxicity` — from the sentiment parquet (note it's a sample; use as secondary).

**Timestamp parsing:** `timestamp` is a STRING — `pd.to_datetime(df['timestamp'])` (no
unit kwarg; it's already a formatted datetime string, NOT epoch). Verify min/max match
the ranges above after parsing. Floor to date for daily grouping.

**Heavy skew warning:** comment volume is bursty (median 42, max 6,633). For modeling,
also compute `log1p(n_comments)`. A handful of viral days dominate the raw scale.

**Pair the interventions:** from the combined CSV, build a table of milestone pairs. Join
`First Public Announcement` rows to `Milestone / Occurrence` rows by matching the
underlying milestone (match on `Event_Title` similarity / shared `Category` + nearest
date; several are obvious: Novel pre-order 2021-03-09 -> publication 2021-04-20; Avanguardia
announce 2024-06-14 -> ticket milestone 2024-06-19 / premiere 2025-01-12; Etna announce ==
occurrence 2025-04-10; Traitors announce 2025-11-07 -> unveiling 2026-06-12, etc.).
Produce `Ali/outputs/virality/intervention_pairs.csv` with
`milestone, announce_date, occurrence_date, gap_days, category`. Some announcements have
no clean occurrence in-window and vice versa — keep them, mark `unpaired`.

> **CHECKPOINT — STOP and report to the user here.** Show: (a) the daily series built
> (shape, date coverage, a plot of `n_comments` over time with announcement dates marked),
> (b) the `intervention_pairs` table with `gap_days`, (c) per-intervention data-coverage
> (does each announce/occurrence date have enough pre-period days for a baseline?). Do NOT
> run the counterfactual models until the user confirms the series and pairs look right.

---

## 3. The method (counterfactual daily forecasting — this is the design)

For EACH intervention date (both announcement and occurrence dates, tested separately so
they can be compared), measure the **causal excess activity** as deviation from a
forecasted counterfactual baseline. This is the single-series analog of CausalImpact.

### 3a. Primary model — SARIMAX counterfactual (per intervention)
- Training window = the **pre-intervention** daily series (use a bounded lookback, e.g.
  the 180–365 days before the date, to keep the baseline locally relevant; STATE the
  choice and test 2 lookbacks for robustness). Model `log1p(n_comments)`.
- Fit **SARIMAX** with weekly seasonality (`m=7`; Instagram has a strong day-of-week
  cycle). Use `pmdarima.auto_arima` if available to pick (p,d,q)(P,D,Q,7); otherwise a
  sensible default like SARIMAX(1,1,1)(1,0,1,7) and check residual ACF. Add holiday/
  day-of-week exogenous dummies if it improves residuals.
- **Forecast forward** over a post-window (default **14 days**, also run 7 and 28) and get
  the prediction interval (95%).
- **Effect = observed − forecast** over the post-window. Report:
  - `cumulative_excess` (sum of observed−forecast),
  - `peak_excess` and `day_of_peak` (lag from intervention date to the spike — KEY for
    the announcement-vs-occurrence comparison: a smaller lag = the spike tracks this date),
  - `point_significance` = fraction of post-window days where observed is above the 95% PI,
  - `relative_lift` = cumulative observed / cumulative forecast.
- Back-transform from log when reporting human-readable comment counts.

### 3b. The actual hypothesis test — announcement vs occurrence (paired)
For each milestone pair, you now have an excess metric for the announcement date and one
for the occurrence date. The hypothesis "**the announcement, not the event, is the viral
moment**" is supported for that milestone when:
- announcement `cumulative_excess` (and/or `peak_excess`, `relative_lift`) **>** occurrence's, AND
- the announcement spike's `day_of_peak` is **tight** (peak within a few days of the
  announce date), indicating the burst tracks the news.
Then aggregate across the 9 pairs: **paired Wilcoxon signed-rank test** on
(announcement_excess − occurrence_excess) across milestones — this is the formal test of
the overall hypothesis. Report effect size and the per-pair breakdown, not just the
aggregate p. With only 9 pairs, power is limited — report it descriptively too.

### 3c. Robustness / alternative estimators (do at least ONE besides SARIMAX)
- **Bayesian structural time series** (`causalimpact`/`tfcausalimpact` if installed) on
  the same pre/post split — cleaner counterfactual + credible intervals. If the package
  is unavailable, skip and note it; do NOT spend long fighting installs.
- **Simple seasonal-naive / Prophet baseline** as a sanity floor: if SARIMAX excess and a
  dumb seasonal-naive excess disagree wildly, investigate before trusting either.
- **Interrupted time series (segmented OLS with HAC SEs)** on the daily log-volume around
  each date as a model-light cross-check of the level jump.

---

## 4. Things to be wary of (PAY EXTRA ATTENTION)
1. **Confounded announce/occurrence windows.** When `gap_days` is small (Etna = 0 days;
   Avanguardia announce→ticket-milestone = 5 days), the two interventions' post-windows
   OVERLAP and you CANNOT separate them — the announcement window already contains the
   occurrence. Flag every pair with `gap_days < post_window` as **inseparable**; for those,
   the comparison is meaningless — say so, don't force a verdict.
2. **Comment date != event virality, strictly.** Comments accrue on POSTS. A spike in
   daily comments can come from (a) a new post going viral, or (b) old posts getting
   re-activated. Cross-check spikes against `n_active_posts` and, if useful, the post
   `timestamp` from `ig_multimodal_final` — a true "she posted about it and it blew up"
   spike should coincide with a new post. Distinguish "new-post virality" from
   "backlog chatter."
3. **Exposure/observability bias.** Our comment data only exists for posts we scraped.
   If a viral moment happened on a post NOT in the dataset, we undercount it. Daily volume
   is a proxy for *comment activity on captured posts*, not total internet buzz. STATE this.
4. **Pre-period sufficiency.** The earliest announcement is 2017-11-13; daily data starts
   2016-08-18, so it has ~15 months of pre-period — OK. But early-period volume is thin
   (the account was smaller), so baselines there are noisier; widen PI expectations.
5. **Heavy-tailed counts -> model on log1p**, and still expect over-dispersion. Consider a
   Poisson/NegBin GLM ITS as an alternative if SARIMAX residuals are ugly.
6. **Multiple comparisons.** 9 pairs x 2 dates x several windows = many tests. The PRIMARY
   inference is the single paired Wilcoxon (one test). For the per-date significance flags,
   FDR-correct (Benjamini-Hochberg) and report q-values.
7. **Weekly seasonality is real and strong** — a model without `m=7` will mistake normal
   weekend dips/peaks for effects. Always include it.
8. **Anticipation / leakage.** News sometimes leaks before the official announcement date;
   if you see the excess starting a few days BEFORE the announce date, that's a finding
   (pre-announcement buzz), not a bug — report the lead/lag honestly.
9. **Single-subject, observational, no control creator** — this is a counterfactual
   forecast, not an RCT. Do not over-claim causation; the baseline rests on the pre-trend.

---

## 5. Deliverables
- `Ali/virality_pipeline.ipynb` (top-to-bottom headless, ASCII only).
- `Ali/outputs/virality/daily_series.parquet` (the reindexed daily frame).
- `Ali/outputs/virality/intervention_pairs.csv` (milestone, announce/occurrence dates, gap, paired flags).
- `Ali/outputs/virality/virality_results.csv` — one row per (intervention_date x window):
  cumulative_excess, peak_excess, day_of_peak, relative_lift, point_significance, FDR q,
  inseparable flag, new-post-coincidence flag.
- `Ali/outputs/virality/announce_vs_event.csv` — per-milestone announcement vs occurrence
  excess, the difference, and the aggregate paired-Wilcoxon result.
- Per-intervention figures: observed daily volume + SARIMAX counterfactual + PI band +
  intervention date marker, saved to `Ali/outputs/virality/`.
- Final markdown summary: for each milestone, "was the announcement or the event the viral
  moment (or neither/inseparable)?", the aggregate verdict, and the §4 caveats stated.

## 6. Definition of done
- Analysis is DAILY with weekly seasonality; virality = excess over a forecasted baseline.
- Announcement vs occurrence is tested PAIRWISE (Wilcoxon) AND shown per-milestone.
- `gap_days < window` pairs are flagged INSEPARABLE, not force-verdicted.
- Spikes are cross-checked against new-post activity (volume vs new-post coincidence).
- Framing is "single-subject counterfactual forecast on observed-comment activity," not RCT.
- Only the §1 files were read; all forbidden files untouched.
