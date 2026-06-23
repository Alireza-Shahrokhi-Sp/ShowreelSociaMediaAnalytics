# Event-Impact / Turning-Point Analysis Plan (for Sonnet)

**You are Sonnet (Engineering & Modeling tier). Read this file fully, then follow it.**
This is a `/code` task: implement the statistical tests an Opus agent already designed
below. Do NOT redesign the methodology or pick different proxies — the proxy choice,
the join keys, and the "what C1 means" semantics here are VERIFIED FACTS (an Opus agent
inspected every file on 2026-06-14). Re-print `df.shape`/`df.dtypes` after each load to
guard against drift, but trust the design.

**Research question:** Were Camihawke's major career events *meaningful turning points*?
We test this by treating each event as an intervention and measuring whether her
audience's composition / sentiment / loyalty changed around it.

---

## 0. Guardrails (READ FIRST)

- **Instagram only.** This is a single-creator (Camihawke) study.
- **No new LLM/Vertex/GCS calls.** Everything is local. All labels already exist.
- **Work in a NEW notebook** `Ali/event_impact_pipeline.ipynb`. Do not edit Mickey's
  notebooks or the persona/sentiment pipelines.
- Save artifacts/figures to `Ali/outputs/event_impact/` (create it).
- **Follow CLAUDE.md "context frugality":** build a chunked todo list (tasks + subtasks)
  first; after each task give the user the key result pointers so they can `/compact`.
- ASCII only in code/notebook (no emoji) per CLAUDE.md formatting rules.

### Files you MUST read (and ONLY these)
| Role | Path |
|------|------|
| **Events (the interventions)** | `Ali/camihawke_major_events_timeline.csv` (17 events, 2016–2026; `Date`,`Event_Title`,`Category`,`Details`,`Metrics_or_Scope`) |
| **Proxy 1 — loyalty cluster over time** | `Mickey/RFM_IG_rolling_quadrimesters/Ig_RCEDTG_k4_monthly_cluster_matrix.csv` |
| Cluster meaning (what C1 is) | `Mickey/RFM_IG_3_years/Ig_RCEDTG_nonoverlapping_3y_kmedoids_k4_cluster_descriptions.csv` |
| **Proxy 2 — sentiment over time** | `Ali/outputs/stage2_sentiment/sentiment_instagram.parquet` |
| **Proxy 3 — persona composition** | `Ali/outputs/stage2_persona_combined/user_personas_combined.parquet` + `Ali/outputs/comments_ml.parquet` (for per-comment timestamps to time-bucket personas) |
| Post dates (optional, to attribute events to posts) | `Ali/outputs/ig_multimodal_final.parquet` |

### Files STRICTLY FORBIDDEN (irrelevant — do not open)
- Any other file under `Mickey/` (the `C1_*_model.csv`, RFM radar/profile CSVs, Mickey's
  notebooks). They are *post-level* models for a different question and use a different
  unit of analysis. Use ONLY the one monthly_cluster_matrix file named above.
- `reza/`, `Ali/Backup/`, `Ali/Archive/`, `Ali/batch_results/`, `Ali/multimodal_dataset*/`,
  `Ali/Output/ig_raw/`, any download/scraping/`*.info.json`/`*.ipynb` producer code.
- YouTube/TikTok/Facebook data in `Data/`.
- The persona/sentiment *notebook internals* — you only need their output parquets above.

---

## 1. VERIFIED facts about the proxies (trust these)

> **All three proxies are CO-EQUAL.** None is primary or superior. C1-share, sentiment,
> and persona-composition are three independent lenses on the same question; the verdict
> (§2) weights them equally and a turning point must show up across proxies to be
> "supported," not just in the cluster proxy. Do not rank or privilege one over another.

### Proxy 1: C1 monthly cluster matrix — loyalty/composition lens

- `Ig_RCEDTG_k4_monthly_cluster_matrix.csv`: **18,752 users (rows)** × monthly columns
  `2023-01 … 2026-03` (**39 months**). Plus `user_key`, `user_id`, `username`.
- Each cell ∈ {`C1`,`C2`,`C3`,`C4`,`Inactive`,`Not yet active`} = that user's RFM cluster
  that month.
- **C1 = "Brand Advocates" — the STRONGEST, most valuable loyalty cluster** (very recent,
  expressive, fast-responding, highly tenured/dispersed). So **higher C1 share = more
  success.** This is the directional anchor for every hypothesis. C4="Occasional
  Newcomers" (weakest). Ordering of value: C1 > C2 > C3 > C4.
- **Derive the monthly time series** by melting wide→long: for each month compute
  `c1_share = (#users==C1) / (#users active that month)` where "active" =
  cell ∈ {C1,C2,C3,C4} (exclude `Inactive`/`Not yet active` from the denominator —
  decide and STATE this; also report an alternative denominator = all-ever-active users,
  so the metric isn't an artifact of the denominator). Also track raw `n_C1`, `n_active`,
  and `c1_to_c4` ratio. These monthly series are the dependent variables for Proxy 1.

### Proxy 2: sentiment_instagram.parquet
- 487,604 IG comments, already IG-only. Has `timestamp`, `sentiment_cat`
  (positive/negative/neutral), `sentiment_score` (-1..1), `toxicity`, `sarcasm`,
  `intent`, `target`, `author_id`, `media_id`.
- Aggregate to a **monthly** series: mean `sentiment_score`, %positive, %negative,
  sarcasm_rate, mean toxicity. These are secondary dependent variables.
- **Caveat (state it):** this is a *sample* that prioritized media-bearing posts, so
  monthly sentiment rates are not a clean population mean — use for relative change, not
  absolute level.

### Proxy 3: persona composition
- `user_personas_combined.parquet`: 40,019 users, `author_id` + `persona_codename`
  (10 classes incl. THE_SUPERFAN, THE_CRITIC, THE_HATER, THE_TAGGER …). This is a
  **single static label per user** (no time dimension on its own).
- To get composition *over time*, join persona onto `comments_ml` IG comments by
  `author_id` (string key, ~98% overlap), then bucket comments by their `timestamp`
  month and compute the monthly share of comments (or active users) from each persona —
  especially **%SUPERFAN (loyalty up) and %CRITIC+%HATER (backlash up)**.

### Join keys (all string — keep string)
- persona/sentiment/comments ↔ user: `author_id`.
- The cluster matrix uses `user_id`/`user_key`/`username`. **VERIFY the id-space match**
  before cross-using it with author_id: print samples of `user_id` from the matrix vs
  `author_id` from sentiment. If they don't align, treat Proxy 1 as a **standalone
  aggregate series** (you do NOT need to join it to individual comments — the monthly
  c1_share is computed entirely within Mickey's matrix). Most of the design only needs
  Proxy 1 as its own monthly series, so an id mismatch is NOT a blocker — note it and
  proceed. Do not force a join that doesn't exist.

### Timestamp handling
- Check dtype/magnitude of every `timestamp` and the event `Date` (US format M/D/YYYY).
  Convert all to monthly `Period('M')` for alignment. State the unit (s vs ms) you used.

---

## 2. The core design (this is the methodology — implement it, don't change it)

Each of the 17 events is a candidate **intervention** on a monthly time series. Because
this is observational single-subject time-series data, "A/B testing" is realized as
**interrupted time series (ITS) / structural-break tests**, NOT a randomized A/B test
(there is no control creator). Be precise with the user about this framing.

### Pre-step: align events to the data window
- The cluster matrix covers **2023-01 → 2026-03**. Events before 2023 (the 2016–2021
  ones) have **no pre-data in Proxy 1** — they CANNOT be tested with Proxy 1; note them
  as out-of-window. Sentiment/persona windows may differ — compute each proxy's actual
  date coverage first and tabulate which events are testable by which proxy.
- Classify events by `Category` into hypotheses:
  - **Career/commercial milestones** (tours, book, TV, awards) → expected to *raise* C1
    share / %SUPERFAN / sentiment.
  - **Personal/relationship events** (breakup confirmation, separation interview) →
    ambiguous: could raise *interest* (engagement) but also *backlash* (%CRITIC/%HATER,
    toxicity). Test both directions.

### Test 1 — Pre/Post window comparison (the "A/B" comparison)
For each testable event and each proxy series:
- Define a pre-window and post-window of equal length (default **±3 months**, also run
  ±2 and ±6 as robustness). Exclude the event month itself from both (washout).
- Compare pre vs post with: **Mann–Whitney U** (non-parametric, small N) as primary, and
  Welch's t as secondary. Report effect size (difference in means / **Cliff's delta**),
  not just p-value.
- **Multiple-comparison correction is MANDATORY:** 17 events × several proxies = many
  tests. Apply **Benjamini–Hochberg FDR** across the full family and report q-values.
  Without this, something will look "significant" by chance — call this out loudly.
- **Overlapping windows caveat:** events in 2024–2025 cluster close together (the tour
  events are months apart), so pre/post windows overlap and effects are confounded —
  you cannot cleanly attribute a change to a single event. Flag every event whose window
  overlaps another event's window.

### Test 2 — Interrupted Time Series regression (the rigorous version)
On the monthly c1_share series (and sentiment series):
- Fit a segmented regression: `y_t = b0 + b1*time + b2*post_t + b3*(time*post_t) + e`,
  where `post_t`=1 after the event. `b2` = **level change** (immediate jump),
  `b3` = **slope change** (trend shift) — the two classic turning-point signatures.
- Use Newey–West / HAC standard errors (monthly autocorrelation is guaranteed).
- Do this only for events with enough pre AND post points (>= ~4 each). Most pre-2024
  events won't qualify for Proxy 1.
- If `statsmodels` is available use it; ITS is just OLS with these regressors + HAC SEs.

### Test 3 (optional, strongest) — CausalImpact-style counterfactual
If `pip` has `causalimpact`/`tfcausalimpact` available locally, OR build a simple
Bayesian-structural / pre-trend-extrapolation counterfactual: model the pre-period
trend+seasonality, project it forward, and measure the post-period deviation. With no
control series, the counterfactual rests on pre-trend only — STATE this limitation.
This is "nice to have"; Tests 1+2 are the required deliverable.

### What makes an event a "meaningful turning point" (decision rule — state it up front)
**The three proxies (C1-share, sentiment, persona-composition) are weighted EQUALLY** —
no proxy is primary or a tie-breaker. For each event, evaluate the evidence in each proxy
independently, then count how many proxies show a *significant* effect (after FDR, in the
expected direction). Within one proxy, "significant" = a significant pre/post difference
(Test 1, q<0.05, |Cliff's delta| not negligible) AND/OR a significant ITS level OR slope
change (Test 2), and the event does not fail the overlapping-window confound.

Tiered verdict (by count of the equally-weighted proxies that fire, among those where the
event is in-window for that proxy):
- **Supported** — effect in **≥2 of the 3** proxies (a real turning point shows across
  lenses, not just one metric).
- **Suggestive** — effect in exactly **1 of 3** proxies, or mixed/conflicting directions.
- **Not supported** — significant in **0** proxies.
- **Untestable** — out of window for so many proxies that <2 proxies could even be tested
  (e.g. pre-2023 events with no Proxy-1 pre-data; say which proxies were testable).

Report the per-proxy result for every event (not just the aggregate) so the equal
weighting is auditable. If proxies disagree in *direction* (e.g. C1-share up but
sentiment down), surface that explicitly — it is itself a finding, not noise to average
away.

---

## 3. Things to be wary of (PAY EXTRA ATTENTION)
1. **Pre-2023 events are untestable with Proxy 1** — don't fabricate a result for them.
2. **Multiple comparisons** — FDR-correct or every conclusion is suspect.
3. **Autocorrelation** — monthly series are serially correlated; naive t-tests
   understate variance. Use HAC SEs in ITS; use non-parametric tests in Test 1.
4. **Confounded/overlapping events** — 2024-2025 tour events are too close to isolate.
5. **Denominator artifacts in c1_share** — report >=2 denominator definitions.
6. **Sampling bias in sentiment proxy** — relative change only, not absolute level.
7. **Persona is static** — its "over time" comes only from comment timestamps; a user's
   persona doesn't change, so persona-composition shifts reflect *who is commenting*,
   not *who converted* — phrase findings accordingly.
8. **id-space between Mickey's matrix and author_id may not match** — verify; Proxy 1
   works standalone if not, so don't block on it.
9. **Seasonality** — Instagram activity has yearly seasonality; ITS with a time term
   partly handles it, but note December/summer effects near some events.
10. **Reverse causality / selection** — milestones may be *caused by* rising audience,
    not the reverse. This is correlational; do not over-claim causation.

---

## 4. Deliverables
- `Ali/event_impact_pipeline.ipynb` (runs top-to-bottom, headless, ASCII only).
- `Ali/outputs/event_impact/monthly_series.parquet` (c1_share + sentiment + persona
  monthly series, one row per month).
- `Ali/outputs/event_impact/event_test_results.csv` — one row per (event × proxy ×
  window): pre/post means, effect size, raw p, **FDR q**, ITS b2/b3 + SE, overlap flag,
  testable flag, per-proxy significant-yes/no + direction. Plus a per-event rollup
  (`n_proxies_testable`, `n_proxies_fired`, `verdict`) so the equal-weighting is auditable.
- Figures per testable event: the proxy series with the event date marked + pre/post
  means + ITS fit, saved to `Ali/outputs/event_impact/`.
- A final markdown summary: ranked list of events by evidence strength, with the §3
  caveats stated explicitly, and a one-line plain-business verdict per event.

## 5. Definition of done

- The three proxies are weighted EQUALLY; the verdict comes from counting how many fired,
  and per-proxy results are reported, never collapsed into one "primary" metric.
- Every claim is FDR-corrected and direction-checked (C1="Brand Advocates" up=good;
  %SUPERFAN/positive-sentiment up=good; %CRITIC/%HATER/toxicity up=backlash).
- Out-of-window and overlapping-confound events are labeled, not silently dropped.
- Framing is "interrupted time series on observational single-creator data," not RCT.
- Only §0 files were read; all forbidden files untouched.
