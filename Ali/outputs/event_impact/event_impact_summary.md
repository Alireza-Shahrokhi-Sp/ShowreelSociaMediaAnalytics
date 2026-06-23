# Camihawke Event-Impact Analysis — Final Summary

**Study type:** Interrupted Time Series (ITS) on observational single-creator Instagram data.
This is NOT a randomised controlled trial. No control creator exists. Causal language is not warranted;
findings are stated as "associated changes" or "turning-point signals".

**Analysis date:** 2026-06-14
**Notebook:** `Ali/event_impact_pipeline.ipynb`
**Artifacts:** `Ali/outputs/event_impact/`

---

## Caveats (front-loaded, per §3 of the plan)

1. **Multiple comparisons.** 379 testable comparisons were FDR-corrected (Benjamini-Hochberg).
   After correction, **zero tests reach q<0.05 and zero reach q<0.10.** The individual p-values
   below are ALL pre-correction and should not be read as standalone evidence.
   Significance language below refers to ITS HAC-SE regression, which is NOT part of the
   FDR family (it uses the full time series), and is a *second* corroborating signal only.

2. **Overlapping windows.** All events from 2025-01 onward are within 6 months of at least
   one other event. Their pre/post windows overlap, so no single event's effect can be
   cleanly isolated from neighbours. P1 (cluster) results for these events are flagged
   `confounded`.

3. **Pre-2023 events are untestable on Proxy 1** (cluster matrix starts 2023-01). Only
   Proxy 2 (sentiment, from 2016-08) and Proxy 3 (persona, from 2008) apply.

4. **Persona proxy (P3) has a 2.6% match rate.** Only 96k of 3.66M comments carry a
   persona label. Monthly persona shares reflect the labelled subset only. Persona is also
   static per user (assigned once), so shifts in monthly persona share reflect *who is
   commenting that month*, not conversion of individual users.

5. **Sentiment sample (P2) is biased** toward media-bearing posts. Monthly rates are not
   population means; only relative change is interpretable.

6. **Seasonality.** C1 share dips in Q4 each year (Dec 2023: c1_share=0.154; Dec 2024:
   0.186) with large C4 (newcomers) spikes. ITS includes a linear time term which absorbs
   trend but not seasonality; some apparent "changes" near year-end events may be seasonal.

7. **Reverse causality.** Career milestones may follow from a rising audience rather than
   causing it. This is correlational.

---

## Proxy 1 — C1 share series (Brand Advocates, primary proxy)

Coverage: 2023-01 to 2026-03 (39 months). 18,752 users.

**Denominator note:** `c1_share_active` = C1 / (C1+C2+C3+C4 that month); used as primary.
`c1_share_total` = C1 / 18,752 (stable denominator); both series trend similarly.

**Baseline level:** ~0.17-0.23 in 2023, declining to 0.12-0.19 in 2024, then stabilising
~0.16-0.21 in 2025, with a late-2025 spike (Nov 0.254, Dec 0.252 — driven simultaneously
by C4 newcomers surge, so the denominator effect is partly mechanical).

---

## Event Verdicts (ranked by evidence strength)

### Tier 1 — Suggestive (corroborating ITS signal, but FDR-corrected q not significant)

**1. Solo Tour Launch — 'Il saggio di fine anno' (2023-04)**
Cleanest event in the dataset: no overlapping events within 6 months.
- P1 (c1_share_active): pre mean 0.168 → post mean 0.220, diff +0.052, Cliff's delta=1.0 (large).
  ITS: level change b2=-0.015 (p=0.44, ns), slope change b3=-0.007 (p=0.40, ns).
  Pre/post comparison is striking in magnitude but the ITS finds no structural break —
  the rise was already in the pre-period trend.
- P2 (sentiment): pre 0.553 → post 0.596, ITS slope b3=-0.005 (p=0.0001, sig) — sentiment
  rose pre-tour but the slope *declined* post-tour (regression to mean after peak).
- **Verdict: suggestive on P1 raw comparison; ITS does not confirm a structural break.
  Business read: the tour coincided with a high-C1 period but may have caught a naturally
  rising phase rather than causing a step-change.**

**2. Debut Novel Publication (2021-04)** [P1 untestable; P2 + P3 only]
- P2 sentiment: pre 0.566 → post 0.622 (+0.056), Cliff's delta=0.78 (large), ITS slope
  b3=-0.004 (p<0.0001 sig) — same pattern as tour: level up then declining slope.
- P3 superfan: pre 0.407 → post 0.430 (+0.023), ITS slope b3=-0.005 (p<0.0001 sig).
- **Verdict: suggestive on P2/P3. The novel launch appears associated with a temporary
  sentiment and superfan-share elevation that subsequently reverted.**

**3. Initial Digital Footprint Expansion (2016-11)** [P1 untestable; P2 + P3 only]
- P3 superfan: ITS level b2=+0.266 (p<0.0001) — a very large structural level jump in
  superfan share after this baseline period. Contextually expected: audience composition
  changes dramatically as Camihawke builds her community from scratch.
- P2 sentiment: not significant.
- **Verdict: suggestive on P3, but this is a baseline formation event, not a pivot.**

**4. Macchianera Award (2019-11)** [P1 untestable]
- P2 sentiment: pre 0.486 → post 0.509, ITS slope b3=-0.003 (p=0.001 sig, HAC).
  Sentiment marginally higher post-award but declining trend.
- **Verdict: suggestive (small effect). Award may have attracted positive attention briefly.**

**5. TEDxRimini (2019-12)** [P1 untestable]
- ITS slope b3=-0.003 (p=0.0003) on sentiment — declining trend post-TEDx, not rising.
- **Verdict: suggestive (small effect, wrong direction for sentiment). Possibly confounded
  by proximity to Macchianera Award one month earlier.**

---

### Tier 2 — Confounded (P1 testable but window overlaps; cannot isolate)

**6. Avanguardia Pura Launch + Ticket Sales Milestone (2024-06)**
Two events in the same month; inseparable. Both overlap with 2024-05 and 2024-07.
- P1: pre 0.157 → post 0.189, Cliff's delta=0.56 (large). ITS slope b3=+0.004 (p=0.019 sig).
  This is the only P1 event showing a significant positive ITS slope change — but it is
  confounded (same month, dual events) and the pre/post FDR q is 0.91.
- P2 sentiment: *drops* post (pre 0.558 → post 0.453, wrong direction). P3 superfan also
  drops. This may reflect the content shift or the comment sample composition changing.
- **Verdict: confounded on P1. Cannot attribute. ITS slope signal is notable but
  cannot be cleanly credited to this event cluster.**

**7. Avanguardia Pura Tour Premiere (2025-01)**
- P1: c1_share falls post (pre 0.164 → post 0.152, wrong direction). Confounded by
  2025-02, 2025-03, 2025-04.
- P3 superfan: pre 0.326 → post 0.367 but ITS slope steeply negative (b3=-0.011, p<0.001).
- **Verdict: confounded; no clean signal.**

**8. Media Confirmation of Relational Dissolution (2025-02)**
Personal event — direction ambiguous by design.
- P1: slight drop post (pre 0.173 → post 0.151). ITS slope b3=+0.006 (p=0.004, sig) —
  suggests a recovering slope after a dip, which could be a rebound from breakup news.
- P2 sentiment: drops (pre 0.502 → post 0.448), ITS level b2=-0.069 (p=0.029 sig).
- **Verdict: confounded. Signal reads as a sentiment dip and C1 dip around the breakup
  news period, with subsequent recovery. Cannot isolate from neighbouring events.**

**9. Public Clarification of Interpersonal Separation (2025-09)**
Personal event — direction ambiguous.
- P1: slight rise post (pre 0.200 → post 0.229). ITS level b2=+0.051 (p=0.051, borderline).
- P2 sentiment: ITS shows very strong slope reversal b3=+0.015 (p<0.0001) — sentiment
  *recovering* after the interview, which corroborates a "closure narrative" interpretation.
- **Verdict: confounded (overlaps 2025-03/04/10). The sentiment slope reversal is
  notable but cannot be isolated.**

**10. Autumn Leg Re-ignition of Avanguardia Pura (2025-10)**
- P1 ITS: level b2=+0.092 (p<0.0001), slope b3=-0.015 (p<0.0001). Large immediate C1
  level jump followed by declining slope. This is the strongest ITS signal in P1 —
  but it is confounded with 2025-09 and 2025-04 windows.
- **Verdict: confounded. The Oct 2025 tour restart coincides with the largest P1 ITS
  signal, but attribution is impossible given the Sep 2025 interview and the broader
  2025 touring run.**

---

### Tier 3 — Not Supported

**11. Rai Radio2 (2017-09)** — Sentiment and superfan move in wrong direction post-event.
ITS level on sentiment b2=+0.066 (p=0.02) but this is likely a gradual pre-trend effect.

**12. Co-Hosting National TV debut (2019-09)** — ITS slope on sentiment b3=-0.003 (p=0.004)
and superfan b3=-0.004 (p<0.0001) — both declining post-debut, not rising. This is
counterintuitive; may reflect audience composition shift (more casual viewers from TV).

**13. Autumn Tour Circuit Expansion Announcement (2025-03)** — Confounded; P3 superfan
and P2 sentiment both decline post-announcement (wrong direction).

**14. Etna Comics Marquee Guest Booking (2025-04)** — Confounded; all proxies wrong direction.

**15. Autumn Leg Tour Events (2025-10, P2/P3)** — Sentiment and superfan move wrong direction.

---

### Tier 4 — Untestable

**16. WMF Creators Fest (2026-06)** — After data end (2026-03).
**17. The Traitors Italia Cast Unveiling (2026-06)** — After data end.

---

## Summary Table

| Event | Month | P1 verdict | P2 verdict | P3 verdict | Overall |
|-------|-------|-----------|-----------|-----------|---------|
| Initial Digital Footprint Expansion | 2016-11 | untestable | not_supported | suggestive | suggestive (baseline formation) |
| Rai Radio2 Broadcast | 2017-09 | untestable | not_supported | not_supported | not supported |
| TV Co-host Debut | 2019-09 | untestable | not_supported | not_supported | not supported |
| Macchianera Award | 2019-11 | untestable | suggestive | not_supported | suggestive (small) |
| TEDxRimini | 2019-12 | untestable | suggestive (small) | not_supported | suggestive (small, mixed) |
| Novel Publication | 2021-04 | untestable | suggestive | suggestive | suggestive |
| **Solo Tour Launch 2023** | **2023-04** | **suggestive** | suggestive | not_supported | **suggestive (best clean signal)** |
| Avanguardia Pura Launch | 2024-06 | confounded | not_supported | not_supported | confounded |
| Tour Premiere | 2025-01 | confounded | not_supported | suggestive | confounded |
| Breakup confirmation | 2025-02 | confounded | suggestive (ambig) | suggestive (ambig) | confounded |
| Tour expansion announcement | 2025-03 | confounded | not_supported | not_supported | confounded |
| Etna Comics | 2025-04 | confounded | not_supported | not_supported | confounded |
| Breakup interview | 2025-09 | confounded | suggestive (ambig) | suggestive (ambig) | confounded |
| Autumn tour restart | 2025-10 | confounded | not_supported | not_supported | confounded (strongest raw ITS signal) |
| WMF Creators Fest | 2026-06 | untestable | untestable | untestable | untestable |
| The Traitors Italia | 2026-06 | untestable | untestable | untestable | untestable |

---

## Key findings for stakeholders

1. **No event passes the plan's dual-gate criterion** (FDR q<0.05 AND ITS sig AND correct
   direction AND not confounded). The FDR correction absorbs all pre/post signals across
   379 comparisons; the monthly series are too short and noisy for any individual event
   to survive it.

2. **The 2023 Solo Tour is the strongest unconfounded signal.** C1 share rose from 0.168
   to 0.220 in the three months after the tour launch (Cliff's delta=1.0, large). However,
   the ITS finds no structural break — the rise was part of a pre-existing trend.
   Verdict: the tour *captured* and *maintained* an already-rising advocacy base rather
   than creating a new one.

3. **The 2025 tour run (Oct restart especially) has the largest raw ITS magnitude**
   (b2=+0.09 on c1_share), but is hopelessly confounded by the 2025 events cluster.

4. **Personal events (breakup) show sentiment dips, not spikes.** The Feb 2025 breakup
   confirmation is associated with a sentiment level drop (b2=-0.069, p=0.03) and a C1
   dip, followed by a recovery slope. The Sep 2025 interview shows the sentiment slope
   recovering. Consistent with a "turbulence then stabilisation" narrative, but confounded.

5. **Pre-2023 events (2016-2021) show declining superfan-share trends** after each event
   in the ITS (negative b3 on P3_pct_superfan). This is a denominator/composition
   artifact: as the total audience grew, the superfan share naturally compressed even if
   absolute superfan counts rose.

6. **Bottom line:** with 39 monthly observations on Proxy 1 and highly autocorrelated data,
   this dataset is underpowered for isolating individual event effects. The analysis
   correctly identifies that the data cannot support strong causal claims.
   A CausalImpact / Bayesian structural-break approach (Test 3, not implemented here)
   with a synthetic control or at minimum a seasonal ARIMA counterfactual would be the
   next step to sharpen inference.
