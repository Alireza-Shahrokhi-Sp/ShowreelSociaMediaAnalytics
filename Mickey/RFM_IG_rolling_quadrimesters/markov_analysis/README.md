# Markov-chain stationarity tests & models — IG window-level RCEDTG panel

Analysis of whether the Instagram window-level RCEDTG cluster panel is stationary
and suitable for Markov-chain modelling, plus two fitted models.

**Source data:** `../Ig_RCEDTG_windowlevel_kmeans_k4_monthly_cluster_matrix.csv`
(18,752 users × 39 monthly windows, 2023-01 → 2026-03). Cell = state in
{C1, C2, C3, C4, Inactive, Not yet active}. `Not yet active` is treated as
left-censoring (user not yet in panel); each user's chain starts at first entry.

## Layout — calculation (Python) vs plotting (Jupyter)

Heavy computation lives in **Python scripts** (efficient, headless). The
**notebook only reads their CSV outputs and draws figures** — no recomputation.

| file | role |
|---|---|
| `markov_common.py` | shared loaders, left-censoring, transition counting, Anderson-Goodman test, stationary dist |
| `markov_stationarity_tests.py` | (A) order tests (B) homogeneity + Monte-Carlo effect size (C) marginal drift (D) asymptotics (E) partial stationarity |
| `markov_mature_refit.py` | recurrent 5-state chain refit on mature window (2024-09+); full-vs-mature compare; per-step TV; reactivation hazard |
| `markov_absorbing_churn.py` | duration-based absorbing-churn chain (full + mature): P, fundamental matrix, time-to-churn, survival curves |
| `markov_plots.ipynb` | **plots only** — reads `outputs/*.csv`, writes `outputs/fig*.png` |
| `outputs/` | all CSV/NPY results + PNG figures |

### Reproduce
```bash
cd markov_analysis
python markov_stationarity_tests.py      # prints test report, writes pooled P + shares
python markov_mature_refit.py            # full-vs-mature comparison
python markov_absorbing_churn.py         # absorbing-churn + survival
jupyter nbconvert --to notebook --execute --inplace markov_plots.ipynb   # figures
```
(Use the `ma_env` interpreter: `D:/conda_envs/ma_env/python.exe`.)

## Figures (`outputs/`)
1. `fig1_transition_heatmaps` — recurrent 5-state vs absorbing 6-state P
2. `fig2_full_vs_mature_P` — recurrent P full vs mature + difference
3. `fig3_stationary_pi` — stationary π, full vs mature
4. `fig4_marginal_drift_convergence` — state shares over time + TV distance to π
5. `fig5_partial_stationarity_perstep` — per-step departure from homogeneity
6. `fig6_reactivation_hazard` — return hazard by inactivity duration
7. `fig7_survival_curves` — P(not churned) over 36-month horizon
8. `fig8_time_to_churn` — expected months to churn by start state

## Findings

**Markov order.** Order-1 hugely beats iid (BIC 0.50M vs 1.16M). Mild residual
order-2 memory exists but is small relative to the order-0→1 gain — first-order
is a reasonable working assumption.

**Time-homogeneity.** Anderson-Goodman rejects equality of per-step P_t
(G2/df = 26 vs ~1.0 under a Monte-Carlo homogeneous null). But the **magnitude is
small**: mean per-step row TV from pooled P ≈ 0.04. The deviation is concentrated
in the early, immature-cohort periods (within-year G2/df: 2023 ≈ 30, 2024 ≈ 31,
2025 ≈ 18, 2026 ≈ 6) — the process is **progressively settling**, not breaking.

**Marginal drift.** Strong full-span trends are mostly **panel maturation**: on
the mature panel (windows ≥90% of max size) most drift vanishes (C3/C4 stable).
The latest window sits at TV ≈ 0.017 from the stationary π.

**Asymptotics.** Recurrent chain is irreducible & aperiodic → unique
π (Inactive ≈ 0.81); SLEM ≈ 0.78, spectral gap ≈ 0.22, mixing ≈ 19 months.

**Inactive is NOT absorbing.** 1-step reactivation = 4.5%; monthly return hazard
stays ~5-7% for 6 months then decays to ~3%. So churn is modelled as a
**duration-based absorbing state**: ≥6 consecutive Inactive months → `Churned`.
Expected months-to-churn: C1 ≈ 16 (stickiest), C2/C3/C4 ≈ 13-14, short-Inactive ≈ 9.

## Verdict
**Conditionally suitable for first-order Markov modelling.** Strong order-1
structure, ergodic with well-defined π, recent marginals already at π. Not
strictly time-homogeneous, but deviations are modest and explained by cohort
maturation. **Recommended: fit on the mature window (2024-09 onward)** where
homogeneity holds best (G2/df 26 → 17), or use a piecewise-by-year chain. Use the
**absorbing-churn variant for retention/survival** questions.
