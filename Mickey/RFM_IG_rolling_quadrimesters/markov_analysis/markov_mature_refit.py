"""
Refit the RECURRENT 5-state chain on the mature/stationary window
(2024-09 .. 2026-03), where the Anderson-Goodman homogeneity statistic is
lowest. Compare against the full-span fit and export everything the notebook
plots.

Outputs (-> outputs/):
  markov_pooled_P_mature.csv            mature-window pooled P
  markov_mature_summary.csv             pi, spectral gap, mixing, homogeneity (full vs mature)
  markov_homogeneity_per_step.csv       per-step G2/df + row TV vs pooled (full span)
  markov_reactivation_hazard.csv        return hazard by inactive-spell month
"""
import os
import numpy as np
import pandas as pd
from markov_common import (load, left_censor, transition_counts_per_step,
                           row_normalize, homogeneity_test, stationary_dist,
                           ACTIVE, OUT)

MATURE_START = "2024-09"


def spectral(P):
    w = np.linalg.eigvals(P)
    eig = np.sort(np.abs(w))[::-1]
    slem = eig[1] if len(eig) > 1 else 0.0
    gap = 1 - slem
    mixing = np.log(0.01) / np.log(slem) if 0 < slem < 1 else np.inf
    return eig, slem, gap, mixing


def fit_block(mats):
    pooled = np.sum(mats, axis=0)
    P = row_normalize(pooled)
    pi = stationary_dist(P)
    eig, slem, gap, mixing = spectral(P)
    H = homogeneity_test(mats)
    return P, pi, eig, slem, gap, mixing, H


def per_step_homogeneity(mats):
    """G2/df treating each step vs pooled is not standard; instead report each
    step's mass-weighted row TV from pooled P + a 1-step-vs-pooled G2/df."""
    pooled = np.sum(mats, axis=0)
    Pp = row_normalize(pooled)
    rows = []
    for N in mats:
        Pt = row_normalize(N)
        rmass = N.sum(1)
        tv = 0.5 * np.abs(Pt - Pp).sum(1)
        w = rmass / rmass.sum() if rmass.sum() > 0 else rmass
        rows.append((tv * w).sum())
    return np.array(rows)


def reactivation_hazard(O):
    T = O.shape[1]
    atrisk = np.zeros(40); ret = np.zeros(40)
    for i in range(O.shape[0]):
        seq = [v for v in O[i] if v is not None]
        t = 0
        while t < len(seq):
            if seq[t] == "Inactive":
                k = 0
                while t + k < len(seq) and seq[t + k] == "Inactive":
                    k += 1
                for j in range(1, k + 1):
                    if t + j < len(seq):
                        atrisk[j] += 1
                        if seq[t + j] != "Inactive":
                            ret[j] += 1
                t += k
            else:
                t += 1
    months = np.arange(1, 25)
    haz = np.array([ret[j] / atrisk[j] if atrisk[j] > 0 else np.nan for j in months])
    return pd.DataFrame({"inactive_month": months, "return_hazard": haz,
                         "at_risk": atrisk[months].astype(int)})


def main():
    S, win = load()
    O = left_censor(S)
    T = len(win)
    t0 = win.index(MATURE_START)

    mats_full, _ = transition_counts_per_step(O, ACTIVE, 0, T - 1)
    mats_mat, _ = transition_counts_per_step(O, ACTIVE, t0, T - 1)

    Pf, pif, eigf, slemf, gapf, mixf, Hf = fit_block(mats_full)
    Pm, pim, eigm, slemm, gapm, mixm, Hm = fit_block(mats_mat)

    print(f"{'metric':28s} {'FULL (2023-01..)':>20s} {'MATURE (2024-09..)':>20s}")
    print("-" * 70)
    print(f"{'homogeneity G2/df':28s} {Hf['G2_per_df']:20.2f} {Hm['G2_per_df']:20.2f}")
    print(f"{'SLEM':28s} {slemf:20.4f} {slemm:20.4f}")
    print(f"{'spectral gap':28s} {gapf:20.4f} {gapm:20.4f}")
    print(f"{'mixing (mo to <1% TV)':28s} {mixf:20.1f} {mixm:20.1f}")
    for k, s in enumerate(ACTIVE):
        print(f"{'pi['+s+']':28s} {pif[k]:20.4f} {pim[k]:20.4f}")

    print("\nMature-window pooled P (rows=from):")
    print("        " + "  ".join(f"{s:>9s}" for s in ACTIVE))
    for s in range(len(ACTIVE)):
        print(f"  {ACTIVE[s]:>8s}  " + "  ".join(f"{Pm[s,c]:9.4f}" for c in range(len(ACTIVE))))

    # exports
    pd.DataFrame(Pm, index=ACTIVE, columns=ACTIVE).to_csv(
        os.path.join(OUT, "markov_pooled_P_mature.csv"))

    summ = pd.DataFrame({
        "metric": ["homogeneity_G2_per_df", "SLEM", "spectral_gap", "mixing_months"]
                  + [f"pi_{s}" for s in ACTIVE],
        "full": [Hf["G2_per_df"], slemf, gapf, mixf] + list(pif),
        "mature": [Hm["G2_per_df"], slemm, gapm, mixm] + list(pim),
    })
    summ.to_csv(os.path.join(OUT, "markov_mature_summary.csv"), index=False)

    tv = per_step_homogeneity(mats_full)
    pd.DataFrame({"step_from": win[:T - 1], "step_to": win[1:T],
                  "rowTV_vs_pooled": tv}).to_csv(
        os.path.join(OUT, "markov_homogeneity_per_step.csv"), index=False)

    reactivation_hazard(O).to_csv(
        os.path.join(OUT, "markov_reactivation_hazard.csv"), index=False)

    print("\nSaved: markov_pooled_P_mature.csv, markov_mature_summary.csv,")
    print("       markov_homogeneity_per_step.csv, markov_reactivation_hazard.csv")


if __name__ == "__main__":
    main()
