"""
Absorbing-churn Markov variant + survival framing.

Empirically 'Inactive' is NOT absorbing: 1-step reactivation = 4.5%, and the
monthly return hazard stays ~5-7% for the first ~6 months before decaying to
~3%. So we model churn as a DURATION-BASED absorbing state: once a user has
been Inactive for >= CHURN_K consecutive months they enter an absorbing
'Churned' state. C1-C4 and (short) Inactive remain transient.

Outputs (-> outputs/):
  markov_absorbing_P[_mature].csv        transition matrix
  markov_absorbing_fundamental[_mature].csv   expected visits / time-to-churn
  markov_absorbing_survival[_mature].csv  P(not churned) by horizon & start state

Run from inside markov_analysis/.
"""
import os
import numpy as np
import pandas as pd
from markov_common import load, OUT

PRE = "Not yet active"
CHURN_K = 6  # consecutive inactive months -> absorbed
STATES = ["C1", "C2", "C3", "C4", "Inactive", "Churned"]
TRANS = ["C1", "C2", "C3", "C4", "Inactive"]
HORIZON = 36  # months for survival curve


def recode(row):
    """Leading PRE -> None; once Inactive run reaches CHURN_K, absorb to Churned."""
    out, run, churned = [], 0, False
    for v in row:
        if v == PRE:
            out.append(None); continue
        if churned:
            out.append("Churned"); continue
        if v == "Inactive":
            run += 1
            out.append("Churned" if run >= CHURN_K else "Inactive")
            if run >= CHURN_K:
                churned = True
        else:
            run = 0
            out.append(v)
    return out


def fit(O, t0, t1, idx, K):
    N = np.zeros((K, K))
    for i in range(O.shape[0]):
        for t in range(t0, t1):
            a, b = O[i, t], O[i, t + 1]
            if a in idx and b in idx:
                N[idx[a], idx[b]] += 1
    P = np.divide(N, N.sum(1, keepdims=True), out=np.zeros_like(N),
                  where=N.sum(1, keepdims=True) > 0)
    P[idx["Churned"]] = 0.0
    P[idx["Churned"], idx["Churned"]] = 1.0
    return N, P


def survival(P, idx):
    """P(not yet churned) at horizon h for each transient start state."""
    ti = [idx[s] for s in TRANS]
    surv = np.zeros((HORIZON + 1, len(TRANS)))
    for j, s in enumerate(TRANS):
        v = np.zeros(len(STATES)); v[idx[s]] = 1.0
        for h in range(HORIZON + 1):
            surv[h, j] = 1.0 - v[idx["Churned"]]
            v = v @ P
    return surv


def run(O, t0, t1, tag, win):
    idx = {s: k for k, s in enumerate(STATES)}
    K = len(STATES)
    N, P = fit(O, t0, t1, idx, K)
    ti = [idx[s] for s in TRANS]
    Q = P[np.ix_(ti, ti)]
    R = P[np.ix_(ti, [idx["Churned"]])]
    Fund = np.linalg.inv(np.eye(len(TRANS)) - Q)
    t_abs = Fund.sum(1)

    suffix = "" if tag == "full" else f"_{tag}"
    print(f"\n=== Absorbing-churn chain [{tag}: {win[t0]}..{win[t1]}] "
          f"(churn = >={CHURN_K} consecutive Inactive mo) ===")
    print(f"Transitions used: {int(N.sum()):,}")
    print("Transition matrix P (rows=from):")
    print("        " + "  ".join(f"{s:>9s}" for s in STATES))
    for s in range(K):
        print(f"  {STATES[s]:>8s}  " + "  ".join(f"{P[s,c]:9.4f}" for c in range(K)))
    print("Expected months until churn, by current state:")
    for s, e in zip(TRANS, t_abs):
        print(f"  {s:>9s}: {e:6.1f} months")

    pd.DataFrame(P, index=STATES, columns=STATES).to_csv(
        os.path.join(OUT, f"markov_absorbing_P{suffix}.csv"))
    fdf = pd.DataFrame(Fund, index=TRANS, columns=TRANS)
    fdf["expected_months_to_churn"] = t_abs
    fdf.to_csv(os.path.join(OUT, f"markov_absorbing_fundamental{suffix}.csv"))
    surv = survival(P, idx)
    pd.DataFrame(surv, columns=TRANS,
                 index=pd.Index(range(HORIZON + 1), name="horizon_months")).to_csv(
        os.path.join(OUT, f"markov_absorbing_survival{suffix}.csv"))
    return P, t_abs


def main():
    S, win = load()
    O = np.array([recode(S[i]) for i in range(S.shape[0])], dtype=object)
    T = O.shape[1]
    # full span
    run(O, 0, T - 1, "full", win)
    # mature window: 2024-09 onward (where homogeneity holds best)
    t0 = win.index("2024-09")
    run(O, t0, T - 1, "mature", win)
    print("\nSaved absorbing-churn outputs (full + mature) to outputs/")


if __name__ == "__main__":
    main()
