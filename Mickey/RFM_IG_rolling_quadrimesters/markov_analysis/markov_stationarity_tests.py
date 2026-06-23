"""
Markov-chain suitability & stationarity diagnostics for the IG window-level
RCEDTG cluster panel.

Data: Ig_RCEDTG_windowlevel_kmeans_k4_monthly_cluster_matrix.csv
  one row per user, one column per monthly window (2023-01 .. 2026-03),
  cell = state in {C1,C2,C3,C4,Inactive,Not yet active}.

We treat the panel as N realizations of a discrete-time stochastic process on a
common time axis and ask, rigorously:

  (A) MARKOV PROPERTY / ORDER
      order-0 (iid) vs order-1 vs order-2 via LR + AIC/BIC.
  (B) TIME-HOMOGENEITY (the core "stationary transition mechanism" test)
      Anderson-Goodman LR test: are the per-step transition matrices P_t equal?
  (C) MARGINAL / DISTRIBUTIONAL STATIONARITY
      is the cross-sectional state distribution stable over time (trend tests)?
  (D) ASYMPTOTICS
      irreducibility, aperiodicity, unique stationary dist pi, spectral gap,
      mixing time, and whether empirical marginals approach pi.
  (E) PARTIAL STATIONARITY
      sliding-window homogeneity scan + regime (yearly) sub-period homogeneity.

Conventions
-----------
"Not yet active" = left-censoring (user not yet in panel). We start each user's
chain at first non-"Not yet active" window. Transitions FROM "Not yet active"
are dropped. "Inactive" IS a behavioral state (churned / dormant) and is kept;
re-activation (Inactive -> Cx) is observed in the data so the chain is not
absorbing in practice -- we verify this.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)
CSV = os.path.join(DATA_DIR, "Ig_RCEDTG_windowlevel_kmeans_k4_monthly_cluster_matrix.csv")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ACTIVE = ["C1", "C2", "C3", "C4", "Inactive"]   # states in the chain
PRE = "Not yet active"
np.set_printoptions(precision=4, suppress=True, linewidth=140)


def load():
    df = pd.read_csv(CSV)
    meta = ["user_key", "user_id", "username"]
    win_cols = [c for c in df.columns if c not in meta]
    win_cols = sorted(win_cols)  # YYYY-MM sorts chronologically
    S = df[win_cols].astype(str).values  # (n_users, T)
    return S, win_cols


def left_censor(S):
    """Replace leading 'Not yet active' with NaN-marker; keep the rest as-is.
    Also any 'Not yet active' that appears AFTER entry (shouldn't happen) -> drop
    that cell as missing. Returns object array with None for 'not in panel'."""
    n, T = S.shape
    out = np.empty((n, T), dtype=object)
    for i in range(n):
        entered = False
        for t in range(T):
            v = S[i, t]
            if v == PRE:
                out[i, t] = None  # not in panel yet (or anomalous)
            else:
                entered = True
                out[i, t] = v
        # nothing special needed; None marks censored cells
    return out


def transition_counts_per_step(O):
    """List of (T-1) count matrices N_t[s,s'] counting i:O[i,t]=s, O[i,t+1]=s'.
    Skips pairs where either side is None (censored)."""
    n, T = O.shape
    idx = {s: k for k, s in enumerate(ACTIVE)}
    K = len(ACTIVE)
    mats = []
    for t in range(T - 1):
        N = np.zeros((K, K))
        col0, col1 = O[:, t], O[:, t + 1]
        for a, b in zip(col0, col1):
            if a is None or b is None:
                continue
            N[idx[a], idx[b]] += 1
        mats.append(N)
    return mats, idx


def row_normalize(N):
    r = N.sum(1, keepdims=True)
    P = np.divide(N, r, out=np.zeros_like(N), where=r > 0)
    return P


# ---------- (A) Markov order ---------------------------------------------------
def markov_order_test(O):
    """LR tests order0<order1<order2. Uses pooled (time-homogeneous) counts.
    For order m we build counts over (m+1)-grams."""
    K = len(ACTIVE)
    idx = {s: k for k, s in enumerate(ACTIVE)}

    def grams(m):
        # count m-history -> next, pooled over all t and users
        c = {}
        n, T = O.shape
        for i in range(n):
            seq = [idx[x] for x in O[i] if x is not None]
            for t in range(len(seq) - m):
                hist = tuple(seq[t:t + m])
                nxt = seq[t + m]
                c.setdefault(hist, np.zeros(K))[nxt] += 1
        return c

    def loglik(c):
        ll = 0.0
        for hist, row in c.items():
            tot = row.sum()
            if tot == 0:
                continue
            p = row / tot
            nz = row > 0
            ll += (row[nz] * np.log(p[nz])).sum()
        return ll

    c0 = grams(0)   # order-0: empty history
    c1 = grams(1)   # order-1
    c2 = grams(2)   # order-2

    ll0, ll1, ll2 = loglik(c0), loglik(c1), loglik(c2)

    # free params: order-m has |states|^m * (K-1)
    def dof(observed_histories):
        return observed_histories * (K - 1)

    # use observed (non-empty) histories to avoid inflating df with impossible ones
    p0 = dof(len(c0))
    p1 = dof(len(c1))
    p2 = dof(len(c2))

    def lr(llr, llf, df):
        stat = 2 * (llf - llr)
        df = max(df, 1)
        pval = stats.chi2.sf(stat, df)
        return stat, df, pval

    res = {}
    res["LR order0->1"] = lr(ll0, ll1, p1 - p0)
    res["LR order1->2"] = lr(ll1, ll2, p2 - p1)

    def aic_bic(ll, k, ntr):
        aic = 2 * k - 2 * ll
        bic = k * np.log(ntr) - 2 * ll
        return aic, bic

    ntr = sum(int(r.sum()) for r in c1.values())  # total transitions
    res["AIC/BIC order0"] = aic_bic(ll0, p0, ntr)
    res["AIC/BIC order1"] = aic_bic(ll1, p1, ntr)
    res["AIC/BIC order2"] = aic_bic(ll2, p2, ntr)
    return res


# ---------- (B) Anderson-Goodman time-homogeneity -----------------------------
def homogeneity_test(mats, label=""):
    """LR test H0: P_t = P (pooled) for all t.
    G2 = 2 * sum_t sum_{s,s'} n_t(s,s') log( p_t(s,s') / p_pooled(s,s') ).
    df = (T-1 effective)*K*(K-1) using only cells with positive pooled mass and
    rows that are observed in that step."""
    K = mats[0].shape[0]
    pooled = np.sum(mats, axis=0)
    Pp = row_normalize(pooled)
    G2 = 0.0
    df = 0
    for N in mats:
        Pt = row_normalize(N)
        for s in range(K):
            rt = N[s].sum()
            if rt == 0:
                continue
            for sp in range(K):
                nts = N[s, sp]
                if nts > 0 and Pp[s, sp] > 0:
                    G2 += 2 * nts * np.log(Pt[s, sp] / Pp[s, sp])
            # df contribution: this (t,s) row contributes up to (#cols with pooled>0 -1)
            cols = int((Pp[s] > 0).sum())
            df += max(cols - 1, 0)
    # subtract pooled params: K rows * (cols-1)
    pooled_df = sum(max(int((Pp[s] > 0).sum()) - 1, 0) for s in range(K))
    df = df - pooled_df
    df = max(df, 1)
    pval = stats.chi2.sf(G2, df)
    return {"label": label, "G2": G2, "df": df, "pval": pval,
            "G2_per_df": G2 / df}


# ---------- (D) Asymptotics on pooled P ---------------------------------------
def asymptotics(mats):
    pooled = np.sum(mats, axis=0)
    P = row_normalize(pooled)
    K = P.shape[0]

    # stationary distribution via left eigenvector for eigenvalue 1
    w, V = np.linalg.eig(P.T)
    i1 = np.argmin(np.abs(w - 1))
    pi = np.real(V[:, i1])
    pi = pi / pi.sum()

    eig = np.sort(np.abs(w))[::-1]
    slem = eig[1] if len(eig) > 1 else 0.0   # second largest eigenvalue modulus
    spectral_gap = 1 - slem
    mixing = np.log(0.01) / np.log(slem) if 0 < slem < 1 else np.inf

    # irreducibility & aperiodicity (on states reachable / communicating)
    reach = (np.linalg.matrix_power((P > 0).astype(int) + np.eye(K, dtype=int), K) > 0)
    irreducible = bool(reach.all())
    # aperiodic if any diagonal of P>0 (self-loop) -> period 1 for that class
    aperiodic = bool((np.diag(P) > 0).any())

    return {"P": P, "pi": pi, "eig": eig, "slem": slem,
            "spectral_gap": spectral_gap, "mixing_steps_to_1pct": mixing,
            "irreducible": irreducible, "aperiodic": aperiodic}


# ---------- (C) Marginal stationarity / drift ---------------------------------
def marginal_drift(O):
    """Cross-sectional state shares per window (among in-panel users) + trend
    test (Spearman of share vs time) per state."""
    K = len(ACTIVE)
    idx = {s: k for k, s in enumerate(ACTIVE)}
    n, T = O.shape
    shares = np.zeros((T, K))
    npanel = np.zeros(T)
    for t in range(T):
        col = O[:, t]
        cnt = np.zeros(K)
        for v in col:
            if v is not None:
                cnt[idx[v]] += 1
        tot = cnt.sum()
        npanel[t] = tot
        if tot > 0:
            shares[t] = cnt / tot
    # trend test per state on the steady-panel region (use all t with panel>0)
    valid = npanel > 0
    tt = np.arange(T)[valid]
    out = {}
    for k, s in enumerate(ACTIVE):
        rho, p = stats.spearmanr(tt, shares[valid, k])
        out[s] = (rho, p)
    return shares, npanel, out


# ---------- (E) partial stationarity ------------------------------------------
def sliding_homogeneity(mats, win_cols, w=6):
    """Homogeneity G2/df within each sliding window of w consecutive steps."""
    res = []
    for start in range(0, len(mats) - w + 1):
        block = mats[start:start + w]
        r = homogeneity_test(block, label=f"{win_cols[start]}..{win_cols[start+w]}")
        res.append(r)
    return res


def regime_homogeneity(mats, win_cols):
    """Group steps by calendar year of the source window; test homogeneity
    within each year, and equality of the per-year pooled matrices."""
    years = {}
    for t in range(len(mats)):
        y = win_cols[t][:4]
        years.setdefault(y, []).append(mats[t])
    within = {y: homogeneity_test(blk, label=f"within-{y}") for y, blk in years.items()}
    # between-year: pooled-per-year as the units
    year_pooled = [np.sum(blk, axis=0) for blk in years.values()]
    between = homogeneity_test(year_pooled, label="between-years")
    return within, between, list(years.keys())


def effect_sizes(mats):
    """Beyond p-values (which are ~0 at this N): how *large* is the per-step
    deviation from the pooled P? Report mean/max row-wise TV distance of each
    P_t from pooled P, and a Monte-Carlo calibration of G2/df expected under
    TRUE homogeneity at the observed row totals (so we can judge whether the
    rejection reflects real non-stationarity or just huge-N over-power)."""
    K = mats[0].shape[0]
    pooled = np.sum(mats, axis=0)
    Pp = row_normalize(pooled)

    # observed per-step row TV vs pooled, weighted by row mass
    tv_steps = []
    for N in mats:
        Pt = row_normalize(N)
        rmass = N.sum(1)
        tv_rows = 0.5 * np.abs(Pt - Pp).sum(1)
        w = rmass / rmass.sum() if rmass.sum() > 0 else rmass
        tv_steps.append((tv_rows * w).sum())
    tv_steps = np.array(tv_steps)

    # Monte-Carlo null: simulate each step's rows ~ Multinomial(row_total, Pp)
    rng = np.random.default_rng(0)
    g2df_null = []
    for _ in range(200):
        sim = []
        for N in mats:
            Ns = np.zeros((K, K))
            for s in range(K):
                rt = int(N[s].sum())
                if rt > 0:
                    Ns[s] = rng.multinomial(rt, Pp[s])
            sim.append(Ns)
        g2df_null.append(homogeneity_test(sim)["G2_per_df"])
    g2df_null = np.array(g2df_null)
    return {"tv_mean": tv_steps.mean(), "tv_max": tv_steps.max(),
            "tv_steps": tv_steps,
            "g2df_null_mean": g2df_null.mean(), "g2df_null_p95": np.percentile(g2df_null, 95)}


def main():
    S, win_cols = load()
    print(f"Panel: {S.shape[0]} users x {len(win_cols)} windows "
          f"({win_cols[0]} .. {win_cols[-1]})")
    O = left_censor(S)
    mats, idx = transition_counts_per_step(O)
    pooled = np.sum(mats, axis=0)
    print(f"States (chain): {ACTIVE}")
    print(f"Total observed transitions: {int(pooled.sum()):,} over {len(mats)} steps")

    print("\n" + "=" * 70)
    print("(A) MARKOV ORDER")
    print("=" * 70)
    A = markov_order_test(O)
    for k in ["LR order0->1", "LR order1->2"]:
        st, df, p = A[k]
        print(f"  {k:14s}: G2={st:12.1f}  df={df:5d}  p={p:.3g}")
    for k in ["AIC/BIC order0", "AIC/BIC order1", "AIC/BIC order2"]:
        aic, bic = A[k]
        print(f"  {k:16s}: AIC={aic:14.1f}  BIC={bic:14.1f}")

    print("\n" + "=" * 70)
    print("(B) TIME-HOMOGENEITY (Anderson-Goodman): H0 all P_t equal pooled P")
    print("=" * 70)
    H = homogeneity_test(mats, "global")
    print(f"  G2={H['G2']:.1f}  df={H['df']}  p={H['pval']:.3g}  "
          f"G2/df={H['G2_per_df']:.2f}")
    print("  [effect size: p~0 is expected at N=595k; judge by magnitude]")
    E = effect_sizes(mats)
    print(f"  mean per-step TV(P_t, P_pooled) = {E['tv_mean']:.4f}  "
          f"(max {E['tv_max']:.4f})  -- 0=identical, 1=disjoint")
    print(f"  Monte-Carlo G2/df under TRUE homogeneity: mean={E['g2df_null_mean']:.2f} "
          f"p95={E['g2df_null_p95']:.2f}  vs observed {H['G2_per_df']:.2f}")
    print(f"  -> observed G2/df is {H['G2_per_df']/E['g2df_null_mean']:.1f}x the "
          f"homogeneous-null level")

    print("\n" + "=" * 70)
    print("(D) ASYMPTOTICS on pooled (time-homogeneous) P")
    print("=" * 70)
    Z = asymptotics(mats)
    print("  Pooled transition matrix P (rows=from, cols=to):")
    hdr = "        " + "  ".join(f"{s:>9s}" for s in ACTIVE)
    print(hdr)
    for s in range(len(ACTIVE)):
        print(f"  {ACTIVE[s]:>6s}  " + "  ".join(f"{Z['P'][s,c]:9.4f}" for c in range(len(ACTIVE))))
    print(f"\n  Stationary dist pi: " +
          "  ".join(f"{ACTIVE[k]}={Z['pi'][k]:.4f}" for k in range(len(ACTIVE))))
    print(f"  |eigenvalues|: {Z['eig']}")
    print(f"  SLEM (2nd largest |eig|): {Z['slem']:.4f}")
    print(f"  spectral gap: {Z['spectral_gap']:.4f}")
    print(f"  mixing steps to <1% TV (approx): {Z['mixing_steps_to_1pct']:.1f}")
    print(f"  irreducible: {Z['irreducible']}   aperiodic: {Z['aperiodic']}")

    print("\n" + "=" * 70)
    print("(C) MARGINAL DISTRIBUTION DRIFT (Spearman trend of share vs time)")
    print("=" * 70)
    shares, npanel, trend = marginal_drift(O)
    for s in ACTIVE:
        rho, p = trend[s]
        flag = "  <-- significant drift" if p < 0.05 else ""
        print(f"  {s:>9s}: rho={rho:+.3f}  p={p:.3g}{flag}")
    # compare last-year empirical marginal vs pi
    last_share = shares[-1]
    print("\n  Last-window empirical share vs stationary pi:")
    for k, s in enumerate(ACTIVE):
        print(f"    {s:>9s}: emp={last_share[k]:.4f}  pi={Z['pi'][k]:.4f}  "
              f"diff={last_share[k]-Z['pi'][k]:+.4f}")
    tv_last = 0.5 * np.abs(last_share - Z["pi"]).sum()
    print(f"  TV distance (last window vs pi): {tv_last:.4f}")
    # disentangle panel-maturation from regime change: re-test drift only over
    # windows where the in-panel size is within 10% of its max (mature region)
    mx = npanel.max()
    mature = npanel >= 0.90 * mx
    if mature.sum() >= 4:
        tt = np.arange(len(npanel))[mature]
        print(f"\n  Mature-panel drift (windows with panel>=90% of max, "
              f"n={mature.sum()} windows, {win_cols[tt[0]]}..{win_cols[tt[-1]]}):")
        for k, s in enumerate(ACTIVE):
            rho, p = stats.spearmanr(tt, shares[mature, k])
            flag = "  <-- drift" if p < 0.05 else "  (stable)"
            print(f"    {s:>9s}: rho={rho:+.3f}  p={p:.3g}{flag}")

    print("\n" + "=" * 70)
    print("(E) PARTIAL STATIONARITY")
    print("=" * 70)
    print("  Sliding-window (w=6 steps) homogeneity G2/df (lower=more stationary):")
    sw = sliding_homogeneity(mats, win_cols, w=6)
    for r in sw:
        flag = "" if r["pval"] > 0.05 else "  (reject homog.)"
        print(f"    {r['label']:>20s}: G2/df={r['G2_per_df']:6.2f}  p={r['pval']:.2g}{flag}")
    print("\n  Within-year homogeneity:")
    within, between, yrs = regime_homogeneity(mats, win_cols)
    for y in yrs:
        r = within[y]
        flag = "" if r["pval"] > 0.05 else "  (reject)"
        print(f"    {y}: G2={r['G2']:9.1f}  df={r['df']:4d}  p={r['pval']:.2g}  "
              f"G2/df={r['G2_per_df']:.2f}{flag}")
    print(f"  Between-year (yearly-pooled P equal?): G2={between['G2']:.1f} "
          f"df={between['df']} p={between['pval']:.2g} G2/df={between['G2_per_df']:.2f}")

    # save the period transition matrices and pooled P for downstream use
    np.save(os.path.join(OUT, "markov_pooled_P.npy"), Z["P"])
    pd.DataFrame(Z["P"], index=ACTIVE, columns=ACTIVE).to_csv(os.path.join(OUT, "markov_pooled_P.csv"))
    pd.DataFrame(shares, index=win_cols, columns=ACTIVE).to_csv(os.path.join(OUT, "markov_marginal_shares_by_window.csv"))
    print("\nSaved: markov_pooled_P.csv, markov_marginal_shares_by_window.csv, markov_pooled_P.npy")


if __name__ == "__main__":
    main()
