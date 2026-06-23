"""
Shared loaders / estimators for the IG window-level RCEDTG Markov analysis.

All scripts in this folder import from here so the data path, state coding,
left-censoring, and transition-counting logic live in one place.

Run scripts from inside markov_analysis/ (paths are resolved relative to this
file's parent: the RFM_IG_rolling_quadrimesters folder).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)  # RFM_IG_rolling_quadrimesters
CSV = os.path.join(DATA_DIR, "Ig_RCEDTG_windowlevel_kmeans_k4_monthly_cluster_matrix.csv")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

ACTIVE = ["C1", "C2", "C3", "C4", "Inactive"]   # recurrent-model states
PRE = "Not yet active"
META = ["user_key", "user_id", "username"]


def load():
    """Return (S, win_cols): S is (n_users, T) object array of raw state labels,
    win_cols the chronological YYYY-MM window names."""
    df = pd.read_csv(CSV)
    win_cols = sorted([c for c in df.columns if c not in META])
    return df[win_cols].astype(str).values, win_cols


def left_censor(S):
    """Replace 'Not yet active' cells with None (user not in panel)."""
    n, T = S.shape
    O = np.empty((n, T), dtype=object)
    for i in range(n):
        for t in range(T):
            O[i, t] = None if S[i, t] == PRE else S[i, t]
    return O


def transition_counts_per_step(O, states=ACTIVE, t0=0, t1=None):
    """List of count matrices N_t for steps t in [t0, t1). Skips censored cells."""
    n, T = O.shape
    t1 = T - 1 if t1 is None else t1
    idx = {s: k for k, s in enumerate(states)}
    K = len(states)
    mats = []
    for t in range(t0, t1):
        N = np.zeros((K, K))
        for a, b in zip(O[:, t], O[:, t + 1]):
            if a in idx and b in idx:
                N[idx[a], idx[b]] += 1
        mats.append(N)
    return mats, idx


def row_normalize(N):
    r = N.sum(1, keepdims=True)
    return np.divide(N, r, out=np.zeros_like(N), where=r > 0)


def homogeneity_test(mats, label=""):
    """Anderson-Goodman LR test H0: all P_t equal pooled P. Returns dict."""
    K = mats[0].shape[0]
    pooled = np.sum(mats, axis=0)
    Pp = row_normalize(pooled)
    G2 = 0.0
    df = 0
    for N in mats:
        Pt = row_normalize(N)
        for s in range(K):
            if N[s].sum() == 0:
                continue
            for sp in range(K):
                if N[s, sp] > 0 and Pp[s, sp] > 0:
                    G2 += 2 * N[s, sp] * np.log(Pt[s, sp] / Pp[s, sp])
            df += max(int((Pp[s] > 0).sum()) - 1, 0)
    df -= sum(max(int((Pp[s] > 0).sum()) - 1, 0) for s in range(K))
    df = max(df, 1)
    return {"label": label, "G2": G2, "df": df, "pval": stats.chi2.sf(G2, df),
            "G2_per_df": G2 / df}


def stationary_dist(P):
    w, V = np.linalg.eig(P.T)
    i1 = np.argmin(np.abs(w - 1))
    pi = np.real(V[:, i1])
    return pi / pi.sum()


def window_index(win_cols, name):
    return win_cols.index(name)
