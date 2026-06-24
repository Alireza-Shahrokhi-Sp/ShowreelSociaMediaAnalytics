"""
Causal inference for sentiment dips at 2024-01 and 2025-10.

Methods:
1. ATT via post-removal counterfactual (direct effect of identified posts)
2. DiD — treated = comments on flagged posts; control = other comments same month
3. Permutation (placebo) test — shuffle treatment assignment across months
4. Synthetic Control — donor-weighted pre-period match, in-time placebo gaps
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── data ─────────────────────────────────────────────────────────────────────
SENTIMENT = r"d:\Polythecninco di Milano\AFB_Lab\Ali\batch_results\sentiment\sentiment_instagram.parquet"
POSTS     = r"d:\Polythecninco di Milano\AFB_Lab\Ali\outputs\ig_multimodal_final.parquet"
RFM       = r"d:\Polythecninco di Milano\AFB_Lab\Ali\outputs\pathway_b_assignments_instagram.parquet"
OUT       = r"d:\Polythecninco di Milano\AFB_Lab\Ali\Sentiment_EDA\outputs\CI_dips.png"

sent  = pd.read_parquet(SENTIMENT)
posts = pd.read_parquet(POSTS)
rfm   = pd.read_parquet(RFM)

sent["ts"]    = pd.to_datetime(sent["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
sent["month"] = sent["ts"].dt.to_period("M")
sent = sent.merge(rfm, on="author_id", how="left")
sent["is_neg"] = (sent["sentiment"] == "negative").astype(float)

# restrict to 2022+ for stability
sent = sent[sent["month"] >= pd.Period("2022-01", "M")].copy()

# ── treatment posts (from EDA) ────────────────────────────────────────────────
# 2024-01: "colors-of-months" reel + captionless reel + ticket announcement
TREAT_A = {"18289555210155460", "17967996401546766", "18081875413420073"}
# 2025-10: train-story post + satire reel (both had >50% neg rate)
TREAT_B = {"17868188040465895", "17940696519056981"}

MONTH_A = pd.Period("2024-01", "M")
MONTH_B = pd.Period("2025-10", "M")

# monthly neg-rate series (all comments)
monthly = (sent.groupby("month")["is_neg"]
               .agg(neg_rate="mean", n="count")
               .reset_index()
               .sort_values("month"))
obs_series = monthly.set_index("month")["neg_rate"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. ATT via post-removal counterfactual
# ─────────────────────────────────────────────────────────────────────────────
def att_removal(df, treat_ids, month):
    in_month = df[df["month"] == month]
    obs = in_month["is_neg"].mean()
    cf  = in_month[~in_month["media_id"].isin(treat_ids)]["is_neg"].mean()
    n_treat   = in_month["media_id"].isin(treat_ids).sum()
    n_control = (~in_month["media_id"].isin(treat_ids)).sum()
    return obs, cf, obs - cf, n_treat, n_control

obs_A, cf_A, att_A, nT_A, nC_A = att_removal(sent, TREAT_A, MONTH_A)
obs_B, cf_B, att_B, nT_B, nC_B = att_removal(sent, TREAT_B, MONTH_B)

print("=== 1. ATT (post-removal counterfactual) ===")
print(f"  2024-01  obs={obs_A:.4f}  cf={cf_A:.4f}  ATT=+{att_A:.4f} ({att_A*100:.1f} pp)  n_treat={nT_A}  n_ctrl={nC_A}")
print(f"  2025-10  obs={obs_B:.4f}  cf={cf_B:.4f}  ATT=+{att_B:.4f} ({att_B*100:.1f} pp)  n_treat={nT_B}  n_ctrl={nC_B}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DiD — OLS with HC0 robust SE
#    Y = neg_rate; treated = comment on a flagged post; post = dip month
#    DiD coeff = interaction treat × post
# ─────────────────────────────────────────────────────────────────────────────
def did_estimate(df, treat_ids, treat_month, pre_months_str):
    pre_months = [pd.Period(m, "M") for m in pre_months_str]
    d = df[df["month"].isin(pre_months + [treat_month])].copy()
    d["treated"] = d["media_id"].isin(treat_ids).astype(float)
    d["post"]    = (d["month"] == treat_month).astype(float)
    d["did"]     = d["treated"] * d["post"]
    X = d[["post","treated","did"]].assign(const=1)[["const","post","treated","did"]].values
    y = d["is_neg"].values
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    e    = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    meat = X.T @ np.diag(e**2) @ X
    V    = XtXi @ meat @ XtXi
    se   = np.sqrt(np.diag(V))
    t    = beta[3] / se[3]
    p    = 2 * stats.t.sf(abs(t), df=len(y) - 4)
    ci95 = (beta[3] - 1.96*se[3], beta[3] + 1.96*se[3])
    return beta[3], se[3], t, p, ci95

did_A_coef, did_A_se, did_A_t, did_A_p, did_A_ci = did_estimate(
    sent, TREAT_A, MONTH_A, ["2023-10","2023-11","2023-12"])
did_B_coef, did_B_se, did_B_t, did_B_p, did_B_ci = did_estimate(
    sent, TREAT_B, MONTH_B, ["2025-07","2025-08","2025-09"])

print("\n=== 2. DiD (HC0 robust SE) ===")
print(f"  2024-01  DiD=+{did_A_coef:.4f} ({did_A_coef*100:.1f} pp)  SE={did_A_se:.4f}  t={did_A_t:.2f}  p={did_A_p:.4f}  95%CI=[{did_A_ci[0]*100:.1f},{did_A_ci[1]*100:.1f}]")
print(f"  2025-10  DiD=+{did_B_coef:.4f} ({did_B_coef*100:.1f} pp)  SE={did_B_se:.4f}  t={did_B_t:.2f}  p={did_B_p:.4f}  95%CI=[{did_B_ci[0]*100:.1f},{did_B_ci[1]*100:.1f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Permutation test
#    For each donor month, compute the same ATT statistic (remove the highest
#    neg-rate posts in that month, same count as the treated set).
#    The treated months' ATTs should sit in the tail of this null distribution.
# ─────────────────────────────────────────────────────────────────────────────
def att_top_k(df, month, k):
    """Remove the k posts with highest neg rate (neg/total) in 'month'."""
    in_m = df[df["month"] == month]
    top_k = (in_m.groupby("media_id")["is_neg"]
               .mean()
               .nlargest(k)
               .index.tolist())
    obs = in_m["is_neg"].mean()
    cf  = in_m[~in_m["media_id"].isin(top_k)]["is_neg"].mean()
    return obs - cf

all_months = sorted(sent["month"].unique())
null_effects = []
for m in all_months:
    if m in (MONTH_A, MONTH_B):
        continue
    # use same k as treated set
    e_a = att_top_k(sent, m, len(TREAT_A))
    null_effects.append(e_a)

null_arr = np.array(null_effects)

# actual: use true identified posts, not top-k proxy
pval_A_perm = (np.abs(null_arr) >= np.abs(att_A)).mean()
pval_B_perm = (np.abs(null_arr) >= np.abs(att_B)).mean()

z_A = (att_A - null_arr.mean()) / null_arr.std()
z_B = (att_B - null_arr.mean()) / null_arr.std()

print(f"\n=== 3. Permutation test (n_placebo={len(null_arr)}) ===")
print(f"  Null dist: mean={null_arr.mean():.4f}  std={null_arr.std():.4f}")
print(f"  2024-01  ATT={att_A:.4f}  z={z_A:.2f}  p={pval_A_perm:.3f}")
print(f"  2025-10  ATT={att_B:.4f}  z={z_B:.2f}  p={pval_B_perm:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Synthetic Control
#    Match pre-treatment neg-rate trajectory; compute treated-month gap.
#    In-time placebo: run SC on every donor month; treated gaps should be outliers.
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_control_nnls(series, treat_month, pre_window=12):
    """
    series: pd.Series indexed by Period
    Returns (obs, sc_val, gap).
    """
    from scipy.optimize import nnls
    idx_all = series.index.tolist()
    t_pos   = idx_all.index(treat_month)

    pre_start = max(0, t_pos - pre_window)
    pre_pos   = list(range(pre_start, t_pos))
    if len(pre_pos) < 3:
        return np.nan, np.nan, np.nan

    # donor pool: exclude ±1 month around treat_month
    exclude = {treat_month + o for o in [-1, 0, 1]}
    donor_pos = [i for i, m in enumerate(idx_all) if m not in exclude]

    # build matrices: rows = pre-period months, cols = donors
    # Each donor column = donor's value at those same pre-period positions
    donor_pre = np.column_stack([
        [series.iloc[p] if 0 <= p < len(series) else np.nan
         for p in [dp - (t_pos - pp) for pp in pre_pos]]
        for dp in donor_pos
    ])
    y_pre = series.iloc[pre_pos].values

    # drop donors with NaNs
    valid = ~np.isnan(donor_pre).any(axis=0)
    donor_pre = donor_pre[:, valid]
    donor_vals_at_treat = series.iloc[[dp for dp, v in zip(donor_pos, valid) if v]].values

    if donor_pre.shape[1] == 0:
        return np.nan, np.nan, np.nan

    w, _ = nnls(donor_pre, y_pre)
    if w.sum() == 0:
        return np.nan, np.nan, np.nan
    w = w / w.sum()

    sc_val  = float(w @ donor_vals_at_treat)
    obs_val = series[treat_month]
    return obs_val, sc_val, obs_val - sc_val

obs_A_sc, sc_A, gap_A = synthetic_control_nnls(obs_series, MONTH_A)
obs_B_sc, sc_B, gap_B = synthetic_control_nnls(obs_series, MONTH_B)

# in-time placebo gaps
sc_placebo = {}
for m in all_months:
    if m in (MONTH_A, MONTH_B):
        continue
    o, s, g = synthetic_control_nnls(obs_series, m)
    if not np.isnan(g):
        sc_placebo[m] = g

placebo_gaps = np.array(list(sc_placebo.values()))
sc_pval_A = (np.abs(placebo_gaps) >= np.abs(gap_A)).mean()
sc_pval_B = (np.abs(placebo_gaps) >= np.abs(gap_B)).mean()

print(f"\n=== 4. Synthetic Control ===")
print(f"  2024-01  obs={obs_A_sc:.4f}  SC={sc_A:.4f}  gap=+{gap_A:.4f} ({gap_A*100:.1f} pp)  p={sc_pval_A:.3f}")
print(f"  2025-10  obs={obs_B_sc:.4f}  SC={sc_B:.4f}  gap=+{gap_B:.4f} ({gap_B*100:.1f} pp)  p={sc_pval_B:.3f}")
print(f"  Placebo gaps: mean={placebo_gaps.mean():.4f}  std={placebo_gaps.std():.4f}  n={len(placebo_gaps)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE
# ─────────────────────────────────────────────────────────────────────────────
months_plot = obs_series.index
x = np.arange(len(months_plot))
x_labels = [str(m) for m in months_plot]
tick_step = max(1, len(x) // 14)

def midx(m):
    return list(months_plot).index(m)

fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

# Panel A — observed vs counterfactual (post removal)
ax1 = fig.add_subplot(gs[0, :])
# counterfactual: remove treated posts
def cf_series(exclude_ids):
    s = sent.copy()
    s = s[~s["media_id"].isin(exclude_ids)]
    m2 = s.groupby("month")["is_neg"].mean().reindex(months_plot)
    return m2

cf_A_series = cf_series(TREAT_A)
cf_B_series = cf_series(TREAT_B)
cf_both     = cf_series(TREAT_A | TREAT_B)

ax1.plot(x, obs_series.values * 100, "o-", color="#d62728", lw=2, ms=3, label="Observed neg rate")
ax1.plot(x, cf_both.values * 100,    "s--", color="#1f77b4", lw=1.5, ms=3, alpha=0.8, label="Counterfactual (all treated posts removed)")
ax1.fill_between(x,
                 obs_series.values * 100,
                 cf_both.values * 100,
                 where=(obs_series.values > cf_both.values),
                 alpha=0.15, color="#d62728", label="ATT region")
for m, col, lab in [(MONTH_A, "purple", "2024-01"), (MONTH_B, "darkorange", "2025-10")]:
    xi = midx(m)
    ax1.axvline(xi, color=col, ls=":", lw=1.8)
    ax1.text(xi + 0.4, 19, lab, color=col, fontsize=9, fontweight="bold")
ax1.set_xticks(x[::tick_step])
ax1.set_xticklabels(x_labels[::tick_step], rotation=45, fontsize=8)
ax1.set_ylabel("Negative rate %", fontsize=10)
ax1.set_title("Panel A — Observed vs Counterfactual Negative Rate", fontsize=11, fontweight="bold", y=1.05)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.25)

# Panel B — Permutation null
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(null_arr * 100, bins=20, color="#aec7e8", edgecolor="white", label=f"Null (n={len(null_arr)})")
ax2.axvline(att_A * 100, color="purple",     lw=2.5, label=f"2024-01: +{att_A*100:.1f}pp  z={z_A:.1f}  p={pval_A_perm:.2f}")
ax2.axvline(att_B * 100, color="darkorange", lw=2.5, label=f"2025-10: +{att_B*100:.1f}pp  z={z_B:.1f}  p={pval_B_perm:.2f}")
ax2.set_xlabel("Removal ATT (pp)", fontsize=9)
ax2.set_ylabel("Count", fontsize=9)
ax2.set_title("Panel B — Permutation Test\n(treated posts vs same-size random removal)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)

# Panel C — DiD parallel-trends visualisation
ax3 = fig.add_subplot(gs[1, 1])
def did_plot(df, treat_ids, treat_month, pre_months_str, color, label):
    pre_months = [pd.Period(m, "M") for m in pre_months_str]
    d = df[df["month"].isin(pre_months + [treat_month])].copy()
    d["treated"] = d["media_id"].isin(treat_ids)
    d["period"]  = (d["month"] == treat_month).astype(int)
    means = d.groupby(["treated","period"])["is_neg"].mean() * 100
    for trt, ls, mark in [(False, "--", "s"), (True, "-", "o")]:
        pre_val  = means.get((trt, 0), np.nan)
        post_val = means.get((trt, 1), np.nan)
        lbl = f"{label} {'treated' if trt else 'control'}"
        ax3.plot([0, 1], [pre_val, post_val], ls, color=color, lw=2.5 if trt else 1.2,
                 marker=mark, ms=7, alpha=1 if trt else 0.5, label=lbl)

did_plot(sent, TREAT_A, MONTH_A, ["2023-10","2023-11","2023-12"], "purple",     "2024-01")
did_plot(sent, TREAT_B, MONTH_B, ["2025-07","2025-08","2025-09"], "darkorange", "2025-10")
ax3.set_xticks([0,1]); ax3.set_xticklabels(["Pre-period\n(3m avg)","Dip month"], fontsize=9)
ax3.set_ylabel("Neg rate %", fontsize=9)
ax3.set_title(f"Panel C — DiD Parallel Trends\n2024-01: t={did_A_t:.1f} p={did_A_p:.4f} | 2025-10: t={did_B_t:.1f} p={did_B_p:.4f}", fontsize=10, fontweight="bold")
ax3.legend(fontsize=7.5, ncol=2)
ax3.grid(alpha=0.25)

# Panel D — Synthetic control in-time placebo
ax4 = fig.add_subplot(gs[2, :])
placebo_months = sorted(sc_placebo.keys())
px = [midx(m) for m in placebo_months if m in months_plot]
py = [sc_placebo[m] * 100 for m in placebo_months if m in months_plot]

ax4.bar(px, py, color="#aec7e8", alpha=0.6, width=0.7, label=f"Placebo SC gaps (n={len(px)})")
for m, col, lab, g, p in [
    (MONTH_A, "purple",     "2024-01", gap_A, sc_pval_A),
    (MONTH_B, "darkorange", "2025-10", gap_B, sc_pval_B),
]:
    if m in months_plot:
        ax4.bar(midx(m), g * 100, color=col, width=0.7, alpha=0.9,
                label=f"{lab}: +{g*100:.1f}pp  p={p:.2f}")
ax4.axhline(0, color="black", lw=0.8)
# add RMSPE ratio annotation
rmspe_null = np.sqrt(np.mean(placebo_gaps**2))
ratio_A = abs(gap_A) / rmspe_null
ratio_B = abs(gap_B) / rmspe_null
ax4.text(0.02, 0.95, f"RMSPE ratio — 2024-01: {ratio_A:.1f}×  |  2025-10: {ratio_B:.1f}×  (>2 = significant)",
         transform=ax4.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax4.set_xticks(x[::tick_step])
ax4.set_xticklabels(x_labels[::tick_step], rotation=45, fontsize=8)
ax4.set_ylabel("SC gap (obs − synthetic) pp", fontsize=9)
ax4.set_title("Panel D — Synthetic Control In-Time Placebo\n(treated months should be outlier bars; RMSPE ratio > 2 indicates significance)", fontsize=10, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(alpha=0.25, axis="y")

plt.suptitle("Causal Inference: Sentiment Dips Driven by Specific Posts", fontsize=14, fontweight="bold", y=1.01)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.savefig(OUT.replace(".png", ".svg"), dpi=300, bbox_inches="tight")  # --- IGNORE ---
print(f"\nFigure saved → {OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("CAUSAL EVIDENCE SUMMARY")
print("="*72)
header = f"{'Method':<28}{'2024-01 effect':>16}{'p-value':>10}{'2025-10 effect':>18}{'p-value':>10}"
print(header)
print("-"*82)
rows = [
    ("ATT (post removal)",    f"+{att_A*100:.1f} pp",         "—",                    f"+{att_B*100:.1f} pp",         "—"),
    ("Permutation test",      f"+{att_A*100:.1f} pp (z={z_A:.1f})",  f"{pval_A_perm:.3f}", f"+{att_B*100:.1f} pp (z={z_B:.1f})",  f"{pval_B_perm:.3f}"),
    ("DiD (HC0 SE)",          f"+{did_A_coef*100:.1f} pp (t={did_A_t:.1f})", f"{did_A_p:.4f}", f"+{did_B_coef*100:.1f} pp (t={did_B_t:.1f})", f"{did_B_p:.4f}"),
    ("Synthetic Control",     f"+{gap_A*100:.1f} pp (r={ratio_A:.1f}×)", f"{sc_pval_A:.3f}", f"+{gap_B*100:.1f} pp (r={ratio_B:.1f}×)", f"{sc_pval_B:.3f}"),
]
for r in rows:
    print(f"  {r[0]:<26}{r[1]:>18}{r[2]:>10}{r[3]:>20}{r[4]:>10}")
print("="*72)
print("\nNote: DiD p-values are comment-level (large N inflates t).")
print("Permutation and SC p-values use month-level null distributions (more conservative).")

# ─────────────────────────────────────────────────────────────────────────────
# TREATED POSTS DETAIL
# ─────────────────────────────────────────────────────────────────────────────
def print_treated_posts(label, treat_ids, df_sent, df_posts):
    print(f"\n{'='*72}")
    print(f"TREATED POSTS — {label}")
    print(f"{'='*72}")
    for mid in sorted(treat_ids):
        post_row = df_posts[df_posts["media_id"] == mid]
        cmt_slice = df_sent[df_sent["media_id"] == mid]
        n_cmt      = len(cmt_slice)
        neg_rate   = cmt_slice["is_neg"].mean() * 100 if n_cmt else 0
        avg_score  = cmt_slice["sentiment_score"].mean() if n_cmt else 0
        top_emot   = cmt_slice["emotion"].value_counts().head(3).to_dict() if n_cmt else {}

        print(f"\n  media_id : {mid}")
        if len(post_row):
            r = post_row.iloc[0]
            permalink = r.get("permalink", "N/A")
            caption   = str(r.get("caption", ""))[:120]
            print(f"  link     : {permalink}")
            print(f"  caption  : {caption}{'...' if len(str(r.get('caption',''))) > 120 else ''}")
            print(f"  type     : {r.get('media_product_type','?')} / {r.get('media_type','?')}")
            print(f"  posted   : {r.get('timestamp','?')}")
            print(f"  likes    : {r.get('like_count','?'):,.0f}   comments_count: {r.get('comments_count','?'):,.0f}   reach: {r.get('reach','?')}")
            print(f"  paid     : {r.get('is_paid_partnership', False)}   music: {r.get('music_source','none')}   song: {r.get('song_title','—')}")
            if r.get("content_form"):
                print(f"  content  : {r.get('content_form')}   slides: {r.get('n_slides','—')}   video_dur: {r.get('video_duration','—')}s")
        else:
            print(f"  (post not in ig_multimodal_final — comment-only reference)")

        print(f"  -- comment-level sentiment --")
        print(f"  n_comments (in dip month) : {n_cmt}")
        print(f"  neg rate                  : {neg_rate:.1f}%")
        print(f"  avg sentiment_score       : {avg_score:.3f}")
        print(f"  top emotions              : {top_emot}")
        tox_dist = cmt_slice["toxicity"].value_counts(normalize=True).mul(100).round(1).to_dict() if n_cmt else {}
        print(f"  toxicity distribution     : {tox_dist}")
        intent_dist = cmt_slice["intent"].value_counts(normalize=True).mul(100).round(1).head(4).to_dict() if n_cmt else {}
        print(f"  top intents               : {intent_dist}")
        target_dist = cmt_slice["target"].value_counts(normalize=True).mul(100).round(1).head(3).to_dict() if n_cmt else {}
        print(f"  target distribution       : {target_dist}")

print_treated_posts("2024-01", TREAT_A, sent[sent["month"] == MONTH_A], posts)
print_treated_posts("2025-10", TREAT_B, sent[sent["month"] == MONTH_B], posts)

# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────
CSV_DIR = r"d:\Polythecninco di Milano\AFB_Lab\Ali\Sentiment_EDA\outputs"

# --- Table 1: Causal evidence summary ---
summary_rows = [
    {"method": "ATT (post removal)", "effect_2024_01": f"+{att_A*100:.1f} pp", "p_2024_01": "",
     "effect_2025_10": f"+{att_B*100:.1f} pp", "p_2025_10": ""},
    {"method": "Permutation test",
     "effect_2024_01": f"+{att_A*100:.1f} pp (z={z_A:.2f})", "p_2024_01": f"{pval_A_perm:.3f}",
     "effect_2025_10": f"+{att_B*100:.1f} pp (z={z_B:.2f})", "p_2025_10": f"{pval_B_perm:.3f}"},
    {"method": "DiD (HC0 SE)",
     "effect_2024_01": f"+{did_A_coef*100:.1f} pp (t={did_A_t:.2f})", "p_2024_01": f"{did_A_p:.4f}",
     "effect_2025_10": f"+{did_B_coef*100:.1f} pp (t={did_B_t:.2f})", "p_2025_10": f"{did_B_p:.4f}"},
    {"method": "Synthetic Control",
     "effect_2024_01": f"+{gap_A*100:.1f} pp (RMSPE={ratio_A:.1f}x)", "p_2024_01": f"{sc_pval_A:.3f}",
     "effect_2025_10": f"+{gap_B*100:.1f} pp (RMSPE={ratio_B:.1f}x)", "p_2025_10": f"{sc_pval_B:.3f}"},
]
pd.DataFrame(summary_rows).to_csv(f"{CSV_DIR}/CI_causal_evidence_summary.csv", index=False)

# --- Table 2: Treated posts detail ---
def build_post_detail(treat_ids, df_sent, df_posts, dip_label):
    rows = []
    for mid in sorted(treat_ids):
        post_row = df_posts[df_posts["media_id"] == mid]
        cmt = df_sent[df_sent["media_id"] == mid]
        n = len(cmt)
        r = post_row.iloc[0] if len(post_row) else pd.Series(dtype=object)
        rows.append({
            "dip_month": dip_label,
            "media_id": mid,
            "permalink": r.get("permalink", ""),
            "caption": str(r.get("caption", ""))[:200],
            "media_product_type": r.get("media_product_type", ""),
            "media_type": r.get("media_type", ""),
            "content_form": r.get("content_form", ""),
            "posted": r.get("timestamp", ""),
            "like_count": r.get("like_count", ""),
            "comments_count": r.get("comments_count", ""),
            "reach": r.get("reach", ""),
            "is_paid_partnership": r.get("is_paid_partnership", ""),
            "music_source": r.get("music_source", ""),
            "song_title": r.get("song_title", ""),
            "video_duration": r.get("video_duration", ""),
            "n_slides": r.get("n_slides", ""),
            "n_comments_in_dip": n,
            "neg_rate_pct": round(cmt["is_neg"].mean() * 100, 1) if n else 0,
            "avg_sentiment_score": round(cmt["sentiment_score"].mean(), 3) if n else 0,
            "top_emotion_1": cmt["emotion"].value_counts().index[0] if n else "",
            "top_emotion_1_n": int(cmt["emotion"].value_counts().iloc[0]) if n else 0,
            "top_emotion_2": cmt["emotion"].value_counts().index[1] if n > 1 and len(cmt["emotion"].value_counts()) > 1 else "",
            "top_emotion_2_n": int(cmt["emotion"].value_counts().iloc[1]) if n > 1 and len(cmt["emotion"].value_counts()) > 1 else 0,
            "top_emotion_3": cmt["emotion"].value_counts().index[2] if n > 2 and len(cmt["emotion"].value_counts()) > 2 else "",
            "top_emotion_3_n": int(cmt["emotion"].value_counts().iloc[2]) if n > 2 and len(cmt["emotion"].value_counts()) > 2 else 0,
            "top_intent_1": cmt["intent"].value_counts().index[0] if n else "",
            "top_intent_1_pct": round(cmt["intent"].value_counts(normalize=True).iloc[0] * 100, 1) if n else 0,
            "top_intent_2": cmt["intent"].value_counts().index[1] if n > 1 and len(cmt["intent"].value_counts()) > 1 else "",
            "top_intent_2_pct": round(cmt["intent"].value_counts(normalize=True).iloc[1] * 100, 1) if n > 1 and len(cmt["intent"].value_counts()) > 1 else 0,
            "top_target_1": cmt["target"].value_counts().index[0] if n else "",
            "top_target_1_pct": round(cmt["target"].value_counts(normalize=True).iloc[0] * 100, 1) if n else 0,
            "top_target_2": cmt["target"].value_counts().index[1] if n > 1 and len(cmt["target"].value_counts()) > 1 else "",
            "top_target_2_pct": round(cmt["target"].value_counts(normalize=True).iloc[1] * 100, 1) if n > 1 and len(cmt["target"].value_counts()) > 1 else 0,
            "toxicity_none_pct": round((cmt["toxicity"] == "none").mean() * 100, 1) if n else 0,
            "toxicity_mild_pct": round((cmt["toxicity"] == "mild").mean() * 100, 1) if n else 0,
            "toxicity_moderate_plus_pct": round((cmt["toxicity"].isin(["moderate","severe","extreme"])).mean() * 100, 1) if n else 0,
        })
    return rows

detail_rows = (
    build_post_detail(TREAT_A, sent[sent["month"] == MONTH_A], posts, "2024-01")
    + build_post_detail(TREAT_B, sent[sent["month"] == MONTH_B], posts, "2025-10")
)
pd.DataFrame(detail_rows).to_csv(f"{CSV_DIR}/CI_treated_posts_detail.csv", index=False, encoding="utf-8-sig")

# --- Table 3: Monthly neg rate time series (observed + counterfactual) ---
cf_both_series = cf_series(TREAT_A | TREAT_B)
ts_df = pd.DataFrame({
    "month": [str(m) for m in months_plot],
    "observed_neg_rate": obs_series.values,
    "counterfactual_neg_rate": cf_both_series.values,
    "att_gap": obs_series.values - cf_both_series.values,
}).round(4)
ts_df.to_csv(f"{CSV_DIR}/CI_monthly_negrate_timeseries.csv", index=False)

# --- Table 4: Permutation null distribution ---
perm_df = pd.DataFrame({
    "month": [str(m) for m in all_months if m not in (MONTH_A, MONTH_B)],
    "placebo_att": null_arr,
})
perm_df.to_csv(f"{CSV_DIR}/CI_permutation_null.csv", index=False)

# --- Table 5: SC placebo gaps ---
sc_df = pd.DataFrame([
    {"month": str(m), "sc_gap": round(g, 4), "is_treated": m in (MONTH_A, MONTH_B)}
    for m, g in sorted({**sc_placebo, MONTH_A: gap_A, MONTH_B: gap_B}.items())
])
sc_df.to_csv(f"{CSV_DIR}/CI_synthetic_control_gaps.csv", index=False)

print(f"\n--- CSVs exported to {CSV_DIR}/ ---")
for f in ["CI_causal_evidence_summary", "CI_treated_posts_detail",
          "CI_monthly_negrate_timeseries", "CI_permutation_null", "CI_synthetic_control_gaps"]:
    print(f"  {f}.csv")
