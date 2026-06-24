"""
Investigate sentiment dips at 2024-01 and 2025-10.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np

SENTIMENT = r"d:\Polythecninco di Milano\AFB_Lab\Ali\batch_results\sentiment\sentiment_instagram.parquet"
POSTS     = r"d:\Polythecninco di Milano\AFB_Lab\Ali\outputs\ig_multimodal_final.parquet"
RFM       = r"d:\Polythecninco di Milano\AFB_Lab\Ali\outputs\pathway_b_assignments_instagram.parquet"

sent  = pd.read_parquet(SENTIMENT)
posts = pd.read_parquet(POSTS)
rfm   = pd.read_parquet(RFM)

# ── parse timestamps ──────────────────────────────────────────────────────────
sent["ts"]  = pd.to_datetime(sent["timestamp"],  format="mixed", utc=True).dt.tz_localize(None)
posts["ts"] = pd.to_datetime(posts["timestamp"], format="mixed", utc=True, errors="coerce").dt.tz_localize(None)

sent["month"]  = sent["ts"].dt.to_period("M")
posts["month"] = posts["ts"].dt.to_period("M")

# ── join RFM cluster onto comments ───────────────────────────────────────────
sent = sent.merge(rfm, on="author_id", how="left")

# is_negative flag
sent["is_neg"] = (sent["sentiment"] == "negative").astype(int)

# ── define dip windows ───────────────────────────────────────────────────────
DIP_A = "2024-01"
DIP_B = "2025-10"
BEFORE_A = ["2023-10", "2023-11", "2023-12"]
AFTER_A  = ["2024-02", "2024-03", "2024-04"]
BEFORE_B = ["2025-07", "2025-08", "2025-09"]
AFTER_B  = ["2025-11", "2025-12", "2026-01"]

def str_period(p): return pd.Period(p, "M")

def month_slice(df, months):
    return df[df["month"].isin([str_period(m) for m in months])]

def dip_summary(label, dip_month, before_months, after_months):
    dip    = month_slice(sent, [dip_month])
    before = month_slice(sent, before_months)
    after  = month_slice(sent, after_months)

    print(f"\n{'='*60}")
    print(f"DIP: {label} ({dip_month})")
    print(f"{'='*60}")
    print(f"  Comments in dip month    : {len(dip)}")
    print(f"  Comments in before period: {len(before)}")
    print(f"  Comments in after period : {len(after)}")

    # sentiment score
    print(f"\n  -- Avg sentiment_score --")
    print(f"  Before : {before['sentiment_score'].mean():.3f}")
    print(f"  DIP    : {dip['sentiment_score'].mean():.3f}")
    print(f"  After  : {after['sentiment_score'].mean():.3f}")

    # negative rate
    print(f"\n  -- Negative rate % --")
    print(f"  Before : {before['is_neg'].mean()*100:.1f}%")
    print(f"  DIP    : {dip['is_neg'].mean()*100:.1f}%")
    print(f"  After  : {after['is_neg'].mean()*100:.1f}%")

    # sentiment mix in dip
    print(f"\n  -- Sentiment mix in dip --")
    print(dip["sentiment"].value_counts(normalize=True).mul(100).round(1).to_string())

    # toxicity in dip vs before
    tox_map = {"none":0,"mild":1,"moderate":2,"severe":3,"extreme":4}
    dip_t = dip["toxicity"].map(tox_map)
    bef_t = before["toxicity"].map(tox_map)
    print(f"\n  -- Toxicity (ordinal avg) --")
    print(f"  Before: {bef_t.mean():.3f}  |  Dip: {dip_t.mean():.3f}")

    # cluster breakdown in dip
    if "macro_cluster" in dip.columns:
        print(f"\n  -- Cluster composition (dip) --")
        print(dip["macro_cluster"].value_counts(normalize=True).mul(100).round(1).to_string())
        print(f"\n  -- Cluster neg rate in dip --")
        print(dip.groupby("macro_cluster")["is_neg"].mean().mul(100).round(1).sort_values(ascending=False).to_string())
        print(f"\n  -- Cluster neg rate BEFORE dip --")
        print(before.groupby("macro_cluster")["is_neg"].mean().mul(100).round(1).sort_values(ascending=False).to_string())

    # Which posts drove comments in the dip month?
    dip_posts = posts[posts["month"] == str_period(dip_month)].copy()
    print(f"\n  -- Posts published in {dip_month}: {len(dip_posts)} posts --")
    if len(dip_posts):
        cols = [c for c in ["caption","media_product_type","media_type","comments_count","like_count","music_source","is_paid_partnership"] if c in dip_posts.columns]
        pd.set_option("display.max_colwidth", 80)
        print(dip_posts[cols].sort_values("comments_count", ascending=False).head(10).to_string())

    # Posts that attracted the most comments IN the dip month (comments ts = dip, not post ts)
    media_neg = dip.groupby("media_id").agg(
        n_comments=("comment_id","count"),
        neg_rate=("is_neg","mean"),
        avg_score=("sentiment_score","mean")
    ).sort_values("n_comments", ascending=False).head(10)
    print(f"\n  -- Top media_ids by comment volume in dip month --")
    print(media_neg.round(3).to_string())

    # merge with post info
    top_media = media_neg.reset_index().merge(
        posts[["media_id","caption","media_product_type","comments_count","like_count","timestamp"]].drop_duplicates("media_id"),
        on="media_id", how="left"
    )
    print(f"\n  -- Top posts detail --")
    print(top_media.to_string())

dip_summary("2024-Jan dip", DIP_A, BEFORE_A, AFTER_A)
dip_summary("2025-Oct dip", DIP_B, BEFORE_B, AFTER_B)

# ── Global monthly trend for context ─────────────────────────────────────────
print("\n\n=== Monthly global neg rate (all clusters) ===")
monthly = sent.groupby("month").agg(
    n=("comment_id","count"),
    neg_rate=("is_neg","mean"),
    avg_score=("sentiment_score","mean")
).reset_index()
monthly["neg_pct"] = (monthly["neg_rate"]*100).round(1)
print(monthly[["month","n","neg_pct","avg_score"]].to_string())

print("\n\n=== Monthly post count ===")
print(posts.groupby("month").size().rename("n_posts").to_string())
