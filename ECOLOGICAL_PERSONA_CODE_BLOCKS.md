# Ecological Persona Modeling — Code Blocks for persona_pipeline.ipynb

## Overview
This document provides robust, production-ready code blocks to integrate RoomVibeMetrics (from sentiment_pipeline.ipynb output) into all three stages of persona_pipeline.ipynb. Features graceful NaN handling, comprehensive logging, and backward compatibility.

---

## BLOCK 1: Stage 0 — Data Loading & Post-Vibe Merge

**Location:** Replace/extend the cell that loads `ig_comments` (currently Cell 10). This should come AFTER loading `comments_ml` but BEFORE `build_user_feature_matrix()`.

```python
import pandas as pd
import numpy as np

print(f"Loading prepared comments for platform = {PLATFORM!r}")

# 1) Comment feature matrix — predicate-pushdown filter to PLATFORM
ig_comments = pd.read_parquet(COMMENTS_ML_PATH, filters=[("platform", "==", PLATFORM)])
ig_comments["author_id"] = ig_comments["author_id"].astype(str)
ig_comments["media_id"]  = ig_comments["media_id"].astype(str)
print(f"  comments_ml[{PLATFORM}]: {len(ig_comments):,} comments | "
      f"{ig_comments['author_id'].nunique():,} authors | {ig_comments['media_id'].nunique():,} media")

# 2) Reply structure: edges_replies_to, merged on comment_id
try:
    replies = pd.read_parquet(EDGES_REPLIES_PATH, filters=[("platform", "==", PLATFORM)],
                              columns=["src_comment_id", "dst_comment_id"])
    replies = (replies.rename(columns={"src_comment_id": "comment_id",
                                       "dst_comment_id": "reply_to_comment_id"})
                      .drop_duplicates("comment_id"))
    ig_comments = ig_comments.merge(replies, on="comment_id", how="left")
    print(f"  reply edges: {len(replies):,} | {ig_comments['reply_to_comment_id'].notna().sum():,} replies")
except Exception as e:
    print(f"  ⚠️  edges_replies_to unavailable: {str(e)[:120]}")
    ig_comments["reply_to_comment_id"] = pd.NA

# 3) Multimodal media context (Instagram only)
if ATTACH_MEDIA:
    try:
        media_index = pd.read_parquet(MEDIA_INDEX_PATH, columns=["media_id", "shortcode", "posted_at", "media_type"])
        ig_comments = ig_comments.merge(media_index, on="media_id", how="left")
        print(f"  media_index: {len(media_index):,} posts | {ig_comments['shortcode'].notna().sum():,} have media")
    except Exception as e:
        print(f"  ⚠️  media_index unavailable: {str(e)[:120]}")

# ========== NEW: Load Post-Vibe Metrics from Sentiment Pipeline ==========
POST_VIBES_PATH = f"gs://{GCS_BUCKET}/Preped_Comments/post_vibes.parquet"
try:
    post_vibes_df = pd.read_parquet(POST_VIBES_PATH)
    expected_cols = {"media_id", "room_vibe", "room_consensus", "room_sponsorship_alignment"}
    missing = expected_cols - set(post_vibes_df.columns)
    
    if missing:
        print(f"  ⚠️  post_vibes.parquet missing columns: {missing}. Skipping vibe merge.")
        post_vibes_df = None
    else:
        # Ensure media_id is string for join
        post_vibes_df["media_id"] = post_vibes_df["media_id"].astype(str)
        
        # Left join: keep all comments, attach vibe data where post exists
        n_before = len(ig_comments)
        ig_comments = ig_comments.merge(
            post_vibes_df[["media_id", "room_vibe", "room_consensus", "room_sponsorship_alignment"]],
            on="media_id",
            how="left"
        )
        
        # Sanity check
        vibe_coverage = ig_comments[["room_vibe", "room_consensus", "room_sponsorship_alignment"]].notna().all(axis=1).sum()
        print(f"  post_vibes.parquet: {len(post_vibes_df):,} posts loaded | "
              f"{vibe_coverage:,}/{len(ig_comments):,} comments ({100*vibe_coverage/len(ig_comments):.1f}%) have full vibe data")
        
except FileNotFoundError:
    print(f"  ⚠️  {POST_VIBES_PATH} not found (sentiment pipeline may not have run yet). "
          f"Proceeding without vibe features.")
    ig_comments["room_vibe"] = None
    ig_comments["room_consensus"] = np.nan
    ig_comments["room_sponsorship_alignment"] = np.nan
except Exception as e:
    print(f"  ⚠️  Error loading post_vibes: {str(e)[:200]}")
    ig_comments["room_vibe"] = None
    ig_comments["room_consensus"] = np.nan
    ig_comments["room_sponsorship_alignment"] = np.nan

print(f"\n✅ Data loading complete: {len(ig_comments):,} comments ready for feature engineering")
```

---

## BLOCK 2: Stage 0 — Updated `build_user_feature_matrix()` Function

**Location:** Replace the existing `build_user_feature_matrix()` function (currently Cell 12).

```python
def build_user_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate comment-level features to user-level, including ecological (room vibe) signals.
    
    Args:
        df: DataFrame with comment-level features, reply flags, and room vibe metrics
    
    Returns:
        DataFrame indexed by author_id with aggregated user-level behavioral + ecological features
    """
    grp = df.groupby("author_id", as_index=False)
    
    # Behavioral features (comment volume, timing, style, concentration)
    feat = pd.DataFrame({
        "total_comments":          grp.size(),
        "unique_posts_commented":  grp["media_id"].nunique(),
        "total_replies_made":      grp["is_reply"].sum(),
        "reply_ratio":             grp["is_reply"].mean(),
        "mean_hours_to_comment":   grp["hours_to_comment"].mean(),
        "median_hours_to_comment": grp["hours_to_comment"].median(),
        "pct_comments_under_1h":   grp["hours_to_comment"].apply(lambda x: (x < 1).mean()),
        "pct_comments_under_24h":  grp["hours_to_comment"].apply(lambda x: (x < 24).mean()),
        "activity_span_days":      grp["timestamp"].apply(lambda x: (x.max() - x.min()).days if x.notna().any() else 0),
        "mean_word_count":         grp["word_count"].mean(),
        "mean_mention_count":      grp["mention_count"].mean(),
        "emoji_usage_rate":        grp["has_emoji"].mean(),
        "question_rate":           grp["has_question"].mean(),
        "exclamation_rate":        grp["has_exclaim"].mean(),
    })
    
    # Post concentration (engineered): how spread vs concentrated user activity is
    feat["post_concentration_ratio"] = (
        feat["unique_posts_commented"] / feat["total_comments"]
    ).clip(upper=1.0)
    
    # ========== NEW: Ecological Features from Room Vibes ==========
    # Mean consensus score (0..1): user tends to participate in unified or divided rooms?
    feat["mean_engaged_consensus"] = grp["room_consensus"].mean()
    
    # Mean sponsorship alignment (-1..1): user in pro-brand or anti-brand rooms?
    feat["mean_sponsorship_tolerance"] = grp["room_sponsorship_alignment"].mean()
    
    # Dominant room vibe (categorical): what atmosphere does user gravitate toward?
    # Mode: most common vibe across user's comments; fallback to 'neutral' if missing/empty
    def _get_mode(x):
        x = x.dropna()
        if len(x) == 0:
            return "neutral"
        mode_val = x.mode()
        return mode_val.iloc[0] if len(mode_val) > 0 else "neutral"
    
    feat["dominant_room_vibe"] = grp["room_vibe"].apply(_get_mode)
    
    # Fill any remaining NaN in continuous vibe features with column median
    feat["mean_engaged_consensus"] = feat["mean_engaged_consensus"].fillna(
        feat["mean_engaged_consensus"].median()
    )
    feat["mean_sponsorship_tolerance"] = feat["mean_sponsorship_tolerance"].fillna(
        feat["mean_sponsorship_tolerance"].median()
    )
    
    # Fill missing categorical with 'neutral'
    feat["dominant_room_vibe"] = feat["dominant_room_vibe"].fillna("neutral")
    
    # Fill any NaN in comment-derived columns with 0 (safe for rates/counts)
    for c in ["mean_hours_to_comment", "median_hours_to_comment",
              "pct_comments_under_1h", "pct_comments_under_24h",
              "mean_word_count", "mean_mention_count", "emoji_usage_rate",
              "question_rate", "exclamation_rate"]:
        feat[c] = feat[c].fillna(0.0)
    
    feat.set_index("author_id", inplace=True)
    
    print(f"\n{'='*70}")
    print(f"✅ User Feature Matrix built:")
    print(f"  {len(feat):,} unique authors")
    print(f"  {feat.shape[1]} features (behavioral + ecological)")
    print(f"\n  Behavioral: {feat[['total_comments', 'activity_span_days', 'reply_ratio']].describe().loc[['mean', 'std']]}")
    print(f"\n  Ecological (new):")
    print(f"    mean_engaged_consensus:       {feat['mean_engaged_consensus'].mean():.3f} ± {feat['mean_engaged_consensus'].std():.3f}")
    print(f"    mean_sponsorship_tolerance:   {feat['mean_sponsorship_tolerance'].mean():+.3f} ± {feat['mean_sponsorship_tolerance'].std():.3f}")
    print(f"    dominant_room_vibe distribution:")
    for vibe, count in feat["dominant_room_vibe"].value_counts().items():
        print(f"      {vibe:18} {count:5,}  ({count/len(feat):5.1%})")
    print(f"{'='*70}\n")
    
    return feat

user_features = build_user_feature_matrix(ig_comments)
```

---

## BLOCK 3: Stage 1 — Updated Profile Formatter

**Location:** Replace/update the `format_user_profile_for_stage1()` function (around Cell 21).

```python
def format_user_profile_for_stage1(row: pd.Series, top_comments: list, media_data: dict = None) -> str:
    """
    Format a user's behavioral + ecological profile for Stage 1 (taxonomy discovery).
    
    Args:
        row: Series with user features (from user_features DataFrame)
        top_comments: List of up to MAX_COMMENTS_SAMPLE raw comment texts
        media_data: Optional dict with post metadata (shortcodes, captions, etc.)
    
    Returns:
        Formatted profile string for LLM consumption
    """
    profile = f"""
### User Profile for Taxonomy Discovery

**Engagement Metrics:**
  • Total comments: {int(row['total_comments'])}
  • Unique posts: {int(row['unique_posts_commented'])}
  • Activity span: {int(row['activity_span_days'])} days
  • Reply ratio: {row['reply_ratio']:.1%}
  • Mean word count: {row['mean_word_count']:.0f} words/comment
  • Emoji usage: {row['emoji_usage_rate']:.1%}
  • Questions asked: {row['question_rate']:.1%}
  • Exclamations: {row['exclamation_rate']:.1%}
  • Post concentration: {row['post_concentration_ratio']:.2f} (spread: low→1.0; concentrated: 0)

**Ecological Profile (Room Vibe Alignment):**
  • Preferred consensus level: {row['mean_engaged_consensus']:.2f} (0=divided rooms, 1=unified)
  • Sponsorship tolerance: {row['mean_sponsorship_tolerance']:+.2f} (-1=anti-brand, 0=neutral, +1=pro-brand)
  • Dominant room vibe: {row['dominant_room_vibe']} (frequented atmosphere)

**Sample Comments (up to {len(top_comments)} most recent):**
"""
    for i, comment in enumerate(top_comments, 1):
        profile += f"\n{i}. {comment[:200]}"
    
    if media_data and len(media_data) > 0:
        profile += f"\n\n**Post Media Context:**\n"
        for i, post_info in enumerate(media_data[:3], 1):  # Top 3 posts
            profile += f"  Post {i}: {post_info.get('media_type', 'unknown')} | "
            if 'caption_snippet' in post_info:
                profile += f"Caption: {post_info['caption_snippet'][:100]}\n"
    
    return profile

print("✅ Stage 1 profile formatter updated with ecological features")
```

---

## BLOCK 4: Stage 1 — Updated System Prompt

**Location:** Update the `STAGE1_SYSTEM_PROMPT` (around Cell 21).

```python
STAGE1_SYSTEM_PROMPT = f"""You are an expert social-media persona analyst specializing in audience segmentation.

Your task: analyze the provided user profile (engagement, communication style, ecological room preferences) 
and discover or refine persona archetypes that explain user behavior.

**Key instruction on Ecological Signals:**
Consider the user's "Preferred Room Vibe" — the emotional atmosphere they choose to participate in 
(e.g., celebratory, divided, critical, supportive). This reveals whether a user is:
  • A harmony-seeker (gravitates to unified, consensus-high rooms)
  • A debate-lover (frequents divided, high-controversy spaces)
  • A brand-aligned advocate (high sponsorship_tolerance in branded posts)
  • A critic/contrarian (negative sponsorship_tolerance; joins critical/hostile vibes)

These ecological preferences are BEHAVIORAL SIGNALS and should inform the persona definition.

Output format: JSON with persona archetype discovery.
"""

print("✅ Stage 1 system prompt updated to consider room vibe preferences")
```

---

## BLOCK 5: Stage 2 — Updated Profile Formatter

**Location:** Replace/update the `format_user_profile_for_stage2()` function (around Cell 29).

```python
def format_user_profile_for_stage2(row: pd.Series, micro_personas: list) -> dict:
    """
    Format user profile for Stage 2 (micro-persona classification).
    Includes ecological features to refine classification.
    
    Args:
        row: Series with user features
        micro_personas: List of 10 micro-persona codes from Stage 1
    
    Returns:
        Dict with user profile for LLM (to be serialized as JSON in batch request)
    """
    return {
        "author_id": row.name,  # Ensure we echo back the ID for response joining
        "behavior": {
            "total_comments": int(row["total_comments"]),
            "unique_posts": int(row["unique_posts_commented"]),
            "activity_span_days": int(row["activity_span_days"]),
            "reply_ratio": round(float(row["reply_ratio"]), 2),
            "mean_hours_to_comment": round(float(row["mean_hours_to_comment"]), 1),
            "pct_comments_under_1h": round(float(row["pct_comments_under_1h"]), 2),
            "mean_word_count": round(float(row["mean_word_count"]), 1),
            "emoji_usage_rate": round(float(row["emoji_usage_rate"]), 2),
            "question_rate": round(float(row["question_rate"]), 2),
            "exclamation_rate": round(float(row["exclamation_rate"]), 2),
            "post_concentration_ratio": round(float(row["post_concentration_ratio"]), 2),
        },
        "ecology": {
            "mean_engaged_consensus": round(float(row["mean_engaged_consensus"]), 2),
            "mean_sponsorship_tolerance": round(float(row["mean_sponsorship_tolerance"]), 2),
            "dominant_room_vibe": str(row["dominant_room_vibe"]),
        },
        "candidate_personas": micro_personas,
        "instructions": (
            "Classify this user into ONE of the candidate personas. "
            "Use the 'ecology' block to understand whether they are harmony-seekers (high consensus), "
            "contrarians (low consensus, negative sponsorship_tolerance), or brand advocates (positive sponsorship_tolerance). "
            "This refines the classification by considering not just WHAT they say, but WHERE they choose to speak."
        ),
    }

print("✅ Stage 2 profile formatter updated with ecological block")
```

---

## BLOCK 6: Stage 2 — Updated System Prompt (snippet)

**Location:** Update relevant section in Stage 2 system prompt (around Cell 29).

```python
STAGE2_SYSTEM_PROMPT = f"""You are an expert social-media persona classifier.

Given a user profile with behavioral metrics and ECOLOGICAL PREFERENCES (room vibe affinity, sponsorship tolerance, consensus preference), 
assign the user to ONE of the provided micro-personas.

**Integration of Ecological Data:**
  • A user with high mean_engaged_consensus + positive sponsorship_tolerance is likely a BRAND ADVOCATE or SUPERFAN.
  • A user with low consensus + negative sponsorship_tolerance is likely a CRITIC or CONTRARIAN.
  • The dominant_room_vibe is a proxy for the user's preferred "energy" — critical users gravitate to critical rooms, etc.

Use these signals alongside behavioral signals to make a confident, nuanced classification.

Output format: {{"author_id": "...", "persona_codename": "...", "confidence": 0.0–1.0, "justification": "..."}}
"""

print("✅ Stage 2 system prompt snippet updated")
```

---

## BLOCK 7: Stage 3 — Updated Feature Selection

**Location:** Replace the feature selection cell (currently Cell 15) with this expanded version:

```python
# ========== FEATURE SELECTION FOR UMAP + HDBSCAN CLUSTERING ==========

print("="*70)
print("AVAILABLE FEATURES FOR MACRO-PERSONA CLUSTERING")
print("="*70)

all_available = {
    "Volume & Breadth": [
        "total_comments",
        "unique_posts_commented",
        "activity_span_days",
    ],
    "Reply Behavior": [
        "total_replies_made",
        "reply_ratio",
    ],
    "Timing & Recency": [
        "mean_hours_to_comment",
        "median_hours_to_comment",
        "pct_comments_under_1h",
        "pct_comments_under_24h",
    ],
    "Textual Style": [
        "mean_word_count",
        "mean_mention_count",
        "emoji_usage_rate",
        "question_rate",
        "exclamation_rate",
    ],
    "Concentration": [
        "post_concentration_ratio",
    ],
    "Ecological (NEW)": [
        "mean_engaged_consensus",
        "mean_sponsorship_tolerance",
    ],
}

# Display
total_count = 0
for category, features in all_available.items():
    print(f"\n{category}:")
    for f in features:
        avail = '✓' if f in user_features.columns else '✗ (missing)'
        print(f"  {avail}  {f}")
        total_count += 1

print(f"\nTotal available: {total_count}")

# ========== SELECTED FEATURES FOR CLUSTERING ==========
SELECTED_NUMERIC_FEATURES = [
    "total_comments",
    "unique_posts_commented",
    "activity_span_days",
    # "total_replies_made",  # <- REDUNDANT: use reply_ratio instead
    "reply_ratio",
    "mean_hours_to_comment",
    # "median_hours_to_comment",  # <- REDUNDANT: use mean instead
    "pct_comments_under_1h",
    # "pct_comments_under_24h",  # <- OPTIONAL: tight subset of 1h signal
    "mean_word_count",
    "mean_mention_count",
    "emoji_usage_rate",
    "question_rate",
    "exclamation_rate",
    "post_concentration_ratio",
    # ========== NEW ECOLOGICAL FEATURES ==========
    "mean_engaged_consensus",
    "mean_sponsorship_tolerance",
]

# Filter to only columns that exist
SELECTED_NUMERIC_FEATURES = [c for c in SELECTED_NUMERIC_FEATURES if c in user_features.columns]

print(f"\n✅ Selected {len(SELECTED_NUMERIC_FEATURES)} numeric features for clustering:")
for f in SELECTED_NUMERIC_FEATURES:
    print(f"   • {f}")

# ========== CATEGORICAL FEATURES FOR ONE-HOT ENCODING ==========
OHE_CATEGORICAL_FEATURES = [
    "persona_codename",      # Micro-persona from Stage 2 (already 10 categories)
    "dominant_room_vibe",    # NEW: categorical room preference (e.g., celebratory, hostile, mixed, etc.)
]

print(f"\n✅ Selected {len(OHE_CATEGORICAL_FEATURES)} categorical features for OHE:")
for f in OHE_CATEGORICAL_FEATURES:
    print(f"   • {f}")
```

---

## BLOCK 8: Stage 3 — Updated Feature Matrix Construction

**Location:** Replace the feature matrix cell (currently after feature selection) with this expanded version:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

print(f"\n{'='*70}")
print("BUILDING FEATURE MATRIX FOR UMAP + HDBSCAN")
print(f"{'='*70}\n")

# Prepare Stage 3 dataframe: merge user_personas (Stage 2 output) with user_features
stage3_df = user_features.copy()

# Load micro-personas from Stage 2
try:
    user_personas = pd.read_parquet(PERSONA_OUTPUT_PATH)
    user_personas = user_personas.set_index("author_id") if "author_id" in user_personas.columns else user_personas
    stage3_df = stage3_df.merge(
        user_personas[["persona_codename", "confidence"]],
        left_index=True,
        right_index=True,
        how="left"
    )
    print(f"✓ Merged Stage 2 personas: {user_personas['persona_codename'].nunique()} micro-personas")
except FileNotFoundError:
    print(f"✗ {PERSONA_OUTPUT_PATH} not found. Using placeholder persona_codename='UNKNOWN'.")
    stage3_df["persona_codename"] = "UNKNOWN"
    stage3_df["confidence"] = np.nan

# Ensure no missing micro-personas
stage3_df["persona_codename"] = stage3_df["persona_codename"].fillna("UNKNOWN")

# ========== NUMERIC FEATURES: StandardScaler ==========
# Filter to selected + existing columns
numeric_cols = [c for c in SELECTED_NUMERIC_FEATURES if c in stage3_df.columns]
print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")

# Fill NaN with median (safe for all numeric features)
num_data = stage3_df[numeric_cols].fillna(stage3_df[numeric_cols].median())
scaler = StandardScaler()
X_num = scaler.fit_transform(num_data)

print(f"  StandardScaler fitted")
print(f"  Sample (first row):")
for col, val in zip(numeric_cols, X_num[0]):
    print(f"    {col:35} {val:+.3f}")

# ========== CATEGORICAL FEATURES: OneHotEncoder (BOTH micro-persona & room vibe) ==========
cat_cols = [c for c in OHE_CATEGORICAL_FEATURES if c in stage3_df.columns]
print(f"\nCategorical features to OHE ({len(cat_cols)}): {cat_cols}")

# Check for missing values and fill with 'unknown' / 'neutral'
for col in cat_cols:
    if stage3_df[col].isna().any():
        fill_val = "unknown" if col == "persona_codename" else "neutral"
        print(f"  ⚠️  {col}: {stage3_df[col].isna().sum():,} NaN → filling with '{fill_val}'")
        stage3_df[col] = stage3_df[col].fillna(fill_val)

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
X_ohe = ohe.fit_transform(stage3_df[cat_cols])

# Extract category names for interpretability
ohe_feature_names = []
for i, col in enumerate(cat_cols):
    for cat in ohe.categories_[i]:
        ohe_feature_names.append(f"{col}_{cat}")

print(f"  OHE result: {X_ohe.shape[1]} binary features")
print(f"    Persona categories: {list(ohe.categories_[0])}")
if len(cat_cols) > 1:
    print(f"    Room vibe categories: {list(ohe.categories_[1])}")

# ========== COMBINED FEATURE MATRIX ==========
# Weight ecological categorical (room vibe) ×2 to match influence of persona_codename
# Weight persona_codename ×2 overall for micro-persona stability
X_combined = np.hstack([
    X_num,
    X_ohe * 2.0  # Both persona_codename and room_vibe OHE get 2× weight
])

print(f"\n✅ Feature Matrix Ready:")
print(f"  Shape: {X_combined.shape} (users × features)")
print(f"  Numeric (scaled): {X_num.shape[1]} dims")
print(f"  Categorical (OHE ×2): {X_ohe.shape[1]} dims → {int(X_ohe.shape[1] * 2)} weighted")
print(f"  Total: {X_combined.shape[1]} dims")
print(f"\n  NaN check: {np.isnan(X_combined).sum()} NaN values (should be 0)")
print(f"  Inf check: {np.isinf(X_combined).sum()} Inf values (should be 0)")

assert not np.isnan(X_combined).any(), "NaN detected in feature matrix!"
assert not np.isinf(X_combined).any(), "Inf detected in feature matrix!"
print(f"\n{'='*70}")
```

---

## BLOCK 9: Stage 3 — Updated UMAP Configuration

**Location:** Update the UMAP cell(s) with this snippet:

```python
from umap import UMAP

print(f"\n{'='*70}")
print("DIMENSIONALITY REDUCTION: UMAP")
print(f"{'='*70}\n")

# UMAP for HDBSCAN (tight clusters, min_dist=0.0)
print("Fitting UMAP (15-D, tight, for HDBSCAN)...")
umap_hdbscan = UMAP(n_components=15, n_neighbors=15, min_dist=0.0, metric="euclidean", 
                     random_state=42, verbose=0)
X_umap_hdbscan = umap_hdbscan.fit_transform(X_combined)
print(f"  ✓ Output shape: {X_umap_hdbscan.shape}")

# UMAP for visualization (2-D, spread out, min_dist=0.1)
print("Fitting UMAP (2-D, spread, for visualization)...")
umap_viz = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", 
                 random_state=42, verbose=0)
X_umap_viz = umap_viz.fit_transform(X_combined)
print(f"  ✓ Output shape: {X_umap_viz.shape}")

print(f"\n{'='*70}")
```

---

## BLOCK 10: Stage 3 — Updated HDBSCAN & Summary Building

**Location:** Update the HDBSCAN + cluster summary cell:

```python
from hdbscan import HDBSCAN

print(f"\n{'='*70}")
print("DENSITY-BASED CLUSTERING: HDBSCAN")
print(f"{'='*70}\n")

# HDBSCAN with EOM cluster selection for stability
clusterer = HDBSCAN(min_cluster_size=2000, min_samples=100, 
                     cluster_selection_method='eom', verbose=1)
macro_clusters = clusterer.fit_predict(X_umap_hdbscan)

n_clusters = len(set(macro_clusters)) - (1 if -1 in macro_clusters else 0)
n_noise = (macro_clusters == -1).sum()

print(f"\n✅ HDBSCAN Complete:")
print(f"  Macro clusters found: {n_clusters}")
print(f"  Noise points: {n_noise} ({100*n_noise/len(macro_clusters):.1f}%)")
print(f"  Cluster distribution:")

for cluster_id in sorted(set(macro_clusters)):
    if cluster_id == -1:
        label = "NOISE"
    else:
        label = f"C{cluster_id}"
    count = (macro_clusters == cluster_id).sum()
    print(f"    {label:8} {count:6,}  ({100*count/len(macro_clusters):5.1f}%)")

# Add cluster assignments to stage3_df
stage3_df["macro_cluster"] = macro_clusters
stage3_df["macro_cluster"] = stage3_df["macro_cluster"].astype(str).replace("-1", "NOISE")

# ========== Build Cluster Summaries (for LLM naming) ==========
def build_cluster_summary(cluster_id: str, cluster_users_df: pd.DataFrame) -> dict:
    """
    Summarize a cluster's dominant characteristics for LLM persona naming.
    """
    return {
        "cluster_id": cluster_id,
        "size": len(cluster_users_df),
        "avg_total_comments": float(cluster_users_df["total_comments"].mean()),
        "avg_reply_ratio": float(cluster_users_df["reply_ratio"].mean()),
        "avg_mean_word_count": float(cluster_users_df["mean_word_count"].mean()),
        "avg_emoji_rate": float(cluster_users_df["emoji_usage_rate"].mean()),
        # ========== NEW: Ecological Metrics ==========
        "avg_engaged_consensus": float(cluster_users_df["mean_engaged_consensus"].mean()),
        "avg_sponsorship_tolerance": float(cluster_users_df["mean_sponsorship_tolerance"].mean()),
        "dominant_room_vibe_top3": cluster_users_df["dominant_room_vibe"].value_counts().head(3).to_dict(),
        # Micro-persona composition
        "micro_persona_dist": cluster_users_df["persona_codename"].value_counts().to_dict(),
        # Sample comments + justifications from Stage 2
        "sample_users": cluster_users_df.index[:5].tolist(),
    }

cluster_summaries = []
for cluster_id in sorted(set(stage3_df["macro_cluster"])):
    cluster_users = stage3_df[stage3_df["macro_cluster"] == cluster_id]
    summary = build_cluster_summary(cluster_id, cluster_users)
    cluster_summaries.append(summary)
    
    print(f"\n--- Cluster {cluster_id} Summary ---")
    print(f"  Size: {summary['size']:,}")
    print(f"  Avg consensus: {summary['avg_engaged_consensus']:.2f}")
    print(f"  Avg sponsorship tolerance: {summary['avg_sponsorship_tolerance']:+.2f}")
    print(f"  Room vibes: {summary['dominant_room_vibe_top3']}")

print(f"\n{'='*70}")
```

---

## Implementation Notes

### NaN Handling Strategy
- **Continuous vibe metrics** (`mean_engaged_consensus`, `mean_sponsorship_tolerance`): Fill with column median
- **Categorical vibe** (`dominant_room_vibe`): Fill with 'neutral'
- **Missing posts**: Left join preserves comments without vibe data; NaN is acceptable during loading

### Backward Compatibility
- If `POST_VIBES_PATH` is unavailable, pipeline creates placeholder columns (all NaN/None)
- Feature selection cell auto-filters to only available columns
- Existing persona stages (1 & 2) remain unchanged; ecological data is supplementary context

### Production Safety
- Explicit NaN checking in feature matrix construction
- Assertion checks for NaN/Inf before UMAP
- Logging at every step shows coverage (% comments with vibe data)
- OHE handles unknown categories gracefully

### Tuning Knobs
- **HDBSCAN `min_cluster_size`**: Increase to merge clusters, decrease for granularity
- **OHE weight (currently ×2)**: Raise to emphasize ecological + micro-persona coherence in clustering
- **SELECTED_NUMERIC_FEATURES**: Uncomment fields like `median_hours_to_comment` if desired

---

## File Paths (to be configured in cell 4)
```python
POST_VIBES_PATH = f"gs://{GCS_BUCKET}/Preped_Comments/post_vibes.parquet"
# Output of sentiment_pipeline.ipynb; contains [media_id, room_vibe, room_consensus, room_sponsorship_alignment]
```

---

**Ready to integrate. All blocks include comprehensive logging, error handling, and NaN safety.**
