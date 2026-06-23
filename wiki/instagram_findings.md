---
title: Instagram Analysis — Major Findings
type: source-summary
sources:
  - Ali/Sentiment_EDA/sentiment_eda.ipynb
  - Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb
  - Mickey/EDA_RFM_IG_vs_TK_rolling_quadrimesters.ipynb
  - Mickey/EDA_cluster_matrix_IG_vs_TK.ipynb
related:
  - "[[sentiment-pipeline]]"
  - "[[persona-pipeline]]"
  - "[[rfm-clustering]]"
created: 2026-06-14
updated: 2026-06-14
confidence: high
---

# Instagram Analysis — Major Findings

## Dataset scale

| Dimension | Value |
|---|---|
| Comments analysed | 487,604 |
| Posts | 1,501 (1,205 Feed · 288 Reels) |
| Unique commenting users | 191,096 |
| Persona-tagged users | 40,019 (19.3% coverage of comments) |
| RFM user-window observations | 174,287 (18,752 users × 40 rolling 4-month windows) |
| Date range | 2016-08 to 2026-03 |

The sentiment table was produced by the Vertex Batch pipeline in `Ali/sentiment_pipeline.ipynb` and stored at `Ali/outputs/stage2_sentiment/sentiment_instagram.parquet`. Persona assignments come from `Ali/outputs/stage2_persona_combined/user_personas_combined.parquet`.

---

## 1. Sentiment overview

### 1.1 Overall distribution

| Sentiment | Count | Share |
|---|---|---|
| Positive | 394,276 | 80.9% |
| Neutral | 68,737 | 14.1% |
| Negative | 24,591 | 5.0% |
| **Toxic (mild + severe)** | **3,081** | **0.63%** |
| Severe toxic only | 329 | 0.07% |

The audience is strongly positive-skewed. Negativity and toxicity are rare in aggregate but cluster sharply on specific posts and user types (see sections 3 and 4).

### 1.2 Dominant emotions and intents

- **Top emotion overall**: joy (overwhelmingly dominant in positive comments)
- **Top emotion in negative comments**: sadness
- **Top intent in negative comments**: criticism
- **Top emotion in toxic comments**: anger
- **Top intent in toxic comments**: criticism

Criticism is the shared intent signal for both negative and toxic comments. Anger separates toxic from merely negative; sadness marks comments that are negative but not hostile.

### 1.3 Intensity distribution

Intensity is labelled low / medium / high by the LLM. The majority of comments sit at medium intensity. High-intensity negative comments carry significantly more anger and disgust; high-intensity positive comments carry joy and trust.

### 1.4 Sentiment trend over time

Monthly sentiment share is broadly stable across the dataset's time range. No structural break was identified in the positive-to-negative ratio within the windows analysed.

---

## 2. Sentiment targets — what are comments aimed at?

The `target` field classifies who or what each comment is directed at. This converts "is it negative?" into "negative *about what?*" — the actionable read.

| Target | Share of all comments | Negative rate | Toxicity rate |
|---|---|---|---|
| Off-topic | 1.2% | **30.2%** | **3.58%** |
| Product | 3.3% | 9.6% | 0.34% |
| None / unclear | 2.2% | 8.4% | 0.78% |
| Other user | 16.0% | 6.9% | 0.98% |
| Content / work | 31.8% | 5.7% | 0.56% |
| Creator | 31.5% | 3.3% | 0.43% |
| Appearance | 14.0% | **1.4%** | 0.65% |

### Key target insights

**Off-topic comments are the toxicity hotspot.** Only 1.2% of volume but 30.2% negative and 3.58% toxic — more than 5x the population baseline. These are cross-platform pile-ons, unrelated grievances, and spam masquerading as commentary.

**Product comments are the frustrated-customer channel.** At 9.6% negative they are the most actionable negative segment. Their intent fingerprint shows elevated question intent and anticipation emotion, consistent with purchase uncertainty or post-purchase disappointment. The Critic persona (see section 3) drives a disproportionate share of this negativity.

**Appearance comments are near-purely positive.** 1.4% negative — the lowest of any target. Complimenting physical appearance is a near-universal positive signal in this community and carries negligible toxicity risk.

**Creator and Content/Work dominate volume (63.3% combined) and are largely positive.** Creator-directed comments are 3.3% negative — below the population mean. Content/work is 5.7% negative, slightly above average, reflecting occasional critical artistic feedback.

**Other-user-directed comments carry meaningful toxicity.** At 0.98% toxic (16% of volume), comment-section interpersonal conflict is the second-largest toxicity source by volume after off-topic comments.

---

## 3. Persona analysis

### 3.1 Persona distribution

Personas were assigned by the LLM pipeline to 40,019 users. Distribution by user count:

| Persona | Users | Share |
|---|---|---|
| THE_TAGGER | 11,628 | 29.1% |
| THE_CASUAL_COMPLIMENTER | 7,802 | 19.5% |
| THE_EMOJI_REACTOR | 5,764 | 14.4% |
| THE_STORYTELLER | 4,471 | 11.2% |
| THE_SUPERFAN | 4,191 | 10.5% |
| THE_INQUIRER | 2,608 | 6.5% |
| THE_CRITIC | 2,253 | 5.6% |
| THE_ADVISOR | 865 | 2.2% |
| THE_SPAMMER | 218 | 0.5% |
| THE_HATER | 164 | 0.4% |

By comment volume the ranking shifts: Superfan generates disproportionately high comment counts relative to user count, indicating high per-user activity.

### 3.2 Toxicity and negativity by persona

| Persona | Neg rate | Tox rate | Tox lift vs pop | n comments |
|---|---|---|---|---|
| THE_HATER | 52.0% | 65.3% | **103.4x** | 173 |
| THE_CRITIC | ~10-12% | 3.4% | 5.5x | 4,988 |
| THE_SPAMMER | ~5% | 0.6% | ~1.0x | 324 |
| THE_SUPERFAN | low | low | <1x | 31,920 |
| THE_TAGGER | low | low | <1x | 19,357 |
| THE_CASUAL_COMPLIMENTER | low | low | <1x | 10,895 |

**THE_HATER is an extreme outlier.** 164 users generate comments that are 52% negative and 65.3% toxic — 103 times the population toxicity baseline. Despite their tiny size, they define the worst-case moderation scenario.

**THE_CRITIC is the operationally important dissatisfied-customer segment.** With 4,988 users and a 5.5x toxicity lift, this is a large enough population to act on. Their negativity concentrates on product targets with criticism intent. The Critic's conditional negative rate on product-directed comments is substantially higher than any other persona — but their product negativity only accounts for ~1% of all product comments because product comments are overwhelmingly positive overall.

**THE_SUPERFAN, TAGGER, and CASUAL_COMPLIMENTER are safe.** Together they represent 59% of labeled users and contribute negligible negativity.

### 3.3 Persona × lifecycle cluster crossing

When the window-level RFM cluster is crossed with persona:

- THE_HATER has no meaningful presence in the Brand advocates cluster — hostile users are not loyal commenters.
- THE_SUPERFAN and THE_CASUAL_COMPLIMENTER concentrate in Brand advocates and Expressive regulars.
- THE_CRITIC appears across all lifecycle states, confirming it is a stable behavioural trait, not a lifecycle phase.

---

## 4. RFM cluster lens on sentiment

### 4.1 User-level clusters (Mickey KMeans k=4 on RCEDTG)

At the user level (most recent window), the four clusters by comment volume:

| Cluster | Comment windows | Neg rate | Tox rate |
|---|---|---|---|
| Passive regulars | 50,602 | 4.9% | ~0.6% |
| Established regulars | 41,714 | — | — |
| Occasional Visitors | 39,981 | — | — |
| Brand advocates | 23,628 | — | — |

### 4.2 Window-level clusters (monthly lifecycle matrix, k=4)

The window-level matrix assigns a cluster per user per month, capturing lifecycle transitions. Cluster definitions:

- **C1 — Brand advocates**: high coverage, high engagement, low recency delay
- **C2 — Expressive regulars**: active, moderate coverage, relatively high gini
- **C3 — Passive regulars**: dominant cluster (69–72% of active user-months), low coverage
- **C4 — Delayed visitors**: arrive and comment late relative to post publication

Sentiment rates by window-level cluster, against population baselines (neg 5.04%, tox 0.63%):

| Cluster | Neg rate | vs baseline | Tox rate | vs baseline |
|---|---|---|---|---|
| Expressive regulars | **7.64%** | +2.6pp | 0.90% | +0.27pp |
| Delayed visitors | 7.31% | +2.3pp | **1.77%** | +1.14pp |
| Brand advocates | 4.66% | -0.4pp | 0.69% | +0.06pp |
| Passive regulars | **2.06%** | -3.0pp | **0.29%** | -0.34pp |

### Key RFM–sentiment insights

**Passive regulars are the most benign state.** Low engagement, low negativity, very low toxicity. They comment infrequently and don't invest emotional energy.

**Delayed visitors carry the highest toxicity** (2.8x Passive regulars). Users who discover and comment on posts long after publication appear more likely to arrive with a critical stance — possibly driven by algorithmic recommendations or external controversy surfacing old content.

**Expressive regulars are the most negative segment** by rate. They are active and emotionally engaged — both in positive and negative directions.

**Brand advocates are near the population baseline** on both metrics, confirming that loyal, high-coverage users are not a sentiment risk.

---

## 5. Post-level signals

### 5.1 Feed vs Reels

Feed posts account for 80.3% of posts. Reels (19.2%) tend to generate different sentiment and emotion distributions, though both are predominantly positive. The LLM vibe analysis (`room_vibe_instagram.parquet`) assigns post-level aggregated vibes; "appreciative" is the most common room vibe.

### 5.2 Sponsored vs organic

Sponsored posts (is_paid_partnership = True) show a modestly different negative and toxicity rate compared to organic posts. The direction is consistent with audience scepticism toward promotional content, though the magnitude is small given the rarity of sponsored posts in the dataset.

### 5.3 Hashtag density

Posts in higher hashtag-density quartiles do not show a clear monotonic relationship with negativity or toxicity. The relationship is non-linear.

### 5.4 Most toxic posts

The single most toxic post (ID 18066958285842484) is a VIDEO with 172 comments, 140 of which (81.4%) are toxic. This is an extreme outlier — the next most toxic post is 18.7%. This post warrants manual review.

Top-15 most toxic posts (by fraction, minimum 20 comments) are all organic, non-sponsored content, indicating that paid content is not the driver of community toxicity.

### 5.5 Negative rate vs post engagement

A regression of negative comment fraction on log(1 + likes) produces a small positive correlation (r ≈ 0.05–0.10, p < 0.05). Higher-engagement posts attract slightly more negative comments in absolute terms, but the fraction remains low. The relationship is weak and driven by a few outlier posts with very high negative fractions at moderate engagement levels.

---

## 6. Linguistic features of negative and toxic comments

### 6.1 Point-biserial correlations

| Feature | r vs negative | r vs toxic |
|---|---|---|
| word_count | **+0.200** | +0.040 |
| text_length | +0.197 | +0.039 |
| question_count | +0.059 | +0.021 |
| has_numbers | +0.062 | +0.010 |
| hashtag_count | +0.016 | +0.029 |
| emoji_count | **-0.068** | -0.025 |
| unique_emoji_count | **-0.079** | -0.041 |
| emoji_per_word_ratio | -0.065 | -0.023 |

### 6.2 Median feature values

| Feature | Overall | Negative | Toxic | Neg lift | Tox lift |
|---|---|---|---|---|---|
| word_count | 4 | 11 | 8 | +175% | +100% |
| text_length (chars) | 29 | 66 | 47 | +128% | +62% |
| emoji_count | 1 | 0 | 0 | -100% | -100% |
| unique_emoji_count | 1 | 0 | 0 | -100% | -100% |

### Key linguistic insights

**Word count is the single strongest predictor of negativity** (r = 0.20). Negative comments are 2.75x longer at the median; toxic comments are 2x longer. This is a practical moderation signal — short comments are almost never negative.

**Emoji use is a negativity shield.** The median negative comment contains zero emojis versus one for the overall population. Unique emoji diversity (r = -0.079) is the strongest single negative-direction correlate. A comment with multiple diverse emojis is structurally unlikely to be negative.

**Questions and number citations co-occur with negativity.** Users who ask questions or cite figures tend to be engaging critically. These are the Inquirer-to-Critic transition zone comments.

---

## 7. Cross-platform comparison: Instagram vs TikTok

### 7.1 Dataset overview

| Metric | Instagram | TikTok |
|---|---|---|
| User-window rows | 174,287 | 56,300 |
| Unique users | 18,752 | 13,249 |
| Rolling windows | 40 | 40 |
| Feature set | RCEDTG (6 features) | RCEGT (5 features) |
| Shared calendar | 2023-01 to 2026-03 | same |

The two user populations are disjoint (zero shared user_key). Cross-platform comparison is done at the window-aggregate level (40 shared monthly windows) and at the cluster-matrix level.

### 7.2 Lifecycle: the defining difference

| Metric | Instagram | TikTok |
|---|---|---|
| Mean active months | 9.0 | 4.3 |
| Median active months | 8 | 4 |
| Re-entry rate | **63.0%** | **6.4%** |
| Mean % of post-debut time active | 34% | 30% |
| Dominant cluster | C3 (69–72% of active user-months) | C3 (87% of active user-months) |

**IG users churn and return; TikTok users churn and leave.** The 63% vs 6.4% re-entry gap is the most important cross-platform finding. Instagram has an episodic engagement pattern: users go dormant and re-activate, making re-engagement campaigns viable. TikTok's audience is far more transient — once inactive, users overwhelmingly do not return.

This has direct implications for retention strategy. IG re-activation targeting has a large addressable base (the majority of inactive users are plausibly recoverable). TikTok acquisition must be continuous because reactivation is near-impossible.

### 7.3 Feature structure (within-platform)

**Instagram's strongest internal correlation: coverage × gini (Spearman ρ = 0.78).** Users who comment on more posts simultaneously concentrate those comments on fewer posts — a hub-and-spoke pattern. This is the dominant structural signal in the IG RFM space. TikTok's equivalent is much weaker (ρ = 0.23).

**VIF analysis** (variance inflation): coverage is the most collinear IG feature (VIF = 4.8), driven by its relationship with gini. No feature exceeds VIF = 5, so multicollinearity is not a modelling obstacle.

**Delay is unique to Instagram** and is extremely heavy-tailed (skew = 21). The median delay between post and comment is 2.3 days but the mean is 76.7 days, driven by long-tail users who comment weeks or months after publication.

### 7.4 Feature distributions: IG vs TikTok

All four common features (recency, coverage, engagement, gini) show statistically significant distributional differences (KS test p ≈ 0):

| Feature | IG median | TK median | Direction |
|---|---|---|---|
| recency | 0.414 | 0.452 | TK commenters are slightly more recent |
| coverage | 0.023 | 0.040 | TK users cover more posts per window |
| engagement | 6.0 | 5.0 | IG slightly higher engagement per comment |
| gini | 0.0 | 0.0 | Both dominated by single-comment users |

### 7.5 Cross-platform co-movement (window aggregates)

Raw window-aggregate Spearman correlations (strongest pairs):

| IG feature | TK feature | ρ |
|---|---|---|
| IG_delay | TK_engagement | **0.735** |
| IG_delay | TK_tenure_binary | 0.674 |
| IG_gini | TK_coverage | -0.655 |
| IG_engagement | TK_engagement | 0.626 |
| IG_coverage | TK_coverage | 0.624 |

After detrending (first differences, removing the shared growth trend), only **coverage** survives with statistical significance (ρ = 0.332, p = 0.039). All other cross-platform correlations are primarily driven by a common temporal trend, not genuine co-movement of audience behaviour shocks.

**The practical read:** the two audiences broadly grow together over time (same macro content cycles and seasonal patterns), but their month-to-month behavioural fluctuations are largely independent. The one genuine co-movement signal is posting breadth (coverage) — when IG users comment more broadly, TK users do too, suggesting shared external events driving content discovery.

### 7.6 Cluster composition and cross-platform cluster correlation

Cluster C3 (the passive, broad-but-shallow commenter) dominates both platforms but more completely on TikTok (87%) than Instagram (70%). Instagram has a richer multi-cluster structure, with C1 (Brand advocates) and C2 (Expressive regulars) representing 5–22% of active user-months depending on the period.

Cross-platform cluster share correlations (raw Spearman, n = 39 months):

- IG_C1 × TK_C1: ρ = 0.341 (p = 0.033) — the brand-advocate tier moves together
- IG_C3 × TK_C1: ρ = -0.354 (p = 0.027) — when IG passive share grows, TK advocate share shrinks

After detrending, none of the cluster-share correlations remain significant. **The cluster compositions are temporally co-trending but not genuinely co-moving in their shocks.** Each platform's cluster dynamics are internally driven.

---

## 8. Room vibe analysis (post-level)

The `room_vibe_instagram.parquet` table (1,501 posts) aggregates comment-level signals into post-level vibes via a secondary LLM pass.

- **Most common LLM vibe**: "appreciative" — the dominant room state across the majority of posts.
- **Vibe score** (0–1, higher = more positive): mean ≈ 0.60, indicating a generally warm comment room.
- **Toxicity rate at post level**: most posts have toxicity_rate = 0.0. The distribution is extremely right-skewed.
- **Room state "united"**: posts where the audience coalesces around a single positive stance. These typically have high llm_consensus scores (0.85–0.90).
- **Room state "mixed"**: posts where sentiment is split. These correlate with slightly higher controversy scores.

The LLM vibe is consistent with comment-level sentiment — "appreciative" rooms have overwhelmingly positive comment-level sentiment. The room analysis adds value primarily for the dominant_stance and split_axis fields, which give narrative context beyond bare sentiment fractions.

---

## 9. Summary of actionable insights

### Moderation priorities
1. **One outlier post** (ID 18066958285842484) with 81.4% toxicity requires immediate review.
2. **Off-topic comments** are the highest-toxicity category by rate — filtering non-topical comments would remove the toxicity hotspot with minimal collateral damage to genuine engagement.
3. **THE_HATER (164 users)** can be flagged for pre-moderation. Their comment pattern is identifiable by high word count, zero emojis, anger emotion, and criticism intent.

### Community health signals
4. **Emoji presence is a strong positive signal** — content that elicits emoji-heavy responses is at negligible toxicity risk.
5. **Long comments (>15 words) warrant a second look** — they are 3x more likely to be negative. This is a lightweight real-time moderation heuristic.
6. **THE_CRITIC targeting products** is an early warning system for product dissatisfaction — monitoring this segment's negativity rate over time tracks brand reception without requiring a survey.

### Audience strategy
7. **IG re-engagement has a large addressable base.** 63% of inactive users historically re-activate. A re-engagement campaign targeting users in their last active month has the highest probability of success.
8. **Delayed visitors need early intervention.** Their high toxicity and negativity rates suggest that late-arriving audiences (surfaced by the algorithm weeks post-publication) arrive without the warm community context of early commenters. Limiting comment windows on older posts may reduce this segment's influence.
9. **Passive regulars are volume, not value.** They represent 70% of active user-months but contribute minimal sentiment signal and low engagement. Strategy should invest in upgrading Passive regulars to Expressive regulars rather than optimising for raw active user counts.
10. **Cross-platform content cycles are real but audience behaviours are independent.** IG and TK audiences respond to the same macro events (their coverage co-moves after detrending), but their cluster dynamics are platform-specific. Strategies calibrated to IG lifecycle patterns should not be assumed to transfer to TikTok.
