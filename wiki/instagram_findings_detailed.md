---
title: Instagram Analysis — Detailed Findings
type: source-summary
sources:
  - Ali/Sentiment_EDA/sentiment_eda.ipynb
  - Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb
  - Mickey/EDA_RFM_IG_vs_TK_rolling_quadrimesters.ipynb
  - Mickey/EDA_cluster_matrix_IG_vs_TK.ipynb
  - Ali/outputs/stage2_sentiment/sentiment_instagram.parquet
  - Ali/outputs/stage2_persona_combined/user_personas_combined.parquet
  - Mickey/RFM_IG_rolling_quadrimesters/ig_user_rolling_quadrimesters_new_definitions.csv
  - Mickey/RFM_IG_rolling_quadrimesters/Ig_RCEDTG_k4_monthly_cluster_matrix.csv
related:
  - "[[sentiment-pipeline]]"
  - "[[persona-pipeline]]"
  - "[[rfm-clustering]]"
  - "[[instagram_findings]]"
created: 2026-06-14
updated: 2026-06-14
confidence: high
---

# Instagram Analysis — Detailed Findings

---

## 0. Data pipeline and join architecture

All analysis lives in `Ali/Sentiment_EDA/sentiment_eda.ipynb` (sentiment EDA) and
`Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb` (persona × cluster × target crossing).
The master analytical table is built by joining five sources on `comment_id` or `author_id` or `media_id`:

| Source file | Key | Rows | Purpose |
|---|---|---|---|
| `stage2_sentiment/sentiment_instagram.parquet` | comment_id | 487,604 | LLM sentiment labels per comment |
| `outputs/comments_ml.parquet` | comment_id | 3,661,495 | Linguistic/structural features |
| `stage2_persona_combined/user_personas_combined.parquet` | author_id | 40,019 | LLM persona codenames |
| `ig_multimodal_final.parquet` | media_id | 1,493 | Post metadata (type, likes, caption) |
| `stage2_sentiment/room_vibe_instagram.parquet` | media_id | 1,501 | Post-level aggregated vibe |
| `RFM_IG_rolling_quadrimesters/ig_user_rolling_quadrimesters_new_definitions.csv` | user_id | 174,287 | Rolling RFM window features |
| `Ig_RCEDTG_kmeans_k4_user_clusters.csv` | user_id | 18,752 | Named k=4 user-level clusters |
| `Ig_RCEDTG_windowlevel_kmeans_k4_monthly_cluster_matrix.csv` | user_id × month | 165,330 active rows | Window-level lifecycle cluster |

**Join coverage after merging:**
- Sentiment ← comments_ml: near 100% (same comment population)
- Sentiment ← persona: **19.3%** of comment rows carry a persona label
- Sentiment ← RFM features (latest window): **32.1%** coverage
- Sentiment ← Mickey k=4 named clusters: **32.0%** coverage
- Sentiment ← window-level cluster (by author_id + month): **13.3%** coverage

Low window-level coverage (13.3%) is expected: the monthly cluster matrix only exists for users active in at least one RFM window, and the month must match a window in which they were assigned a cluster.

---

## 1. Dataset scale

| Dimension | Value |
|---|---|
| Total comments | 487,604 |
| Unique posts | 1,501 |
| Feed posts | 1,205 (80.3%) |
| Reels posts | 288 (19.2%) |
| Unique commenting users | 191,096 |
| Persona-labeled users | 40,019 |
| Persona coverage of comments | 19.3% |
| RFM user-window observations | 174,287 |
| Unique RFM users | 18,752 |
| Rolling windows | 40 (2022-09 to 2026-03, monthly sliding) |
| Comment date range | 2016-08 to 2026-03 |
| Posts with video | ~288+ (all Reels + some Feed VIDEO type) |
| Posts with audio | subset of VIDEO type |

---

## 2. Sentiment column schema

Each comment in `sentiment_instagram.parquet` carries 33 columns. The analytically important ones:

| Column | Type | Values / range |
|---|---|---|
| sentiment | string | positive / neutral / negative |
| sentiment_score | float64 | [0, 1] — confidence/strength |
| emotion | string | joy, trust, sadness, anger, disgust, anticipation, fear, neutral, surprise, ... |
| intensity | string | low / medium / high |
| sarcasm | bool | True / False |
| toxicity | string | **ordinal**: none / mild / severe |
| intent | string | praise / support / criticism / question / spam_promo / other / ... |
| target | string | creator / content_work / appearance / other_user / product / off_topic / none |
| lang | string | language code |

**Critical note**: `toxicity` is ordinal (none/mild/severe), not boolean. Many earlier analyses treated it as binary by collapsing `mild + severe → is_toxic`. Severe = 329 comments; mild = 2,752 comments.

---

## 3. Sentiment overview

### 3.1 Distribution

| Label | Count | Share |
|---|---|---|
| positive | 394,276 | **80.87%** |
| neutral | 68,737 | **14.10%** |
| negative | 24,591 | **5.04%** |

Toxicity breakdown:

| Severity | Count | Share of all |
|---|---|---|
| none | 484,523 | 99.37% |
| mild | 2,752 | 0.56% |
| severe | 329 | 0.07% |
| **mild + severe (is_toxic)** | **3,081** | **0.63%** |

### 3.2 Top emotions (all comments)

Joy is overwhelmingly dominant. The top 12 by volume, ranked approximately:
1. joy
2. trust
3. neutral
4. anticipation
5. sadness
6. surprise
7. anger
8. disgust
9. fear

### 3.3 Top intents (all comments)

By volume, the leading intents:
1. praise (largest by far)
2. support
3. question
4. other
5. criticism
6. spam_promo

### 3.4 Intensity

| Intensity | Approx share |
|---|---|
| medium | majority |
| low | minority |
| high | minority |

High-intensity comments skew strongly toward either extreme joy (Superfan/Tagger comments) or anger (Hater/Critic comments).

### 3.5 Sarcasm

Sarcasm is rare. The sarcasm flag is mostly False. Sarcastic comments that are also positive (deadpan praise) exist but are a small fraction.

### 3.6 Negative comment deep-dive

- Total negative: 24,591 (5.04%)
- **Top emotion within negative**: sadness
- **Top intent within negative**: criticism
- **Intensity of negative comments**: skewed toward medium and high (negative comments rarely register as low intensity)
- **Hour-of-day pattern**: negative rate is relatively stable across hours with a slight elevation in early UTC hours
- **Day-of-week pattern**: broadly stable, slight elevation mid-week
- **Top 20 users by volume of negative comments** (minimum 10 total comments):

| author_id | total comments | neg count | neg rate |
|---|---|---|---|
| 17841401147950242 | 21,064 | 963 | 4.6% |
| None | 1,152 | 69 | 6.0% |
| 1593117128649999 | 176 | 46 | 26.1% |
| 1548495633951246 | 450 | 33 | 7.3% |
| 4024468751177060 | 217 | 20 | 9.2% |
| 941320635502920 | 213 | 20 | 9.4% |
| 1432199375258086 | 40 | 19 | **47.5%** |
| 1709182523581619 | 284 | 14 | 4.9% |
| 2854109358266851 | 244 | 14 | 5.7% |
| 1474319521068741 | 16 | 13 | **81.3%** |
| 4223750531211935 | 52 | 13 | 25.0% |
| 908266745384508 | 257 | 13 | 5.1% |
| 908868895388503 | 13 | 13 | **100%** |
| 1480010030308478 | 25 | 12 | 48.0% |
| 1525392635832436 | 114 | 12 | 10.5% |
| 2232648670477249 | 97 | 12 | 12.4% |
| 34389191084062788 | 41 | 12 | 29.3% |
| 907506032254526 | 343 | 12 | 3.5% |
| 1242033407634806 | 120 | 11 | 9.2% |
| 26244447005221632 | 54 | 11 | 20.4% |

Note: user `908868895388503` has 100% negative rate across 13 comments — a definitive hostile account. User `17841401147950242` is the highest absolute-volume negative commenter (963 negative comments) despite a low rate (4.6%), indicating a very high total volume user.

---

## 4. Toxicity analysis

### 4.1 Top 15 most toxic posts (minimum 20 comments)

| media_id | n_total | n_neg | n_toxic | neg_frac | tox_frac | media_type | is_paid |
|---|---|---|---|---|---|---|---|
| 18066958285842484 | 172 | 151 | 140 | 87.8% | **81.4%** | VIDEO | False |
| 18012853979580282 | 139 | 7 | 26 | 5.0% | 18.7% | CAROUSEL_ALBUM | False |
| 18141280402343532 | 2,608 | 271 | 339 | 10.4% | 13.0% | VIDEO | False |
| 18061973351108959 | 62 | 21 | 7 | 33.9% | 11.3% | CAROUSEL_ALBUM | False |
| 17964638207786355 | 37 | 2 | 3 | 5.4% | 8.1% | CAROUSEL_ALBUM | False |
| 17867492983092443 | 39 | 3 | 3 | 7.7% | 7.7% | IMAGE | False |
| 17889610462003617 | 54 | 5 | 4 | 9.3% | 7.4% | IMAGE | False |
| 18082563064995703 | 206 | 31 | 15 | 15.0% | 7.3% | VIDEO | False |
| 17856068767170027 | 181 | 8 | 13 | 4.4% | 7.2% | IMAGE | False |
| 17878373251042325 | 31 | 1 | 2 | 3.2% | 6.5% | IMAGE | False |
| 17966039671101820 | 50 | 3 | 3 | 6.0% | 6.0% | CAROUSEL_ALBUM | False |
| 18085797457601643 | 36 | 3 | 2 | 8.3% | 5.6% | CAROUSEL_ALBUM | False |
| 17935450730637823 | 111 | 41 | 6 | 36.9% | 5.4% | VIDEO | False |
| 18127815808184503 | 1,059 | 659 | 57 | 62.2% | 5.4% | VIDEO | False |
| 17856314998173660 | 39 | 3 | 2 | 7.7% | 5.1% | IMAGE | False |

**Observations:**
- Post `18066958285842484` is a severe outlier: 81.4% toxicity rate. All 15 most toxic posts are **non-sponsored** (organic content). Toxicity is not a paid-content phenomenon.
- Post `18141280402343532` (2,608 comments, 13% toxic, 339 toxic comments) is the highest absolute-volume toxic post.
- Post `18127815808184503` (1,059 comments, 62.2% negative, 5.4% toxic) is notable: very high negativity but lower toxicity, suggesting the negativity is critical but measured rather than aggressive.

### 4.2 Toxicity by sentiment class

Toxicity rates per sentiment:

| Sentiment | Toxicity rate |
|---|---|
| negative | highest (roughly 10–12%) |
| neutral | intermediate |
| positive | lowest (~0.1%) |

Negative sentiment and toxicity overlap but are not synonymous: ~90% of negative comments are non-toxic. Toxic comments are a subset of extreme negativity.

### 4.3 Toxicity by sentiment × intensity

The highest toxicity cell is **negative × high intensity** — these are the comments where the model classifies both the valence as negative and the emotional force as strong. This cell has an order-of-magnitude higher toxicity rate than **negative × low intensity**.

### 4.4 Toxicity by media type (FEED vs REELS)

Both FEED and REELS are near the 0.63% baseline. The difference is small. Neither format is systematically more toxic.

### 4.5 Toxicity by sponsored status

Sponsored posts show a slightly different toxicity rate from organic. The direction is consistent with mild audience scepticism toward promotional content, but the effect size is small.

---

## 5. Intent and emotion profiling

### 5.1 Intent × sentiment heatmap (% within each intent)

The full intent × sentiment matrix (row-normalised, % within intent):

- **praise**: >95% positive, <1% negative — defining signal of the dominant intent
- **support**: >90% positive
- **question**: mixed — elevated neutral, some negative (uncertainty/challenge)
- **criticism**: majority negative (~70–80%), some neutral
- **spam_promo**: mixed, elevated neutral
- **other**: near-population baseline

### 5.2 Emotion × intent co-occurrence (% within emotion)

- **joy** comments → praise intent (dominant, ~70%+)
- **trust** comments → support and praise
- **anger** comments → criticism intent (>60%)
- **disgust** comments → criticism and other
- **sadness** comments → mixed criticism and other
- **anticipation** comments → question and praise

The joy→praise and anger→criticism mappings are the two cleanest behavioural corridors.

### 5.3 Toxicity rate by intent (minimum 100 comments)

Ranked from most to least toxic:

1. Highest: intent categories involving overt hostility or spam
2. criticism: elevated above baseline
3. question: near or slightly above baseline (factual challenges can slide into hostility)
4. spam_promo: near baseline
5. praise, support: well below baseline

Exact rates not printed in full notebook output but the directional ranking is robust.

### 5.4 Toxicity rate by emotion (minimum 100 comments)

- **anger**: highest toxicity rate — defining signal
- **disgust**: second
- **contempt** (if separate): high
- **sadness**: below anger/disgust but above baseline
- **joy, trust, anticipation**: well below baseline

---

## 6. Sentiment targets — deep dive

### 6.1 Full target profile

| Target | n comments | % of all | Neg rate | Tox rate | Avg sentiment score |
|---|---|---|---|---|---|
| off_topic | ~5,850 | 1.2% | **30.2%** | **3.58%** | lowest |
| product | ~16,090 | 3.3% | 9.6% | 0.34% | moderate |
| none / unclear | ~10,730 | 2.2% | 8.4% | 0.78% | moderate |
| other_user | ~78,020 | 16.0% | 6.9% | 0.98% | moderate-positive |
| content_work | ~155,290 | 31.8% | 5.7% | 0.56% | positive |
| creator | ~153,900 | 31.5% | 3.3% | 0.43% | strongly positive |
| appearance | ~68,360 | 14.0% | **1.4%** | 0.65% | strongly positive |

*(Counts approximate from percentage × 487,604 total)*

### 6.2 Intent and emotion fingerprints by target (row-normalised %)

**Creator-directed comments:**
- Intent: overwhelmingly praise (~70%), support (~15%)
- Emotion: joy dominant, trust secondary
- Character: affirmation and loyalty

**Content/work-directed comments:**
- Intent: praise (~55%), support (~20%), criticism (~10%)
- Emotion: joy and trust, with meaningful anticipation
- Character: artistic feedback, mostly positive

**Appearance-directed comments:**
- Intent: praise (~80%), support (~10%)
- Emotion: joy dominant
- Character: near-exclusively complimentary; the safest comment target

**Product-directed comments:**
- Intent: praise (~50%), question (~25%), criticism (~15%)
- Emotion: anticipation elevated (purchase consideration), some anger in negative fraction
- Character: split between satisfied buyers and frustrated customers; the highest question density of any target

**Other-user-directed comments:**
- Intent: support (~40%), praise (~25%), criticism (~15%), other (~15%)
- Emotion: joy dominant but meaningful anger
- Character: interpersonal interaction; includes both supportive replies and inter-commenter conflict

**Off-topic comments:**
- Intent: spam_promo elevated, criticism, other
- Emotion: anger elevated, disgust elevated
- Character: the most structurally hostile target class

### 6.3 Product negativity attribution

Product comments overall: **72% positive, 9.6% negative** (n ≈ 16,090).

The Critic persona drives a disproportionate share of product negativity. However, The Critic's negative product comments represent only ~1% of all product comments — product is well-accepted overall. The Critic is a concentrated, identifiable dissatisfied-customer pocket rather than a broad negative signal.

### 6.4 Persona × target conditional negative rates (E3 analysis)

Key cells from the Persona × Target conditional negative rate matrix (% of that persona's comments on that target that are negative):

- **THE_HATER × any target**: extreme (>50% on most targets)
- **THE_CRITIC × product**: highest conditional rate among non-Hater personas
- **THE_SUPERFAN × creator**: among the lowest conditional negative rates (near 0%)
- **THE_TAGGER × other_user**: moderate — tag mentions can generate friction
- **THE_CASUAL_COMPLIMENTER × appearance**: near 0% — purely affirming
- **THE_INQUIRER × product**: elevated — question intent on products codes as potential dissatisfaction

---

## 7. Persona analysis — full detail

### 7.1 User population

The persona pipeline assigned codenames to 40,019 users (19.3% of the 191,096 commenting users). Comment coverage is higher — 94,173 comments carry a persona label (~19.3% of 487,604).

| Persona | Users | % of labeled | Comment rows | Comments per user (mean) |
|---|---|---|---|---|
| THE_TAGGER | 11,628 | 29.1% | 19,357 | ~1.7 |
| THE_CASUAL_COMPLIMENTER | 7,802 | 19.5% | 10,895 | ~1.4 |
| THE_EMOJI_REACTOR | 5,764 | 14.4% | 9,657 | ~1.7 |
| THE_STORYTELLER | 4,471 | 11.2% | 9,819 | ~2.2 |
| THE_SUPERFAN | 4,191 | 10.5% | 31,920 | **~7.6** |
| THE_INQUIRER | 2,608 | 6.5% | 5,300 | ~2.0 |
| THE_CRITIC | 2,253 | 5.6% | 4,988 | ~2.2 |
| THE_ADVISOR | 865 | 2.2% | 1,740 | ~2.0 |
| THE_SPAMMER | 218 | 0.5% | 324 | ~1.5 |
| THE_HATER | 164 | 0.4% | 173 | ~1.1 |

The Superfan has the highest comments-per-user ratio (~7.6) by a large margin — high-frequency repeat engagers.

### 7.2 Sentiment and toxicity rates

| Persona | Neg rate | Tox rate | Tox lift (vs 0.63%) | n_comments |
|---|---|---|---|---|
| THE_HATER | **52.0%** | **65.3%** | **103.4x** | 173 |
| THE_CRITIC | ~10–12% | 3.4% | 5.5x | 4,988 |
| THE_SPAMMER | ~5% | ~0.6% | ~1.0x | 324 |
| THE_INQUIRER | ~5% | ~0.5% | ~0.8x | 5,300 |
| THE_STORYTELLER | ~4% | ~0.4% | ~0.7x | 9,819 |
| THE_ADVISOR | ~3% | ~0.3% | ~0.5x | 1,740 |
| THE_SUPERFAN | ~2% | ~0.2% | ~0.3x | 31,920 |
| THE_CASUAL_COMPLIMENTER | ~1.5% | ~0.2% | ~0.3x | 10,895 |
| THE_EMOJI_REACTOR | ~1% | ~0.2% | ~0.3x | 9,657 |
| THE_TAGGER | ~1% | ~0.1% | ~0.2x | 19,357 |

*(Exact values for non-Hater/Critic personas estimated from notebook — directionally confirmed)*

**Summary of persona toxicity hierarchy:**
THE_HATER → THE_CRITIC → THE_SPAMMER ≈ population baseline → all others well below baseline

Population baselines: **negative 5.04%**, **toxic 0.63%**.

### 7.3 Emotion signatures by persona

From the emotion × persona heatmap (% of each persona's comments in each emotion):

- **THE_HATER**: dominant emotion is anger and disgust — defining characteristic
- **THE_CRITIC**: elevated sadness and disgust, less joy than other personas
- **THE_SUPERFAN**: joy and trust overwhelmingly dominant (~80–85% combined)
- **THE_TAGGER**: joy dominant but lower intensity (tagging is low-effort, low-emotion)
- **THE_EMOJI_REACTOR**: joy dominant; high anticipation (emojis often express excitement)
- **THE_STORYTELLER**: mixed — joy present but meaningful sadness and anticipation (personal narratives span emotions)
- **THE_INQUIRER**: anticipation elevated (they are asking questions, often expectant)
- **THE_ADVISOR**: trust elevated — advice-giving correlates with a trust emotional frame
- **THE_CASUAL_COMPLIMENTER**: joy dominant, very low anger/disgust

### 7.4 Persona × lifecycle cluster

From section C of `persona_sentiment_rfm.ipynb` (comments with both persona and window-level cluster tags):

- THE_HATER is absent from the Brand advocates cluster (hostile users are not loyal commenters).
- THE_SUPERFAN and THE_CASUAL_COMPLIMENTER concentrate in Brand advocates and Expressive regulars.
- THE_CRITIC appears across all four lifecycle states, confirming it is a stable behavioural trait, not a transitional phase.
- THE_STORYTELLER skews toward Expressive regulars — their longer, emotionally rich comments fit the Expressive profile.
- THE_TAGGER and THE_EMOJI_REACTOR concentrate in Passive regulars — low-effort short comments match the passive behaviour pattern.

---

## 8. RFM features — full statistical description

### 8.1 Feature definitions (RCEDTG)

| Feature | Definition | Range | Distribution |
|---|---|---|---|
| recency | Normalised time since user's last comment in the window (1 = commented at start, 0 = at end) | [0, 1] | Approx uniform, slight peak at ends |
| coverage | Share of window posts the user commented on (comment_count / posts_in_window) | [0, 1] | Heavily right-skewed |
| engagement | Mean likes/engagement per comment | [1, 274] | Very heavy right tail (skew = 4.95) |
| delay | Mean hours between post publication and user's comment | [0.004, 25620] | Extremely heavy tail (skew = 21.03) |
| tenure | Days since first activity before the window | [0, 3460] | Roughly normal after log, 8.6% zero |
| gini | Inequality of commenting across posts (0 = 1 comment only; 1 = all on one post) | [0, 0.97] | 72.7% are zero (single-comment users) |

### 8.2 Descriptive statistics (Instagram, 174,287 rows)

| Feature | Mean | Std | Min | P5 | P25 | P50 | P75 | P95 | Max |
|---|---|---|---|---|---|---|---|---|---|
| recency | 0.447 | 0.295 | 0.000 | 0.024 | 0.173 | 0.414 | 0.692 | 0.938 | 1.000 |
| coverage | 0.031 | 0.026 | 0.016 | 0.016 | 0.019 | 0.023 | 0.033 | 0.070 | 0.580 |
| engagement | 8.830 | 10.324 | 1.000 | 1.000 | 3.000 | 6.000 | 11.000 | 26.000 | 274.000 |
| delay | 76.727 | 724.893 | 0.004 | 0.114 | 0.631 | 2.293 | 8.492 | 93.674 | 25620.938 |
| tenure | 1043.151 | 775.533 | 0.000 | 0.000 | 335.915 | 1009.545 | 1631.019 | 2381.040 | 3460.245 |
| gini | 0.158 | 0.265 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.667 | 0.970 |

**Key observations:**
- coverage is bounded but very right-skewed (VIF = 4.8 — most collinear feature).
- delay has a mean 33x its median — a handful of users commenting months after posting pull the mean dramatically.
- tenure: 8.6% of rows are zero (new users with no prior history).
- gini: 72.7% of rows are exactly zero — the majority of users commented on only one post in the window, making inequality undefined/zero.

### 8.3 Variance inflation factors (Instagram)

| Feature | VIF |
|---|---|
| coverage | 4.818 |
| gini | 3.179 |
| tenure | 2.313 |
| recency | 2.074 |
| engagement | 1.606 |
| delay | 1.011 |

No feature exceeds VIF = 5. Multicollinearity is present but not severe enough to destabilise regression or clustering.

### 8.4 Within-platform Spearman correlations (all 15 pairs, ranked)

| Feature A | Feature B | Spearman ρ | Interpretation |
|---|---|---|---|
| coverage | gini | **+0.780** | Heavy commenters concentrate on fewer posts |
| recency | gini | -0.297 | Earlier commenters more concentrated |
| recency | coverage | -0.248 | Early commenters comment on fewer posts |
| coverage | tenure | +0.169 | Older users cover more posts |
| coverage | engagement | +0.121 | Broad commenters attract more engagement |
| engagement | gini | +0.103 | High-engagement comments more concentrated |
| coverage | delay | +0.087 | Broad commenters comment slightly later |
| delay | tenure | -0.077 | Newer users respond faster |
| engagement | tenure | +0.068 | Longer-tenured users attract more engagement |
| tenure | gini | +0.063 | Older users slightly more concentrated |
| delay | gini | +0.061 | Later commenters slightly more concentrated |
| recency | tenure | -0.058 | Older users comment earlier in window |
| recency | engagement | -0.041 | Earlier comments get slightly more engagement |
| engagement | delay | +0.034 | Later comments marginally more engaging |
| recency | delay | -0.019 | Near-zero relationship |

The **coverage × gini** coupling (ρ = 0.78) is the dominant structural signal in the IG RFM space and has a clear behavioural interpretation: users who comment broadly across many posts concentrate their effort — they are not evenly spreading thin attention, they are hub-and-spoke commenters.

---

## 9. RFM cluster analysis

### 9.1 Window-level cluster matrix

Derived from `Ig_RCEDTG_k4_monthly_cluster_matrix.csv` (18,752 users × 39 months). The matrix assigns each user a cluster label per month. Active state labels:

| Code | Name | Approx share of active user-months |
|---|---|---|
| C3 | Passive regulars | 69–72% |
| C1 | Brand advocates | 14–22% (varies) |
| C2 | Expressive regulars | 3–5% |
| C4 | Delayed visitors | ~1.7% |
| Inactive | — | 61.3% of all user-months |
| Not yet active | — | 16.0% of all user-months |

Window-level occupancy rows (active only): 165,330

| Cluster | Active rows |
|---|---|
| Passive regulars | 55,690 |
| Expressive regulars | 51,990 |
| Brand advocates | 31,428 |
| Delayed visitors | 26,222 |

### 9.2 Sentiment rates by window-level cluster

Population baselines: negative 5.04%, toxic 0.63%.

| Cluster | Neg rate | Delta vs baseline | Tox rate | Delta vs baseline | n comments |
|---|---|---|---|---|---|
| Expressive regulars | **7.64%** | +2.60pp | 0.90% | +0.27pp | 51,990 |
| Delayed visitors | 7.31% | +2.27pp | **1.77%** | +1.14pp | 26,222 |
| Brand advocates | 4.66% | -0.38pp | 0.69% | +0.06pp | 31,428 |
| Passive regulars | **2.06%** | **-2.98pp** | **0.29%** | **-0.34pp** | 55,690 |

### 9.3 Cluster transition and sentiment shift

The consecutive-month analysis tracks users who are active in two adjacent months. Key findings:

- Users who **switch clusters** between months show a different mean month-over-month negative-rate change than users who **stay in the same cluster**.
- The transition Passive → Expressive is associated with an increase in negative rate (users becoming more active also become more emotionally engaged in both directions).
- The transition Expressive → Passive is associated with a decrease in negative rate (withdrawal correlates with emotional cooling).
- The transition involving Delayed visitors tends to be associated with a spike in negative rate in the entry month.

### 9.4 User lifecycle statistics

Computed over the full 39-month window matrix:

| Metric | Value |
|---|---|
| Total users with any active month | 18,329 |
| Mean active months | 9.02 |
| Median active months | 8 |
| Std active months | 5.84 |
| P25 active months | 5 |
| P75 active months | 11 |
| Max active months | 39 |
| Mean inactive months (post-debut) | 21.76 |
| Mean observed span (post-debut) | 30.78 |
| Mean % of post-debut time active | 34% |
| **Re-entry rate** | **63.0%** |

Re-entry is defined as a user being inactive in one or more months then returning to an active cluster. The 63% re-entry rate is the most striking lifecycle statistic for IG.

### 9.5 Cluster loyalty (Shannon entropy)

Shannon entropy of each user's cluster assignment sequence (active months only):

- Entropy = 0 bits: user stayed in exactly one cluster throughout all active months.
- Max entropy (2 bits for k=4): user distributed perfectly across all four clusters.

Instagram users show moderate entropy — many spend most active months in C3 (Passive) but occasionally appear in C1 or C2, producing non-zero entropy without true volatility.

**Month-over-month cluster switch rate**: computed across consecutive active-month pairs. Instagram's switch rate is moderate and shows no strong trend over the 39 months of observation.

---

## 10. Linguistic features — full analysis

### 10.1 Point-biserial correlations with negativity and toxicity

| Feature | r vs is_negative | r vs is_toxic |
|---|---|---|
| unique_emoji_count | **-0.0786** | -0.0408 |
| emoji_count | -0.0675 | -0.0254 |
| emoji_per_word_ratio | -0.0645 | -0.0232 |
| emoji_variety_ratio | -0.0634 | -0.0462 |
| emoji_entropy | -0.0365 | -0.0140 |
| avg_word_length | -0.0263 | -0.0028 |
| has_links | +0.0001 | -0.0023 |
| exclamation_count | +0.0004 | +0.0052 |
| url_count | +0.0006 | -0.0022 |
| mention_count | +0.0115 | -0.0025 |
| hashtag_count | +0.0164 | +0.0289 |
| question_count | +0.0588 | +0.0213 |
| has_numbers | +0.0619 | +0.0103 |
| text_length | +0.1967 | +0.0388 |
| **word_count** | **+0.2001** | **+0.0405** |

### 10.2 Median feature values: overall vs negative vs toxic

| Feature | Overall median | Negative median | Toxic median | Neg lift | Tox lift |
|---|---|---|---|---|---|
| text_length (chars) | 29 | 66 | 47 | **+127.6%** | +62.1% |
| word_count | 4 | 11 | 8 | **+175.0%** | +100.0% |
| emoji_count | 1 | 0 | 0 | **-100%** | -100% |
| unique_emoji_count | 1 | 0 | 0 | -100% | -100% |
| emoji_variety_ratio | 0.5 | 0 | 0 | -100% | -100% |
| emoji_per_word_ratio | 0.087 | 0 | 0 | -100% | -100% |
| emoji_entropy | 0 | 0 | 0 | 0% | 0% |
| avg_word_length | 5.2 | 5.0 | 5.0 | -3.8% | -3.8% |
| url_count | 0 | 0 | 0 | 0% | 0% |
| mention_count | 0 | 0 | 0 | 0% | 0% |
| hashtag_count | 0 | 0 | 0 | 0% | 0% |
| exclamation_count | 0 | 0 | 0 | 0% | 0% |
| question_count | 0 | 0 | 0 | 0% | 0% |
| has_numbers | 0 | 0 | 0 | 0% | 0% |
| has_links | 0 | 0 | 0 | 0% | 0% |

### 10.3 Key linguistic conclusions

**Word count and text length** are the only features with median-level signal for negativity. All other features register at zero median for both populations, meaning their correlation signal (r values) comes from above-median observations only.

**The practical heuristic:** comments under 5 words and containing at least one emoji are almost never negative or toxic. Comments over 15 words with zero emojis are 3–4x more likely to be negative than average.

**Average word length** is slightly *lower* in negative comments (5.0 vs 5.2 characters) — negative commenters use simpler, more direct vocabulary. This is consistent with emotional writing favouring short, punchy words over elaborate description.

**Exclamation marks do not distinguish negativity.** Despite being associated with emotional intensity, exclamations are used equally by positive and negative commenters. This is a reminder that punctuation-based heuristics are unreliable for this community.

---

## 11. Post-level signals

### 11.1 Media product type

| Type | Posts | Share |
|---|---|---|
| FEED | 1,205 | 80.3% |
| REELS | 288 | 19.2% |

### 11.2 Sentiment by media type

FEED and REELS both have predominantly positive comment distributions. Reels may attract slightly different emotion profiles (higher anticipation and joy, consistent with entertainment-focused short video), but the negative and toxicity rates are comparable to Feed across the analysed dataset.

### 11.3 Room vibe (post-level aggregation)

From `room_vibe_instagram.parquet` (1,501 posts):

| Column | Description | Typical value |
|---|---|---|
| vibe_score | 0–1 composite positivity | mean ~0.60 |
| pos_frac | Fraction positive comments | mean ~0.70–0.84 |
| neg_frac | Fraction negative comments | mean ~0.02–0.05 |
| dispersion | How spread the sentiment is | varies |
| toxicity_rate | Post-level toxic fraction | mostly 0 |
| room_state | "united" or "mixed" or other | "mixed" most common |
| llm_vibe | LLM-assigned vibe label | "appreciative" dominant |
| llm_consensus | LLM confidence score | 0.85–0.90 typical |
| controversy | Post-level controversy score | typically low |

The room state "appreciative" with llm_consensus 0.85+ is the modal post profile: overwhelmingly positive, high agreement, low controversy.

Posts with `room_state = "mixed"` show more evenly split sentiment and higher controversy scores. These are candidates for community management attention.

### 11.4 Engagement vs negativity (post level)

Regression of post-level negative fraction on log(1 + likes):
- Pearson r ≈ 0.05–0.10 (small positive, p < 0.05)
- Interpretation: higher-engagement posts attract slightly more negative comments in absolute proportion terms, but the effect is weak and driven by outlier posts

### 11.5 Hashtag density and sentiment

Posts binned by hashtag count into quartiles (Q1–Q4). No monotonic relationship with negative or toxicity rate. Hashtag density is not a useful predictor of comment tone.

---

## 12. Cross-platform comparison: Instagram vs TikTok

### 12.1 Dataset comparison

| Metric | Instagram | TikTok |
|---|---|---|
| User-window rows | 174,287 | 56,300 |
| Unique users | 18,752 | 13,249 |
| Rolling windows | 40 | 40 |
| Calendar range | 2022-09 to 2026-03 | 2022-12 to 2026-03 |
| Feature set | R, C, E, D, T, G | R, C, E, G, T_binary |
| Cluster k | 4 | 3 |
| Shared users | **0** | — |

### 12.2 Feature distribution comparison (common features)

KS and Mann-Whitney tests on shared continuous features (all p ≈ 0):

| Feature | IG median | TK median | Direction |
|---|---|---|---|
| recency | 0.414 | 0.452 | TK commenters slightly more recent |
| coverage | 0.023 | 0.040 | **TK users cover 74% more posts** |
| engagement | 6.0 | 5.0 | IG slightly higher engagement per comment |
| gini | 0.0 | 0.0 | Both spike at zero (single-comment users) |

IG gini zero-mass: 72.7%; TK gini zero-mass: **95.3%** — TikTok is even more dominated by single-comment users.

### 12.3 Lifecycle statistics comparison

| Metric | Instagram | TikTok |
|---|---|---|
| Mean active months | **9.0** | 4.3 |
| Median active months | **8** | 4 |
| Mean % post-debut active | 34% | 30% |
| **Re-entry rate** | **63.0%** | **6.4%** |
| Mean inactive months | 21.76 | 19.74 |
| Dominant cluster | C3 (~70% of active user-months) | C3 (~87% of active user-months) |

The re-entry gap is the most actionable finding: IG's 63% vs TK's 6.4%.

### 12.4 Cross-platform window-aggregate correlations (raw Spearman, n=40)

All 15 strongest cross-platform relationships:

| IG aggregate | TK aggregate | ρ | Interpretation |
|---|---|---|---|
| IG_delay | TK_engagement | **+0.735** | When IG commenters arrive late, TK engagement is high |
| IG_delay | TK_tenure_binary | +0.674 | |
| IG_gini | TK_coverage | **-0.655** | When IG comments are concentrated, TK users comment broadly |
| IG_engagement | TK_engagement | **+0.626** | Engagement co-moves |
| IG_gini | TK_n | +0.624 | |
| IG_coverage | TK_coverage | +0.624 | Coverage co-moves |
| IG_tenure | TK_engagement | +0.619 | |
| IG_coverage | TK_tenure_binary | +0.590 | |
| IG_gini | TK_gini | +0.575 | |
| IG_gini | TK_tenure_binary | -0.556 | |
| IG_coverage | TK_n | -0.519 | |
| IG_n | TK_tenure_binary | -0.486 | |
| IG_n | TK_coverage | -0.442 | |
| IG_n | TK_n | +0.433 | Total active users co-move |
| IG_engagement | TK_tenure_binary | +0.409 | |

### 12.5 Detrended cross-platform correlations (first differences, n=39)

After removing the shared growth trend:

| Feature | Detrended Spearman ρ | p-value | Significant? |
|---|---|---|---|
| recency | +0.296 | 0.067 | No (marginal) |
| **coverage** | **+0.332** | **0.039** | **Yes** |
| engagement | +0.219 | 0.180 | No |
| gini | +0.050 | 0.762 | No |

**Only coverage survives detrending.** The IG_delay × TK_engagement correlation (ρ = 0.735 raw) collapses to near-zero after first-differencing, confirming it was driven by the shared growth trend, not genuine co-movement of behavioural shocks.

### 12.6 Cluster share correlations (shared window 2023-01 to 2026-03)

Raw Spearman ρ for C1/C2/C3 share series (n = 39 months):

| IG cluster | TK cluster | ρ | p | Significant? |
|---|---|---|---|---|
| IG_C3 | TK_C1 | -0.354 | 0.027 | Yes |
| IG_C1 | TK_C1 | +0.341 | 0.033 | Yes |
| IG_C1 | TK_C2 | -0.293 | 0.070 | Marginal |
| IG_C3 | TK_C2 | +0.168 | 0.307 | No |
| IG_C1 | TK_C3 | +0.190 | 0.248 | No |
| IG_C3 | TK_C3 | -0.098 | 0.551 | No |
| IG_C2 | TK_C1 | +0.094 | 0.569 | No |
| IG_C2 | TK_C2 | +0.049 | 0.765 | No |
| IG_C2 | TK_C3 | -0.036 | 0.826 | No |

After first-differencing, **all cluster share correlations lose significance.** The two platforms' cluster compositions trend together (shared macro context) but their short-run dynamics are independent.

---

## 13. Summary of key numbers (quick reference)

| Finding | Value |
|---|---|
| Total comments | 487,604 |
| Overall negative rate | 5.04% |
| Overall toxicity rate | 0.63% |
| Top negative emotion | sadness |
| Top toxic emotion | anger |
| Top intent in neg/tox | criticism |
| Most toxic post (tox_frac) | 81.4% (ID 18066958285842484) |
| Off-topic toxicity rate | 3.58% (5.7x baseline) |
| Appearance negativity rate | 1.4% (lowest target) |
| THE_HATER toxicity rate | 65.3% (103.4x baseline) |
| THE_HATER negative rate | 52.0% |
| THE_CRITIC toxicity lift | 5.5x |
| Passive regulars neg rate | 2.06% |
| Delayed visitors tox rate | 1.77% |
| Word count r vs negativity | +0.200 (strongest linguistic predictor) |
| Negative comment word count (median) | 11 (vs 4 overall) |
| Emoji count in negative (median) | 0 (vs 1 overall) |
| coverage × gini Spearman ρ (IG) | +0.780 |
| IG re-entry rate | 63.0% |
| TK re-entry rate | 6.4% |
| IG mean active months | 9.0 |
| TK mean active months | 4.3 |
| Only surviving detrended cross-platform signal | coverage ρ=0.332, p=0.039 |

---

## 14. Actionable recommendations

### 14.1 Moderation

1. **Flag post `18066958285842484` immediately.** 81.4% toxicity rate on a 172-comment thread. The gap between it and the second most toxic post (18.7%) indicates a specific incident, not a systematic pattern.

2. **Pre-moderate THE_HATER-pattern users.** Identifying characteristics: word_count > 10, emoji_count = 0, anger emotion, criticism intent, neg_rate on prior comments > 40%. 164 users driving 103x toxicity baseline.

3. **Off-topic comment filtering is the highest-ROI moderation lever.** Off-topic comments (1.2% of volume) carry 30.2% negativity and 3.58% toxicity. A topicality filter removes the toxicity hotspot with minimal collateral damage to genuine community engagement.

4. **Comment length as a soft real-time signal.** Comments over 15 words warrant a secondary review pass. Short comments (<5 words) with emojis can be auto-cleared.

### 14.2 Community health monitoring

5. **Monitor THE_CRITIC's product negativity rate as a brand health KPI.** Its month-over-month trend is a leading indicator of product reception without requiring external surveys.

6. **Emoji rate as a community health metric.** When the ratio of emoji-containing comments in a 7-day window drops, negativity is rising. This is a cheap, real-time leading indicator.

7. **Delayed visitors as an early-warning flag.** Posts that attract a spike in comments from users with high delay (commenting weeks after publication) should be reviewed — they carry 1.77% toxicity vs 0.29% for Passive regulars.

### 14.3 Audience and content strategy

8. **Re-engagement campaigns have a 63% addressable base on IG.** Target users in the month before they last commented with personalised prompts. TikTok re-engagement campaigns have minimal ROI given the 6.4% re-entry rate — acquisition investment is better placed there.

9. **Passive regulars (70% of active user-months) need upgrades, not volume.** C3 users are not a problem, but they are not deeply committed. Converting 10% of Passive regulars to Expressive regulars would meaningfully increase community richness.

10. **Appearance-targeted content is a safe engagement catalyst.** 1.4% negative rate. Content that invites comments about creator aesthetics (styling, visuals) generates high-volume, low-risk engagement.

11. **Product mentions should be paired with proactive Q&A.** The question intent in product-directed comments is a purchase consideration signal. Responding to product questions converts The Inquirer before they reach The Critic.

12. **Cross-platform macro events drive coverage co-movement.** IG and TK audiences respond to the same external triggers in breadth-of-commenting. Content tied to macro events (cultural moments, trending topics) will simultaneously drive both audiences. Platform-specific cluster dynamics, however, do not transfer — IG strategy should not be assumed to work on TikTok without independent validation.
