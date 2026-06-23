---
title: Instagram persona pipeline
type: source-summary
sources:
  - Ali/persona_pipeline.ipynb
  - Ali/outputs/stage1_persona/
  - Ali/outputs/stage2_persona_combined/
related:
  - "[[instagram_findings]]"
  - "[[persona-pipeline]]"
created: 2026-06-14
updated: 2026-06-14
confidence: high
---

# Instagram persona pipeline

## Overview

Two-stage Vertex AI Batch pipeline that classifies users into 10 behavioural personas based on their comment patterns.

**Stage 1**: Initial classification and user sampling
**Stage 2**: Refined classification with multimodal context (comment body + post metadata)

Final output: `user_personas_combined.parquet` (40,019 users with persona labels)

---

## Stage 1: Initial persona assignment

**Notebook**: `Ali/persona_pipeline.ipynb` (Stage 1 section)

**Input**: User-aggregated comment statistics

For each user, computed:
- Comment frequency (total comments, comments per post)
- Emoji usage pattern (emoji count, diversity, frequency)
- Engagement patterns (avg likes per comment, reply count)
- Activity span (days between first and last comment)
- Reply ratio (replies vs top-level comments)
- Mean word count per comment

**LLM prompt**: Vertex AI Claude model reads user's aggregated comment profile and assigns one of 10 personas:

1. **THE_TAGGER** — High volume, short comments, primarily tagging/mentioning others
2. **THE_CASUAL_COMPLIMENTER** — Frequent, brief positive comments with emojis
3. **THE_EMOJI_REACTOR** — Comments are primarily emoji responses (minimal text)
4. **THE_STORYTELLER** — Longer, narrative comments sharing personal anecdotes
5. **THE_SUPERFAN** — Extremely high frequency, mostly praise, strong engagement signals
6. **THE_INQUIRER** — Frequent questions, seeking information or clarification
7. **THE_CRITIC** — Regular critical/constructive comments, balanced view
8. **THE_ADVISOR** — Offers advice and guidance to creator
9. **THE_SPAMMER** — Repetitive promotional or off-topic content
10. **THE_HATER** — Hostile, frequently negative, aggressive tone

### Stage 1 output files

**Location**: `Ali/outputs/stage1_persona/`

**File**: `stage1_sample_users_instagram.parquet`

| Column | Type | Description |
|---|---|---|
| author_id | string | User ID (key) |
| persona_codename | string | One of 10 personas above |
| confidence | float64 | [0, 1] LLM confidence in assignment |
| reason | string | Brief explanation of assignment |

**Size**: 40,000+ rows (stratified sample of users)

---

**Pathway assignments** (experimental variants)

Two pathway files explore alternative groupings:

- `pathway_a_assignments_instagram.parquet` — Persona tree A (earlier variant)
- `pathway_b_assignments_instagram.parquet` — Persona tree B (final variant, used in analyses)

Both have schema: author_id, persona_codename, confidence, additional_context columns.

**Size**: ~40,000 rows each

---

**Feature cache**

**File**: `user_features_instagram_cache.parquet`

Cached aggregated user features used in Stage 1 LLM prompting:

| Column | Type | Description |
|---|---|---|
| author_id | string | Key |
| total_comments | int64 | All-time comment count |
| activity_span_days | float64 | Days between first and last comment |
| reply_ratio | float64 | [0, 1] share of comments that are replies |
| mean_word_count | float64 | Mean words per comment |
| emoji_ratio | float64 | [0, 1] share of comments with emojis |
| avg_emoji_per_comment | float64 | Mean emoji count per comment |
| engagement_score | float64 | Aggregated likes/replies feedback |

**Size**: ~40,000 rows

---

## Stage 2: Refined persona with multimodal context

**Notebook**: `Ali/persona_pipeline.ipynb` (Stage 2 section)

**Input**: 
- Comment body (full text)
- Post metadata (caption, media type, is_paid_partnership)
- Comment-level LLM labels (sentiment, emotion, intent, target — from sentiment pipeline)

**LLM prompt**: Vertex AI Claude sees comment body, post context, and sentiment label and may refine the persona assessment or confirm Stage 1 assignment.

### Stage 2 output

**File**: `Ali/outputs/stage2_persona_combined/user_personas_combined.parquet`

Final canonical persona assignments. Schema:

| Column | Type | Description |
|---|---|---|
| author_id | string | User ID (key) |
| persona_codename | string | Final persona (one of 10) |
| confidence | float64 | [0, 1] confidence (Stage 2 refined) |
| total_comments | int64 | All-time comment count |
| activity_span_days | float64 | Days from first to last comment |
| reply_ratio | float64 | [0, 1] share of replies |
| mean_word_count | float64 | Mean words per comment |
| persona_source | string | "stage1" / "stage2_refined" (which stage assigned it) |

**Size**: 40,019 rows

**Coverage**: 19.3% of all 487,604 comments (191,096 unique users; 40,019 labeled)

---

## Persona frequency distribution

| Persona | User count | Comment rows | % of labeled users |
|---|---|---|---|
| THE_TAGGER | 11,628 | 19,357 | 29.1% |
| THE_CASUAL_COMPLIMENTER | 7,802 | 10,895 | 19.5% |
| THE_EMOJI_REACTOR | 5,764 | 9,657 | 14.4% |
| THE_STORYTELLER | 4,471 | 9,819 | 11.2% |
| THE_SUPERFAN | 4,191 | 31,920 | 10.5% |
| THE_INQUIRER | 2,608 | 5,300 | 6.5% |
| THE_CRITIC | 2,253 | 4,988 | 5.6% |
| THE_ADVISOR | 865 | 1,740 | 2.2% |
| THE_SPAMMER | 218 | 324 | 0.5% |
| THE_HATER | 164 | 173 | 0.4% |
| **Total** | **40,019** | **94,173** | **100%** |

---

## Persona sentiment and toxicity

Key findings from sentiment analysis:

| Persona | Neg rate | Tox rate | Tox lift | Interpretation |
|---|---|---|---|---|
| THE_HATER | 52.0% | 65.3% | 103.4x | Extreme outlier; hostile comment content |
| THE_CRITIC | ~10–12% | 3.4% | 5.5x | Dissatisfied-customer segment; product-focused |
| THE_SPAMMER | ~5% | ~0.6% | ~1.0x | Near population baseline; promotional noise |
| THE_INQUIRER | ~5% | ~0.5% | ~0.8x | Questions sometimes code as sceptical |
| THE_STORYTELLER | ~4% | ~0.4% | ~0.7x | Narrative engagement; largely positive |
| THE_ADVISOR | ~3% | ~0.3% | ~0.5x | Helpful tone; rarely negative |
| THE_SUPERFAN | ~2% | ~0.2% | ~0.3x | Extreme loyalty; lowest negativity |
| THE_CASUAL_COMPLIMENTER | ~1.5% | ~0.2% | ~0.3x | Affirming; safe high-frequency engagement |
| THE_EMOJI_REACTOR | ~1% | ~0.2% | ~0.3x | Low-effort positive response; minimal friction |
| THE_TAGGER | ~1% | ~0.1% | ~0.2x | Mechanical tagging; near-neutral sentiment |

---

## Persona × sentiment target crossing

From `persona_sentiment_rfm.ipynb` Section E3:

**THE_CRITIC × product target** has the highest conditional negative rate among non-Hater personas. Despite high rate (>35% of The Critic's product comments), The Critic represents <1% of all product comments.

**THE_SUPERFAN × creator target** has near-zero negativity conditional rate (all praise).

**THE_TAGGER × other_user target** moderately elevated — tag mentions can generate interpersonal friction.

---

## Persona × lifecycle (window-level RFM cluster)

From `persona_sentiment_rfm.ipynb` Section C:

| Observation | Evidence |
|---|---|
| THE_HATER absent from Brand advocates | Hostile users are not loyal commenters |
| THE_SUPERFAN in Brand advocates | Loyalty traits cluster together |
| THE_CASUAL_COMPLIMENTER in Passive regulars | Low-effort comments match passive behaviour |
| THE_CRITIC in all lifecycle states | Stable trait, not phase-dependent |
| THE_STORYTELLER in Expressive regulars | Long narratives match expressive profile |

---

## Persona assignment algorithm notes

**LLM model**: Claude (Vertex AI Batch)

**Batching**: Processed in large batches for cost efficiency

**Prompting strategy**: Two-stage refinement (initial classification, then multimodal context)

**Validation**: No ground truth; persona assignments are exploratory labels. Validity measured by:
- Internal consistency (users in same persona have similar behaviour profiles)
- Predictive signal for sentiment/toxicity
- Interpretability of persona names

**Known limitations**:
- Users with very few comments (<5) may receive high-confidence but unstable labels
- Persona names are archetypal, not deterministic — overlap and ambiguity exist
- THE_HATER and THE_SPAMMER are small classes (0.4% and 0.5%) and may not generalise to other creators

---

## Batch processing parameters

**Vertex AI Batch Input/Output**:
- Input format: JSONL (one user record per line)
- Prompt template: User aggregate profile + optional multimodal context
- Model parameters: temperature ~0.3–0.5 (low randomness, consistent assignments)
- Cost: ~$X per 1M tokens (standard Claude Batch pricing)

**Processing time**: ~2–4 hours per stage for full 40k user dataset

---

## Integration with downstream analysis

Persona labels are merged into the master sentiment dataframe on `author_id`:

```
df = sentiment_table.merge(
    personas[['author_id','persona_codename','confidence']],
    on='author_id',
    how='left'
)
```

Coverage: 19.3% of comment rows inherit a persona label.

Persona-based analyses (sentiment aggregation, moderation policies, community profiling) use only the labeled subset and often exclude small classes (Hater, Spammer) or flag them separately due to small n.
