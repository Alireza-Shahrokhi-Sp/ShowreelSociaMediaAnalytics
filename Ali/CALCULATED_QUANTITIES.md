# Calculated Quantities & Formulas

Complete list of quantities calculated in the sentiment and persona pipelines with their exact formulas.

---

## Sentiment Pipeline

### Text-Level Features (Per-Comment)

| Quantity | Formula | Notes |
|----------|---------|-------|
| `text_length` | `len(text)` | Character count |
| `word_count` | `len(text.split())` | Word count (whitespace-split) |
| `avg_word_length` | `text_length / word_count` | Average characters per word |
| `emoji_count` | Count of emoji characters in text | All emoji occurrences |
| `unique_emoji_count` | Count of distinct emoji Unicode points | Deduplicated emoji set |
| `emoji_entropy` | `-sum(p_i * log2(p_i)) / log2(3)` where `p_i = count_i / total_emojis` | Shannon entropy normalized to [-1, 1]; 0 = all same emoji, 1 = max diversity; log2(3) for 3-emoji case |
| `emoji_variety_ratio` | `unique_emoji_count / emoji_count` | Ratio of distinct to total emojis |
| `emoji_per_word_ratio` | `emoji_count / word_count` | Emojis per word |
| `url_count` | Count of URLs (http/https/www patterns) | Links in text |
| `mention_count` | Count of @username patterns | @mentions |
| `hashtag_count` | Count of #hashtag patterns | Hashtags |
| `exclamation_count` | `text.count('!')` | Exclamation marks |
| `question_count` | `text.count('?')` | Question marks |
| `has_numbers` | `bool(any(c.isdigit() for c in text))` | Contains digits |
| `has_links` | `bool(url_count > 0)` | Has any URL |

### Post-Level Room Metrics (Aggregated from Comments)

Aggregation scope: all comments under a single post with `sentiment_cat` derived from LLM output.

Let:
- `s` = array of `sentiment_score` values for all comments in post
- `lab` = array of `sentiment_cat` values ("positive", "negative", "neutral")
- `pos` = fraction of positive comments = `(lab == "positive").mean()`
- `neg` = fraction of negative comments = `(lab == "negative").mean()`
- `neu` = fraction of neutral comments = `(lab == "neutral").mean()`

| Quantity | Formula | Notes |
|----------|---------|-------|
| `vibe_score` | `s.mean()` | Overall post polarity; range: [-1, +1] |
| `dispersion` | `s.std(ddof=0)` | Standard deviation of sentiment scores; 0 = unanimous, high = divergent |
| `pos_frac` | `(lab == "positive").mean()` | Fraction of positive comments |
| `neg_frac` | `(lab == "negative").mean()` | Fraction of negative comments |
| `neu_frac` | `(lab == "neutral").mean()` | Fraction of neutral comments |
| `stance_alignment` | `sign(s).mean()` | Mean sign of sentiment_score; +1 = all positive, -1 = all negative, 0 = split |
| `polarization` | `2 * min(pos, neg)` | How evenly split pos vs neg; 0 = one-sided, 1 = perfectly balanced |
| `controversy` | `4 * pos * neg` | Strength of opposing viewpoints; peaks at 1 when pos=neg=0.5 |
| `consensus` | `1 - entropy(pos, neg, neu)` | Agreement level; 1 = all same label, 0 = max disagreement |
| `sarcasm_rate` | `sarcasm.fillna(False).mean()` | Fraction of sarcastic comments |
| `toxicity_rate` | `(toxicity != "none").mean()` | Fraction of comments with any toxicity |

#### Entropy Function (for consensus)

```
entropy(pos, neg, neu) = -sum(p_i * log2(p_i)) / log2(3)
  where p_i ∈ {pos, neg, neu}, log2(3) normalizes to [0, 1]
```

If all comments are "positive": entropy = 0, consensus = 1.
If evenly split 3-way: entropy = 1, consensus = 0.

#### Room State Classification

```
if controversy >= 0.3:
    room_state = "divided"
elif consensus >= 0.6:
    room_state = "united"
else:
    room_state = "mixed"
```

---

## Persona Pipeline

### Stage 1: Candidate Discovery

Per-persona aggregation (implicit in LLM output):

| Quantity | Calculation |
|----------|-----------|
| `frequency_estimate` | Subjective LLM estimate based on comment prevalence |
| `signal_markers` | Linguistic/behavioral patterns identified by LLM |
| `example_comments` | Representative comments selected by LLM |

### Stage 2: Per-User Classification

| Quantity | Formula | Notes |
|----------|---------|-------|
| `comment_count` | `count(comments by author_id)` | Total comments by user |
| `avg_sentiment_score` | `sentiment_score.mean()` for all comments by user | User's typical sentiment |
| `dominant_emotion` | `mode(emotion)` for all comments by user | Most frequent emotion |
| `sarcasm_rate` | `sarcasm.mean()` for all comments by user | User's sarcasm tendency |
| `target_preference` | `mode(target)` for all comments by user | What user most comments on |

---

## Cross-Pipeline Features

### Per-Author Aggregation (If Joining Sentiment → Persona)

Derived by grouping all comments by `author_id`:

| Quantity | Formula |
|----------|---------|
| `total_comments` | `count(comment_id)` |
| `mean_sentiment` | `sentiment_score.mean()` |
| `positive_ratio` | `(sentiment_cat == "positive").mean()` |
| `negative_ratio` | `(sentiment_cat == "negative").mean()` |
| `mean_toxicity` | `toxicity.mean()` (if numeric) |
| `sarcasm_rate` | `sarcasm.mean()` |
| `avg_text_length` | `text_length.mean()` |
| `avg_emoji_density` | `(emoji_count / word_count).mean()` |

---

## Parameters & Thresholds

| Name | Value | Used For |
|------|-------|----------|
| `NEUTRAL_BAND` | 0.15 | Sentiment categorization: `\|score\| <= 0.15` → neutral |
| `toxicity_high` | ≥ 0.7 | Toxicity label categorization |
| `toxicity_moderate` | 0.3–0.7 | Toxicity label categorization |
| `controversy_threshold` | ≥ 0.3 | Room state = "divided" |
| `consensus_threshold` | ≥ 0.6 | Room state = "united" |
| `entropy_normalization` | log₂(3) | Consensus entropy scaling (3-class sentiment) |

---

## Notes on Calculations

1. **Sentiment categorization** happens in the notebook *after* retrieval:
   ```python
   _band = NEUTRAL_BAND  # 0.15
   sentiment_cat = "positive" if score > _band
                   else "negative" if score < -_band
                   else "neutral"
   ```

2. **Polarity metrics** (`polarization`, `controversy`) use simplified formulas for fast aggregation; they don't perfectly match information-theoretic versions but are computationally efficient.

3. **Entropy** is always normalized by log₂(3) to scale [0, 1] assuming 3-class sentiment.

4. **Sign function** in stance_alignment:
   ```python
   sign(x) = +1 if x > 0
             -1 if x < 0
              0 if x == 0
   ```

5. **Emoji entropy** uses base-2 logarithm (Shannon); values range [0, 1]:
   - 0 = all comments use same emoji
   - 1 = emojis evenly distributed (max diversity)
