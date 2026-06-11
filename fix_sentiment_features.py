import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'Ali/persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

# ── Patch 1: feature engineering cell (76e3bb29) ─────────────────────────────
# Insert sentiment/toxicity join + aggregation columns
OLD_BINARY_FLAGS = (
    'ig_comments["has_emoji"]    = (ig_comments["emoji_count"] > 0).astype(int)\n'
    'ig_comments["has_question"] = (ig_comments["question_count"] > 0).astype(int)\n'
    'ig_comments["has_exclaim"]  = (ig_comments["exclamation_count"] > 0).astype(int)\n'
    'ig_comments["is_reply"]     = ig_comments["reply_to_comment_id"].notna().astype(int)\n'
    'print("Binary flags derived.")'
)

NEW_BINARY_FLAGS = (
    'ig_comments["has_emoji"]    = (ig_comments["emoji_count"] > 0).astype(int)\n'
    'ig_comments["has_question"] = (ig_comments["question_count"] > 0).astype(int)\n'
    'ig_comments["has_exclaim"]  = (ig_comments["exclamation_count"] > 0).astype(int)\n'
    'ig_comments["is_reply"]     = ig_comments["reply_to_comment_id"].notna().astype(int)\n'
    'print("Binary flags derived.")\n'
    '\n'
    '# Join per-comment sentiment/toxicity from sentiment pipeline output.\n'
    '# sentiment_score: -1 (very negative) -> +1 (very positive)\n'
    '# is_toxic: 1 if toxicity in {mild, severe}\n'
    '# is_negative: 1 if sentiment == negative\n'
    '_sent_path = SENTIMENT_PATH\n'
    'if os.path.exists(_sent_path):\n'
    '    _sent = pd.read_parquet(\n'
    '        _sent_path,\n'
    '        columns=["comment_id", "sentiment_score", "toxicity", "sentiment"],\n'
    '    )\n'
    '    _sent["is_toxic"]    = _sent["toxicity"].isin(["mild", "severe"]).astype(int)\n'
    '    _sent["is_negative"] = (_sent["sentiment"] == "negative").astype(int)\n'
    '    ig_comments = ig_comments.merge(\n'
    '        _sent[["comment_id", "sentiment_score", "is_toxic", "is_negative"]],\n'
    '        on="comment_id", how="left",\n'
    '    )\n'
    '    ig_comments["sentiment_score"] = ig_comments["sentiment_score"].fillna(0.0)\n'
    '    ig_comments["is_toxic"]        = ig_comments["is_toxic"].fillna(0).astype(int)\n'
    '    ig_comments["is_negative"]     = ig_comments["is_negative"].fillna(0).astype(int)\n'
    '    print(f"Sentiment joined: {_sent["comment_id"].nunique():,} comments enriched.")\n'
    'else:\n'
    '    ig_comments["sentiment_score"] = 0.0\n'
    '    ig_comments["is_toxic"]        = 0\n'
    '    ig_comments["is_negative"]     = 0\n'
    '    print(f"WARNING: {_sent_path} not found — sentiment features will be zero.")'
)

OLD_AGG_BLOCK = (
    '            "emoji_usage_rate":        grp["has_emoji"].mean(),\n'
    '            "question_rate":           grp["has_question"].mean(),\n'
    '            "exclamation_rate":        grp["has_exclaim"].mean(),\n'
    '        }'
)

NEW_AGG_BLOCK = (
    '            "emoji_usage_rate":        grp["has_emoji"].mean(),\n'
    '            "question_rate":           grp["has_question"].mean(),\n'
    '            "exclamation_rate":        grp["has_exclaim"].mean(),\n'
    '            "mean_sentiment_score":    grp["sentiment_score"].mean(),\n'
    '            "pct_negative":            grp["is_negative"].mean(),\n'
    '            "toxicity_rate":           grp["is_toxic"].mean(),\n'
    '        }'
)

for cell in nb['cells']:
    if cell['id'] != '76e3bb29':
        continue
    src = ''.join(cell['source'])
    assert OLD_BINARY_FLAGS in src, "binary flags block not found"
    assert OLD_AGG_BLOCK in src, "agg block not found"
    src = src.replace(OLD_BINARY_FLAGS, NEW_BINARY_FLAGS)
    src = src.replace(OLD_AGG_BLOCK, NEW_AGG_BLOCK)
    compile(src, '76e3bb29', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('patched 76e3bb29')
    break

# ── Patch 2: feature selection cell ──────────────────────────────────────────
OLD_FEATURES = (
    '    "post_concentration_ratio",\n'
    ']'
)

NEW_FEATURES = (
    '    "post_concentration_ratio",\n'
    '    "mean_sentiment_score",   # average sentiment polarity per user (-1 negative, +1 positive)\n'
    '    "pct_negative",           # share of comments flagged negative\n'
    '    "toxicity_rate",          # share of comments flagged toxic (mild or severe)\n'
    ']'
)

for cell in nb['cells']:
    if cell['id'] != 'select-features-for-clustering':
        continue
    src = ''.join(cell['source'])
    assert OLD_FEATURES in src, "feature list end not found"
    src = src.replace(OLD_FEATURES, NEW_FEATURES)
    compile(src, 'select-features-for-clustering', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('patched select-features-for-clustering')
    break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
