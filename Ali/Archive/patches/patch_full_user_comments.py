import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'Ali/persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

changes = []

# ─── 1. Increase comment history stored per user: .head(5) -> .head(20) ──────
for cell in nb['cells']:
    if cell.get('id') != '76e3bb29':
        continue
    src = ''.join(cell['source'])
    OLD = (
        '    top_comments = (\n'
        '        txt.groupby("author_id").head(5)\n'
        '           .groupby("author_id")["text"]\n'
        '           .apply(lambda ts: " ||| ".join(ts.astype(str).tolist()))\n'
        '           .reset_index()\n'
        '           .rename(columns={"text": "top_comments_sample"})\n'
        '    )'
    )
    NEW = (
        '    top_comments = (\n'
        '        txt.groupby("author_id").head(20)\n'
        '           .groupby("author_id")["text"]\n'
        '           .apply(lambda ts: " ||| ".join(ts.astype(str).tolist()))\n'
        '           .reset_index()\n'
        '           .rename(columns={"text": "top_comments_sample"})\n'
        '    )'
    )
    assert OLD in src, "head(5) pattern not found in 76e3bb29"
    src = src.replace(OLD, NEW, 1)
    compile(src, '76e3bb29', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    changes.append('76e3bb29: top_comments_sample stores up to 20 comments per user (was 5)')
    break

# ─── 2. Rewrite pb-summaries to give whole-user comment profiles ──────────────
for cell in nb['cells']:
    if cell.get('id') != 'pb-summaries':
        continue
    src = ''.join(cell['source'])

    OLD_SAMPLES = (
        '    samples = []\n'
        '    if "top_comments_sample" in sub.columns:\n'
        '        picks = sub["top_comments_sample"].dropna()\n'
        '        if len(picks):\n'
        '            picks = picks.sample(n=min(MAX_SAMPLE_USERS_PER_CLUSTER, len(picks)), random_state=42)\n'
        '        for txt in picks:\n'
        '            frags = [f.strip() for f in str(txt).split("|||") if f.strip()]\n'
        '            samples.extend(frags[:2])\n'
        '            if len(samples) >= 30:\n'
        '                break\n'
        '    # pct_audience is relative to ALL users (incl. noise) so noise share is meaningful.\n'
        '    return {\n'
        '        "cluster_id": int(cluster_id),\n'
        '        "is_noise": cluster_id == -1,\n'
        '        "n_users": n,\n'
        '        "pct_audience": round(100 * n / max(len(df), 1), 1),\n'
        '        "mean_behavioral_stats": mean_stats,\n'
        '        "dominant_room_vibe_mix": vibe_mix,\n'
        '        "sample_comments": samples[:30],\n'
        '    }'
    )
    NEW_SAMPLES = (
        '    # Pick N_USERS_FOR_LLM representative users and include ALL their stored comments.\n'
        '    N_USERS_FOR_LLM = 5\n'
        '    sample_users = []\n'
        '    if "top_comments_sample" in sub.columns:\n'
        '        picks = sub[["author_id", "top_comments_sample"]].dropna(subset=["top_comments_sample"])\n'
        '        if len(picks):\n'
        '            picks = picks.sample(n=min(N_USERS_FOR_LLM, len(picks)), random_state=42)\n'
        '        for _, row in picks.iterrows():\n'
        '            all_comments = [c.strip() for c in str(row["top_comments_sample"]).split("|||") if c.strip()]\n'
        '            sample_users.append({"user_id": str(row["author_id"]), "comments": all_comments})\n'
        '    # pct_audience is relative to ALL users (incl. noise) so noise share is meaningful.\n'
        '    return {\n'
        '        "cluster_id": int(cluster_id),\n'
        '        "is_noise": cluster_id == -1,\n'
        '        "n_users": n,\n'
        '        "pct_audience": round(100 * n / max(len(df), 1), 1),\n'
        '        "mean_behavioral_stats": mean_stats,\n'
        '        "dominant_room_vibe_mix": vibe_mix,\n'
        '        "sample_users": sample_users,\n'
        '    }'
    )
    assert OLD_SAMPLES in src, "pb-summaries samples pattern not found"
    src = src.replace(OLD_SAMPLES, NEW_SAMPLES, 1)
    compile(src, 'pb-summaries', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    changes.append('pb-summaries: sample_users = 5 users with full comment history each')
    break

# ─── 3. Update _pb_block in pb-label to render user profiles ─────────────────
for cell in nb['cells']:
    if cell.get('id') != 'pb-label':
        continue
    src = ''.join(cell['source'])

    OLD_BLOCK = (
        'def _pb_block(s):\n'
        '    stats = "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"\n'
        '                      for k, v in s["mean_behavioral_stats"].items())\n'
        '    vibe = "  ".join(f"{k}: {v}%" for k, v in s["dominant_room_vibe_mix"].items())\n'
        '    comments = "\\n    - ".join(s["sample_comments"][:8])\n'
        '    noise_tag = "  *** NOISE — low-cohesion, no dominant pattern ***" if s["is_noise"] else ""\n'
        '    return (f"CLUSTER {s[\'cluster_id\']}{noise_tag}  (n={s[\'n_users\']:,}, {s[\'pct_audience\']}% of all users)\\n"\n'
        '            f"  Mean behavioural stats: {stats}\\n"\n'
        '            f"  Dominant room-vibe mix: {vibe}\\n"\n'
        '            f"  Sample comments:\\n    - {comments}")'
    )
    NEW_BLOCK = (
        'def _pb_block(s):\n'
        '    stats = "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"\n'
        '                      for k, v in s["mean_behavioral_stats"].items())\n'
        '    vibe = "  ".join(f"{k}: {v}%" for k, v in s["dominant_room_vibe_mix"].items())\n'
        '    noise_tag = "  *** NOISE — low-cohesion, no dominant pattern ***" if s["is_noise"] else ""\n'
        '    users_block = ""\n'
        '    for u in s.get("sample_users", []):\n'
        '        comment_lines = "\\n      ".join(f\'"{c}"\' for c in u["comments"])\n'
        '        users_block += f\'\\n  USER {u["user_id"]}:\\n      {comment_lines}\'\n'
        '    return (f"CLUSTER {s[\'cluster_id\']}{noise_tag}  (n={s[\'n_users\']:,}, {s[\'pct_audience\']}% of all users)\\n"\n'
        '            f"  Mean behavioural stats: {stats}\\n"\n'
        '            f"  Dominant room-vibe mix: {vibe}\\n"\n'
        '            f"  Representative users (full comment history):{users_block}")'
    )
    assert OLD_BLOCK in src, "_pb_block pattern not found in pb-label"
    src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    compile(src, 'pb-label', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    changes.append('pb-label: _pb_block renders 5 user profiles with full comment lists')
    break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Saved.')
for c in changes:
    print(' -', c)
