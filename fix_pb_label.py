import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'd:\Polythecninco di Milano\AFB_Lab\Ali\persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

src = (
    '# LLM-label each cluster into the SAME 5-key persona schema, then write final_taxonomy.\n'
    'from google.genai import types\n'
    '\n'
    'PB_NAMING_SYSTEM = (\n'
    '    "You are a strategic audience analyst for an Italian Instagram influencer agency.\\n"\n'
    '    "Commenters were clustered into behavioural macro-segments (UMAP + HDBSCAN). Each cluster is\\n"\n'
    '    "described by mean behavioural stats, a dominant room-vibe mix, and representative comments.\\n"\n'
    '    "One special cluster is labelled NOISE (cluster_id=-1): these are users that HDBSCAN could not\\n"\n'
    '    "assign to any dense cluster. Treat them as a distinct low-cohesion segment, not as a real\\n"\n'
    '    "behavioural archetype. Give them codename NOISE_UNCLASSIFIED and a description that clearly\\n"\n'
    '    "notes they are statistically scattered / do not fit any dominant pattern.\\n"\n'
    '    "Turn every other cluster into ONE audience persona. Personas must be MUTUALLY EXCLUSIVE.\\n"\n'
    '    f"Return AT MOST {TARGET_PERSONAS} personas (NOISE_UNCLASSIFIED + up to {TARGET_PERSONAS-1} real),\\n"\n'
    '    "ordered by audience share (NOISE_UNCLASSIFIED last).\\n"\n'
    '    "For each persona output exactly these 5 keys: codename (UPPER_SNAKE_CASE), label (short Title Case),\\n"\n'
    '    "description (1-2 sentences), quantitative_signals (3-5 distinguishing behavioural signals as strings),\\n"\n'
    '    "example_comments (2-3 verbatim fragments drawn from the cluster\'s sample comments).\\n"\n'
    '    "Output ONLY a valid JSON array of objects with exactly those 5 keys. No preamble, no markdown fences."\n'
    ')\n'
    '\n'
    'def _pb_block(s):\n'
    '    stats = "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"\n'
    '                      for k, v in s["mean_behavioral_stats"].items())\n'
    '    vibe = "  ".join(f"{k}: {v}%" for k, v in s["dominant_room_vibe_mix"].items())\n'
    '    comments = "\\n    - ".join(s["sample_comments"][:8])\n'
    '    noise_tag = "  *** NOISE CLASS: users HDBSCAN could not cluster ***" if s["is_noise"] else ""\n'
    '    return (f"CLUSTER {s[\'cluster_id\']}{noise_tag}  (n={s[\'n_users\']:,}, {s[\'pct_audience\']}% of all users)\\n"\n'
    '            f"  Mean behavioural stats: {stats}\\n"\n'
    '            f"  Dominant room-vibe mix: {vibe}\\n"\n'
    '            f"  Sample comments:\\n    - {comments}")\n'
    '\n'
    'pb_prompt = PB_NAMING_SYSTEM + "\\n\\n=== CLUSTER DATA ===\\n\\n" + "\\n\\n".join(\n'
    '    _pb_block(s) for s in pb_cluster_summaries)\n'
    '\n'
    'pb_cfg = types.GenerateContentConfig(\n'
    '    temperature=0.0, top_p=1.0, max_output_tokens=8192,\n'
    '    response_mime_type="application/json",\n'
    '    thinking_config=types.ThinkingConfig(thinking_budget=4096),\n'
    ')\n'
    'print(f"Labelling {len(pb_cluster_summaries)} clusters via {MODEL_STAGE3_NAMING} (incl. noise) ...")\n'
    'pb_resp = client.models.generate_content(model=MODEL_STAGE3_NAMING, contents=pb_prompt, config=pb_cfg)\n'
    '\n'
    '_pb_text = strip_fences(pb_resp.text or "")\n'
    'try:\n'
    '    pb_macro_personas = json.loads(_pb_text)\n'
    'except Exception as e:\n'
    '    print("Could not parse LLM output:", str(e)[:120], "\\n", _pb_text[:400])\n'
    '    pb_macro_personas = []\n'
    '\n'
    '_keys = ["codename", "label", "description", "quantitative_signals", "example_comments"]\n'
    'pb_macro_personas = [{k: p.get(k, "" if k in ("codename", "label", "description") else []) for k in _keys}\n'
    '                     for p in pb_macro_personas if isinstance(p, dict)][:TARGET_PERSONAS]\n'
    '\n'
    'print(f"{len(pb_macro_personas)} personas from Pathway B:")\n'
    'for p in pb_macro_personas:\n'
    '    print(f"   {str(p[\'codename\']):<28} | {p[\'label\']}")\n'
    '\n'
    'import os\n'
    'if os.path.exists(TAXONOMY_JSON_PATH):\n'
    '    data = json.load(open(TAXONOMY_JSON_PATH, encoding="utf-8"))\n'
    'else:\n'
    '    data = {"status": "PENDING_HUMAN_REVIEW", "raw_candidates": []}\n'
    'data["final_taxonomy"] = pb_macro_personas\n'
    'data["status"] = "PENDING_HUMAN_REVIEW"\n'
    'data["pathway"] = "B_CLUSTER"\n'
    'data["instructions"] = ("Pathway B (UMAP+HDBSCAN) taxonomy. Review/merge, ensure MECE, "\n'
    '                        "set status=\'APPROVED\' before Stage 2. Set CONSOLIDATION_PATHWAY=\'B_CLUSTER\' "\n'
    '                        "so the Pathway A wrapper cell does not overwrite this.")\n'
    'with open(TAXONOMY_JSON_PATH, "w", encoding="utf-8") as f:\n'
    '    json.dump(data, f, ensure_ascii=False, indent=2)\n'
    'print(f"Written -> {TAXONOMY_JSON_PATH} (pathway=B_CLUSTER). "\n'
    '      "Set CONSOLIDATION_PATHWAY=\'B_CLUSTER\' and status=\'APPROVED\' before Stage 2.")\n'
    '\n'
    '_cluster_map = {int(s["cluster_id"]): pb_macro_personas[i]["codename"]\n'
    '                for i, s in enumerate(pb_cluster_summaries) if i < len(pb_macro_personas)}\n'
    'with open(CLUSTER_PERSONA_MAP_PATH, "w", encoding="utf-8") as f:\n'
    '    json.dump(_cluster_map, f, ensure_ascii=False, indent=2)\n'
    'pb_df[["author_id", "macro_cluster"]].to_parquet(\n'
    '    f"{LOCAL_DIR}/pathway_b_assignments_{PLATFORM}.parquet", index=False)\n'
    'print(f"Cluster->persona map -> {CLUSTER_PERSONA_MAP_PATH}; "\n'
    '      f"assignments -> pathway_b_assignments_{PLATFORM}.parquet")\n'
)

compile(src, 'pb-label', 'exec')
print('syntax OK')

for cell in nb['cells']:
    if cell['id'] == 'pb-label':
        lines = src.split('\n')
        cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
