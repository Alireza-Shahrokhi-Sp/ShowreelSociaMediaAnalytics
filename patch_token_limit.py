import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'Ali/persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

changes = []

# ─── 1. Add MAX_BATCH_TOKENS to config cell (db046044) ────────────────────────
for cell in nb['cells']:
    if cell.get('id') != 'db046044':
        continue
    src = ''.join(cell['source'])
    OLD = 'STAGE1_USERS_PER_REQUEST = 3'
    NEW = (
        'STAGE1_USERS_PER_REQUEST = 3\n'
        'MAX_BATCH_TOKENS         = 900_000   # stay under Vertex 1M token limit; each job gets split'
    )
    assert OLD in src, f"config pattern not found: {repr(OLD)}"
    src = src.replace(OLD, NEW, 1)
    compile(src, 'db046044', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    changes.append('db046044: added MAX_BATCH_TOKENS')
    break

# ─── 2. Rewrite submit_stage1 + retrieve_stage1 in cell 4a9b6ed7 ─────────────
for cell in nb['cells']:
    if cell.get('id') != '4a9b6ed7':
        continue
    src = ''.join(cell['source'])

    OLD_SUBMIT = (
        '# ── SUBMIT (returns immediately — safe to close the laptop) ────────────────────\n'
        'def submit_stage1(user_features_df, n_sample=SAMPLE_N_USERS,\n'
        '                  group_size=STAGE1_USERS_PER_REQUEST, seed=SAMPLE_SEED):\n'
        '    print(f"\\n{\'=\'*60}\\nSTAGE 1 (batch submit) — Taxonomy Discovery\\n"\n'
        '          f"  Sample: {n_sample} users | {group_size}/request | Model: {MODEL_STAGE1_EXPLORATORY}\\n{\'=\'*60}")\n'
        '    sample_df = stratified_user_sample(user_features_df, n_sample=n_sample, seed=seed)\n'
        '    # Persist the sampled author set so Pathway B clusters the IDENTICAL users.\n'
        '    sample_df[["author_id"]].to_parquet(\n'
        '        f"{LOCAL_DIR}/stage1_sample_users_{PLATFORM}.parquet", index=False)\n'
        '    groups = [sample_df.iloc[i:i+group_size] for i in range(0, len(sample_df), group_size)]\n'
        '    lines  = [build_stage1_line(g) for g in tqdm(groups, desc="Build Stage 1 requests")]\n'
        '    in_uri  = upload_to_gcs(write_jsonl(lines, f"{LOCAL_DIR}/stage1_input.jsonl"),\n'
        '                            GCS_BUCKET, BATCH_INPUT_PREFIX + "stage1_input.jsonl")\n'
        '    out_uri = f"gs://{GCS_BUCKET}/{BATCH_OUTPUT_PREFIX}stage1/"\n'
        '    job = submit_batch_job(in_uri, out_uri, MODEL_STAGE1_EXPLORATORY)\n'
        '    record_batch_job("stage1", job, out_uri)\n'
        '    print(f"\\n\\U0001f4e4 Stage 1 submitted ({len(lines)} requests). Safe to close the laptop.")\n'
        '    print("   When it finishes, run:  retrieve_stage1()   -> writes taxonomy.json for review")\n'
        '    return job'
    )

    NEW_SUBMIT = (
        '# ── SUBMIT (returns immediately — safe to close the laptop) ────────────────────\n'
        'def _estimate_tokens(lines: list) -> int:\n'
        '    """Rough token estimate: ~4 chars per token across all request JSON."""\n'
        '    total_chars = sum(len(json.dumps(l, ensure_ascii=False)) for l in lines)\n'
        '    return total_chars // 4\n'
        '\n'
        'def submit_stage1(user_features_df, n_sample=SAMPLE_N_USERS,\n'
        '                  group_size=STAGE1_USERS_PER_REQUEST, seed=SAMPLE_SEED):\n'
        '    print(f"\\n{\'=\'*60}\\nSTAGE 1 (batch submit) — Taxonomy Discovery\\n"\n'
        '          f"  Sample: {n_sample} users | {group_size}/request | Model: {MODEL_STAGE1_EXPLORATORY}\\n{\'=\'*60}")\n'
        '    sample_df = stratified_user_sample(user_features_df, n_sample=n_sample, seed=seed)\n'
        '    # Persist the sampled author set so Pathway B clusters the IDENTICAL users.\n'
        '    sample_df[["author_id"]].to_parquet(\n'
        '        f"{LOCAL_DIR}/stage1_sample_users_{PLATFORM}.parquet", index=False)\n'
        '    groups = [sample_df.iloc[i:i+group_size] for i in range(0, len(sample_df), group_size)]\n'
        '    lines  = [build_stage1_line(g) for g in tqdm(groups, desc="Build Stage 1 requests")]\n'
        '\n'
        '    # Split into chunks that fit under Vertex AI 1M token limit.\n'
        '    chunks, chunk, chunk_tokens = [], [], 0\n'
        '    for line in lines:\n'
        '        tok = len(json.dumps(line, ensure_ascii=False)) // 4\n'
        '        if chunk and chunk_tokens + tok > MAX_BATCH_TOKENS:\n'
        '            chunks.append(chunk)\n'
        '            chunk, chunk_tokens = [], 0\n'
        '        chunk.append(line)\n'
        '        chunk_tokens += tok\n'
        '    if chunk:\n'
        '        chunks.append(chunk)\n'
        '\n'
        '    print(f"  Splitting into {len(chunks)} batch job(s) to stay under {MAX_BATCH_TOKENS:,} tokens each.")\n'
        '    jobs = []\n'
        '    for idx, chunk_lines in enumerate(chunks):\n'
        '        tag = f"stage1_chunk{idx}"\n'
        '        local_path = f"{LOCAL_DIR}/stage1_input_chunk{idx}.jsonl"\n'
        '        in_uri  = upload_to_gcs(write_jsonl(chunk_lines, local_path),\n'
        '                                GCS_BUCKET, BATCH_INPUT_PREFIX + f"stage1_input_chunk{idx}.jsonl")\n'
        '        out_uri = f"gs://{GCS_BUCKET}/{BATCH_OUTPUT_PREFIX}stage1_chunk{idx}/"\n'
        '        job = submit_batch_job(in_uri, out_uri, MODEL_STAGE1_EXPLORATORY)\n'
        '        record_batch_job(tag, job, out_uri)\n'
        '        jobs.append(job)\n'
        '\n'
        '    # Also write a manifest so retrieve_stage1() knows how many chunks to expect.\n'
        '    with open(f"{LOCAL_DIR}/stage1_chunks.json", "w") as f:\n'
        '        json.dump({"n_chunks": len(chunks), "tags": [f"stage1_chunk{i}" for i in range(len(chunks))]}, f)\n'
        '    print(f"\\n\\U0001f4e4 Stage 1 submitted ({len(chunks)} job(s), {len(lines)} total requests). Safe to close the laptop.")\n'
        '    print("   When all finish, run:  retrieve_stage1()   -> writes taxonomy.json for review")\n'
        '    return jobs'
    )

    assert OLD_SUBMIT in src, "submit_stage1 pattern not found"
    src = src.replace(OLD_SUBMIT, NEW_SUBMIT, 1)

    # Also update retrieve_stage1 to handle multiple chunks
    OLD_RETRIEVE = (
        '# ── RETRIEVE (run later, even from a fresh kernel) ────────────────────────────\n'
        'def retrieve_stage1(save=True):\n'
        '    job = get_recorded_job("stage1")\n'
        '    if job.state != JobState.JOB_STATE_SUCCEEDED:\n'
        '        print(f"⏳ Stage 1 not ready (state={job.state}). Re-run later.")\n'
        '        return None'
    )
    NEW_RETRIEVE = (
        '# ── RETRIEVE (run later, even from a fresh kernel) ────────────────────────────\n'
        'def retrieve_stage1(save=True):\n'
        '    manifest_path = f"{LOCAL_DIR}/stage1_chunks.json"\n'
        '    if os.path.exists(manifest_path):\n'
        '        with open(manifest_path) as f:\n'
        '            manifest = json.load(f)\n'
        '        tags = manifest["tags"]\n'
        '    else:\n'
        '        tags = ["stage1"]  # legacy single-job path\n'
        '\n'
        '    all_texts = []\n'
        '    for tag in tags:\n'
        '        job = get_recorded_job(tag)\n'
        '        if job.state != JobState.JOB_STATE_SUCCEEDED:\n'
        '            print(f"⏳ {tag} not ready (state={job.state}). Re-run retrieve_stage1() later.")\n'
        '            return None\n'
        '        all_texts.extend(retrieve_response_texts(job, GCS_BUCKET))\n'
        '    print(f"[retrieve] {len(all_texts):,} total response rows from {len(tags)} job(s)")\n'
        '\n'
        '    if False:  # dummy block to match old indentation flow\n'
        '        pass'
    )

    if OLD_RETRIEVE in src:
        src = src.replace(OLD_RETRIEVE, NEW_RETRIEVE, 1)
        changes.append('4a9b6ed7: retrieve_stage1 updated for multi-chunk')
    else:
        print("WARNING: retrieve_stage1 OLD pattern not found — skipping that part")

    compile(src, '4a9b6ed7', 'exec')
    lines_list = src.split('\n')
    cell['source'] = [l + '\n' for l in lines_list[:-1]] + ([lines_list[-1]] if lines_list[-1] else [])
    changes.append('4a9b6ed7: submit_stage1 split into chunks')
    break

# ─── 3. Remove labeling_rationale from pb-label (cell pb-label) ───────────────
for cell in nb['cells']:
    if cell.get('id') != 'pb-label':
        continue
    src = ''.join(cell['source'])

    # Remove from the schema description in PB_NAMING_SYSTEM
    OLD_SCHEMA = (
        '  "  labeling_rationale (1-2 sentences: WHY this label — which stats or comment patterns\\n"\n'
        '    "    were decisive, and what distinguishes it from adjacent clusters).\\n"\n'
        '    "Output ONLY a valid JSON array of objects with exactly those 6 keys. No preamble, no markdown fences."'
    )
    NEW_SCHEMA = (
        '    "Output ONLY a valid JSON array of objects with exactly those 5 keys. No preamble, no markdown fences."'
    )

    # More flexible: just fix the count and remove rationale lines
    src = src.replace(
        '  labeling_rationale (1-2 sentences: WHY this label — which stats or comment patterns\n'
        '    were decisive, and what distinguishes it from adjacent clusters).\n',
        ''
    )
    src = src.replace(
        '"  labeling_rationale (1-2 sentences: WHY this label — which stats or comment patterns\\n"\n'
        '    "    were decisive, and what distinguishes it from adjacent clusters).\\n"\n',
        ''
    )
    # Fix key count 6 -> 5
    src = src.replace('exactly these 6 keys:', 'exactly these 5 keys:')
    src = src.replace('exactly 6 keys', 'exactly 5 keys')
    # Remove labeling_rationale from _keys list
    src = src.replace(
        '["codename", "label", "description", "quantitative_signals", "example_comments", "labeling_rationale"]',
        '["codename", "label", "description", "quantitative_signals", "example_comments"]'
    )
    # Remove labeling_rationale from persona printing
    src = src.replace(
        '    if p.get("labeling_rationale"):\n'
        '        print(f"      rationale: {p[\'labeling_rationale\'][:120]}")\n',
        ''
    )

    try:
        compile(src, 'pb-label', 'exec')
    except SyntaxError as e:
        print(f"SyntaxError in pb-label: {e}")
        sys.exit(1)
    lines_list = src.split('\n')
    cell['source'] = [l + '\n' for l in lines_list[:-1]] + ([lines_list[-1]] if lines_list[-1] else [])
    changes.append('pb-label: removed labeling_rationale from schema (6->5 keys)')
    break

# ─── 4. Update pb-show-rationales to drop rationale column ────────────────────
for cell in nb['cells']:
    if cell.get('id') != 'pb-show-rationales':
        continue
    src = ''.join(cell['source'])
    NEW_SRC = (
        '# Display full persona descriptions\n'
        'from IPython.display import Markdown, display\n'
        '\n'
        'for persona in _taxonomy:\n'
        '    codename = persona["codename"]\n'
        '    label = persona["label"]\n'
        '    description = persona["description"]\n'
        '    signals = persona.get("quantitative_signals", [])\n'
        '    examples = persona.get("example_comments", [])\n'
        '\n'
        '    signals_md = "\\n".join(f"- {s}" for s in signals)\n'
        '    examples_md = "\\n".join(f"> {e}" for e in examples)\n'
        '\n'
        '    md = f"""\n'
        '### **{codename}** — {label}\n'
        '\n'
        '{description}\n'
        '\n'
        '**Key signals:**  \n'
        '{signals_md}\n'
        '\n'
        '**Example comments:**  \n'
        '{examples_md}\n'
        '\n'
        '---\n'
        '"""\n'
        '    display(Markdown(md))\n'
    )
    compile(NEW_SRC, 'pb-show-rationales', 'exec')
    lines_list = NEW_SRC.split('\n')
    cell['source'] = [l + '\n' for l in lines_list[:-1]] + ([lines_list[-1]] if lines_list[-1] else [])
    changes.append('pb-show-rationales: shows codename/label/description/signals/examples (no rationale)')
    break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Saved.')
for c in changes:
    print(' -', c)
