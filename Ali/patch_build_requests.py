import json
import os

notebook_path = r'd:\Polythecninco di Milano\AFB_Lab\Ali\sentiment_pipeline.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'def build_requests(df: "pd.DataFrame") -> tuple[list[str], dict]:' in source and 'for media_id, g in df.groupby' in source:
            new_source = '''def select_comments() -> "pd.DataFrame":
    df = comments
    if MAX_COMMENTS and len(df) > MAX_COMMENTS:
        if PRIORITIZE_MEDIA_POSTS and ATTACH_MEDIA and posts_with_media:
            mask = df["media_id"].map(_norm_id).isin(posts_with_media)
            df = pd.concat([
                df[mask].sample(frac=1, random_state=SAMPLE_SEED),
                df[~mask].sample(frac=1, random_state=SAMPLE_SEED),
            ]).head(MAX_COMMENTS)
        else:
            df = df.sample(n=MAX_COMMENTS, random_state=SAMPLE_SEED)
    return df.reset_index(drop=True)


def build_requests(df: "pd.DataFrame") -> tuple[list[str], dict]:
    """Returns list of raw JSONL line strings + manifest dict."""
    import concurrent.futures
    line_strings, manifest = [], {}
    
    groups = list(df.groupby("media_id", sort=False))
    
    def process_group(args):
        rid_start, media_id, g = args
        lines = []
        mans = {}
        rid = rid_start
        for s in range(0, len(g), COMMENTS_PER_REQUEST):
            chunk  = g.iloc[s : s + COMMENTS_PER_REQUEST]
            req_id = f"R{rid}"
            line_str, man = build_batch_jsonl_line(req_id, chunk, PLATFORM, media_id)
            lines.append(line_str)
            mans[req_id] = man
            rid += 1
        return lines, mans

    group_args = []
    current_rid = 0
    for media_id, g in groups:
        group_args.append((current_rid, media_id, g))
        current_rid += (len(g) + COMMENTS_PER_REQUEST - 1) // COMMENTS_PER_REQUEST

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for lines, mans in executor.map(process_group, group_args):
            line_strings.extend(lines)
            manifest.update(mans)
            
    log.info("built %d JSONL lines for %d posts", len(line_strings), df["media_id"].nunique())
    return line_strings, manifest
'''
            lines = [line + '\n' for line in new_source.split('\n')]
            if lines:
                lines[-1] = lines[-1].rstrip('\n')
            
            cell['source'] = lines
            print('Cell updated successfully!')
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
