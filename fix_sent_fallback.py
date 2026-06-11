import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'Ali/persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

OLD = (
    'else:\n'
    '    ig_comments["sentiment_score"] = 0.0\n'
    '    ig_comments["is_toxic"]        = 0\n'
    '    ig_comments["is_negative"]     = 0\n'
    '    print(f"WARNING: {_sent_path} not found — sentiment features will be zero.")'
)

NEW = (
    'else:\n'
    '    ig_comments["sentiment_score"] = 0.0\n'
    '    ig_comments["is_toxic"]        = 0\n'
    '    ig_comments["is_negative"]     = 0\n'
    '    print(f"WARNING: {_sent_path} not found — sentiment features will be zero.")\n'
    '# Guard: ensure columns exist even if the branch above was skipped for any reason.\n'
    'for _col, _default in [("sentiment_score", 0.0), ("is_toxic", 0), ("is_negative", 0)]:\n'
    '    if _col not in ig_comments.columns:\n'
    '        ig_comments[_col] = _default'
)

for cell in nb['cells']:
    if cell['id'] != '76e3bb29':
        continue
    src = ''.join(cell['source'])
    assert OLD in src, "pattern not found"
    src = src.replace(OLD, NEW)
    compile(src, '76e3bb29', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('patched')
    break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
