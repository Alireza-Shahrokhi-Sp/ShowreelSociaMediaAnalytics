import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'Ali/persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

OLD = '    print(f"Sentiment joined: {_sent["comment_id"].nunique():,} comments enriched.")\n'
NEW = '    _n_sent = _sent["comment_id"].nunique()\n    print(f"Sentiment joined: {_n_sent:,} comments enriched.")\n'

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
