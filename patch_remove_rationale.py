import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'Ali/persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

for cell in nb['cells']:
    if cell.get('id') != 'pb-label':
        continue
    src = ''.join(cell['source'])

    # Fix closing instruction: 6 keys -> 5 keys
    src = src.replace(
        '"Output ONLY a valid JSON array of objects with exactly those 6 keys. No preamble, no markdown fences."',
        '"Output ONLY a valid JSON array of objects with exactly those 5 keys. No preamble, no markdown fences."'
    )

    # Fix default value dict: remove labeling_rationale from the tuple
    src = src.replace(
        'p.get(k, "" if k in ("codename","label","description","labeling_rationale") else [])',
        'p.get(k, "" if k in ("codename","label","description") else [])'
    )

    compile(src, 'pb-label', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('patched pb-label')
    break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
