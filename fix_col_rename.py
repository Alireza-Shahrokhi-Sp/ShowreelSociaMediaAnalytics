import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'Ali/persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

for cell in nb['cells']:
    if cell['id'] != '76e3bb29':
        continue
    src = ''.join(cell['source'])
    src = src.replace(
        '"room_sponsorship_alignment"',
        '"stance_alignment"'
    ).replace(
        '"room_sponsorship_alignment"]',
        '"stance_alignment"]'
    ).replace(
        'room_sponsorship_alignment',
        'stance_alignment'
    )
    compile(src, '76e3bb29', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('patched')
    break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
