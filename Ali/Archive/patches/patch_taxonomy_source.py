import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'Ali/persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

for cell in nb['cells']:
    if cell.get('id') != 'pb-show-rationales':
        continue
    src = ''.join(cell['source'])

    OLD = (
        '# Display full persona descriptions\n'
        'from IPython.display import Markdown, display\n'
        '\n'
        'for persona in _taxonomy:'
    )
    NEW = (
        '# Display full persona descriptions\n'
        'import json, os\n'
        'from IPython.display import Markdown, display\n'
        '\n'
        '_taxonomy = (\n'
        '    _taxonomy if "_taxonomy" in dir()\n'
        '    else json.load(open(TAXONOMY_JSON_PATH, encoding="utf-8")).get("final_taxonomy", [])\n'
        ')\n'
        '\n'
        'for persona in _taxonomy:'
    )

    assert OLD in src, "pattern not found"
    src = src.replace(OLD, NEW, 1)
    compile(src, 'pb-show-rationales', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('patched pb-show-rationales')
    break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
