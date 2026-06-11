import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'Ali/persona_pipeline.ipynb'
nb = json.load(open(path, encoding='utf-8'))

for cell in nb['cells']:
    if cell.get('id') != 'pb-sentiment-persona-plots':
        continue
    src = ''.join(cell['source'])

    OLD = (
        'TOX_ORDER   = ["none", "mild", "severe", "spam_promo"]\n'
        'TOX_COLORS  = {"none": "#9ecae1", "mild": "#fdae6b", "severe": "#d62728", "spam_promo": "#756bb1"}\n'
        '\n'
        '# ── figure 1: stacked 100% bar — sentiment mix per persona'
    )
    NEW = (
        'TOX_ORDER   = ["none", "mild", "severe", "spam_promo"]\n'
        'TOX_COLORS  = {"none": "#9ecae1", "mild": "#fdae6b", "severe": "#d62728", "spam_promo": "#756bb1"}\n'
        '\n'
        '# ── persona descriptions ──────────────────────────────────────────────────────\n'
        '_tax_data = json.load(open(TAXONOMY_JSON_PATH, encoding="utf-8"))\n'
        '_tax_list = _tax_data.get("final_taxonomy", [])\n'
        '_tax_by_code = {p["codename"]: p for p in _tax_list}\n'
        'print("=" * 60)\n'
        'print("PERSONA DESCRIPTIONS")\n'
        'print("=" * 60)\n'
        'for _persona_name in _pct_sent.index:\n'
        '    _p = _tax_by_code.get(_persona_name, {})\n'
        '    _lbl = _p.get("label", "")\n'
        '    _desc = _p.get("description", "No description available.")\n'
        '    print(f"\\n{_persona_name}" + (f" — {_lbl}" if _lbl else ""))\n'
        '    print(f"  {_desc}")\n'
        'print("\\n" + "=" * 60 + "\\n")\n'
        '\n'
        '# ── figure 1: stacked 100% bar — sentiment mix per persona'
    )

    assert OLD in src, "insertion point not found"
    src = src.replace(OLD, NEW, 1)
    compile(src, 'pb-sentiment-persona-plots', 'exec')
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('patched pb-sentiment-persona-plots')
    break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
