"""Patch all old cluster display names in notebook source cells."""
import json, pathlib

NB = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

REPLACEMENTS = [
    ("Expressive regulars", "Established regulars"),
    ("Delayed visitors",    "Occasional visitors"),
]

patched = 0
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    new_src = []
    for line in cell['source']:
        original = line
        for old, new in REPLACEMENTS:
            line = line.replace(old, new)
        if line != original:
            patched += 1
        new_src.append(line)
    cell['source'] = new_src

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Patched {patched} line(s)')
