import json, pathlib
NB = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))
bad = {'j_header','j_load','j_l1','j_l2','j_l3','j_l4'}
before = len(nb['cells'])
nb['cells'] = [c for c in nb['cells'] if c['id'] not in bad]
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Removed {before - len(nb["cells"])}, now {len(nb["cells"])}')
