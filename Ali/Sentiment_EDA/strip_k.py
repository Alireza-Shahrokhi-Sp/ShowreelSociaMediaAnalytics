import json, pathlib
NB = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))
bad = {'k_header','k_load','k_k1','k_k2','k_k3','k_k4','k_k5'}
before = len(nb['cells'])
nb['cells'] = [c for c in nb['cells'] if c['id'] not in bad]
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Removed {before - len(nb["cells"])}, now {len(nb["cells"])}')
