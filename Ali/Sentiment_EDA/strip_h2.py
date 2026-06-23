import json, pathlib
NB = pathlib.Path(r'd:/Polythecninco di Milano/AFB_Lab/Ali/Sentiment_EDA/persona_sentiment_rfm.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))
bad_ids = {'h2_header','h2_1_emo','h2_2_int','h2_3_tgt','h2_4_traj','h2_5_fp','h2_6_bub'}
before = len(nb['cells'])
nb['cells'] = [c for c in nb['cells'] if c['id'] not in bad_ids]
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Removed {before - len(nb["cells"])} cells, now {len(nb["cells"])} total')
