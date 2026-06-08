"""
Patches sentiment_pipeline.ipynb:
  - cell id='client'  : appends Pydantic models + get_batch_generation_config_dict()
  - cell id='requests': schema strings auto-generated from Pydantic models;
                        build_sentiment_line uses get_batch_generation_config_dict()
"""
import json

with open('sentiment_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('outputs/client_append.py', 'r', encoding='utf-8') as f:
    CLIENT_APPEND = f.read()

with open('outputs/requests_cell_new.py', 'r', encoding='utf-8') as f:
    REQUESTS_NEW = f.read()

patched = 0
for cell in nb['cells']:
    cid = cell.get('id')
    if cid == 'client':
        existing = ''.join(cell.get('source', []))
        cell['source'] = existing.rstrip() + '\n' + CLIENT_APPEND
        cell['outputs'] = []
        patched += 1
        print("Patched: client (appended Pydantic models + get_batch_generation_config_dict)")
    elif cid == 'requests':
        cell['source'] = REQUESTS_NEW
        cell['outputs'] = []
        patched += 1
        print("Patched: requests (schema strings auto-generated from Pydantic models)")

with open('sentiment_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write('\n')

print(f"Done. {patched}/2 cells patched.")
