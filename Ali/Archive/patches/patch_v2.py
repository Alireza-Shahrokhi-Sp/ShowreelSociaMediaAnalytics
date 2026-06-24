"""
Patches sentiment_pipeline.ipynb:
  - cell id='client'   : full replacement with new Pydantic models +
                         enum-type-injecting _sanitize_schema +
                         hard-coded get_batch_generation_config_dict()
  - cell id='requests' : full replacement with new prompt + builder
"""
import json

NB_PATH = "sentiment_pipeline.ipynb"

with open("outputs/client_cell_new.py",  "r", encoding="utf-8") as f:
    CLIENT_NEW = f.read()

with open("outputs/requests_cell_new.py", "r", encoding="utf-8") as f:
    REQUESTS_NEW = f.read()

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = 0
for cell in nb["cells"]:
    cid = cell.get("id")
    if cid == "client":
        cell["source"]  = CLIENT_NEW
        cell["outputs"] = []
        patched += 1
        print("Patched: client")
    elif cid == "requests":
        cell["source"]  = REQUESTS_NEW
        cell["outputs"] = []
        patched += 1
        print("Patched: requests")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"Done. {patched}/2 cells patched.")
