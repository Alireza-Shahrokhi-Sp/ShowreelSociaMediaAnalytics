# Sentiment Response Schema

Vertex AI Batch `responseSchema` for `sentiment_pipeline.ipynb`.  
Generated from the Pydantic v2 models in the `client` notebook cell via `SentimentResponse.model_json_schema()` + `_sanitize_schema()`.

---

## Top-level structure

```json
{
  "req":   "<string>  -- echoes the request ID from the manifest",
  "items": [ <SentimentItem>, ... ],
  "room":  <RoomRead>
}
```

One `SentimentItem` per comment (index `i` maps back to the comment order in the prompt).  
One `RoomRead` covering all comments in the chunk collectively.

---

## SentimentItem

| Field | Type | Constraint | Semantics |
|-------|------|-----------|-----------|
| `i` | integer | — | 0-based index matching the comment's position in the prompt |
| `sentiment` | enum | `positive \| negative \| neutral \| mixed` | Categorical sentiment label |
| `score` | number | `[-1.0, 1.0]` | Continuous polarity: -1 = very negative, 0 = neutral, +1 = very positive |
| `emotion` | enum | see below | Plutchik's 8 core emotions + neutral |
| `intensity` | enum | `low \| medium \| high` | Strength of the detected emotion |
| `sarcasm` | boolean | — | True if the comment is ironic/sarcastic |
| `toxicity` | enum | `none \| mild \| severe` | Abusiveness level |
| `intent` | enum | see below | Commenter's communicative purpose |
| `target` | enum | see below | Who or what the comment is directed at |
| `lang` | string | ISO-639-1 | Detected language code (e.g. `it`, `en`) |

**Required:** all fields.

### `emotion` values — Plutchik's 8 core emotions

```
joy | trust | fear | surprise | sadness | disgust | anger | anticipation | neutral
```

Replaces the 19-class GoEmotions vocabulary to prevent class sparsity in downstream Random Forest and UMAP clustering.

### `intent` values

```
praise | affection | support | criticism | question | suggestion | tag_share | spam_promo | joke | other
```

### `target` values

```
creator | appearance | content_work | product | other_user | off_topic | none
```

---

## RoomRead

One judgement across all comments in the chunk, read holistically alongside post media.

| Field | Type | Constraint | Semantics |
|-------|------|-----------|-----------|
| `vibe` | enum | see below | Qualitative atmosphere of the comment section |
| `consensus` | number | `[0.0, 1.0]` | How strongly commenters agree with each other; 1 = unanimous |
| `alignment` | number | `[-1.0, 1.0]` | Overall stance toward creator/post; +1 = all support, -1 = all against, 0 = split |
| `controversy` | number | `[0.0, 1.0]` | Strength of opposing camps actively arguing; 1 = intense debate |
| `sponsorship_alignment` | number | `[0.0, 1.0]` | Audience receptiveness to ad/commercial content; **return 0.5 if post is not sponsored** |
| `split_axis` | string \| null | ≤ 5 words | What divides the room, or `null` if united |
| `dominant_stance` | string | — | Majority position in one short phrase |
| `summary` | string | — | 1–2 sentence narrative of the room vibe |

**Required:** all fields except `split_axis` (nullable).

### `vibe` values

```
celebratory | supportive | appreciative | amused | mixed | debated | divided | critical | hostile | spam_heavy | neutral
```

---

## Flattened OpenAPI 3.0 schema (as sent to Vertex)

This is the exact dict embedded in `generationConfig.responseSchema` of every JSONL line.  
`_sanitize_schema()` removes `$defs`/`$ref` because Vertex does not resolve them inline.

```json
{
  "type": "object",
  "required": ["req", "items", "room"],
  "properties": {
    "req": { "type": "string" },
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["i", "sentiment", "score", "emotion", "intensity", "sarcasm", "toxicity", "intent", "target", "lang"],
        "properties": {
          "i":         { "type": "integer" },
          "sentiment": { "type": "string", "enum": ["positive", "negative", "neutral", "mixed"] },
          "score":     { "type": "number", "minimum": -1.0, "maximum": 1.0 },
          "emotion":   { "type": "string", "enum": ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "neutral"] },
          "intensity": { "type": "string", "enum": ["low", "medium", "high"] },
          "sarcasm":   { "type": "boolean" },
          "toxicity":  { "type": "string", "enum": ["none", "mild", "severe"] },
          "intent":    { "type": "string", "enum": ["praise", "affection", "support", "criticism", "question", "suggestion", "tag_share", "spam_promo", "joke", "other"] },
          "target":    { "type": "string", "enum": ["creator", "appearance", "content_work", "product", "other_user", "off_topic", "none"] },
          "lang":      { "type": "string" }
        }
      }
    },
    "room": {
      "type": "object",
      "required": ["vibe", "consensus", "alignment", "controversy", "sponsorship_alignment", "dominant_stance", "summary"],
      "properties": {
        "vibe":                  { "type": "string", "enum": ["celebratory", "supportive", "appreciative", "amused", "mixed", "debated", "divided", "critical", "hostile", "spam_heavy", "neutral"] },
        "consensus":             { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "alignment":             { "type": "number", "minimum": -1.0, "maximum": 1.0 },
        "controversy":           { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "sponsorship_alignment": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "split_axis":            { "anyOf": [{"type": "string"}, {"type": "null"}], "default": null },
        "dominant_stance":       { "type": "string" },
        "summary":               { "type": "string" }
      }
    }
  }
}
```

---

## Source of truth

The schema is defined in `sentiment_pipeline.ipynb`, `client` cell, as three Pydantic v2 classes:

```
SentimentItem      -> items[*]
RoomRead           -> room
SentimentResponse  -> top-level envelope (req + items + room)
```

To change the schema, edit those classes. `get_batch_generation_config_dict(SentimentResponse, ...)` and `_model_to_prompt_schema()` pick up changes automatically — no manual sync needed.

---

## Notes on Vertex AI compatibility

- `responseMimeType: "application/json"` must be set alongside `responseSchema` (Vertex requires both).
- `thinkingConfig.thinkingBudget` counts against `maxOutputTokens`, so `MAX_OUTPUT_TOKENS = 8192` must cover thinking tokens + the full JSON array.
- Pydantic v2 emits `$defs` for nested models; `_sanitize_schema()` resolves all `$ref` inline before submission because Vertex does not follow `$ref` in batch JSONL.
- `title` and `$schema` keys are stripped — Vertex rejects them.

---

## Implementation Protocol

This section codifies the exact construction rules for any future pipeline that needs a strict JSON schema enforced at the Vertex AI Batch API level. Follow these steps in order every time a new schema is introduced or an existing one is changed.

---

### Rule 1 — Define schema in Pydantic first, never in raw JSON

All schema structure lives in Pydantic v2 `BaseModel` classes. Raw JSON schema is always derived, never hand-written.

```python
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

class MyItem(BaseModel):
    i:         int                                    # index anchor — always first
    reasoning: str = Field(..., description="≤40-word chain-of-thought scratchpad")
    label:     Literal["a", "b", "c"]                # enum — no Field() needed
    score:     float = Field(..., ge=-1.0, le=1.0)   # bounded float
    flag:      bool
    lang:      str = Field(..., description="ISO-639-1")
```

**Why:** Pydantic is the single source of truth. Both `responseSchema` (API enforcement) and system-prompt field descriptions derive from the same class. They cannot drift from each other.

---

### Rule 2 — Field order is token order; put the scratchpad second

```
i  →  reasoning  →  all label fields
```

Vertex AI streams tokens left-to-right in schema field order. Placing `reasoning: str` immediately after the index anchor forces the model to commit its chain-of-thought **before** any label token is generated. Labels produced after the scratchpad are conditioned on it — this is the mechanism that prevents categorical hallucination (e.g. an emotion value leaking into a sentiment field).

Never move `reasoning` after any label field. Never omit it.

---

### Rule 3 — Run `_sanitize_schema()` before embedding; always inject `"type"` on enum nodes

```python
def _sanitize_schema(schema: dict) -> dict:
    defs = schema.pop("$defs", {})

    def _resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return _resolve(defs.get(ref_name, node))
            resolved = {k: _resolve(v) for k, v in node.items()
                        if k not in ("title", "$schema")}
            # CRITICAL: inject "type": "string" on every enum node that lacks it.
            # Without this Vertex silently disables logit masking for that field.
            if "enum" in resolved and "type" not in resolved:
                resolved["type"] = "string"
            return resolved
        if isinstance(node, list):
            return [_resolve(v) for v in node]
        return node

    return _resolve(schema)
```

Three transformations are mandatory, in this order:

| # | Transform | Why |
|---|---|---|
| 1 | Resolve `$ref` by inlining from `$defs` | Vertex does not follow `$ref` in batch JSONL |
| 2 | Strip `"title"` and `"$schema"` keys | Vertex rejects both |
| 3 | Inject `"type": "string"` on enum nodes missing a type | Without it Vertex parses the constraint but silently ignores it — logit masking does not engage and the model produces unconstrained token distributions |

---

### Rule 4 — `get_batch_generation_config_dict()` takes no arguments; settings are hard-coded

```python
def get_batch_generation_config_dict() -> dict:
    raw_schema   = TopLevelResponseModel.model_json_schema()
    clean_schema = _sanitize_schema(raw_schema)
    return {
        "temperature":      0.0,           # deterministic — no stochastic label drift
        "maxOutputTokens":  MAX_OUTPUT_TOKENS,
        "responseMimeType": "application/json",   # required alongside responseSchema
        "responseSchema":   clean_schema,
        "thinkingConfig":   {"thinkingBudget": THINKING_BUDGET},
    }
```

Do not pass `topP` at `temperature=0` — greedy decoding ignores it and extra keys create unnecessary Vertex validation surface.

Do not accept `response_model`, `max_tokens`, or `thinking_budget` as parameters. Call-site misconfiguration is a whole class of bugs; zero-argument functions eliminate it.

---

### Rule 5 — The JSONL builder returns a `str`, not a `dict`

```python
def build_batch_jsonl_line(req_id: str, chunk: pd.DataFrame, ...) -> tuple[str, dict]:
    parts = [...]                          # assemble contents list
    gen_cfg = get_batch_generation_config_dict()
    payload = {
        "request": {
            "contents":         [{"role": "user", "parts": parts}],
            "generationConfig": gen_cfg,
        }
    }
    try:
        line_str = json.dumps(payload, ensure_ascii=False)
    except TypeError as exc:
        raise TypeError(f"req={req_id} not serializable: {exc}") from exc
    return line_str, manifest_entry       # str, not dict
```

The `json.dumps()` call inside the builder (not at write time) is the **serializability guard**. If any Pydantic object, SDK type, or non-serializable value leaked into the payload, the error is raised here — at build time — not silently at GCS upload time when debugging is much harder.

Callers write directly:
```python
with open(path, "w", encoding="utf-8") as f:
    for line in line_strings:
        f.write(line + "\n")
```

---

### Rule 6 — Keep LLM extraction and business logic in separate layers

The LLM labels **raw text valence**. It does not apply business rules.

Post-processing transforms (e.g. zeroing brand scores for off-topic comments) run as vectorized Pandas operations on the retrieved DataFrame — not as prompt instructions.

```
Vertex Batch job
      │
      ▼
retrieve_sentiment()       ← raw LLM outputs: true sentiment, true target
      │
      ▼
calculate_brand_equity_scores(df)   ← business rule applied here, not in the prompt
      │
      ▼
brand_sentiment / brand_score columns
```

**Why:** Prompt-level business rules pollute the neutral class, distort score calibration, and cannot be changed without resubmitting an expensive batch job. A Pandas function can be rerun on the cached results in milliseconds.

---

### Rule 7 — System prompt describes fields in prose; never inject JSON strings

Do not call `_model_to_prompt_schema()` or any helper that stringifies the Pydantic model into the prompt. Field descriptions in the prompt are prose — they tell the model what to *think*, not what to *output*. The schema tells the API what to *enforce*.

The two layers are independent by design:

| Layer | What it does | Where it lives |
|---|---|---|
| `responseSchema` in `generationConfig` | Hardware-level logit masking — constrains which tokens are physically possible | `get_batch_generation_config_dict()` |
| System prompt | Reasoning instructions — tells the model how to arrive at the right token | `sentiment_system_prompt()` |

Injecting a JSON schema string into the prompt conflates these two roles and creates a third source of truth that can drift from both.

---

### Checklist — before every batch submission

- [ ] Schema change? → Edit Pydantic class only. Do not touch the JSON schema directly.
- [ ] New enum field? → Confirm `_sanitize_schema` will inject `"type": "string"` (it will, automatically).
- [ ] `get_batch_generation_config_dict()` called with no arguments.
- [ ] `json.dumps(payload)` guard present inside the JSONL builder.
- [ ] `reasoning` field is the second field in every `BaseModel` that represents a scored item.
- [ ] `responseMimeType: "application/json"` present alongside `responseSchema`.
- [ ] `thinkingBudget + expected output tokens ≤ MAX_OUTPUT_TOKENS`.
- [ ] Post-processing business logic is in a Pandas function, not in the prompt.
