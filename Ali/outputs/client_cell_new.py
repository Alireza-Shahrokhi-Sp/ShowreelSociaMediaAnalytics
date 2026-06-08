from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState

client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)

# ---------------------------------------------------------------------------
# Step 1: Pydantic Structural Contract
# ---------------------------------------------------------------------------
# SentimentItem field order is deliberate:
#   i          -> index anchor (maps comment back to the prompt list)
#   reasoning  -> chain-of-thought scratchpad; forces the model to commit
#                 to its analysis BEFORE emitting any label tokens
#   <labels>   -> all enum/float fields follow after the reasoning anchor
#
# "mixed" removed from sentiment: ambiguity belongs in score ∈ (-0.3, 0.3),
# not in a fourth categorical class that the classifier can hide behind.
# ---------------------------------------------------------------------------

class SentimentItem(BaseModel):
    i:         int
    reasoning: str = Field(
        ...,
        description=(
            "≤40 words of chain-of-thought: sarcasm verdict, target, "
            "then sentiment rationale. Written BEFORE any label is emitted."
        ),
    )
    sentiment: Literal["positive", "negative", "neutral"]
    score:     float = Field(
        ..., ge=-1.0, le=1.0,
        description="-1.0 = very negative, 0.0 = neutral, +1.0 = very positive",
    )
    emotion:   Literal[
        "joy", "trust", "fear", "surprise",
        "sadness", "disgust", "anger", "anticipation", "neutral"
    ]
    intensity: Literal["low", "medium", "high"]
    sarcasm:   bool
    toxicity:  Literal["none", "mild", "severe"]
    intent:    Literal[
        "praise", "affection", "support", "criticism",
        "question", "suggestion", "tag_share", "spam_promo", "joke", "other"
    ]
    target:    Literal[
        "creator", "appearance", "content_work", "product",
        "other_user", "off_topic", "none"
    ]
    lang:      str = Field(..., description="ISO-639-1, e.g. it, en")


class RoomRead(BaseModel):
    vibe: Literal[
        "celebratory", "supportive", "appreciative", "amused",
        "mixed", "debated", "divided", "critical",
        "hostile", "spam_heavy", "neutral"
    ]
    consensus:             float = Field(..., ge=0.0, le=1.0)
    alignment:             float = Field(..., ge=-1.0, le=1.0)
    controversy:           float = Field(..., ge=0.0, le=1.0)
    sponsorship_alignment: float = Field(..., ge=0.0, le=1.0)
    split_axis:            Optional[str] = Field(
        None, description="<=5 words naming the dividing axis, or null if room is united"
    )
    dominant_stance: str
    summary:         str


class SentimentResponse(BaseModel):
    req:   str
    items: list[SentimentItem]
    room:  RoomRead


# ---------------------------------------------------------------------------
# Step 2: Vertex AI Schema Sanitizer
# ---------------------------------------------------------------------------

def _sanitize_schema(schema: dict) -> dict:
    """
    Flatten Pydantic v2 $defs/$ref for Vertex AI OpenAPI 3.0 subset.

    Two fixes applied recursively:
      1. $ref resolution  — Vertex does not follow $ref in batch JSONL;
                            inline the definition directly.
      2. enum type injection — Vertex silently falls back to free-text
                               generation when an enum node lacks "type".
                               Inject "type": "string" whenever "enum" is
                               present and "type" is absent.
      3. key stripping    — "title" and "$schema" are rejected by Vertex.
    """
    defs = schema.pop("$defs", {})

    def _resolve(node: object) -> object:
        if isinstance(node, dict):
            # 1. Resolve $ref before anything else
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return _resolve(defs.get(ref_name, node))

            resolved = {
                k: _resolve(v)
                for k, v in node.items()
                if k not in ("title", "$schema")
            }

            # 2. Inject "type": "string" when enum is present but type is absent
            if "enum" in resolved and "type" not in resolved:
                resolved["type"] = "string"

            return resolved

        if isinstance(node, list):
            return [_resolve(v) for v in node]

        return node

    return _resolve(schema)


def get_batch_generation_config_dict() -> dict:
    """
    Returns the fully json.dumps()-serializable generationConfig for one
    Vertex Batch JSONL line.

    Hard-coded decisions:
      temperature=0.0   — deterministic; avoids stochastic label drift
      thinkingBudget=512 — enough for sarcasm/irony reasoning without
                           eating into the 8192-token output budget
      topP omitted       — irrelevant at temperature=0; fewer keys = fewer
                           Vertex validation surface errors
    """
    raw_schema   = SentimentResponse.model_json_schema()
    clean_schema = _sanitize_schema(raw_schema)

    return {
        "temperature":      0.0,
        "maxOutputTokens":  MAX_OUTPUT_TOKENS,
        "responseMimeType": "application/json",
        "responseSchema":   clean_schema,
        "thinkingConfig":   {"thinkingBudget": THINKING_BUDGET},
    }


import logging
log = logging.getLogger("sentiment_pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

log.info("genai Vertex client ready: %s  %s", GCP_PROJECT_ID, GCP_LOCATION)
log.info("Pydantic models + get_batch_generation_config_dict() ready.")
