from typing import Literal, Optional
from pydantic import BaseModel, Field
import json

class SentimentItem(BaseModel):
    i:         int
    reasoning: str
    sentiment: Literal['positive', 'negative', 'neutral']
    score:     float
    emotion:   Literal[
        'joy', 'trust', 'fear', 'surprise',
        'sadness', 'disgust', 'anger', 'anticipation', 'neutral'
    ]
    intensity: Literal['low', 'medium', 'high']
    sarcasm:   bool
    toxicity:  Literal['none', 'mild', 'severe']
    intent:    Literal[
        'praise', 'affection', 'support', 'criticism',
        'question', 'suggestion', 'tag_share', 'spam_promo', 'joke', 'other'
    ]
    target:    Literal[
        'creator', 'appearance', 'content_work', 'product',
        'other_user', 'off_topic', 'none'
    ]
    lang:      str

class RoomRead(BaseModel):
    vibe: Literal['celebratory', 'supportive']
    consensus: float

class SentimentResponse(BaseModel):
    req:   str
    items: list[SentimentItem]
    room:  RoomRead

def _sanitize_schema(schema: dict) -> dict:
    defs = schema.pop('$defs', {})
    def _resolve(node: object) -> object:
        if isinstance(node, dict):
            if '$ref' in node:
                ref_name = node['$ref'].split('/')[-1]
                return _resolve(defs.get(ref_name, node))
            resolved = {
                k: _resolve(v)
                for k, v in node.items()
                if k not in ('title', '$schema')
            }
            if 'enum' in resolved and 'type' not in resolved:
                resolved['type'] = 'string'
            return resolved
        if isinstance(node, list):
            return [_resolve(v) for v in node]
        return node
    return _resolve(schema)

s = SentimentResponse.model_json_schema()
print(json.dumps(_sanitize_schema(s), indent=2))
