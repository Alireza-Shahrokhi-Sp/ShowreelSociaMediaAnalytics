# Showreel Social Media Analytics

This repository contains the code, notes, and working artifacts for the social media analytics project.

## What this project does

- Prepares YouTube transcript data for analysis.
- Builds local transcript job bundles for Vertex AI batch processing.
- Stores project notes and pipeline design documents.
- Keeps related RFEC / clustering work for Instagram, Facebook, and TikTok in one place.

## Main folders

- `Data_Cleaned/` - transcript pipeline scripts, cleaned datasets, manifests, and pipeline docs.
- `llm_wiki/` - short internal notes on batching, hashing, and deterministic preprocessing.
- `Michele_OneDrive/` - RFEC and related analysis material.
- `.vscode/` - workspace settings.
- `.claude/` - local agent settings.

## Current pipeline entry points

- `Data_Cleaned/add_transcripts.py` - merges local transcript files into the main parquet.
- `Data_Cleaned/pipeline_tools/add_transcripts.py` - relocated transcript merge script.
- `Data_Cleaned/pipeline_tools/prepare_transcript_jobs.py` - builds JSONL job bundles and manifests for GCP.
- `Data_Cleaned/GCP_VERTEX_AI_PIPELINE.md` - pipeline design and architecture notes.

## Notebook usage

The notebook in `Data_Cleaned/Untitled-1.ipynb` is a working analysis notebook. It loads the transcript parquet and explores time-series and duration plots.

## Data expectations

The main transcript dataset is expected at:

- `Data_Cleaned/yt_videos_with_local_transcripts.parquet`

Several large generated artifacts are intentionally ignored or kept outside version control:

- transcript job bundles under `Data_Cleaned/gcp_jobs/`
- parquet / pickle intermediates
- transcript caches and runtime manifests

## Suggested workflow

1. Update transcripts or preprocessors in `Data_Cleaned/`.
2. Regenerate local job bundles with `prepare_transcript_jobs.py`.
3. Review manifests before any cloud upload.
4. Keep generated bulk artifacts out of Git unless they are small and intentionally curated.

## Notes for agents

- Read `.agent.md` first.
- Prefer local deterministic preprocessing before cloud inference.
- Do not deduplicate transcripts unless a task explicitly asks for it.
- Keep cloud payloads compact and idempotent.
