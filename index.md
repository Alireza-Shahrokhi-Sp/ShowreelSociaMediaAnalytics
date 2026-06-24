# Showreel Social Media Analytics

**Politecnico di Milano — AFB Lab**
**Last updated:** 2026-06-24

---

## What this project does

This repository analyses the social media community of **Camihawke** (Show Reel Media Group) across four platforms. It combines traditional analytics (RFM clustering, time-series) with LLM-based analysis (sentiment, personas) powered by Google Vertex AI.

**Dataset scale:**

| Platform | Posts | Comments | Coverage |
|----------|-------|----------|----------|
| Instagram | 1,675 | 573K | 94.7% (Apr 2015 - Mar 2026) |
| Facebook | 390 | 394K | ~100% (Nov 2016 - Mar 2026) |
| TikTok | 335 | 19.6K | 27.6% (Jan 2020 - Mar 2026) |
| YouTube | 11.8K videos | ~4M comments | In progress |

---

## Repository layout

```
AFB_Lab/
|-- Ali/                  Main analytics (sentiment, personas, modeling, EDA)
|   |-- outputs/          Pipeline outputs (parquets, figs, model artifacts)
|   |-- persona/          OOP persona pipeline package
|   |-- Sentiment_EDA/    EDA notebooks + chart outputs
|   |-- Archive/          Superseded scripts, plans, notebooks
|   |-- llm_wiki_afb/     Internal Obsidian wiki
|-- Data/                 Raw cleaned datasets (parquet) — source of truth
|-- Mickey/               RFM/RCEDTG clustering and user lifecycle analysis
|-- reza/                 YouTube transcript pipeline, NER, thumbnail CV
|-- wiki/                 Project findings and methodology documentation
|-- .github/skills/       CI data-profiling utilities
```

Each team member owns a top-level directory. Shared raw data lives in `Data/`.

---

## Ali/ — Sentiment, Personas & Modeling

The primary working directory. All production pipelines live here as Jupyter notebooks.

### Production notebooks

| Notebook | What it does | Key output |
|----------|-------------|------------|
| `sentiment_pipeline.ipynb` | Per-comment LLM sentiment via Vertex Batch | `outputs/stage2_sentiment/sentiment_instagram.parquet` |
| `persona_pipeline.ipynb` | Two-stage user persona classification (10 types) | `outputs/stage2_persona_combined/user_personas_combined.parquet` |
| `Data_Preparation_Pipeline.ipynb` | ETL: normalize IG/FB/TK into unified schema | `outputs/comments_ml.parquet` |
| `modeling_pipeline.ipynb` | 4 supervised ML tasks (XGBoost) | `outputs/modeling/` |
| `event_impact_pipeline.ipynb` | Interrupted time-series for 17 career events | `outputs/event_impact/` |
| `virality_pipeline.ipynb` | SARIMAX counterfactual: announcement vs event | `outputs/virality/` |

### Reference notebooks

| Notebook | Purpose |
|----------|---------|
| `Vertex_Batch_Inference.ipynb` | GCP Vertex AI batch job tutorial |

### EDA notebooks (`Sentiment_EDA/`)

| Notebook | Purpose |
|----------|---------|
| `sentiment_eda.ipynb` | Sentiment distribution validation, stratified samples |
| `persona_sentiment_rfm.ipynb` | Cross-analysis: personas x sentiment x RFM clusters |

### Python scripts

| Script | Purpose |
|--------|---------|
| `prompts.py` | Central prompt registry for all Vertex AI / Gemini calls (single source of truth) |
| `parse_comments.py` | Split prepped HeteroGraph exports per platform; extended community-vibe schema |
| `sentiment_analysis.py` | OOP EDA over sentiment parquets: summary tables, toxicity drill-down, figures |
| `run_stage2_async.py` | High-throughput async persona Stage 2 classification (aiohttp, bypasses SDK) |
| `run_stage2_async_sample.py` | Same as above, 20% sample variant |
| `merge_persona_runs.py` | Merge multiple Stage 2 runs by highest confidence per user |
| `patch_persona_codenames.py` | Normalize hallucinated LLM codenames (e.g. missing `THE_` prefix) |

### Persona package (`persona/`)

Object-oriented CLI port of the persona pipeline. Run with `python -m persona <subcommand>`.

| Module | Purpose |
|--------|---------|
| `__main__.py` | CLI entry point: submit-stage1, retrieve-stage1, consolidate, pathway-b, submit-stage2, etc. |
| `pipeline.py` | `PersonaPipeline` facade wiring all components together |
| `config.py` | `PipelineConfig` dataclass (GCP project, bucket, paths, model settings) |
| `data.py` | `DataLoader` — lazy load comments + user features |
| `features.py` | `FeatureEngineer` — per-user feature aggregation |
| `media.py` | `MediaContextBuilder` — multimodal media attachment |
| `stages.py` | Stage 1 discovery, Stage 2 classification, taxonomy consolidation |
| `batch.py` | `BatchClient` — Vertex AI batch job submit/retrieve |
| `clustering.py` | Pathway B (UMAP + HDBSCAN) and legacy Stage 3 clustering |

### Internal wiki (`llm_wiki_afb/`)

A structured Obsidian-compatible wiki documenting every pipeline, schema, and design decision. Start at [Ali/llm_wiki_afb/index.md](Ali/llm_wiki_afb/index.md).

### Archive (`Archive/`)

Superseded notebooks, one-shot scripts, and historical plans. Nothing here is imported by production code.

| Subdirectory | Contents |
|-------------|----------|
| `patches/` | Notebook cell-patching scripts (patch_cells, client_cell_new, etc.) |
| `eda_scripts/` | One-shot EDA inject/strip/investigate scripts |
| `plans/` | Historical implementation plans (EVENT_IMPACT, MODELING, VIRALITY) |
| `ig_media_scripts/` | Legacy Instagram media download/processing scripts |
| `colab_download_pipeline/` | Colab-based media download pipeline |
| `to_download/` | Completed download queue files |

---

## Data/ — Raw Datasets

Immutable cleaned exports from platform APIs. These are never modified by downstream pipelines.

| File | Rows | Description |
|------|------|-------------|
| `ig_posts_cleaned.parquet` | 1,675 | Instagram posts (caption, engagement, media type) |
| `ig_comments_cleaned.parquet` | 573K | Instagram comments with reply structure |
| `fb_posts_clean.parquet` | 390 | Facebook posts with reaction breakdown |
| `fb_comments_clean.parquet` | 394K | Facebook comments and replies |
| `tk_posts_clean.parquet` | 335 | TikTok videos |
| `tk_comments_clean.parquet` | 19.6K | TikTok comments |
| `YTcomments_[1-4]_cleaned.parquet` | ~4M | YouTube comments (4 shards) |

See [Data/DATASET_README.md](Data/DATASET_README.md) for full column-level schemas.

---

## Mickey/ — RFM Clustering & User Lifecycle

User behavioral segmentation using RCEDTG features (Recency, Coverage, Engagement, Delay, Tenure, Gini).

| Analysis | Location | Method |
|----------|----------|--------|
| IG rolling quadrimester | `RFM_IG_rolling_quadrimesters/` | 40 monthly sliding 4-month windows, KMeans k=4 |
| IG 3-year periods | `RFM_IG_3_years/` | 3 disjoint periods, KMedoids k=4 |
| TK rolling quadrimester | `RFM_TK_rolling_quadrimesters/` | Same methodology on TikTok |
| TK 3-year periods | `RFM_TK_3_years/` | Same methodology on TikTok |
| Cross-platform EDA | `EDA_cluster_matrix_IG_vs_TK.ipynb` | IG vs TK cluster comparison |
| User journey | `User_Cluster_Journey_Analysis.ipynb` | Lifecycle transitions across windows |
| YouTube affinity | `Topic-affinity-index-Camihawke/` | Cosine similarity to Camihawke topic vector |

### Markov analysis scripts (`RFM_IG_rolling_quadrimesters/markov_analysis/`)

| Script | Purpose |
|--------|---------|
| `markov_common.py` | Shared loaders, state coding, left-censoring, transition counting |
| `markov_stationarity_tests.py` | Order testing (LR + AIC/BIC), Anderson-Goodman homogeneity, marginal stationarity |
| `markov_absorbing_churn.py` | Absorbing-state churn model with duration-based threshold; survival curves |
| `markov_mature_refit.py` | Refit recurrent chain on stationary window (2024-09..2026-03) vs full span |

Outputs: `eda_cluster_matrix_outputs/` (17 charts), `eda_rfm_outputs/` (distribution plots), `markov_analysis/outputs/` (transition matrices, survival tables).

---

## reza/ — YouTube Pipeline

Transcript processing, named entity recognition, topic clustering, and thumbnail computer vision.

### NER scripts (`ner/`)

| Script | Purpose |
|--------|---------|
| `extract_entities.py` | spaCy NER (it_core_news_lg) over transcripts + comments; entity resonance |
| `export_entity_data.py` | Convert flat entity_resonance.parquet to nested per-video JSON |
| `reclean_and_rebuild_jobs.py` | Deep-clean local_transcript column (HTML unescape, VTT artifacts, dedup) + rebuild JSONL |

### Pipeline tools (`pipeline_tools/`)

| Script | Purpose |
|--------|---------|
| `add_transcripts.py` | Merge VTT/SRT/TXT transcripts into yt_videos_cleaned.parquet |
| `prepare_transcript_jobs.py` | Build prompt-ready JSONL bundles for GCP inference (chunking, dedup, manifest) |

### Thumbnail CV (`thumbnail_cv/`)

| Script | Purpose |
|--------|---------|
| `fetch_thumbnails.py` | Download YouTube thumbnails from CDN, upload to GCS |
| `thumbnail_features.py` | GCP Batch task: extract colour, face, text, CLIP features per thumbnail |
| `merge_thumbnail_features.py` | Consolidate per-task parquet shards into single thumbnail_features.parquet |

### Topic clustering (`topic_clustering/`)

| Script | Purpose |
|--------|---------|
| `topic_clustering.py` | Chapter-level UMAP + HDBSCAN; per-video compositional topic profiles |

See [reza/README.md](reza/README.md) and [reza/ANALYSIS_PLAN.md](reza/ANALYSIS_PLAN.md) for the full roadmap.

---

## wiki/ — Findings Documentation

Analytical findings synthesized from all pipelines. Start at [wiki/index.md](wiki/index.md).

| Page | What it covers |
|------|---------------|
| `instagram_findings.md` | Executive summary of sentiment, personas, RFM |
| `instagram_findings_detailed.md` | Full-detail reference with all raw numbers |
| `instagram_executive_summary.md` | 7-model inventory with metrics and readiness |
| `instagram_data_sources.md` | Data schemas and join architecture |
| `instagram_sentiment_pipeline.md` | Sentiment methodology and output schema |
| `instagram_persona_pipeline.md` | Persona taxonomy and classification |
| `instagram_rfm_clustering.md` | RFM design and cluster definitions |
| `instagram_advanced_analytics.md` | Modeling, event impact, virality |

---

## How to run

### Prerequisites

- Python 3.10+ with conda (environment spec: `Ali/environment.yml`)
- Google Cloud credentials for Vertex AI batch jobs
- `ma_env` conda environment at `D:\conda_envs\ma_env` (includes pyarrow 24, whisper, umap)

### Typical workflow

1. **Data prep:** Run `Ali/Data_Preparation_Pipeline.ipynb` to produce `comments_ml.parquet`
2. **Sentiment:** Run `Ali/sentiment_pipeline.ipynb` to classify comments via Vertex Batch
3. **Personas:** Run `Ali/persona_pipeline.ipynb` for user segmentation
4. **RFM:** Run notebooks in `Mickey/` for behavioral clustering
5. **Modeling:** Run `Ali/modeling_pipeline.ipynb` for supervised ML
6. **EDA:** Run `Ali/Sentiment_EDA/` notebooks for visualizations

### GCP setup

- Project: configured in `.env` (see `Ali/.env.example`)
- Bucket: `gs://afb_showreel/`
- See [Ali/llm_wiki_afb/concepts/vertex-batch-inference.md](Ali/llm_wiki_afb/concepts/vertex-batch-inference.md) for full setup guide

---

## Key results (Instagram)

- **Sentiment:** 80.9% positive, 14.1% neutral, 5.0% negative, 0.63% toxic
- **Personas:** 10 behavioral types identified (Tagger, Casual Complimenter, Emoji Reactor, Storyteller, Superfan, Inquirer, Critic, Hater, Spammer, Lurker-Liker)
- **RFM clusters:** 4 lifecycle segments (Brand Advocates, Expressive Regulars, Passive Regulars, Delayed Visitors)
- **Best model:** Persona classifier (XGBoost, macro-F1 = 0.36, 3.6x above majority baseline)
- **Causal finding:** Event impacts on loyalty cluster share detectable via ITS

For full results see [wiki/instagram_findings.md](wiki/instagram_findings.md).

---

## Team

| Member | Area | Directory |
|--------|------|-----------|
| Ali | Sentiment, personas, modeling, multimodal, EDA | `Ali/` |
| Mickey | RFM clustering, lifecycle analysis, user journeys | `Mickey/` |
| Reza | YouTube transcripts, NER, topic clustering, thumbnail CV | `reza/` |

**Repository:** [github.com/rezache4/ShowreelSociaMediaAnalytics](https://github.com/rezache4/ShowreelSociaMediaAnalytics)
