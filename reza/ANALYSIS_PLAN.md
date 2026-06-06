# Embedding-to-Insight Analysis Plan

**Date:** June 2026  
**Status:** Phase 1 complete — proceeding to feature engineering  
**Data in hand:** 11,801 video embeddings (chapter-level, 768-dim, multilingual-e5-base), 9,029 VTT files, YouTube comments (4 parquet shards), engagement metrics

---

## What this plan is and isn't

This pipeline produces **associational findings**, not causal ones. The honest framing is:

> "Which content characteristics predict higher engagement, after controlling for channel identity and video format?"

Strict causality would require randomisation or a valid instrument (e.g., A/B tests). What we can claim: associations that survive channel fixed effects and format controls are credible signals for content strategy.

---

## Why not SAE (Sparse Autoencoder)

SAEs are trained on millions of LLM forward-pass activations. With 11,801 video-level vectors and ~25M SAE parameters the model operates ~100× below the data regime needed for clean monosemantic features. Even if training converged, running regression on 16K sparse predictors with 11K samples is severely underpowered (p >> n). Discarded.

---

## Phase 2A — Topic Clustering

**Runtime:** local CPU, ~10 min  
**Input:** `dense_vectors_semantic.parquet` (chapter-level)

1. Mean-pool chapter vectors per `video_id` → one 768-dim vector per video.
2. UMAP: 768 → 20 dims (for clustering) + 768 → 2 dims (for visualisation).
3. HDBSCAN on the 20-dim space → topic cluster assignments. Noise points (cluster = -1) get assigned to nearest cluster centroid.
4. Name each cluster: pull the 5 nearest-to-centroid transcripts → Gemini 2.5 Flash prompt → concise label (3–6 words).
5. Output: `topic_clusters.parquet` — columns `[video_id, cluster_id, cluster_label, umap_x, umap_y]`.

**Regression use:** one-hot dummy per cluster (drop one as reference). Coefficient = engagement lift vs. reference cluster, controlling for everything else.

---

## Phase 2B — Structural / Pacing Features

**Runtime:** trivial, already embedded  
**Input:** `dense_vectors_semantic.parquet` (chapter-level, multiple rows per long video)

Derived directly from the existing embeddings — no new model calls needed:

| Feature | Definition |
|---------|-----------|
| `n_chapters` | Number of semantic chapters detected per video |
| `topic_volatility` | Mean cosine distance between consecutive chapter vectors (long videos only; NULL for shorts) |
| `embedding_norm` | L2 norm of mean-pooled video vector (proxy for semantic density / specificity) |

High volatility = rapid topic switching / fast-paced editing. Low volatility = deep single-topic content.

---

## Phase 2C — Hook Extraction & Classification

**Runtime:** ~30–60 min local CPU (embedding 9K hooks)  
**Input:** `reza/transcripts/*.it.vtt` (9,029 files); fallback to parquet transcript for ~2,772 without VTT

**VTT path (preferred):**
1. Parse each `.it.vtt`: extract all cue text with start timestamp < 60.0 seconds. Deduplicate rolling cues (VTT repeats each cue as words accumulate).
2. Concatenate into a single hook string per video.
3. Embed all hook strings in batch → 768-dim hook vectors.

**Fallback path (no VTT):**
- Take first 120 words of the `local_transcript` column (word-count proxy for ~60s).

**Clustering:**
- HDBSCAN on hook vectors separately from video-level clusters.
- Name hook clusters with Gemini: "Direct problem statement", "Anecdotal opening", "Question hook", etc.
- Output: `hook_features.parquet` — `[video_id, hook_cluster_id, hook_cluster_label, hook_source]`

**Regression use:** hook cluster dummies as predictors of engagement.

---

## Phase 3 — Comment Analysis

**Runtime:** ~1–2 hrs local CPU (embedding aggregated comments)  
**Input:** `yt_comments_1..4_cleaned.parquet`; `reza/transcripts/*.it.vtt`

### 3A — Comment–Video Semantic Alignment
1. Aggregate all comment text per `video_id` into one string (or mean-pool individual comment embeddings if RAM allows).
2. Embed the aggregate → 768-dim comment vector.
3. Compute cosine similarity between comment vector and video's mean embedding.
4. High similarity = comments engage with video content intellectually. Low = generic reactions.
5. Feature: `comment_alignment` (scalar, 0–1).

### 3B — Timestamp Attribution
1. Regex `\b\d{1,2}:\d{2}(?::\d{2})?\b` over all comments per video → extract timestamp strings.
2. Count: `n_timestamp_comments` (engagement intensity proxy).
3. Optional: map timestamps back to VTT chapters → identify which chapter provokes the most comment timestamps (not a regression feature, more of an exploratory/qualitative output).

### 3C — Lexical Engagement Metrics
Per-video aggregates over comments:
- `mean_comment_words`: average comment length (effort proxy)
- `comment_lexical_diversity`: type-token ratio over all comments
- `comment_sentiment_entropy`: distribution entropy of positive/negative/neutral sentiment labels (measures debate vs. consensus)

Sentiment: use a multilingual classifier (e.g., `cardiffnlp/twitter-xlm-roberta-base-sentiment`) or a simple Italian sentiment lexicon if speed matters.

---

## Phase 4 — Regression

**Runtime:** < 5 min, local CPU  
**Input:** all features above + engagement metrics from `yt_videos_with_local_transcripts.parquet`

### Feature matrix

| Feature group | Columns |
|--------------|---------|
| Content type | `topic_cluster_*` dummies |
| Pacing | `n_chapters`, `topic_volatility` |
| Hook | `hook_cluster_*` dummies |
| Comment engagement | `comment_alignment`, `n_timestamp_comments`, `mean_comment_words`, `comment_sentiment_entropy` |
| Controls | `duration_seconds`, `is_short`, `C(channelTitle)` (fixed effects), `upload_year` |

### Outcome variables (separate models)
- `log1p(viewCount)` — reach
- `commentCount / viewCount` (CVR) — community activation
- `likeCount / viewCount` (LVR) — passive approval

### Model sequence
1. **OLS with robust SEs** — interpretable baseline, coefficients in natural units.
2. **Lasso (cross-validated λ)** — feature selection; identifies which predictors survive regularisation.
3. Anything with p < 0.05 in Lasso-selected OLS gets a **partial regression plot** for sense-checking.

### Honest limitations to report
- Channel fixed effects absorb most variance; within-channel variation drives coefficient estimates.
- Comment features are post-hoc (comments follow views, not the reverse) — treat them as engagement *characterisers*, not predictors.
- No causal claims. Findings support content strategy hypotheses, not A/B test conclusions.

---

## Deliverables

| Output | Phase | Format |
|--------|-------|--------|
| `topic_clusters.parquet` | 2A | parquet |
| `structural_features.parquet` | 2B | parquet |
| `hook_features.parquet` | 2C | parquet |
| `comment_features.parquet` | 3 | parquet |
| `regression_results.csv` | 4 | CSV |
| UMAP scatter (topic clusters) | 2A | PNG / notebook |
| Coefficient plot | 4 | PNG / notebook |

---

## Execution order

```
Phase 2B  (minutes — just math on existing embeddings)
Phase 2A  (~10 min — UMAP + HDBSCAN + Gemini labelling)
Phase 2C  (~1 hr — VTT parsing + hook embedding + clustering)
Phase 3   (~1–2 hrs — comment embedding + lexical stats)
Phase 4   (< 5 min — regression)
```

2B first because it's free. 2A before 2C because topic cluster labels help contextualise hook cluster naming.

---

## Phase 5 — Thumbnail Computer Vision

**Runtime:** ~15 min GPU compute + ~30 min VM setup  
**Compute:** GCP Compute Engine `g2-standard-4` (1× L4 GPU) — **must poweroff immediately after output is written to GCS**  
**Model:** CLIP ViT-B/32 via `open-clip-torch` — free, local, 512-dim image+text joint embeddings  
**Cost:** L4 instance (~$0.72/hr) × ~1 hr total ≈ **~$1–2**

### Why CLIP

CLIP encodes images and text in the same embedding space, enabling cosine similarity between a thumbnail image and the video title text. One model, zero API cost, handles all three goals: clustering, regression features, cross-modal alignment.

### 5A — Provision VM and download thumbnails

```bash
gcloud compute instances create clip-embedder \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --metadata="install-nvidia-driver=True"
```

Download thumbnails **directly on the VM** (avoids local download + upload round-trip):

```python
# download_thumbnails.py
import asyncio, aiohttp, pandas as pd
from pathlib import Path

df = pd.read_parquet("yt_videos_with_local_transcripts.parquet")[["videoId"]]
Path("thumbnails").mkdir(exist_ok=True)

async def fetch(session, video_id):
    url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    async with session.get(url) as r:
        if r.status == 200:
            Path(f"thumbnails/{video_id}.jpg").write_bytes(await r.read())

async def main():
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[fetch(session, vid) for vid in df["videoId"]])

asyncio.run(main())
```

Resolution: **`hqdefault.jpg` (480×360)** — CLIP resizes to 224×224 internally; `maxresdefault` wastes ~8× bandwidth with no model-quality gain and is not always available.

### 5B — CLIP embedding

```python
# clip_embed.py
import open_clip, torch, pandas as pd, numpy as np
from pathlib import Path
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.cuda().eval()

df = pd.read_parquet("yt_videos_with_local_transcripts.parquet")[["videoId", "title"]]

visual_rows, text_rows = [], []
BATCH = 256

paths  = [p for p in Path("thumbnails").glob("*.jpg")]
id_map = {p.stem: p for p in paths}

with torch.no_grad():
    # Visual embeddings
    for i in range(0, len(df), BATCH):
        batch = df.iloc[i:i+BATCH]
        imgs, vids = [], []
        for _, row in batch.iterrows():
            p = id_map.get(row["videoId"])
            if p:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
                vids.append(row["videoId"])
        if not imgs:
            continue
        vecs = model.encode_image(torch.stack(imgs).cuda())
        vecs = (vecs / vecs.norm(dim=-1, keepdim=True)).cpu().numpy()
        visual_rows.extend({"video_id": v, "clip_visual": vec.tolist()}
                            for v, vec in zip(vids, vecs))

    # Text embeddings (titles) — same embedding space, enables cross-modal similarity
    for i in range(0, len(df), BATCH):
        batch = df.iloc[i:i+BATCH]
        tokens = tokenizer(batch["title"].fillna("").tolist()).cuda()
        vecs   = model.encode_text(tokens)
        vecs   = (vecs / vecs.norm(dim=-1, keepdim=True)).cpu().numpy()
        text_rows.extend({"video_id": row["videoId"], "clip_title_text": vec.tolist()}
                          for (_, row), vec in zip(batch.iterrows(), vecs))

import subprocess
pd.DataFrame(visual_rows).to_parquet("clip_visual.parquet", index=False)
pd.DataFrame(text_rows).to_parquet("clip_title_text.parquet", index=False)
subprocess.run(["gsutil", "cp", "clip_visual.parquet",
                "gs://showreel-bucket/thumbnails/clip_visual.parquet"], check=True)
subprocess.run(["gsutil", "cp", "clip_title_text.parquet",
                "gs://showreel-bucket/thumbnails/clip_title_text.parquet"], check=True)

print("Done. Shutting down.")
subprocess.run(["sudo", "poweroff"])   # ← billing stops here
```

### 5C — Thumbnail archetype clustering (local, after downloading parquets from GCS)

UMAP (512 → 20 dims) + HDBSCAN on `clip_visual` vectors → thumbnail style clusters.  
Name each cluster: top-5 thumbnail images per cluster → Gemini Vision prompt → archetype label.

Expected archetypes: "talking-head close-up", "reaction/emotion face", "text-dominant minimal", "product or object focus", "action/scene shot", etc.

Output: `thumbnail_clusters.parquet` — `[video_id, thumb_cluster_id, thumb_cluster_label]`

### 5D — Cross-modal alignment feature (free, no extra model calls)

```python
import numpy as np, pandas as pd

vis = pd.read_parquet("clip_visual.parquet")
txt = pd.read_parquet("clip_title_text.parquet")
merged = vis.merge(txt, on="video_id")

merged["thumbnail_title_alignment"] = [
    float(np.dot(v, t))
    for v, t in zip(merged["clip_visual"], merged["clip_title_text"])
]
merged[["video_id", "thumbnail_title_alignment"]].to_parquet("cross_modal.parquet", index=False)
```

High alignment = thumbnail illustrates the title (informative).  
Low alignment = thumbnail diverges from title content (potential clickbait signal).

### 5E — Join into Phase 4 regression

Thumbnail features slot into the existing feature matrix alongside transcript features:

| Added feature group | Columns |
|--------------------|---------|
| Visual archetype | `thumb_cluster_*` dummies |
| Cross-modal | `thumbnail_title_alignment` |

### Execution order within Phase 5

```
5A  Provision VM + download thumbnails directly on VM (~20 min)
5B  CLIP embedding on GPU (~15 min) → write to GCS → VM poweroff
5C  Download parquets locally → UMAP + HDBSCAN + Gemini labelling (~15 min)
5D  Cross-modal alignment (~2 min, free given 5B)
5E  Re-run Phase 4 regression with extended feature matrix
```

Phase 5 is independent of Phases 2–3 and can run in parallel once thumbnail URLs are confirmed available.
