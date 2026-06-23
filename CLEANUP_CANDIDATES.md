# Cleanup Candidates

**Date:** 2026-06-23  
**Purpose:** Files and directories that are redundant, temporary, or should not be in the git repository. Organized by priority.

---

## Priority 1 — Remove from Git (generated/should never be tracked)

| Path | Size | Reason |
|------|------|--------|
| `node_modules/` | 7.4 MB | Generated from `package.json`; should be in `.gitignore`. Run `npm install` to recreate. Currently git-tracked (158 files). |
| `test_env/` | ~200 MB | Python virtual environment. Already partially gitignored but `Lib/site-packages/` has tracked files (pip, numpy, httpx, etc.). Add `test_env/` to `.gitignore` and `git rm -r --cached test_env/`. |
| `Ali/_debug_profile/` | 17 MB | Chromium browser profile (Crashpad, ShaderCache, etc.) — local debugging artifact, no value in VCS. |
| `Ali/_api_dump/` | ~5 MB | Raw Instagram API JSON dumps — intermediate cache for `ig_refetch.py`. Useful locally, not in git. |
| `Ali/.obsidian/` | <1 MB | Obsidian vault settings (personal IDE config). Should be in `.gitignore`. |
| `Ali/__pycache__/` | <1 MB | Python bytecode cache. Should be in `.gitignore` (add `__pycache__/`). |

**Action:** Add all of these to `.gitignore`, then `git rm -r --cached <path>` for any that are currently tracked.

---

## Priority 2 — Redundant files (safe to delete)

| Path | Size | Reason |
|------|------|--------|
| `Ali/persona_pipeline.ipynb.bak` | 820 KB | Superseded backup from Jun 9. The current `persona_pipeline.ipynb` is the production version. Git history preserves all prior states. |
| `Ali/persona_pipeline.ipynb.prestage3del_bak` | 1.2 MB | Development checkpoint from Jun 12. Same as above. |
| `Ali/persona_pipeline.ipynb.preorchdel_bak` | 1.2 MB | Development checkpoint from Jun 12. Same as above. |
| `Ali/persona_pipeline.ipynb.reorder_bak` | 1.2 MB | Development checkpoint from Jun 12. Same as above. |
| `Ali/persona_pipeline.ipynb.split_bak` | 1.2 MB | Development checkpoint from Jun 12. Same as above. |
| `Ali/New folder/sentiment_pipeline_archived.ipynb` | 1.4 MB | Unnamed folder with an archived pipeline. The production version is `Ali/sentiment_pipeline.ipynb`; the old version is also at `Ali/sentiment_pipeline_old.ipynb`. Triple-redundant. |
| `Ali/sentiment_pipeline_old.ipynb` | 1.3 MB | Explicitly named as superseded in the index. Git history has all prior versions. |
| `Ali/camihawke_major_events_timeline.csv` | 4.6 KB | Superseded by `Ali/camihawke_combined_comprehensive_timeline.csv` (26 rows vs 17 rows — the combined version is a strict superset). |
| `dd.png` (root) | 135 KB | Purpose unknown. No documentation references it. Filename is not descriptive. |
| `video duration all.png` (root) | 17 KB | Orphaned visualization at root level. If needed, should be in an outputs folder. |
| `modeling_visual_philosophy.md` (root) | 3.9 KB | Design philosophy document for a single visualization. Not referenced by any pipeline or wiki page. Content is aspirational, not operational. |

**Total recoverable:** ~8.5 MB (small individually, but reduces clutter significantly)

---

## Priority 3 — Consolidate duplicates

| Item | Files | Reason |
|------|-------|--------|
| **Executive summary duplication** | `INSTAGRAM_EXECUTIVE_SUMMARY.md` (root, 350 lines) vs `wiki/instagram_executive_summary.md` (374 lines) | Different documents with overlapping scope. Root version covers 5 modelling paths; wiki version is a 7-model inventory. Recommend keeping the wiki version as canonical and replacing the root version with a short pointer: "See [wiki/instagram_executive_summary.md](wiki/instagram_executive_summary.md)". |
| **Sentiment EDA PNG+SVG pairs** | `Ali/Sentiment_EDA/outputs/*.png` + `*.svg` (100+ pairs) | Every chart is saved in both PNG and SVG. Pick one format (PNG for portability, SVG for editability) and delete the other. Saves ~50% of the outputs folder. |
| **Ali/Output/ vs Ali/outputs/** | `Ali/Output/` (1.4 GB) and `Ali/outputs/` (3.3 GB) | Two output directories with different casing. `Ali/outputs/` is the current production target. `Ali/Output/` contains legacy files (`ig_multimodal_final.parquet`, `ig_media_features.parquet`, `Prepared Comments/`, `Short_to_Long_connection/`). Verify nothing still reads from `Ali/Output/`, then delete or move needed files. |
| **ig_multimodal_final.parquet** | `Ali/Output/ig_multimodal_final.parquet` (604 KB) + `Ali/Output/ig_multimodal_final.csv` (1.1 MB) | The canonical version is in `Ali/outputs/` (referenced by all pipelines). These are older copies. |
| **Archive download link files** | `Ali/Archive/downloaded_links.csv`, `downloaded_links_urls_only.csv`, `failed_downloads.csv` | Also exist in `reza/ner/` (`downloaded_links.csv`, `downloaded_links_urls_only.csv`). Determine which is authoritative; delete the duplicate. |

---

## Priority 4 — Large directories to evaluate

| Path | Size | Reason |
|------|------|--------|
| `Ali/Backup/` | 6.4 GB | Historical multimodal dataset snapshots (sample Instagram media from 2023). Already `.gitignore`d so not tracked, but takes local disk space. Safe to archive to external storage or delete — data can be re-fetched via `ig_refetch.py`. |
| `Ali/multimodal_dataset_fixed/` | 7.0 GB | Already `.gitignore`d. Contains processed carousel/reel/feed media with extracted frames and Whisper transcripts. Needed if re-running multimodal pipeline locally; otherwise archive externally. |
| `Ali/playwright_downloads/` | 239 MB | Already `.gitignore`d. YouTube Shorts-to-Long scraping artifacts. Can be deleted if scraping is complete. |
| `Ali/batch_results/` | 57 MB | Old Vertex AI batch job JSONL outputs. Results have been parsed into parquets in `Ali/outputs/`. Safe to delete unless needed for audit trail. |
| `Mickey/*.csv` (generated) | ~300 MB | 20+ CSV files (30-58 MB each) that are generated by Mickey's notebooks. These are analysis outputs, not source data. If notebooks can regenerate them, consider `.gitignore`-ing them and adding a "run notebook to regenerate" note. |

---

## Priority 5 — Minor cleanup (nice-to-have)

| Path | Reason |
|------|--------|
| `Ali/Archive/ECOLOGICAL_PERSONA_CODE_BLOCKS.md` | 28 KB of code blocks from a superseded persona approach. The current persona pipeline is fully self-contained. |
| `Ali/Archive/DATA_CLEANING_AND_FEATURE_ENGINEERING_REPORT.md` | 54 KB historical report. The wiki pages now cover this content more accurately. |
| `Ali/Archive/CLAUDE_BATCHWORK.MD` | 1.4 KB of old batch workflow notes. Superseded by wiki documentation. |
| `Ali/enriched_post_vibe_matrix.parquet` | Legacy community vibe output — superseded by `outputs/stage2_sentiment/` parquets. |
| `Ali/handoff_temp.md` | Temporary Claude Code handoff file (0 lines). Should be ephemeral, not tracked. |
| `Ali/to_download/` | Download queue/staging. If downloads are complete, can be emptied. |
| `Ali/processed_urls.txt` + `Ali/reels_links.txt` + `Ali/feed_links.txt` | URL lists used during media download phase. If download is complete, these are historical only. |
| `CANVA_PRESENTATION_BRIEF.md` (root) | 46 KB slide-by-slide brief. Useful during presentation creation but not ongoing. Could move to `Ali/Archive/` after the presentation is finalized. |
| `generate_modeling_summary_visual.py` (root) | Script for one-off visual generation. Could move to `Ali/` to keep root clean. |
| `package.json` + `package-lock.json` (root) | Only dependency is `pptxgenjs` for presentation generation. If no longer needed, remove along with `node_modules/`. |

---

## Proposed .gitignore additions

```gitignore
# Generated / local-only
node_modules/
test_env/
__pycache__/
Ali/__pycache__/
Ali/_api_dump/
Ali/_debug_profile/
Ali/.obsidian/
Ali/New folder/
Ali/handoff_temp.md
Ali/batch_results/

# Notebook backups
*.ipynb.bak
*.ipynb.*_bak
```

---

## Summary

| Priority | Items | Disk savings | Effort |
|----------|-------|-------------|--------|
| P1 — Remove from git | 6 paths | ~230 MB tracked | Low (gitignore + git rm --cached) |
| P2 — Delete redundant | 11 files | ~8.5 MB | Low (delete) |
| P3 — Consolidate | 5 groups | ~1.5 GB (Output/) | Medium (verify references first) |
| P4 — Evaluate large dirs | 5 paths | ~14 GB local | Medium (decide archive vs delete) |
| P5 — Minor cleanup | 10 items | ~130 MB | Low (optional) |
