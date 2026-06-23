# Wiki Index

**Last updated:** 2026-06-23

This wiki documents the analytical findings, pipelines, data sources, and modeling results for the **Showreel Social Media Analytics** project (Instagram focus).

---

## Findings & Synthesis

| Page | Type | Description |
|---|---|---|
| [instagram_findings.md](instagram_findings.md) | findings | Executive summary of findings: sentiment overview, targets, personas, RFM clusters, linguistics, cross-platform comparison (487k comments) |
| [instagram_findings_detailed.md](instagram_findings_detailed.md) | findings | Maximum-detail reference: 14 sections, all raw numbers, full correlation tables, exact sentiment rates, 14 actionable recommendations |
| [instagram_executive_summary.md](instagram_executive_summary.md) | modeling | 7-model inventory: all Python & R modeling tasks, metrics, findings, production readiness |

## Data & Pipelines

| Page | Type | Description |
|---|---|---|
| [instagram_data_sources.md](instagram_data_sources.md) | reference | Data sources & schemas: raw cleaned (ig_posts/comments), intermediate (multimodal), analysis-ready (sentiment, room_vibe), HeteroGraph |
| [instagram_sentiment_pipeline.md](instagram_sentiment_pipeline.md) | reference | Sentiment Vertex Batch pipeline: comment labels, room-level vibe, EDA notebooks, toxicity ordinal schema, 25+ figures |
| [instagram_persona_pipeline.md](instagram_persona_pipeline.md) | reference | Persona pipeline: 10-persona LLM taxonomy, Stage 1/2 outputs, sentiment x persona profiling, lifecycle crossing |
| [instagram_rfm_clustering.md](instagram_rfm_clustering.md) | reference | RFM clustering: rolling quadrimester (40 windows, k=4 KMeans), non-overlapping 3-year (k=4 KMedoids), lifecycle & re-entry analysis |

## Advanced Analytics

| Page | Type | Description |
|---|---|---|
| [instagram_advanced_analytics.md](instagram_advanced_analytics.md) | reference | Modeling (4 Python + 3 R tasks), event impact (ITS), virality (SARIMAX announcement vs occurrence). Outputs: predictions, monthly series, 21 counterfactual plots |

---

## Related Documentation

| Location | Description |
|---|---|
| [Ali/llm_wiki_afb/](../Ali/llm_wiki_afb/index.md) | Internal technical wiki (21 pages): pipeline concepts, schemas, Vertex AI setup, entity definitions |
| [INSTAGRAM_EXECUTIVE_SUMMARY.md](../INSTAGRAM_EXECUTIVE_SUMMARY.md) | Standalone executive report (root level): 5 modelling paths, lifecycle clusters, causal findings |
| [CANVA_PRESENTATION_BRIEF.md](../CANVA_PRESENTATION_BRIEF.md) | Slide-by-slide brief for the 7-model Canva presentation |
