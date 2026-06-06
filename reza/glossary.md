# Glossary of Key Concepts

Definitions for recurring terms across pipeline conversations, design docs, and analysis work.

---

## Soft Membership Vector

In **hard clustering** (e.g. k-means, k-medoids) each data point is assigned to exactly one cluster. In **soft clustering** (e.g. fuzzy c-means, Gaussian Mixture Models) each point instead receives a vector of weights — one per cluster — representing the degree to which it belongs to each. These weights sum to 1.0 and form the soft membership vector.

**Example:** A YouTube comment that discusses both cooking and lifestyle might produce a soft membership vector like `[0.62, 0.31, 0.07]` across three topic clusters, rather than being forced into cluster 0 alone.

**Why it matters here:** Hard assignment discards information when a video or comment sits at the boundary of two themes. Soft membership lets downstream models use the full probability distribution as a richer feature.

| Property | Hard assignment | Soft membership vector |
|----------|----------------|------------------------|
| Output per point | Single cluster ID (int) | Vector of floats summing to 1.0 |
| Handles ambiguity | No | Yes |
| Downstream use | One-hot or cluster label | Dense feature directly usable in regression / ranking |

---

## Hook

A **hook** is a shell command that the system automatically executes in response to a specific event, without the user or model having to trigger it manually.

### Git hooks
Scripts stored in `.git/hooks/` that run at fixed points in the git workflow — e.g. `pre-commit` runs before a commit is created, `post-merge` runs after a merge completes. Used to enforce linting, run tests, or block bad commits.

### Claude Code hooks
Configured in `.claude/settings.json`. Claude Code executes these shell commands when certain agent events fire — e.g. when a tool is called, when a session starts, or when the agent stops. Unlike memory or preferences, hooks are executed by the harness, not inferred by the model, so they are reliable and deterministic.

```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "echo tool used" }] }
    ]
  }
}
```

**Key distinction:** A hook runs regardless of what the model "decides" — it is infrastructure, not a suggestion.

---

## Mean-Pooling

Averaging a set of embedding vectors element-wise into a single representative vector.

If a transcript is split into 5 chunks and each chunk is embedded into a 768-dimensional vector, mean-pooling produces one 768-dimensional vector per chapter by averaging all chunk vectors within that chapter.

**Used in this project:** Short videos are collapsed to a single chapter-0 vector via mean-pooling; long videos are mean-pooled within each detected chapter boundary.

---

## Cosine Similarity

A measure of the angle between two vectors in high-dimensional space, ignoring their magnitudes. Ranges from -1 (opposite) to 1 (identical direction).

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Used in this project:** Chapter break detection in long-video transcripts. When cosine similarity between consecutive chunk embeddings drops below `0.75`, a new semantic chapter is declared.

---

## Staging Area (Git Index)

The intermediate zone between your working files and a git commit. Changes must be explicitly added to the staging area (`git add`) before they are included in a commit. Allows you to compose a commit from a selective subset of local changes.

| Zone | What it is |
|------|-----------|
| Working tree | Files on disk as you see them |
| Staging area (index) | Snapshot of what the next commit will contain |
| HEAD | The last committed snapshot |

**Relevant to this session:** The 88 deleted files were in the working tree (physically gone) but still registered in git's index — staging the deletions (`git rm --cached`) reconciled the two.

---

## HEAD

`HEAD` is git's pointer to the currently checked-out commit — the snapshot that your working tree is based on. On a branch, `HEAD` points to the latest commit on that branch. Comparing against `HEAD` shows what has changed since the last commit.

---

## Idempotency

A property of an operation: running it once or many times produces the same result. Critical for pipeline safety — if a step fails halfway and is re-run, it should not create duplicates or corrupt state.

**Used in this project:** SHA-256 content hashes make the transcript ingestion pipeline idempotent — re-processing the same transcript produces the same hash and overwrites rather than duplicates the record.

---

## Fertility (Tokenizer)

The average number of sub-word tokens a tokenizer produces per word. Lower fertility means the model processes text more efficiently (fewer tokens = fewer forward passes = lower cost and latency).

**Used in this project:** Italian text processed by English-centric tokenizers (e.g. Llama-3.1's default) has high fertility because Italian words are split into many unfamiliar sub-tokens. Vocabulary adaptation with the Minerva tokenizer reduces Italian fertility by 16–25%.
