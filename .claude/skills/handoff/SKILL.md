---
name: handoff
description: Aggressively compress the current session context into a minimal handoff brief written to Ali/handoff_temp.md. Use when switching models (Opus/Sonnet/Haiku) or delegating a task to a subagent, so the incoming model loads the minimum context needed to continue — not the full history. Also handles `/handoff clean` to delete all handoff_temp.md files.
---

# Handoff

Produce a dense, structured brief that lets a cold model (different tier or subagent) continue
the session without reading any conversation history. Write it to `Ali/handoff_temp.md`.
Overwrite the file if it already exists — only the latest handoff matters.

The `_temp` suffix marks it as disposable. Run `/handoff clean` when done to purge all temp files.

## Arguments
- `--model <opus|sonnet|haiku>` (optional) — override the "Model tier for next task" section and
  append the matching tier tag to the **Next task** line. Shapes the brief style for the target model.
- `clean` (sub-command, no `--`) — delete all `handoff_temp.md` files in the repo. See section below.

## When to use
- User says: "handoff", "/handoff", "switch to Opus/Sonnet/Haiku", "pass this to a subagent",
  "prepare context for the next model", "compact and switch".
- Before `/model` is changed mid-session.
- Before spawning a subagent that needs to know what has been done and what is next.

## Output file
`Ali/handoff_temp.md` — overwrite every time. The `_temp` suffix signals it is safe to delete.

---

## Sub-command: `clean`

Triggered by: `/handoff clean`
**Intended model: Haiku** — purely mechanical file deletion, no reasoning needed.

### What to do
1. Search the entire repo for any file named `handoff_temp.md` (check all subdirectories).
2. Delete each one found using the Bash tool: `rm "<path>"`.
3. Also delete `Ali/handoff.md` if it exists (legacy name from before `_temp` convention).
4. Report a one-line summary: "Cleaned N handoff_temp file(s): <paths>." or "Nothing to clean."

### What to avoid
- Do NOT delete any other `.md` files.
- Do NOT read or summarize the contents of the files before deleting.
- Do NOT ask for confirmation — the `_temp` naming convention is the authorization.
- Do NOT modify `SKILL.md` or any file in `.claude/`.

---

## Model profiles — adapt the brief based on `--model`

### No `--model` flag (generic)
Standard brief. All sections present. Target 20-40 lines.

---

### `--model opus`
**Strengths:** Deep reasoning, causal inference, strategy, ambiguity resolution.
**Weaknesses:** Expensive — do not waste tokens on low-level detail it will ignore anyway.
**Brief style:** Compress hard. Skip file-level specifics unless Opus must decide on architecture.
Focus on: the *why* behind the session goal, unresolved decisions or trade-offs, and what strategic
output is expected. Omit step-by-step instructions — Opus infers them.

Extra section to add:
```
## Open questions / decisions needed
- <unresolved design or strategy question Opus must answer>
```

Do NOT include: exact column names, parquet schemas, env setup steps, regex patterns.
Target length: 15-25 lines.

---

### `--model sonnet`
**Strengths:** Code generation, data modeling, pipeline engineering, refactoring.
**Weaknesses:** May over-engineer or introduce abstractions; needs scope guardrails.
**Brief style:** Balanced. Include file paths, relevant schemas or function signatures if they
are the gotcha. Be explicit about scope — what is in and out of bounds for this task.

Extra section to add:
```
## Scope guardrails
- Do: <what Sonnet should produce>
- Do NOT: <common overreach to avoid, e.g. "do not refactor surrounding cells", "do not change schema">
```

Target length: 20-35 lines.

---

### `--model haiku`
**Strengths:** Fast parsing, reading files, summarizing, linting, simple transformations.
**Weaknesses:** Limited reasoning depth — will hallucinate if the task is ambiguous; cannot
make architectural decisions; may miss subtle schema constraints; tends to over-summarize
rather than flag issues explicitly.
**Brief style:** Most detailed of the three. Spell out every step. Be explicit about what
to avoid. Provide exact expected output format. Leave zero ambiguity.

Extra sections to add:
```
## Step-by-step instructions
1. <exact first action>
2. <exact second action>
...

## What to avoid
- Do NOT write new code or modify any file except the output target.
- Do NOT infer schema — use only what is explicitly stated in this brief.
- Do NOT summarize findings into prose — use bullet lists with file:line references.
- Do NOT make decisions — flag ambiguities back to the user instead of resolving them.
- <any task-specific pitfall>

## Expected output format
Describe exactly what the output should look like (file path, structure, length).
```

Target length: 35-50 lines (more detail is correct here, not a sign of bloat).

---

## Shared brief structure (all models include these sections)

```markdown
# Handoff Brief — {YYYY-MM-DD HH:MM} [Target: Opus | Sonnet | Haiku]

## Project
One sentence: what project/pipeline/dataset is being worked on and why.

## Session goal
One sentence: what the user is trying to accomplish this session.

## Completed tasks
- <task> → <key result / output file / finding>

## Current state
- <file path> — <what changed or what it contains>

## Next task
[Opus | Sonnet | Haiku] <single concrete action, file, expected output>

## Pending tasks (after next)
- <ordered list, one line each>

## Critical constraints / gotchas
- <max 5 bullets, most dangerous only>

## Model tier for next task
[Opus | Sonnet | Haiku] — <reason in 5 words>
```
Then append the model-specific extra sections above.

## Rules
- **No conversation recap.** Only what was done and what is next.
- **File paths mandatory** for every input/output mentioned.
- **One next task only.**
- **Strip hedging language** ("probably", "might", "could") — facts or omit.
- After writing the file, tell the user: "Handoff written to Ali/handoff_temp.md. Safe to /compact or switch model. Run `/model <haiku|sonnet|opus>` to switch. Run `/handoff clean` when done to delete the temp file."

## Critical constraints to always include (project-level)
These apply to every handoff in this repo — always include in the gotchas section:
- Conda env `ma_env` required for Whisper/multimodal work.
- GCP: project=`gen-lang-client-0792749758`, bucket=`afb_showreel`, location=`us-central1`.
- IG media_id id-spaces differ between instaloader and ig_cleaned — join via permalink shortcode.
- Vertex Batch JSONL responseSchema must be flattened (no `$defs`/`$ref`) — use `_sanitize_schema`.
- Model tiers: Opus=Strategy, Sonnet=Code/Modeling, Haiku=Parsing/Logs (see Ali/CLAUDE.md).
