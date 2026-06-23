---
name: compact-smart
description: Aggressively compact the current session into a dense, in-session summary structured around tasks done + outputs, suggested next moves, and dos & don'ts. Use when the user says "compact", "/compact_smart", "compact smart", "summarize and trim", or wants to shrink context mid-session without losing the actionable thread. Unlike /handoff, the summary stays IN this session (it becomes the working context) — it is not written to a file for a cold model.
---

# Compact Smart

Replace the conversation so far with a single dense summary that preserves only what is
needed to keep working: **what was done (and where the outputs are), what to do next, and
what to avoid.** Everything else — tool chatter, exploratory dead-ends, superseded
attempts, verbatim file dumps — is dropped.

This is `/compact` with a fixed, opinionated shape. The summary stays in-session and
becomes the new working context. It is NOT a file (that is `/handoff`).

## When to use
- User says: "compact", "/compact_smart", "compact smart", "summarize and trim",
  "shrink context", "compact aggressively", "trim the session".
- Context is getting long and the user wants to continue working, not switch model/agent.

## Difference from /handoff
- `/handoff` → writes `Ali/handoff_temp.md` for a *cold* model/subagent. Self-contained file.
- `/compact-smart` → emits the summary *into this chat* to become the live context. No file.
  Assumes the same model continues, so it can lean on the model's own inference.

## Output — print exactly this structure, nothing before or after

```markdown
# Session Compact — {YYYY-MM-DD HH:MM}

## Goal
<one sentence: what this session is ultimately trying to achieve>

## Done — tasks + outputs
- <task> → <result / output file path / key finding>
  (one bullet per completed unit of work; ALWAYS include the output artifact path if one exists)

## State now
- <file path or pipeline stage> — <what it currently contains / its status>

## Next moves (suggested, ordered)
1. <highest-value next action — concrete, names the file/cell/function>
2. <next>
3. <next>

## Dos
- <thing that worked / the right approach to keep using>

## Don'ts
- <dead-end already tried — do NOT repeat it>
- <gotcha that will bite if ignored>

## Open threads
- <anything unresolved, awaiting user input, or deferred>
```

## How to compact (the aggressive part)
- **Collapse, don't transcribe.** A 10-step debugging arc becomes one bullet: the fix +
  the file. Drop every intermediate failure *unless* it's a "don't repeat this" warning.
- **Outputs are mandatory.** Every completed task that produced an artifact MUST cite the
  path (parquet/png/ipynb/py). A "Done" bullet with no output path is suspect — verify it
  actually completed.
- **Merge duplicates.** If three turns touched the same file, one "State now" bullet.
- **Kill verbatim content.** No pasted code blocks, no full schemas, no log dumps. Cite
  `file:line` or a column name instead.
- **Suggested next moves are recommendations, not a contract** — rank by value, but the
  user may redirect. Keep to 3-5.
- **Strip hedging.** "probably", "might", "I think" → state the fact or omit it.
- **Target length: 25-45 lines total.** If longer, you are transcribing, not compacting.

## Dos / Don'ts sourcing
The Dos and Don'ts sections are the highest-value part — they encode what the next stretch
of work should and shouldn't do. Pull them from:
- Approaches that demonstrably worked this session → Dos.
- Things tried that failed, or paths the user explicitly rejected → Don'ts.
- Project gotchas relevant to the next moves (see below) → Don'ts.

## Project gotchas — include in Don'ts only if relevant to the next moves
- Conda env `ma_env` required for Whisper/multimodal work.
- GCP: project=`gen-lang-client-0792749758`, bucket=`afb_showreel`, location=`us-central1`.
- IG `media_id` id-spaces differ between instaloader and `ig_cleaned` — join via permalink shortcode.
- Vertex Batch JSONL responseSchema must be flattened (no `$defs`/`$ref`) — use `_sanitize_schema`.
- Model tiers: Opus=Strategy, Sonnet=Code/Modeling, Haiku=Parsing/Logs (Ali/CLAUDE.md).
- ASCII only in notebooks/configs — avoid emoji/non-ASCII (UTF-8 corruption risk).

## Rules
- **No conversation recap, no narration, no preamble.** Print the summary block only.
- **One artifact path per done-task minimum** where an artifact exists.
- **3-5 next moves, ranked.** Not an exhaustive backlog.
- **This summary is now the context** — after printing it, continue from it if the user
  gives a next instruction. Do not re-read history you just compacted away.
