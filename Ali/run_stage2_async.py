#!/usr/bin/env python3
"""
Stage 2 persona classification - ASYNC, high-throughput standalone script.

Mirrors the Stage 2 logic in persona_pipeline.ipynb (same system prompt, profile
schema, taxonomy, output schema). Transport is RAW async HTTP (aiohttp) straight to
the Vertex generateContent REST endpoint - NO google-genai SDK, so there is no
pydantic schema-build hang at import and no per-request pydantic validation. The
request payload is identical to what the SDK sends, so classifications are unchanged.
Designed to saturate your Vertex Flash 2.5 quota.

Inputs (produced by the notebook, already on disk):
  - outputs/stage1_persona/user_features_instagram_cache.parquet   (all users)
  - outputs/stage1_persona/taxonomy.json   (status must be APPROVED)

Outputs:
  - outputs/stage2_persona/stage2_responses.jsonl   (resumable checkpoint; raw model text per group)
  - outputs/stage2_persona/user_personas.parquet    (final joined table)

Usage:
  python run_stage2_async.py                 # classify ALL users
  python run_stage2_async.py --max-users 30  # smoke test (1 call ~= 10 users)
  python run_stage2_async.py --concurrency 100 --users-per-request 10
  python run_stage2_async.py --dry-run       # build one request, print it, no API call

Resumable: re-running skips groups already in the checkpoint file. Delete the
checkpoint to force a clean full re-run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp
import google.auth
import google.auth.transport.requests
import pandas as pd

# Windows consoles default to cp1252; comment samples contain emoji/non-ASCII.
# Force UTF-8 so progress printing never crashes the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ======================== CONFIG (matches the notebook) ========================
PLATFORM = "instagram"

GCP_PROJECT_ID = "project-a9b99a62-b082-46aa-b54"
GCP_LOCATION = "us-central1"

LOCAL_DIR = "outputs/stage1_persona"
TAXONOMY_JSON_PATH = f"{LOCAL_DIR}/taxonomy.json"
USER_FEATURES_CACHE_PATH = f"{LOCAL_DIR}/user_features_{PLATFORM}_cache.parquet"

OUT_DIR = "outputs/stage2_persona"
CHECKPOINT_PATH = f"{OUT_DIR}/stage2_responses.jsonl"
RESULTS_PATH = f"{OUT_DIR}/user_personas.parquet"

MODEL_STAGE2_CLASSIFY = "gemini-2.5-flash"

# Determinism (academic reproducibility) - identical to the notebook.
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_OUTPUT_TOKENS_STAGE2 = 8192
THINKING_BUDGET_STAGE2 = 512

# Defaults; overridable via CLI.
DEFAULT_USERS_PER_REQUEST = 10   # async tolerates bigger packing than the batch path
DEFAULT_CONCURRENCY = 60         # in-flight requests; raise toward your RPM quota
DEFAULT_MAX_RETRIES = 8
# Requests-per-minute ceiling. Set this AT OR BELOW your Vertex Flash 2.5 RPM quota.
# The rate limiter spaces requests so you don't sustain 429s; on a 429 it auto-throttles down.
DEFAULT_RPM = 200

# Same summary columns the notebook's retrieve_stage2 merges onto the LLM output.
SUMMARY_COLS = ["author_id", "total_comments", "activity_span_days",
                "mean_hours_to_comment", "pct_comments_under_1h",
                "reply_ratio", "mean_word_count"]


# ============================ PROMPT BUILDERS ==================================
# Copied verbatim (logic) from persona_pipeline.ipynb cell 51 so the classification
# is identical to what the notebook would produce.

def build_stage2_system_prompt(taxonomy: list) -> str:
    taxonomy_text = ""
    for p in taxonomy:
        taxonomy_text += (
            f"\nPERSONA: {p['codename']} - {p.get('label', '')}\n"
            f"  Description: {p.get('description', '')}\n"
            f"  Quantitative signals: {'; '.join(p.get('quantitative_signals', []))}\n"
            f"  Example comments: {' | '.join(p.get('example_comments', []))}\n"
        )
    return (
        "You are a deterministic community analyst for Show Reel Media Group.\n"
        "Classify the Instagram commenter into exactly ONE persona from the approved taxonomy,\n"
        "using their behavioural metrics and comment samples.\n"
        "\n\n"
        "=== APPROVED PERSONA TAXONOMY ===\n"
        f"{taxonomy_text}"
        "=================================\n\n"
        "RULES:\n"
        "1. Assign exactly ONE persona - the closest match.\n"
        "2. Output a confidence score between 0.0 and 1.0.\n"
        "3. Cite a specific comment fragment as justification.\n"
        "4. If data is insufficient, assign the most probable persona with confidence <= 0.4.\n"
        "5. Echo the author_id exactly as given.\n"
        "6. Output ONLY a valid JSON ARRAY - one object per user, in the same order. No preamble, no markdown fences.\n\n"
        'Per-user schema: {"author_id": str, "persona_codename": str, "confidence": float, '
        '"justification": str}. Output a JSON array: [ {...}, {...}, ... ] with exactly one '
        "object for every user in the batch."
    )


def format_user_profile_for_stage2(row) -> dict:
    return {
        "author_id":                str(row["author_id"]),
        "total_comments":           int(row["total_comments"]),
        "unique_posts":             int(row["unique_posts_commented"]),
        "activity_span_days":       int(row["activity_span_days"]),
        "pct_comments_under_1h":    round(float(row["pct_comments_under_1h"]), 2),
        "pct_comments_under_24h":   round(float(row["pct_comments_under_24h"]), 2),
        "reply_ratio":              round(float(row["reply_ratio"]), 2),
        "mean_mention_count":       round(float(row["mean_mention_count"]), 2),
        "mean_word_count":          round(float(row["mean_word_count"]), 1),
        "emoji_usage_rate":         round(float(row["emoji_usage_rate"]), 2),
        "question_rate":            round(float(row["question_rate"]), 2),
        "exclamation_rate":         round(float(row["exclamation_rate"]), 2),
        "post_concentration_ratio": round(float(row["post_concentration_ratio"]), 2),
        "sample_comments":          str(row.get("top_comments_sample", ""))[:500],
    }


def build_stage2_parts(group_df, system_prompt: str) -> list:
    profiles = [format_user_profile_for_stage2(row) for _, row in group_df.iterrows()]
    block = ("\n\n=== USERS TO CLASSIFY (" + str(len(profiles)) + ") ===\n"
             + json.dumps(profiles, ensure_ascii=False))
    return [{"text": system_prompt + block +
             "\n\nClassify EACH user into exactly one persona. "
             "Output ONLY a JSON array with one object per user, in the same order."}]


def strip_fences(s: str) -> str:
    import re
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


# ============================ ASYNC INFERENCE (raw REST) =======================
# generationConfig as a plain dict - identical fields to the SDK's GenerateContentConfig.
GENERATION_CONFIG = {
    "temperature": TEMPERATURE,
    "topP": TOP_P,
    "maxOutputTokens": MAX_OUTPUT_TOKENS_STAGE2,
    "responseMimeType": "application/json",
    "thinkingConfig": {"thinkingBudget": THINKING_BUDGET_STAGE2},
}


def vertex_endpoint(model: str) -> str:
    return (f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/"
            f"{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/publishers/google/"
            f"models/{model}:generateContent")


class TokenProvider:
    """Application Default Credentials -> bearer token, refreshed before expiry.

    Uses google.auth (no SDK). Refresh is a blocking call, so we run it in a
    thread and guard it with a lock so concurrent coroutines refresh once.
    """

    def __init__(self):
        self._creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self._req = google.auth.transport.requests.Request()
        self._lock = asyncio.Lock()

    async def token(self) -> str:
        if self._creds.valid:
            return self._creds.token
        async with self._lock:
            if not self._creds.valid:
                await asyncio.to_thread(self._creds.refresh, self._req)
        return self._creds.token


def _extract_text(resp_json: dict) -> str:
    """Pull the text out of a generateContent REST response dict."""
    try:
        parts = resp_json["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError):
        return ""


class AdaptiveRateLimiter:
    """Token-bucket limiter that ADAPTS to 429s.

    - Hands out at most `rpm` permits per minute (spaced evenly, not bursty).
    - On a 429, multiplicatively cuts the effective rate and imposes a global
      cooldown so every coroutine slows down together (AIMD: cut hard on error,
      recover gently when calls succeed).
    """

    def __init__(self, rpm: int, min_rpm: int = 10):
        self.max_interval = 60.0 / max(rpm, 1)   # target seconds between requests at full rpm
        self.min_interval = 60.0 / max(rpm, 1)
        self.max_interval_cap = 60.0 / max(min_rpm, 1)
        self.interval = self.min_interval        # current spacing (grows on 429)
        self._next_time = 0.0
        self._cooldown_until = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            start = max(now, self._next_time, self._cooldown_until)
            self._next_time = start + self.interval
            wait = start - now
        if wait > 0:
            await asyncio.sleep(wait)

    async def on_429(self, cooldown: float = 0.0):
        """Slow down (multiplicative increase of spacing) + optional global cooldown."""
        async with self._lock:
            self.interval = min(self.interval * 1.5, self.max_interval_cap)
            if cooldown > 0:
                now = asyncio.get_event_loop().time()
                self._cooldown_until = max(self._cooldown_until, now + cooldown)

    async def on_success(self):
        """Gently recover toward the configured rpm after sustained success."""
        async with self._lock:
            self.interval = max(self.min_interval, self.interval * 0.97)

    @property
    def current_rpm(self):
        return 60.0 / self.interval if self.interval > 0 else float("inf")


async def classify_group(session, url, tokens, parts, sem, max_retries, limiter) -> str | None:
    """One async generateContent POST: rate-limited, concurrency-capped, AIMD backoff on 429."""
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": GENERATION_CONFIG,
    }
    delay = 5.0
    async with sem:
        for attempt in range(max_retries):
            await limiter.acquire()
            try:
                headers = {"Authorization": f"Bearer {await tokens.token()}",
                           "Content-Type": "application/json"}
                async with session.post(url, json=payload, headers=headers) as r:
                    if r.status == 200:
                        await limiter.on_success()
                        return _extract_text(await r.json()) or None
                    body = (await r.text())[:160]
                    is_429 = r.status == 429
                    retryable = r.status in (429, 500, 503) or r.status == 408
                    if retryable and attempt < max_retries - 1:
                        if is_429:
                            await limiter.on_429(cooldown=delay)
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 120)
                        continue
                    print(f"   [classify] FAILED HTTP {r.status} after {attempt + 1} "
                          f"attempt(s): {body}", file=sys.stderr, flush=True)
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                print(f"   [classify] FAILED (network) after {attempt + 1} attempt(s): "
                      f"{str(exc)[:160]}", file=sys.stderr, flush=True)
                return None
    return None


def _load_done_indices(path: str) -> set:
    """Group indices that SUCCEEDED (non-null text).

    A checkpointed null (a request that failed, e.g. exhausted retries on a 429)
    is NOT counted as done, so re-running automatically retries only the failures.
    """
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("text"):          # only successful groups count
                        done.add(rec["group_idx"])
                except Exception:
                    pass
    return done


async def run(args):
    # --- load taxonomy (must be APPROVED) ---
    tax = json.load(open(TAXONOMY_JSON_PATH, encoding="utf-8"))
    if tax.get("status") != "APPROVED":
        raise SystemExit(f"{TAXONOMY_JSON_PATH} is not APPROVED (status={tax.get('status')!r}).")
    taxonomy = tax["final_taxonomy"]
    system_prompt = build_stage2_system_prompt(taxonomy)
    print(f"Approved Pathway {tax.get('chosen_pathway', '?')} taxonomy: {len(taxonomy)} personas "
          f"({', '.join(p['codename'] for p in taxonomy)})")

    # --- load users ---
    df = pd.read_parquet(USER_FEATURES_CACHE_PATH)
    df["author_id"] = df["author_id"].astype(str)
    if args.max_users is not None:
        df = df.head(args.max_users)
    n_users = len(df)

    upr = args.users_per_request
    groups = [df.iloc[k:k + upr] for k in range(0, n_users, upr)]

    # --- dry run: build + print one request, no API call ---
    if args.dry_run:
        parts = build_stage2_parts(groups[0], system_prompt)
        print("\n===== DRY RUN: first request parts[0] (truncated) =====")
        print(parts[0]["text"][:4000])
        print(f"\n(total request chars: {len(parts[0]['text']):,}; "
              f"{len(groups)} groups of <= {upr} users would be sent)")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    done_idx = _load_done_indices(CHECKPOINT_PATH)
    pending = [(i, g) for i, g in enumerate(groups) if i not in done_idx]

    print(f"\n{'=' * 64}\nSTAGE 2 (async) - Deterministic Classification\n"
          f"  Users: {n_users:,} | {upr}/request | {len(groups)} groups total\n"
          f"  Already done: {len(done_idx)} | Remaining: {len(pending)}\n"
          f"  Model: {MODEL_STAGE2_CLASSIFY} | concurrency: {args.concurrency} | "
          f"target RPM: {args.rpm}\n{'=' * 64}", flush=True)

    if pending:
        url = vertex_endpoint(MODEL_STAGE2_CLASSIFY)
        tokens = TokenProvider()
        sem = asyncio.Semaphore(args.concurrency)
        limiter = AdaptiveRateLimiter(args.rpm)
        ckpt_lock = asyncio.Lock()
        ckpt_file = open(CHECKPOINT_PATH, "a", encoding="utf-8")
        completed = 0
        failed = 0
        n_pending = len(pending)
        t0 = time.time()

        def _should_print(n):
            # Dense early feedback, then throttle.
            if n <= 10:
                return True
            if n <= 100:
                return n % 10 == 0
            return n % 50 == 0 or n == n_pending

        async def worker(session, idx, grp):
            nonlocal completed, failed
            parts = build_stage2_parts(grp, system_prompt)
            text = await classify_group(session, url, tokens, parts,
                                        sem, args.max_retries, limiter)
            async with ckpt_lock:
                ckpt_file.write(json.dumps({"group_idx": idx, "text": text},
                                           ensure_ascii=False) + "\n")
                ckpt_file.flush()
                completed += 1
                if text is None:
                    failed += 1
                if _should_print(completed):
                    rate = completed / max(time.time() - t0, 1e-9)
                    eta = (n_pending - completed) / max(rate, 1e-9)
                    print(f"   [{completed}/{n_pending}] | {rate * 60:.0f} req/min "
                          f"(limiter {limiter.current_rpm:.0f}) | ETA {eta / 60:.1f} min | "
                          f"failed: {failed}", flush=True)

        print(f"   Firing {n_pending:,} requests (in-flight cap {args.concurrency}, "
              f"<= {args.rpm} req/min)...", flush=True)
        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=args.concurrency + 10)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            await asyncio.gather(*(worker(session, i, g) for i, g in pending))
        ckpt_file.close()
        print(f"   All requests done in {(time.time() - t0) / 60:.1f} min. "
              f"Failed: {failed}/{n_pending}", flush=True)

    # --- parse checkpoint -> results ---
    results = []
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                t = rec.get("text")
                if not t:
                    continue
                obj = json.loads(strip_fences(t))
                if isinstance(obj, list):
                    results.extend(obj)
                elif isinstance(obj, dict):
                    results.append(obj)
            except Exception as e:
                print("   parse error:", str(e)[:80], file=sys.stderr)

    results_df = pd.DataFrame(results)
    if "author_id" in results_df.columns:
        results_df["author_id"] = results_df["author_id"].astype(str)
        results_df = results_df.drop_duplicates("author_id", keep="first")

    base = df[SUMMARY_COLS].copy()
    base["author_id"] = base["author_id"].astype(str)
    final_df = base.merge(results_df, on="author_id", how="left")
    final_df.to_parquet(RESULTS_PATH, index=False)

    matched = final_df["persona_codename"].notna().sum() if "persona_codename" in final_df.columns else 0
    print(f"\nStage 2 complete. Classified {matched:,}/{len(final_df):,} users -> {RESULTS_PATH}")
    if "persona_codename" in final_df.columns:
        dist = final_df["persona_codename"].value_counts(normalize=True).mul(100).round(1)
        print("\n   Persona Distribution:")
        for persona, pct in dist.items():
            print(f"      {str(persona):<35} {pct:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Async Stage 2 persona classification.")
    ap.add_argument("--max-users", type=int, default=None,
                    help="Classify only the first N users (smoke test).")
    ap.add_argument("--users-per-request", type=int, default=DEFAULT_USERS_PER_REQUEST,
                    help=f"Users packed per LLM request (default {DEFAULT_USERS_PER_REQUEST}).")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help=f"Max in-flight requests (default {DEFAULT_CONCURRENCY}).")
    ap.add_argument("--rpm", type=int, default=DEFAULT_RPM,
                    help=f"Target requests/minute ceiling - set at/below your Flash 2.5 RPM "
                         f"quota (default {DEFAULT_RPM}). Auto-throttles down on 429s.")
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                    help=f"Retries per request on transient errors (default {DEFAULT_MAX_RETRIES}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build + print one request without calling Vertex.")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
