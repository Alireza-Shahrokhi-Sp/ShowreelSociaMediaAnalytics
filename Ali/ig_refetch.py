"""Re-fetch Instagram media metadata that yt-dlp does NOT capture.

The ``multimodal_dataset_fixed`` ``.info.json`` dumps (from yt-dlp) carry no
music, no tagged accounts, and no product/shopping tags. Instagram's own private
web endpoint does. This module re-queries it per post, using the logged-in
session in ``instagram_cookies.txt``, and extracts the interaction-relevant
features only (NOT captions/comments — those you already have).

Endpoint:  GET https://www.instagram.com/api/v1/media/{pk}/info/
Auth:      session cookies + X-IG-App-ID + X-CSRFToken headers.
Keying:    by SHORTCODE (the post folder name) → numeric ``pk`` via the standard
           base62 decode. NOTE: the cleaned datasets' ``media_id`` is the *Graph
           API* id, a different id-space — do NOT use it here (join via shortcode).

What it pulls per post
----------------------
  music   : source (licensed / original / none), song title, artist,
            audio_id, audio_cluster_id (groups posts sharing the same sound —
            a strong virality signal), duration, is_explicit
  tags    : tagged accounts (usertags), coauthors, product/shopping tags,
            paid-partnership sponsor(s)
  signals : like_count, comment_count, play_count, view_count, reshare_count,
            is_paid_partnership, video_duration, plus n_hashtags / n_mentions
            (numeric only — the caption text itself is discarded)

Stdlib only (urllib) — no requests/instagrapi needed. pandas is used solely to
write the final parquet. Results stream to a JSONL checkpoint so runs resume.

⚠️ This makes authenticated requests against Instagram with a real session.
Keep ``--sleep`` conservative; a 401 means the session died, a sustained 429
means you are rate-limited (back off / stop). Start with ``--limit 1`` to smoke
test before the full 372.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("ig_refetch")

IG_APP_ID = "936619743392459"
INFO_URL = "https://www.instagram.com/api/v1/media/{pk}/info/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36")
SHORTCODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)
MENTION_RE = re.compile(r"@[\w.]+", re.UNICODE)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class RefetchConfig:
    cookies_path: str = "instagram_cookies.txt"
    dataset_root: str = "multimodal_dataset_fixed"
    subdirs: tuple = ("reel", "feed")          # which post types to re-fetch
    out_jsonl: str = "Output/ig_media_features.jsonl"
    out_parquet: str = "Output/ig_media_features.parquet"
    raw_dir: Optional[str] = None              # set to keep raw API responses
    sleep: float = 5.0                         # base seconds between requests
    jitter: float = 3.0                        # + random[0, jitter)
    timeout: float = 30.0
    max_retries: int = 3


# --------------------------------------------------------------------------- #
# Shortcode → pk  (Instagram base62; standard algorithm)
# --------------------------------------------------------------------------- #
def shortcode_to_pk(shortcode: str) -> int:
    pk = 0
    for ch in shortcode:
        pk = pk * 64 + SHORTCODE_ALPHABET.index(ch)
    return pk


# --------------------------------------------------------------------------- #
# Cookie handling — the file is a Cookie-Editor JSON export {url, cookies:[...]}
# --------------------------------------------------------------------------- #
def load_cookies(path: str) -> Dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw["cookies"] if isinstance(raw, dict) and "cookies" in raw else raw
    if isinstance(items, dict):                      # {name: value}
        return {str(k): str(v) for k, v in items.items()}
    return {c["name"]: c["value"] for c in items}    # [{name, value, ...}]


def build_headers(cookies: Dict[str, str]) -> Dict[str, str]:
    missing = [c for c in ("sessionid", "csrftoken", "ds_user_id") if c not in cookies]
    if missing:
        raise ValueError(f"cookie jar missing essential cookies: {missing}")
    cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
    return {
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "X-IG-App-ID": IG_APP_ID,
        "X-CSRFToken": cookies["csrftoken"],
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.instagram.com/",
        "Cookie": cookie_header,
    }


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
class SessionError(RuntimeError):
    """Raised when the session is dead (401) — stop the whole run."""


def fetch_media_info(pk: int, headers: Dict[str, str], cfg: RefetchConfig) -> Dict[str, Any]:
    url = INFO_URL.format(pk=pk)
    req = urllib.request.Request(url, headers=headers)
    last_exc: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            items = payload.get("items") or []
            if not items:
                raise RuntimeError(f"empty items (status={payload.get('status')})")
            return items[0]
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 401:
                raise SessionError("401 Unauthorized — session cookie is dead/expired")
            if e.code == 404:
                raise RuntimeError("404 — media not found (private/deleted?)")
            if e.code == 429 or 500 <= e.code < 600:
                wait = cfg.sleep * (2 ** attempt) + random.uniform(0, cfg.jitter)
                LOGGER.warning("HTTP %s on pk=%s (attempt %d) — backing off %.1fs",
                               e.code, pk, attempt, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_exc = e
            wait = cfg.sleep * attempt + random.uniform(0, cfg.jitter)
            LOGGER.warning("%s on pk=%s (attempt %d) — retry in %.1fs",
                           type(e).__name__, pk, attempt, wait)
            time.sleep(wait)
    raise RuntimeError(f"exhausted retries: {last_exc}")


# --------------------------------------------------------------------------- #
# Parse — keep only interaction-relevant fields
# --------------------------------------------------------------------------- #
def _music(item: Dict[str, Any]) -> Dict[str, Any]:
    clips = item.get("clips_metadata") or {}
    mi = (clips.get("music_info") or {}).get("music_asset_info") or {}
    osi = clips.get("original_sound_info") or {}
    audio_type = clips.get("audio_type")            # 'original_sounds' | 'licensed_music' | ...
    mashups_allowed = (clips.get("mashup_info") or {}).get("mashups_allowed")
    if mi:
        return {
            "music_source": "licensed", "audio_type": audio_type,
            "song_title": mi.get("title"),
            "artist": mi.get("display_artist"),
            "audio_id": mi.get("id"),
            "audio_cluster_id": mi.get("audio_cluster_id"),
            "music_duration_ms": mi.get("duration_in_ms"),
            "is_explicit": mi.get("is_explicit"),
            "mashups_allowed": mashups_allowed,
        }
    if osi:
        return {
            "music_source": "original", "audio_type": audio_type,
            "song_title": osi.get("original_audio_title"),
            "artist": (osi.get("ig_artist") or {}).get("username"),
            "audio_id": osi.get("audio_asset_id"),
            "audio_cluster_id": clips.get("music_canonical_id"),
            "music_duration_ms": osi.get("duration_in_ms"),
            "is_explicit": osi.get("is_explicit"),
            "mashups_allowed": mashups_allowed,
        }
    return {"music_source": "none", "audio_type": audio_type, "song_title": None,
            "artist": None, "audio_id": None, "audio_cluster_id": None,
            "music_duration_ms": None, "is_explicit": None,
            "mashups_allowed": mashups_allowed}


def _tags(item: Dict[str, Any]) -> Dict[str, Any]:
    # Tagged accounts — keep username AND user id (pk), as parallel lists.
    usertags = ((item.get("usertags") or {}).get("in")) or []
    tagged_u = [(u.get("user") or {}) for u in usertags]
    tagged = [u.get("username") for u in tagged_u]
    tagged_ids = [str(u.get("pk")) for u in tagged_u if u.get("pk") is not None]

    # Collaborations (coauthors) — username AND id (pk) both requested.
    coauth = item.get("coauthor_producers") or []
    coauthors = [c.get("username") for c in coauth]
    coauthor_ids = [str(c.get("pk")) for c in coauth if c.get("pk") is not None]

    prod_in = ((item.get("product_tags") or {}).get("in")) or []
    products = []
    for p in prod_in:
        pr = p.get("product") or {}
        products.append({
            "product_id": pr.get("product_id"),
            "name": pr.get("name"),
            "merchant": (pr.get("merchant") or {}).get("username"),
        })
    sponsors = [s.get("sponsor", {}).get("username") for s in (item.get("sponsor_tags") or [])]
    return {
        "tagged_usernames": [t for t in tagged if t],
        "tagged_user_ids": tagged_ids,
        "n_tagged": len([t for t in tagged if t]),
        "coauthors": [c for c in coauthors if c],
        "coauthor_ids": coauthor_ids,
        "n_coauthors": len([c for c in coauthors if c]),
        "product_tags": products,
        "n_product_tags": len(products),
        "n_featured_products": len(item.get("featured_products") or []),
        "sponsor_usernames": [s for s in sponsors if s],
        "is_paid_partnership": item.get("is_paid_partnership"),
    }


def _signals(item: Dict[str, Any]) -> Dict[str, Any]:
    cap = (item.get("caption") or {}).get("text") or ""   # used for counts only, NOT stored
    owner = item.get("owner") or item.get("user") or {}
    loc = item.get("location") or ((item.get("locations") or [None])[0]) or {}
    return {
        "owner_username": owner.get("username"),
        "owner_id": owner.get("pk"),
        "media_type": item.get("media_type"),
        "product_type": item.get("product_type"),
        "taken_at": item.get("taken_at"),
        "like_count": item.get("like_count"),
        "comment_count": item.get("comment_count"),
        "play_count": item.get("play_count") or item.get("ig_play_count"),
        "view_count": item.get("view_count"),
        # like_count/view_count are UNRELIABLE when this is True (likes hidden):
        "like_and_view_counts_disabled": item.get("like_and_view_counts_disabled"),
        "reshare_count": item.get("reshare_count"),
        "share_count_disabled": item.get("share_count_disabled"),
        "has_audio": item.get("has_audio"),
        "video_duration": item.get("video_duration"),
        "location_name": loc.get("name") if isinstance(loc, dict) else None,
        "n_hashtags": len(HASHTAG_RE.findall(cap)),
        "n_mentions": len(MENTION_RE.findall(cap)),
    }


def parse_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ig_pk": item.get("pk"), "ig_code": item.get("code")}
    out.update(_signals(item))
    out.update(_music(item))
    out.update(_tags(item))
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def discover_shortcodes(cfg: RefetchConfig) -> List[str]:
    root = Path(cfg.dataset_root)
    codes: List[str] = []
    for sub in cfg.subdirs:
        d = root / sub
        if d.is_dir():
            codes.extend(p.name for p in sorted(d.iterdir()) if p.is_dir())
    return codes


def _done_set(jsonl: Path) -> set:
    if not jsonl.exists():
        return set()
    done = set()
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                done.add(json.loads(line)["shortcode"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def run(cfg: RefetchConfig, limit: Optional[int] = None,
        only: Optional[List[str]] = None, dry_run: bool = False) -> None:
    headers = build_headers(load_cookies(cfg.cookies_path))
    codes = only or discover_shortcodes(cfg)
    out_jsonl = Path(cfg.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = _done_set(out_jsonl)
    todo = [c for c in codes if c not in done]
    if limit:
        todo = todo[:limit]
    LOGGER.info("%d posts total, %d already done, %d to fetch%s.",
                len(codes), len(done), len(todo), " (dry-run)" if dry_run else "")

    if dry_run:
        for c in todo:
            LOGGER.info("  %s -> pk %d", c, shortcode_to_pk(c))
        return

    raw_dir = Path(cfg.raw_dir) if cfg.raw_dir else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    ok = err = 0
    with out_jsonl.open("a", encoding="utf-8") as sink:
        for i, code in enumerate(todo, 1):
            pk = shortcode_to_pk(code)
            rec: Dict[str, Any] = {"shortcode": code, "fetched_at": int(time.time())}
            try:
                item = fetch_media_info(pk, headers, cfg)
                if raw_dir:
                    (raw_dir / f"{code}.json").write_text(
                        json.dumps(item, ensure_ascii=False), encoding="utf-8")
                rec.update(parse_item(item))
                rec["fetch_status"] = "ok"
                ok += 1
                LOGGER.info("[%d/%d] %s ✓ music=%s tagged=%d products=%d",
                            i, len(todo), code, rec.get("music_source"),
                            rec.get("n_tagged", 0), rec.get("n_product_tags", 0))
            except SessionError as e:
                LOGGER.error("SESSION DEAD at %s: %s — stopping.", code, e)
                break
            except Exception as e:  # noqa: BLE001 — record & continue
                rec["fetch_status"] = f"error: {e}"
                err += 1
                LOGGER.warning("[%d/%d] %s ✗ %s", i, len(todo), code, e)
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            if i < len(todo):
                time.sleep(cfg.sleep + random.uniform(0, cfg.jitter))
    LOGGER.info("Done: %d ok, %d errors. JSONL → %s", ok, err, out_jsonl)
    _write_parquet(cfg)


def reparse_from_raw(cfg: RefetchConfig) -> None:
    """Rebuild the JSONL + parquet from saved raw responses — NO network.

    Use after changing ``parse_item`` (e.g. to add tagged/coauthor IDs): every
    ``<shortcode>.json`` in ``raw_dir`` is re-parsed with the current logic.
    Error rows (shortcodes with no saved raw) are preserved from the old JSONL.
    """
    if not cfg.raw_dir:
        raise ValueError("--raw-dir is required for --reparse")
    raw_dir = Path(cfg.raw_dir)
    raw_files = sorted(raw_dir.glob("*.json"))
    out_jsonl = Path(cfg.out_jsonl)

    # Preserve prior error rows (no raw file exists for them).
    have_raw = {f.stem for f in raw_files}
    preserved: List[Dict[str, Any]] = []
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("shortcode") not in have_raw:
                preserved.append(rec)

    n_ok = 0
    with out_jsonl.open("w", encoding="utf-8") as sink:
        for f in raw_files:
            item = json.loads(f.read_text(encoding="utf-8"))
            rec: Dict[str, Any] = {"shortcode": f.stem}
            rec.update(parse_item(item))
            rec["fetch_status"] = "ok"
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
        for rec in preserved:
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    LOGGER.info("Reparsed %d raw responses (+%d preserved error rows) → %s",
                n_ok, len(preserved), out_jsonl)
    _write_parquet(cfg)


def _write_parquet(cfg: RefetchConfig) -> None:
    import pandas as pd
    rows = [json.loads(l) for l in Path(cfg.out_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        LOGGER.warning("No rows to write to parquet.")
        return
    df = pd.DataFrame(rows)
    Path(cfg.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.out_parquet, index=False)
    LOGGER.info("Wrote %d rows → %s", len(df), cfg.out_parquet)


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-fetch IG music/tags/signals per post.")
    ap.add_argument("--cookies", default=RefetchConfig.cookies_path)
    ap.add_argument("--root", default=RefetchConfig.dataset_root)
    ap.add_argument("--subdirs", nargs="*", default=list(RefetchConfig.subdirs))
    ap.add_argument("--out-jsonl", default=RefetchConfig.out_jsonl)
    ap.add_argument("--out-parquet", default=RefetchConfig.out_parquet)
    ap.add_argument("--raw-dir", default=None, help="keep raw API responses here")
    ap.add_argument("--sleep", type=float, default=RefetchConfig.sleep)
    ap.add_argument("--jitter", type=float, default=RefetchConfig.jitter)
    ap.add_argument("--limit", type=int, default=None, help="fetch at most N (smoke test)")
    ap.add_argument("--only", nargs="*", default=None, help="specific shortcodes")
    ap.add_argument("--dry-run", action="store_true", help="list shortcode→pk, no requests")
    ap.add_argument("--reparse", action="store_true",
                    help="rebuild outputs from saved --raw-dir responses, no network")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    cfg = RefetchConfig(
        cookies_path=a.cookies, dataset_root=a.root, subdirs=tuple(a.subdirs),
        out_jsonl=a.out_jsonl, out_parquet=a.out_parquet, raw_dir=a.raw_dir,
        sleep=a.sleep, jitter=a.jitter,
    )
    if a.reparse:
        reparse_from_raw(cfg)
    else:
        run(cfg, limit=a.limit, only=a.only, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
