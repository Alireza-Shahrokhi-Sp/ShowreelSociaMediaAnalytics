"""
Patch: normalize hallucinated persona codenames in stage2 output parquets.

The Stage 2 LLM occasionally omits the 'THE_' prefix (e.g. outputs
'CASUAL_COMPLIMENTER' instead of 'THE_CASUAL_COMPLIMENTER').  This script
applies a canonical mapping and rewrites the affected files in-place.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Map any hallucinated / alias codename -> canonical codename.
# Extend this dict whenever a new hallucination is observed.
CODENAME_ALIASES: dict[str, str] = {
    "CASUAL_COMPLIMENTER": "THE_CASUAL_COMPLIMENTER",
    "EMOJI_REACTOR":       "THE_EMOJI_REACTOR",
    "TAGGER":              "THE_TAGGER",
    "STORYTELLER":         "THE_STORYTELLER",
    "SUPERFAN":            "THE_SUPERFAN",
    "INQUIRER":            "THE_INQUIRER",
    "CRITIC":              "THE_CRITIC",
    "ADVISOR":             "THE_ADVISOR",
    "SPAMMER":             "THE_SPAMMER",
    "HATER":               "THE_HATER",
}

DEFAULT_TARGETS = [
    "Ali/outputs/stage2_persona/user_personas.parquet",
    "Ali/outputs/stage2_persona_20_percent/user_personas_sample.parquet",
    "Ali/outputs/stage1_persona/pathway_a_assignments_instagram.parquet",
]

CODENAME_COL = {
    "user_personas.parquet":        "persona_codename",
    "user_personas_sample.parquet": "persona_codename",
    "pathway_a_assignments_instagram.parquet": "persona",
}


def patch_file(path: Path, dry_run: bool = False) -> int:
    col = CODENAME_COL.get(path.name)
    if col is None:
        # fall back: try common column names
        df = pd.read_parquet(path)
        for candidate in ("persona_codename", "persona", "codename"):
            if candidate in df.columns:
                col = candidate
                break
        if col is None:
            print(f"  SKIP {path} — no persona column found")
            return 0
    else:
        df = pd.read_parquet(path)

    if col not in df.columns:
        print(f"  SKIP {path} — column '{col}' not present")
        return 0

    mask = df[col].isin(CODENAME_ALIASES)
    n_bad = mask.sum()
    if n_bad == 0:
        print(f"  OK   {path} — no aliases found")
        return 0

    print(f"  FIX  {path} — patching {n_bad} row(s):")
    for alias, canon in CODENAME_ALIASES.items():
        n = (df[col] == alias).sum()
        if n:
            print(f"       {alias!r} -> {canon!r}  ({n} rows)")

    if not dry_run:
        df[col] = df[col].replace(CODENAME_ALIASES)
        df.to_parquet(path, index=False)
        print(f"       Written: {path}")

    return n_bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch hallucinated persona codenames")
    parser.add_argument("files", nargs="*", help="Parquet files to patch (default: standard outputs)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    root = Path(__file__).parent.parent  # d:\Polythecninco di Milano\AFB_Lab
    targets = [Path(f) for f in args.files] if args.files else [root / t for t in DEFAULT_TARGETS]

    total = 0
    for p in targets:
        if not p.exists():
            print(f"  MISS {p} — file not found, skipping")
            continue
        total += patch_file(p, dry_run=args.dry_run)

    print(f"\nDone. {'(dry-run) ' if args.dry_run else ''}Total rows patched: {total}")


if __name__ == "__main__":
    main()
