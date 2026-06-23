"""
Merge two Stage 2 persona output parquets into a single combined file.

Run 1 (stage2_persona):             1,400 users  (early/partial Stage 2 batch)
Run 2 (stage2_persona_20_percent):  38,903 users (larger Stage 2 batch)

For the 284 overlapping users the higher-confidence assignment wins.
Output: Ali/outputs/stage2_persona_combined/user_personas_combined.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent  # AFB_Lab root

SRC_FULL   = ROOT / "Ali/outputs/stage2_persona/user_personas.parquet"
SRC_SAMPLE = ROOT / "Ali/outputs/stage2_persona_20_percent/user_personas_sample.parquet"
OUT_DIR    = ROOT / "Ali/outputs/stage2_persona_combined"
OUT_PATH   = OUT_DIR / "user_personas_combined.parquet"


def merge() -> pd.DataFrame:
    df_full   = pd.read_parquet(SRC_FULL)
    df_sample = pd.read_parquet(SRC_SAMPLE)

    # Apply codename alias fix to both (idempotent)
    from patch_persona_codenames import CODENAME_ALIASES
    for df in (df_full, df_sample):
        df["persona_codename"] = df["persona_codename"].replace(CODENAME_ALIASES)

    overlap = set(df_full["author_id"]) & set(df_sample["author_id"])
    print(f"Full run:   {len(df_full):,} rows")
    print(f"Sample run: {len(df_sample):,} rows")
    print(f"Overlap:    {len(overlap):,} users")

    # Non-overlapping rows from each run
    only_full   = df_full[~df_full["author_id"].isin(overlap)]
    only_sample = df_sample[~df_sample["author_id"].isin(overlap)]

    # Overlapping rows: keep higher-confidence row per author
    ov_full   = df_full[df_full["author_id"].isin(overlap)]
    ov_sample = df_sample[df_sample["author_id"].isin(overlap)]
    ov_both   = pd.concat([ov_full, ov_sample], ignore_index=True)
    ov_best   = (
        ov_both
        .sort_values("confidence", ascending=False)
        .drop_duplicates(subset="author_id", keep="first")
    )

    ov_cmp = ov_full.set_index("author_id")[["persona_codename"]].join(
        ov_sample.set_index("author_id")[["persona_codename"]], lsuffix="_full", rsuffix="_sample"
    )
    agreed = (ov_cmp["persona_codename_full"] == ov_cmp["persona_codename_sample"]).sum()
    print(f"Overlap agreement: {agreed}/{len(overlap)} ({agreed/len(overlap)*100:.1f}%)")

    combined = pd.concat([only_full, only_sample, ov_best], ignore_index=True)
    combined = combined.sort_values("author_id").reset_index(drop=True)

    print(f"\nCombined:   {len(combined):,} unique users")
    print("\nPersona distribution:")
    print(combined["persona_codename"].value_counts(normalize=True).mul(100).round(1).to_string())

    return combined


def main() -> None:
    combined = merge()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
