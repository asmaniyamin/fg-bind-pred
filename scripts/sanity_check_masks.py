# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
scripts/sanity_check_masks.py

Load all JSON files produced by extract_fg_masks.py and print a summary:
  - Total ligand count
  - Functional group name frequency table
  - Average mask density (fraction of atoms in any FG)
  - Outlier flags (0 FGs, all-1 or all-0 masks, suspiciously large/small density)

Usage:
    python scripts/sanity_check_masks.py --output-dir outputs/fg_masks/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [ERROR] Could not read {path}: {exc}")
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarise functional group mask JSON outputs."
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing per-ligand JSON files.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top functional groups to display (default: 25).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()

    json_files = sorted(out_dir.glob("**/*.json"))
    if not json_files:
        print(f"No JSON files found under {out_dir}")
        return 1

    print(f"Loading {len(json_files)} JSON files from {out_dir} ...\n")

    total_ligands = 0
    fg_counter: Counter[str] = Counter()
    mask_densities: list[float] = []
    ligand_fg_counts: list[int] = []
    outliers: list[str] = []

    for jf in json_files:
        data = load_json(jf)
        if data is None:
            continue

        total_ligands += 1
        sid = data.get("structure_id", "?")
        lid = data.get("ligand_id", "?")
        n_atoms: int = data.get("n_atoms", 0)
        mask: list[int] = data.get("functional_group_mask", [])
        fgs: list[dict] = data.get("functional_groups", [])
        smiles: str | None = data.get("smiles")

        # FG name frequency (deduplicate per ligand so each name counts once)
        unique_fg_names = {fg["name"] for fg in fgs}
        for name in unique_fg_names:
            fg_counter[name] += 1
        ligand_fg_counts.append(len(unique_fg_names))

        # Mask density
        if n_atoms > 0 and mask:
            density = sum(mask) / n_atoms
        else:
            density = 0.0
        mask_densities.append(density)

        # Outlier detection
        tag = f"{sid}/{lid}"
        if len(fgs) == 0:
            outliers.append(f"ZERO_FGS      {tag}  (n_atoms={n_atoms}, smiles={smiles})")
        if density == 0.0 and n_atoms > 0:
            outliers.append(f"MASK_ALL_ZERO {tag}  (n_atoms={n_atoms})")
        elif density == 1.0:
            outliers.append(f"MASK_ALL_ONE  {tag}  (n_atoms={n_atoms})")
        elif density > 0.9:
            outliers.append(f"HIGH_DENSITY  {tag}  density={density:.1%}  (n_atoms={n_atoms})")
        if n_atoms == 0:
            outliers.append(f"ZERO_ATOMS    {tag}")
        if smiles is None:
            outliers.append(f"NO_SMILES     {tag}")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    sep = "=" * 60

    print(sep)
    print("  FUNCTIONAL GROUP MASK SUMMARY")
    print(sep)
    print(f"  JSON files loaded   : {len(json_files)}")
    print(f"  Total ligands       : {total_ligands}")
    print()

    if mask_densities:
        avg = sum(mask_densities) / len(mask_densities)
        lo = min(mask_densities)
        hi = max(mask_densities)
        # Histogram buckets
        buckets = [0] * 10
        for d in mask_densities:
            idx = min(int(d * 10), 9)
            buckets[idx] += 1
        print("  Mask density (fraction of atoms in any FG):")
        print(f"    avg = {avg:.1%}   min = {lo:.1%}   max = {hi:.1%}")
        print()
        print("  Density distribution:")
        for b, count in enumerate(buckets):
            bar = "#" * count
            lo_b = b * 10
            hi_b = lo_b + 10
            print(f"    {lo_b:3d}-{hi_b:3d}%  {bar}  ({count})")
        print()

    if ligand_fg_counts:
        avg_fgs = sum(ligand_fg_counts) / len(ligand_fg_counts)
        print(f"  Unique FG types per ligand: avg={avg_fgs:.1f}  "
              f"min={min(ligand_fg_counts)}  max={max(ligand_fg_counts)}")
        print(f"  Total distinct FG types seen: {len(fg_counter)}")
        print()

    print(f"  Top {args.top_n} functional groups (by ligand occurrence):")
    for rank, (name, count) in enumerate(fg_counter.most_common(args.top_n), 1):
        pct = count / total_ligands * 100 if total_ligands else 0
        print(f"    {rank:3d}. {count:4d} ligands ({pct:5.1f}%)  {name}")
    print()

    if outliers:
        print(f"  Outliers ({len(outliers)}):")
        for o in outliers[:40]:
            print(f"    {o}")
        if len(outliers) > 40:
            print(f"    ... and {len(outliers) - 40} more")
    else:
        print("  No outliers detected.")

    print(sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
