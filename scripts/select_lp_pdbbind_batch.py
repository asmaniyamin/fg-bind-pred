#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Select a non-overlapping LP-PDBBind batch of arbitrary size.

Excludes IDs already present in earlier selections or prepared datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a filtered LP-PDBBind batch while excluding IDs that are "
            "already present in earlier selections or prepared dataset roots."
        )
    )
    parser.add_argument(
        "--lp-csv",
        default="PDBind/LP-PDBBind/dataset/LP_PDBBind.csv",
        help="Path to LP_PDBBind.csv.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to write the sampled metadata CSV.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of rows to sample.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Filter `new_split` to this value. Use an empty string to disable.",
    )
    parser.add_argument(
        "--require-cl1",
        action="store_true",
        default=True,
        help="Require CL1 == True (default: enabled).",
    )
    parser.add_argument(
        "--allow-covalent",
        action="store_true",
        help="Include covalent structures instead of excluding them.",
    )
    parser.add_argument(
        "--exclude-csv",
        nargs="*",
        default=[],
        help="Existing metadata CSVs whose PDB IDs should be excluded.",
    )
    parser.add_argument(
        "--exclude-dataset-root",
        nargs="*",
        default=[],
        help="Prepared dataset roots whose child directory names should be excluded.",
    )
    parser.add_argument(
        "--exclude-pdb-id",
        nargs="*",
        default=[],
        help="Additional explicit PDB IDs to exclude.",
    )
    return parser.parse_args(argv)


def load_ids_from_csv(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    if "pdb_id" in frame.columns:
        series = frame["pdb_id"]
    else:
        first_col = frame.columns[0]
        series = frame[first_col]
    return {str(value).strip().lower() for value in series if str(value).strip()}


def load_ids_from_dataset_root(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.is_dir():
        return ids
    for child in path.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        ids.add(child.name.strip().lower())
    return ids


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lp_csv = Path(args.lp_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    if not lp_csv.is_file():
        raise FileNotFoundError(f"LP-PDBBind CSV not found: {lp_csv}")

    frame = pd.read_csv(lp_csv, index_col=0)
    frame.index = frame.index.astype(str).str.lower()

    filtered = frame.copy()
    if args.split:
        filtered = filtered[filtered["new_split"] == args.split]
    if args.require_cl1:
        filtered = filtered[filtered["CL1"] == True]  # noqa: E712
    if not args.allow_covalent:
        filtered = filtered[filtered["covalent"] == False]  # noqa: E712

    excluded: set[str] = {pdb_id.lower() for pdb_id in args.exclude_pdb_id}
    for csv_path in args.exclude_csv:
        path = Path(csv_path).expanduser().resolve()
        if path.is_file():
            excluded.update(load_ids_from_csv(path))
    for dataset_root in args.exclude_dataset_root:
        path = Path(dataset_root).expanduser().resolve()
        excluded.update(load_ids_from_dataset_root(path))

    filtered = filtered[~filtered.index.isin(excluded)]
    if len(filtered) < args.sample_size:
        raise RuntimeError(
            f"Only {len(filtered)} candidate rows remain after exclusions, "
            f"but sample_size={args.sample_size}"
        )

    sample = filtered.sample(n=args.sample_size, random_state=args.random_state)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_csv)

    print(
        f"Wrote {len(sample)} rows to {output_csv} "
        f"from {len(filtered)} eligible candidates "
        f"after excluding {len(excluded)} PDB IDs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
