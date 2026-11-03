#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Download bound PDB files for a selected LP-PDBBind metadata CSV.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download RCSB PDB files for the PDB IDs listed in a metadata CSV."
    )
    parser.add_argument(
        "--selected-csv",
        required=True,
        help="CSV containing a pdb_id column or PDB IDs in the first column.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where downloaded <pdb_id>.pdb files should be written.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not redownload PDB files that already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds for each download.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum parallel download workers.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help=(
            "Return exit 0 even when some downloads fail. The failure list is "
            "still printed. Useful for large batches where a small fraction of "
            "404s/network errors are expected and the caller wants downstream "
            "stages to continue."
        ),
    )
    parser.add_argument(
        "--max-failure-fraction",
        type=float,
        default=0.10,
        help=(
            "When --allow-failures is set, still exit nonzero if the failure "
            "fraction exceeds this threshold (default: 0.10 = 10%%)."
        ),
    )
    return parser.parse_args(argv)


def iter_pdb_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        ids: list[str] = []
        for row in reader:
            pdb_id = str(row.get("pdb_id") or row.get("") or "").strip().lower()
            if pdb_id:
                ids.append(pdb_id)
    return ids


def download_one(pdb_id: str, output_path: Path, timeout: int) -> None:
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    if response.text.startswith("<!DOCTYPE"):
        raise RuntimeError(f"Unexpected HTML response for {pdb_id} from {url}")
    output_path.write_text(response.text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_csv = Path(args.selected_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not selected_csv.is_file():
        raise FileNotFoundError(f"Selected metadata CSV not found: {selected_csv}")

    pdb_ids = iter_pdb_ids(selected_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    skipped = 0
    jobs: list[tuple[int, str, Path]] = []
    for index, pdb_id in enumerate(pdb_ids, start=1):
        output_path = output_dir / f"{pdb_id}.pdb"
        if args.skip_existing and output_path.exists():
            skipped += 1
            print(f"[{index}/{len(pdb_ids)}] {pdb_id}: skip existing")
            continue
        jobs.append((index, pdb_id, output_path))

    downloaded = 0
    failed: list[str] = []
    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=len(pdb_ids),
            initial=skipped,
            desc="Downloading PDBs",
            unit="pdb",
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(download_one, pdb_id, output_path, args.timeout): (index, pdb_id)
            for index, pdb_id, output_path in jobs
        }
        for future in as_completed(future_map):
            index, pdb_id = future_map[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{pdb_id}: {type(exc).__name__}: {exc}")
                print(f"[{index}/{len(pdb_ids)}] {pdb_id}: FAILED {exc}")
            else:
                downloaded += 1
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    downloaded=downloaded,
                    skipped=skipped,
                    failed=len(failed),
                )
            else:
                if future.exception() is None:
                    print(f"[{index}/{len(pdb_ids)}] {pdb_id}: downloaded")

    if progress is not None:
        progress.close()

    print(
        f"Done. downloaded={downloaded} skipped={skipped} failed={len(failed)} "
        f"out_dir={output_dir}"
    )
    if failed:
        for item in failed:
            print(f"  {item}")
        if not args.allow_failures:
            return 1
        fraction = len(failed) / max(1, len(pdb_ids) - skipped)
        if fraction > args.max_failure_fraction:
            print(
                f"failure fraction {fraction:.1%} exceeds "
                f"--max-failure-fraction {args.max_failure_fraction:.1%}; "
                f"returning nonzero so the orchestrator halts."
            )
            return 1
        print(
            f"--allow-failures is set and failure fraction {fraction:.1%} is "
            f"within --max-failure-fraction {args.max_failure_fraction:.1%}; "
            f"returning 0 so downstream stages can proceed."
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
