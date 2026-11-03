#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Rerun a selected subset of CIFs through the MM/GBSA pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from pathlib import Path

import run_mmpbsa as mod


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Rerun a chosen subset of CIFs with the shared MM/GBSA pipeline."
    )
    parser.add_argument("pdb_ids", nargs="+", help="PDB IDs to rerun (e.g. 4flh 5afl)")
    parser.add_argument(
        "--threads-per-complex",
        type=int,
        default=mod.DEFAULT_THREADS_PER_COMPLEX,
        help=f"Thread cap inside each complex workflow (default: {mod.DEFAULT_THREADS_PER_COMPLEX}).",
    )
    parser.add_argument(
        "--max-total-threads",
        type=int,
        default=mod.DEFAULT_MAX_TOTAL_THREADS,
        help=f"Global thread budget across all complexes (default: {mod.DEFAULT_MAX_TOTAL_THREADS}).",
    )
    parser.add_argument(
        "--ligand-selection-mode",
        choices=mod.LIGAND_SELECTION_MODES,
        default=mod.DEFAULT_LIGAND_SELECTION_MODE,
        help="Ligand selection policy to use for this rerun.",
    )
    args = parser.parse_args()

    if args.threads_per_complex < 1:
        parser.error("--threads-per-complex must be >= 1")
    if args.max_total_threads < args.threads_per_complex:
        parser.error("--max-total-threads must be >= --threads-per-complex")

    cifs = []
    for pdb_id in args.pdb_ids:
        cif = ROOT / f"{pdb_id.lower()}.cif"
        if not cif.exists():
            parser.error(f"Missing CIF: {cif}")
        cifs.append(cif)

    workroot = ROOT / "mmpbsa_work"
    workroot.mkdir(exist_ok=True)
    max_workers = mod.compute_max_complex_workers(
        args.max_total_threads,
        args.threads_per_complex,
        len(cifs),
    )
    mod.configure_thread_environment(args.threads_per_complex)

    print(
        f"Configured {args.threads_per_complex} thread(s) per complex; "
        f"running up to {max_workers} complex(es) concurrently within a "
        f"{args.max_total_threads}-thread budget."
    )
    print(f"Ligand selection mode: {args.ligand_selection_mode}")

    failures = {}
    results = {}
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cif = {
            executor.submit(
                mod.process_cif,
                cif,
                workroot,
                None,
                None,
                args.ligand_selection_mode,
            ): cif
            for cif in cifs
        }
        for future in cf.as_completed(future_to_cif):
            cif = future_to_cif[future]
            pdb_id = cif.stem.split("_")[0]
            try:
                results[pdb_id] = str(future.result())
                print(f"DONE {pdb_id}")
            except Exception as exc:
                failures[pdb_id] = str(exc)
                print(f"FAIL {pdb_id} {exc}")

    print("SUMMARY_DONE", len(results), "SUMMARY_FAILED", len(failures))
    print("FAILURES", failures)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
