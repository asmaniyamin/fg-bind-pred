#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Stage a 2000-PDB LP-PDBBind batch on the /mnt mirror.

Thin wrapper around `scripts/run_next_lp_pdbbind_batch.py` that:
  - Defaults to batch_tag = 2000_<today> if not given
  - Auto-discovers prior batches under the mirror and excludes them from sampling
  - Passes --allow-prepwizard-failures so a small (<= 10%) fraction of failed
    PrepWizard jobs doesn't crash the pipeline before pockets/fg can run

Examples:
    # First 2000-batch (already done as 2000_20260523)
    python scripts/run_lp_pdbbind_2000_batch.py --all --workers 20 --threads-per-job 1

    # Second 2000-batch - auto-excludes the first one
    python scripts/run_lp_pdbbind_2000_batch.py --all --workers 20 --threads-per-job 1 \
        --batch-tag 2000_20260524 --random-state 43

    # Dry-run to inspect commands
    python scripts/run_lp_pdbbind_2000_batch.py --all --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = REPO_ROOT / "scripts" / "run_next_lp_pdbbind_batch.py"

MIRROR_REPO_ROOT = "/mnt/data/projects/cs273b/asmani-branch"

# Static exclusions: prior batches that live in the local repo tree (not the mirror).
STATIC_SELECTION_CSVS = [
    "PDBind/lp_pdbbind_100_bound_pdbs/selected_100_metadata.csv",
    "PDBind/lp_pdbbind_1000_bound_pdbs/selected_1000_metadata.csv",
]
STATIC_DATASET_ROOTS = [
    "PDBind/PDB_100_4292026",
]


def _discover_mirror_exclusions(mirror_root: Path, current_tag: str) -> tuple[list[str], list[str]]:
    """Auto-discover every prior PDB_<tag>/ and selected_<tag>_metadata.csv on the
    mirror, excluding the batch we're about to create."""
    csvs: list[str] = []
    roots: list[str] = []
    pdbind = mirror_root / "PDBind"
    if pdbind.is_dir():
        for child in sorted(pdbind.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("PDB_") and name != f"PDB_{current_tag}":
                roots.append(str(child))
            elif name.startswith("lp_pdbbind_") and name.endswith("_bound_pdbs"):
                # Tag is between the prefix and suffix.
                tag = name[len("lp_pdbbind_"):-len("_bound_pdbs")]
                if tag == current_tag:
                    continue
                candidate = child / f"selected_{tag}_metadata.csv"
                if candidate.is_file():
                    csvs.append(str(candidate))
    return csvs, roots


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", action="store_true", help="Run selection.")
    parser.add_argument("--download", action="store_true", help="Run download.")
    parser.add_argument("--preprocess", action="store_true", help="Run PrepWizard preprocessing.")
    parser.add_argument("--pockets", action="store_true", help="Run pocket extraction.")
    parser.add_argument("--fg", action="store_true", help="Run functional-group mask extraction.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run select+download+preprocess+pockets+fg in order (default if no step flags set).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--batch-tag",
        default=None,
        help="Batch tag (default: 2000_<YYYYMMDD>).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42). Use a different value when launching a second batch.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of PDBs to sample (default: 2000).",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=2,
        help="PrepWizard threads per structure (default: 2).",
    )
    parser.add_argument(
        "--max-total-threads",
        type=int,
        default=20,
        help="Global thread budget (default: 20).",
    )
    parser.add_argument(
        "--max-total-memory-gb",
        type=float,
        default=40.0,
        help="Global memory budget in GB (default: 40).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Single knob: pin download/prep/pocket/fg worker counts all to "
            "this value (overrides budget-based planning)."
        ),
    )
    parser.add_argument(
        "--strict-prepwizard",
        action="store_true",
        help=(
            "Abort the orchestrator on ANY PrepWizard failure (the old default). "
            "By default this wrapper sets --allow-prepwizard-failures so up to "
            "10%% of structures can fail without halting the pipeline."
        ),
    )
    parser.add_argument(
        "--max-prepwizard-failure-fraction",
        type=float,
        default=0.10,
        help="Failure-fraction ceiling when --strict-prepwizard is NOT set (default: 0.10).",
    )
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace) -> list[str]:
    step_flags = [
        flag
        for flag in ("--select", "--download", "--preprocess", "--pockets", "--fg")
        if getattr(args, flag.lstrip("-").replace("-", "_"))
    ]
    if args.all or not step_flags:
        step_flags = ["--all"]

    batch_tag = args.batch_tag or f"2000_{_dt.date.today().strftime('%Y%m%d')}"
    mirror_root = Path(MIRROR_REPO_ROOT)
    auto_csvs, auto_roots = _discover_mirror_exclusions(mirror_root, batch_tag)

    cmd = [
        sys.executable,
        str(ORCHESTRATOR),
        "--mirror-repo-root", MIRROR_REPO_ROOT,
        "--batch-tag", batch_tag,
        "--sample-size", str(args.sample_size),
        "--random-state", str(args.random_state),
        "--threads-per-job", str(args.threads_per_job),
        "--max-total-threads", str(args.max_total_threads),
        "--max-total-memory-gb", str(args.max_total_memory_gb),
        "--exclude-csv", *STATIC_SELECTION_CSVS, *auto_csvs,
        "--exclude-dataset-root", *STATIC_DATASET_ROOTS, *auto_roots,
        *step_flags,
    ]
    if args.workers is not None:
        cmd.extend([
            "--download-workers", str(args.workers),
            "--prep-concurrency", str(args.workers),
            "--pocket-workers",   str(args.workers),
            "--fg-workers",       str(args.workers),
        ])
    if not args.strict_prepwizard:
        cmd.extend([
            "--allow-prepwizard-failures",
            "--max-prepwizard-failure-fraction", str(args.max_prepwizard_failure_fraction),
            "--allow-download-failures",
            "--max-download-failure-fraction", str(args.max_prepwizard_failure_fraction),
        ])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cmd = build_command(args)
    print("Invoking:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
