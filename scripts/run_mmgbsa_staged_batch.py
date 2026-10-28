#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Run baseline MM/GBSA and per-FG masked MM/GBSA in memory-safe stages.

Designed for large mirrored LP-PDBBind batches where:
  - baseline jobs are memory-heavy
  - masked jobs are lighter
  - total machine thread + memory limits must be respected

Defaults are tuned for the 1000_20260520 mirrored batch under /mnt/data.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MIRROR_REPO_ROOT = Path("/mnt/data/projects/cs273b/asmani-branch")
DEFAULT_BATCH_TAG = "1000_20260520"
DEFAULT_THREADS_PER_COMPLEX = 2
DEFAULT_BASELINE_THREADS_PER_COMPLEX = 2
DEFAULT_MASKED_THREADS_PER_COMPLEX = 1
DEFAULT_MAX_TOTAL_THREADS = 20
DEFAULT_MAX_TOTAL_MEMORY_GB = 40.0
# Profiled on 2026-05-20 with a large successful case (2ogz):
# - baseline MMPBSA peak RSS ~0.45 GB
# - masked ghost MMPBSA peak RSS ~0.45 GB
# The staged planner should therefore be thread-limited under the current
# 20-thread / 40-GB machine budget rather than artificially memory-limited.
DEFAULT_BASELINE_MEM_GB = 0.5
DEFAULT_MASKED_MEM_GB = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage baseline MM/GBSA and masked MM/GBSA under explicit "
            "thread and memory budgets."
        )
    )
    p.add_argument(
        "--mirror-repo-root",
        default=str(DEFAULT_MIRROR_REPO_ROOT),
        help="Mirror repo root containing the prepared batch dataset.",
    )
    p.add_argument(
        "--batch-tag",
        default=DEFAULT_BATCH_TAG,
        help="Batch tag, e.g. 1000_20260520.",
    )
    p.add_argument(
        "--selected-csv",
        default=None,
        help="Optional explicit selected metadata CSV.",
    )
    p.add_argument(
        "--prepared-root",
        default=None,
        help="Optional explicit prepared dataset root containing {pdb}/{pdb}.cif.",
    )
    p.add_argument(
        "--fg-masks-dir",
        default=None,
        help="Optional explicit FG JSON root.",
    )
    p.add_argument(
        "--mmpbsa-dir",
        default=str(REPO_ROOT / "temp_dirs" / "mmgbsa-temp"),
        help="Directory containing run_mmpbsa.py.",
    )
    p.add_argument(
        "--workroot",
        default=None,
        help="Baseline workroot for the staged run.",
    )
    p.add_argument(
        "--masked-output-root",
        default=None,
        help="Masked output root for the staged run.",
    )
    p.add_argument(
        "--threads-per-complex",
        type=int,
        default=DEFAULT_THREADS_PER_COMPLEX,
        help=(
            "Legacy shared thread cap. If phase-specific values are not given, "
            "baseline and masked jobs both inherit this value."
        ),
    )
    p.add_argument(
        "--baseline-threads-per-complex",
        type=int,
        default=None,
        help=(
            f"Thread cap inside each baseline structure job. Default: "
            f"{DEFAULT_BASELINE_THREADS_PER_COMPLEX}."
        ),
    )
    p.add_argument(
        "--masked-threads-per-complex",
        type=int,
        default=None,
        help=(
            f"Thread cap inside each masked structure job. Default: "
            f"{DEFAULT_MASKED_THREADS_PER_COMPLEX}."
        ),
    )
    p.add_argument(
        "--max-total-threads",
        type=int,
        default=DEFAULT_MAX_TOTAL_THREADS,
        help="Global machine thread budget.",
    )
    p.add_argument(
        "--max-total-memory-gb",
        type=float,
        default=DEFAULT_MAX_TOTAL_MEMORY_GB,
        help="Global machine memory budget in GB.",
    )
    p.add_argument(
        "--baseline-mem-gb",
        type=float,
        default=DEFAULT_BASELINE_MEM_GB,
        help="Estimated memory footprint per baseline structure job.",
    )
    p.add_argument(
        "--masked-mem-gb",
        type=float,
        default=DEFAULT_MASKED_MEM_GB,
        help="Estimated memory footprint per masked structure job.",
    )
    p.add_argument(
        "--baseline-workers",
        type=int,
        default=None,
        help="Optional explicit baseline concurrency override.",
    )
    p.add_argument(
        "--masked-workers",
        type=int,
        default=None,
        help="Optional explicit masked concurrency override.",
    )
    p.add_argument(
        "--pdb-ids",
        nargs="+",
        default=None,
        help="Optional subset of PDB IDs to process.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit after selection/filtering.",
    )
    p.add_argument("--baseline-only", action="store_true", help="Run only baseline MM/GBSA.")
    p.add_argument("--masked-only", action="store_true", help="Run only masked MM/GBSA.")
    p.add_argument("--force-baseline", action="store_true", help="Rerun baseline even if FINAL_RESULTS_MMGBSA.dat exists.")
    p.add_argument("--force-masked", action="store_true", help="Rerun masked outputs even if per-group results already exist.")
    p.add_argument("--amber-prefix", default=None, help="Optional AmberTools prefix passed to masked runs.")
    p.add_argument("--dry-run", action="store_true", help="Print the planned stages without executing them.")
    return p.parse_args(argv)


def build_default_path(mirror_repo_root: Path, batch_tag: str, kind: str) -> Path:
    if kind == "selected_csv":
        return mirror_repo_root / "PDBind" / f"lp_pdbbind_{batch_tag}_bound_pdbs" / f"selected_{batch_tag}_metadata.csv"
    if kind == "prepared_root":
        return mirror_repo_root / "PDBind" / f"PDB_{batch_tag}"
    if kind == "fg_masks_dir":
        return mirror_repo_root / "outputs" / f"fg_masks_{batch_tag}"
    if kind == "workroot":
        return mirror_repo_root / "temp_dirs" / f"mmgbsa_{batch_tag}" / "mmpbsa_work"
    if kind == "masked_output_root":
        return mirror_repo_root / "temp_dirs" / f"mmgbsa_{batch_tag}" / "per_fg_masked_gbsa"
    raise KeyError(kind)


def compute_workers(
    *,
    max_total_threads: int,
    threads_per_complex: int,
    max_total_memory_gb: float,
    mem_gb_per_job: float,
    explicit_override: int | None,
) -> int:
    if explicit_override is not None:
        return max(1, explicit_override)
    by_threads = max(1, max_total_threads // max(1, threads_per_complex))
    by_memory = max(1, int(max_total_memory_gb // max(mem_gb_per_job, 0.01)))
    return max(1, min(by_threads, by_memory))


def load_selected_ids(selected_csv: Path) -> list[str]:
    ids: list[str] = []
    with selected_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pdb_id = str(row.get("pdb_id") or row.get("") or "").strip().lower()
            if pdb_id:
                ids.append(pdb_id)
    return ids


def import_run_mmpbsa_module(mmpbsa_dir: Path):
    module_name = "run_mmpbsa"
    if str(mmpbsa_dir) not in sys.path:
        sys.path.insert(0, str(mmpbsa_dir))
    cached = sys.modules.get(module_name)
    if cached is not None:
        cached_file = getattr(cached, "__file__", None)
        if cached_file and Path(cached_file).resolve() == (mmpbsa_dir / "run_mmpbsa.py").resolve():
            return cached
        del sys.modules[module_name]
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def discover_prepared_ids(prepared_root: Path, requested_ids: list[str] | None = None) -> tuple[list[str], list[str]]:
    requested = set(requested_ids or [])
    if requested_ids is not None:
        ordered = [p.lower() for p in requested_ids]
    else:
        ordered = sorted(p.name.lower() for p in prepared_root.iterdir() if p.is_dir())

    ready: list[str] = []
    missing_cif: list[str] = []
    for pdb_id in ordered:
        cif_path = prepared_root / pdb_id / f"{pdb_id}.cif"
        if cif_path.is_file():
            ready.append(pdb_id)
        else:
            missing_cif.append(pdb_id)
    return ready, missing_cif


def baseline_done(workroot: Path, pdb_id: str) -> bool:
    return (workroot / pdb_id / "FINAL_RESULTS_MMGBSA.dat").is_file()


def run_baseline_job(
    *,
    pdb_id: str,
    prepared_root: Path,
    workroot: Path,
    mmpbsa_dir: Path,
    threads_per_complex: int,
    ligand_selection_mode: str = "lp_pdbbind_strict",
) -> tuple[str, str]:
    mod = import_run_mmpbsa_module(mmpbsa_dir)
    mod.configure_thread_environment(threads_per_complex)
    cif_path = prepared_root / pdb_id / f"{pdb_id}.cif"
    try:
        mod.process_cif(
            cif_path,
            workroot,
            None,
            None,
            ligand_selection_mode,
        )
    except Exception as exc:  # noqa: BLE001
        return pdb_id, f"failed: {exc}"
    if not baseline_done(workroot, pdb_id):
        return pdb_id, "failed: missing FINAL_RESULTS_MMGBSA.dat"
    return pdb_id, "ok"


def build_mask_command(
    *,
    pdb_id: str,
    workroot: Path,
    fg_masks_dir: Path,
    masked_output_root: Path,
    prepared_root: Path,
    mmpbsa_dir: Path,
    threads_per_complex: int,
    amber_prefix: str | None,
    force_masked: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_per_fg_masked_gbsa_batch.py"),
        "--mmpbsa-dir",
        str(mmpbsa_dir),
        "--workroot",
        str(workroot),
        "--fg-masks-dir",
        str(fg_masks_dir),
        "--output-root",
        str(masked_output_root),
        "--pdbbind-root",
        str(prepared_root),
        "--pdb-ids",
        pdb_id,
        "--threads-per-complex",
        str(threads_per_complex),
        "--skip-summary",
    ]
    if amber_prefix:
        cmd.extend(["--amber-prefix", amber_prefix])
    if force_masked:
        cmd.append("--force")
    return cmd


def run_mask_job(
    *,
    pdb_id: str,
    workroot: Path,
    fg_masks_dir: Path,
    masked_output_root: Path,
    prepared_root: Path,
    mmpbsa_dir: Path,
    threads_per_complex: int,
    amber_prefix: str | None,
    force_masked: bool,
) -> tuple[str, str]:
    cmd = build_mask_command(
        pdb_id=pdb_id,
        workroot=workroot,
        fg_masks_dir=fg_masks_dir,
        masked_output_root=masked_output_root,
        prepared_root=prepared_root,
        mmpbsa_dir=mmpbsa_dir,
        threads_per_complex=threads_per_complex,
        amber_prefix=amber_prefix,
        force_masked=force_masked,
    )
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        return pdb_id, f"failed: exit {completed.returncode}"
    return pdb_id, "ok"


def run_stage(
    *,
    label: str,
    items: list[str],
    workers: int,
    fn,
    dry_run: bool,
    job_desc: str,
):
    if not items:
        print(f"[{label}] nothing to do")
        return {}

    print(f"\n[{label}] {len(items)} structure(s), concurrency={workers}")
    if dry_run:
        for pdb_id in items[:20]:
            print(f"  would run {job_desc}: {pdb_id}")
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more")
        return {pdb_id: "dry-run" for pdb_id in items}

    outcomes: dict[str, str] = {}
    progress = tqdm(total=len(items), desc=label, unit="pdb") if tqdm is not None else None
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(fn, pdb_id): pdb_id for pdb_id in items}
        for future in cf.as_completed(future_map):
            pdb_id = future_map[future]
            try:
                done_id, status = future.result()
            except Exception as exc:  # noqa: BLE001
                done_id, status = pdb_id, f"failed: {exc}"
            outcomes[done_id] = status
            print(f"  {done_id}: {status}", flush=True)
            if progress is not None:
                ok = sum(1 for s in outcomes.values() if s == "ok")
                fail = len(outcomes) - ok
                progress.update(1)
                progress.set_postfix(ok=ok, failed=fail)
    if progress is not None:
        progress.close()
    return outcomes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    baseline_threads_per_complex = (
        args.baseline_threads_per_complex
        if args.baseline_threads_per_complex is not None
        else DEFAULT_BASELINE_THREADS_PER_COMPLEX
    )
    masked_threads_per_complex = (
        args.masked_threads_per_complex
        if args.masked_threads_per_complex is not None
        else DEFAULT_MASKED_THREADS_PER_COMPLEX
    )

    mirror_repo_root = Path(args.mirror_repo_root).expanduser().resolve()
    selected_csv = Path(args.selected_csv).expanduser().resolve() if args.selected_csv else build_default_path(mirror_repo_root, args.batch_tag, "selected_csv")
    prepared_root = Path(args.prepared_root).expanduser().resolve() if args.prepared_root else build_default_path(mirror_repo_root, args.batch_tag, "prepared_root")
    fg_masks_dir = Path(args.fg_masks_dir).expanduser().resolve() if args.fg_masks_dir else build_default_path(mirror_repo_root, args.batch_tag, "fg_masks_dir")
    mmpbsa_dir = Path(args.mmpbsa_dir).expanduser().resolve()
    workroot = Path(args.workroot).expanduser().resolve() if args.workroot else build_default_path(mirror_repo_root, args.batch_tag, "workroot")
    masked_output_root = Path(args.masked_output_root).expanduser().resolve() if args.masked_output_root else build_default_path(mirror_repo_root, args.batch_tag, "masked_output_root")

    baseline_workers = compute_workers(
        max_total_threads=args.max_total_threads,
        threads_per_complex=baseline_threads_per_complex,
        max_total_memory_gb=args.max_total_memory_gb,
        mem_gb_per_job=args.baseline_mem_gb,
        explicit_override=args.baseline_workers,
    )
    masked_workers = compute_workers(
        max_total_threads=args.max_total_threads,
        threads_per_complex=masked_threads_per_complex,
        max_total_memory_gb=args.max_total_memory_gb,
        mem_gb_per_job=args.masked_mem_gb,
        explicit_override=args.masked_workers,
    )

    if not prepared_root.is_dir():
        print(f"[ERROR] Missing prepared root: {prepared_root}")
        return 1
    if not fg_masks_dir.is_dir():
        print(f"[ERROR] Missing FG masks dir: {fg_masks_dir}")
        return 1
    if not (mmpbsa_dir / "run_mmpbsa.py").is_file():
        print(f"[ERROR] Missing run_mmpbsa.py under {mmpbsa_dir}")
        return 1

    selected_ids = [p.lower() for p in args.pdb_ids] if args.pdb_ids else load_selected_ids(selected_csv)
    ready_ids, missing_cif = discover_prepared_ids(prepared_root, selected_ids)
    if args.limit is not None:
        ready_ids = ready_ids[: args.limit]

    if not args.dry_run:
        workroot.mkdir(parents=True, exist_ok=True)
        masked_output_root.mkdir(parents=True, exist_ok=True)

    print("Staged MMGBSA plan:")
    print(f"  selected_csv        : {selected_csv}")
    print(f"  prepared_root       : {prepared_root}")
    print(f"  fg_masks_dir        : {fg_masks_dir}")
    print(f"  mmpbsa_dir          : {mmpbsa_dir}")
    print(f"  workroot            : {workroot}")
    print(f"  masked_output_root  : {masked_output_root}")
    print(f"  baseline_threads    : {baseline_threads_per_complex}")
    print(f"  masked_threads      : {masked_threads_per_complex}")
    print(f"  max_total_threads   : {args.max_total_threads}")
    print(f"  max_total_memory_gb : {args.max_total_memory_gb:.1f}")
    print(f"  baseline_workers    : {baseline_workers} (est {args.baseline_mem_gb:.1f} GB/job)")
    print(f"  masked_workers      : {masked_workers} (est {args.masked_mem_gb:.1f} GB/job)")
    print(f"  selected_ids        : {len(selected_ids)}")
    print(f"  prepared_cifs       : {len(ready_ids)}")
    print(f"  missing_prepared    : {len(missing_cif)}")

    run_baseline_phase = not args.masked_only
    run_masked_phase = not args.baseline_only

    baseline_queue = [
        pdb_id for pdb_id in ready_ids
        if args.force_baseline or not baseline_done(workroot, pdb_id)
    ]
    mask_candidates = [
        pdb_id for pdb_id in ready_ids
        if baseline_done(workroot, pdb_id)
    ]

    baseline_outcomes: dict[str, str] = {}
    if run_baseline_phase:
        baseline_outcomes = run_stage(
            label="baseline_mmgbsa",
            items=baseline_queue,
            workers=baseline_workers,
            dry_run=args.dry_run,
            job_desc="baseline",
            fn=lambda pdb_id: run_baseline_job(
                pdb_id=pdb_id,
                prepared_root=prepared_root,
                workroot=workroot,
                mmpbsa_dir=mmpbsa_dir,
                threads_per_complex=baseline_threads_per_complex,
            ),
        )

    if run_masked_phase:
        mask_candidates = [
            pdb_id for pdb_id in ready_ids
            if baseline_done(workroot, pdb_id)
        ]
        run_stage(
            label="masked_mmgbsa",
            items=mask_candidates,
            workers=masked_workers,
            dry_run=args.dry_run,
            job_desc="masked",
            fn=lambda pdb_id: run_mask_job(
                pdb_id=pdb_id,
                workroot=workroot,
                fg_masks_dir=fg_masks_dir,
                masked_output_root=masked_output_root,
                prepared_root=prepared_root,
                mmpbsa_dir=mmpbsa_dir,
                threads_per_complex=masked_threads_per_complex,
                amber_prefix=args.amber_prefix,
                force_masked=args.force_masked,
            ),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
