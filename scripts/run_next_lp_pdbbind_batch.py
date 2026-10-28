#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Plan or launch a mirrored LP-PDBBind batch under /mnt/data/projects/... style roots.

This script does not force every step to run at once; instead it gives us one
place to compute the canonical mirrored paths and dispatch the existing tools in
the correct order.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BatchLayout:
    mirror_repo_root: Path
    batch_tag: str
    selected_csv: Path
    download_dir: Path
    prepared_root: Path
    fg_output_dir: Path
    pocket_summary: Path
    pocket_log: Path
    schrodinger_runtime_root: Path
    schrodinger_home: Path
    jobserver_dir: Path
    schrodinger_tmpdir: Path


_RUNTIME_CONFIG: dict[str, int] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select, download, prepare, pocket-process, and FG-process a future "
            "LP-PDBBind batch directly under a mirrored repo-style root."
        )
    )
    parser.add_argument(
        "--mirror-repo-root",
        default="/mnt/data/projects/cs273b/asmani-branch",
        help="Mirror repo root to use for all generated dataset paths.",
    )
    parser.add_argument(
        "--batch-tag",
        default="1000_20260520",
        help="Tag used in output directory names, e.g. 1000_20260520.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="How many LP-PDBBind rows to sample.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the LP-PDBBind sampler.",
    )
    parser.add_argument(
        "--exclude-csv",
        nargs="*",
        default=None,
        help="Existing metadata CSVs to exclude from sampling.",
    )
    parser.add_argument(
        "--exclude-dataset-root",
        nargs="*",
        default=None,
        help="Prepared dataset roots whose PDB IDs should be excluded from sampling.",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=2,
        help="PrepWizard threads per structure.",
    )
    parser.add_argument(
        "--prep-concurrency",
        type=int,
        default=None,
        help="Optional explicit PrepWizard concurrency override.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=None,
        help="Optional explicit download worker override.",
    )
    parser.add_argument(
        "--pocket-workers",
        type=int,
        default=None,
        help="Optional explicit pocket worker override.",
    )
    parser.add_argument(
        "--fg-workers",
        type=int,
        default=None,
        help="Optional explicit FG worker override.",
    )
    parser.add_argument(
        "--max-total-threads",
        type=int,
        default=20,
        help="Global thread budget for planning concurrency.",
    )
    parser.add_argument(
        "--max-total-memory-gb",
        type=float,
        default=40.0,
        help="Global memory budget in GB for planning concurrency.",
    )
    parser.add_argument(
        "--download-mem-gb-per-worker",
        type=float,
        default=0.25,
        help="Estimated memory footprint per download worker.",
    )
    parser.add_argument(
        "--preprocess-mem-gb-per-job",
        type=float,
        default=4.0,
        help="Estimated memory footprint per PrepWizard job.",
    )
    parser.add_argument(
        "--pocket-mem-gb-per-worker",
        type=float,
        default=2.0,
        help="Estimated memory footprint per pocket worker.",
    )
    parser.add_argument(
        "--fg-mem-gb-per-worker",
        type=float,
        default=2.0,
        help="Estimated memory footprint per FG worker.",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Run the LP-PDBBind selection step.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run the PDB download step.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run the PrepWizard preparation step.",
    )
    parser.add_argument(
        "--pockets",
        action="store_true",
        help="Run the pocket extraction step.",
    )
    parser.add_argument(
        "--fg",
        action="store_true",
        help="Run the functional-group extraction step.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable all steps in order.",
    )
    parser.add_argument(
        "--allow-prepwizard-failures",
        action="store_true",
        help=(
            "Pass --allow-failures through to preprocess_batch_linux.py so a "
            "small fraction of PrepWizard failures doesn't abort the "
            "orchestrator before pockets/fg can run."
        ),
    )
    parser.add_argument(
        "--max-prepwizard-failure-fraction",
        type=float,
        default=0.10,
        help=(
            "When --allow-prepwizard-failures is set, still halt if the "
            "failure fraction exceeds this threshold (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--allow-download-failures",
        action="store_true",
        help=(
            "Pass --allow-failures through to download_selected_pdbs.py so "
            "transient 404s / RCSB timeouts don't abort the orchestrator."
        ),
    )
    parser.add_argument(
        "--max-download-failure-fraction",
        type=float,
        default=0.10,
        help=(
            "When --allow-download-failures is set, still halt if the failure "
            "fraction exceeds this threshold (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    parser.add_argument(
        "prepwizard_args",
        nargs=argparse.REMAINDER,
        help="Additional args passed through to preprocess_batch_linux.py after '--'.",
    )
    return parser.parse_args(argv)


def build_layout(mirror_repo_root: Path, batch_tag: str) -> BatchLayout:
    selected_dir = mirror_repo_root / "PDBind" / f"lp_pdbbind_{batch_tag}_bound_pdbs"
    prepared_root = mirror_repo_root / "PDBind" / f"PDB_{batch_tag}"
    fg_output_dir = mirror_repo_root / "outputs" / f"fg_masks_{batch_tag}"
    schrodinger_runtime_root = mirror_repo_root / ".schrodinger_runtime"
    return BatchLayout(
        mirror_repo_root=mirror_repo_root,
        batch_tag=batch_tag,
        selected_csv=selected_dir / f"selected_{batch_tag}_metadata.csv",
        download_dir=selected_dir,
        prepared_root=prepared_root,
        fg_output_dir=fg_output_dir,
        pocket_summary=prepared_root / "_batch_pocket_summary.json",
        pocket_log=prepared_root / "_batch_pocket_log.txt",
        schrodinger_runtime_root=schrodinger_runtime_root,
        schrodinger_home=schrodinger_runtime_root / "home",
        jobserver_dir=schrodinger_runtime_root / "jobserverd",
        schrodinger_tmpdir=schrodinger_runtime_root / "tmp",
    )


def base_python() -> str:
    return sys.executable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compute_parallelism(
    *,
    max_total_threads: int,
    threads_per_worker: int,
    max_total_memory_gb: float,
    mem_gb_per_worker: float,
    explicit_override: int | None,
) -> int:
    if explicit_override is not None:
        return explicit_override
    by_threads = max(1, max_total_threads // max(1, threads_per_worker))
    by_memory = max(1, int(max_total_memory_gb // max(mem_gb_per_worker, 0.01)))
    return max(1, min(by_threads, by_memory))


def set_runtime_config(config: dict[str, int]) -> None:
    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = config


def runtime_config() -> dict[str, int]:
    assert _RUNTIME_CONFIG is not None
    return _RUNTIME_CONFIG


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_or_print(cmd: list[str], *, dry_run: bool, cwd: Path | None = None) -> None:
    if dry_run:
        prefix = f"(cd {cwd} && " if cwd else ""
        suffix = ")" if cwd else ""
        print(prefix + quote_cmd(cmd) + suffix)
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def selection_command(args: argparse.Namespace, layout: BatchLayout) -> list[str]:
    cmd = [
        base_python(),
        str(repo_root() / "scripts" / "select_lp_pdbbind_batch.py"),
        "--output-csv",
        str(layout.selected_csv),
        "--sample-size",
        str(args.sample_size),
        "--random-state",
        str(args.random_state),
    ]
    exclude_csvs = list(args.exclude_csv or [])
    if not exclude_csvs:
        exclude_csvs = [str(repo_root() / "PDBind" / "lp_pdbbind_100_bound_pdbs" / "selected_100_metadata.csv")]
    cmd.extend(["--exclude-csv", *exclude_csvs])

    exclude_roots = list(args.exclude_dataset_root or [])
    if not exclude_roots:
        exclude_roots = [str(repo_root() / "PDBind" / "PDB_100_4292026")]
    cmd.extend(["--exclude-dataset-root", *exclude_roots])
    return cmd


def download_command(args: argparse.Namespace, layout: BatchLayout) -> list[str]:
    cmd = [
        base_python(),
        str(repo_root() / "scripts" / "download_selected_pdbs.py"),
        "--selected-csv",
        str(layout.selected_csv),
        "--output-dir",
        str(layout.download_dir),
        "--skip-existing",
        "--workers",
        str(runtime_config()["download_workers"]),
    ]
    if args.allow_download_failures:
        cmd.extend([
            "--allow-failures",
            "--max-failure-fraction", str(args.max_download_failure_fraction),
        ])
    return cmd


def preprocess_command(args: argparse.Namespace, layout: BatchLayout) -> list[str]:
    cmd = [
        base_python(),
        str(repo_root() / "protein" / "clean" / "preprocess_batch_linux.py"),
        str(layout.download_dir),
        str(layout.prepared_root),
        "--glob",
        "*.pdb",
        "--concurrency",
        str(runtime_config()["prep_concurrency"]),
        "--threads-per-job",
        str(args.threads_per_job),
        "--schrodinger-home",
        str(layout.schrodinger_home),
        "--jobserver-dir",
        str(layout.jobserver_dir),
        "--schrodinger-tmpdir",
        str(layout.schrodinger_tmpdir),
        "--overwrite",
    ]
    if args.allow_prepwizard_failures:
        cmd.extend([
            "--allow-failures",
            "--max-failure-fraction", str(args.max_prepwizard_failure_fraction),
        ])
    passthrough = list(args.prepwizard_args or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    cmd.extend(passthrough)
    return cmd


def pocket_command(layout: BatchLayout) -> list[str]:
    return [
        base_python(),
        str(repo_root() / "scripts" / "run_pocket_batch.py"),
        "--dataset-root",
        str(layout.prepared_root),
        "--summary-json",
        str(layout.pocket_summary),
        "--log-path",
        str(layout.pocket_log),
        "--workers",
        str(runtime_config()["pocket_workers"]),
    ]


def fg_command(layout: BatchLayout) -> list[str]:
    return [
        base_python(),
        str(repo_root() / "scripts" / "extract_fg_masks.py"),
        "--input-dir",
        str(layout.prepared_root),
        "--output-dir",
        str(layout.fg_output_dir),
        "--workers",
        str(runtime_config()["fg_workers"]),
        "--lp-pdbbind-only",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mirror_repo_root = Path(args.mirror_repo_root).expanduser().resolve()
    layout = build_layout(mirror_repo_root, args.batch_tag)
    runtime = {
        "download_workers": compute_parallelism(
            max_total_threads=args.max_total_threads,
            threads_per_worker=1,
            max_total_memory_gb=args.max_total_memory_gb,
            mem_gb_per_worker=args.download_mem_gb_per_worker,
            explicit_override=args.download_workers,
        ),
        "prep_concurrency": compute_parallelism(
            max_total_threads=args.max_total_threads,
            threads_per_worker=args.threads_per_job,
            max_total_memory_gb=args.max_total_memory_gb,
            mem_gb_per_worker=args.preprocess_mem_gb_per_job,
            explicit_override=args.prep_concurrency,
        ),
        "pocket_workers": compute_parallelism(
            max_total_threads=args.max_total_threads,
            threads_per_worker=1,
            max_total_memory_gb=args.max_total_memory_gb,
            mem_gb_per_worker=args.pocket_mem_gb_per_worker,
            explicit_override=args.pocket_workers,
        ),
        "fg_workers": compute_parallelism(
            max_total_threads=args.max_total_threads,
            threads_per_worker=1,
            max_total_memory_gb=args.max_total_memory_gb,
            mem_gb_per_worker=args.fg_mem_gb_per_worker,
            explicit_override=args.fg_workers,
        ),
    }
    set_runtime_config(runtime)

    selected = args.all or args.select
    download = args.all or args.download
    preprocess = args.all or args.preprocess
    pockets = args.all or args.pockets
    fg = args.all or args.fg

    print("Mirrored LP-PDBBind batch layout:")
    print(f"  mirror_repo_root : {layout.mirror_repo_root}")
    print(f"  selected_csv     : {layout.selected_csv}")
    print(f"  download_dir     : {layout.download_dir}")
    print(f"  prepared_root    : {layout.prepared_root}")
    print(f"  fg_output_dir    : {layout.fg_output_dir}")
    print(f"  pocket_summary   : {layout.pocket_summary}")
    print(f"  pocket_log       : {layout.pocket_log}")
    print(f"  schrodinger_home : {layout.schrodinger_home}")
    print(f"  jobserver_dir    : {layout.jobserver_dir}")
    print(f"  schrodinger_tmp  : {layout.schrodinger_tmpdir}")
    print("Resource plan:")
    print(f"  max_total_threads: {args.max_total_threads}")
    print(f"  max_total_memory : {args.max_total_memory_gb:.1f} GB")
    print(f"  download_workers : {runtime['download_workers']}")
    print(f"  prep_concurrency : {runtime['prep_concurrency']} @ {args.threads_per_job} threads/job")
    print(f"  pocket_workers   : {runtime['pocket_workers']}")
    print(f"  fg_workers       : {runtime['fg_workers']}")
    print()

    commands: list[tuple[str, list[str]]] = []
    if selected:
        commands.append(("select", selection_command(args, layout)))
    if download:
        commands.append(("download", download_command(args, layout)))
    if preprocess:
        commands.append(("preprocess", preprocess_command(args, layout)))
    if pockets:
        commands.append(("pockets", pocket_command(layout)))
    if fg:
        commands.append(("fg", fg_command(layout)))

    if not commands:
        print("No steps selected. Use one of --select/--download/--preprocess/--pockets/--fg or --all.")
        return 0

    for label, cmd in commands:
        print(f"[{label}]")
        run_or_print(cmd, dry_run=args.dry_run, cwd=repo_root())
        if args.dry_run:
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
