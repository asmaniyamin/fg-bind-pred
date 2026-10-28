# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/clean/preprocess_batch_linux.py

Batch Schrodinger Protein Preparation Wizard runner for Linux.

Each input structure is prepared once to a Maestro master file (`.maegz`) and
then converted to `.pdb` and `.cif`, so all output formats reflect the same
prepared structure.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import preprocess as base
import preprocess_linux as linux
from tqdm import tqdm


@dataclass(frozen=True, slots=True)
class BatchJobResult:
    input_path: Path
    maegz_path: Path
    pdb_path: Path
    cif_path: Path


def parse_batch_args(argv: Iterable[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare every structure in a directory with Schrodinger PrepWizard "
            "on Linux, running the preparation once to .maegz and converting the "
            "result to .pdb and .cif."
        )
    )
    parser.add_argument("input_dir", help="Directory containing input structures.")
    parser.add_argument("output_dir", help="Directory where maegz/pdb/cif outputs are written.")
    parser.add_argument(
        "--glob",
        default="*.pdb",
        help="Glob used to select input files inside input_dir. Default: %(default)s",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of PrepWizard jobs to run at once. Default: %(default)s",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=2,
        help="Threads to request per PrepWizard job via --host localhost:N. Default: %(default)s",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on how many matching files to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N matching files before processing. Default: %(default)s",
    )
    parser.add_argument(
        "--jobserver-dir",
        help="Writable directory to use for Schrodinger local job server state.",
    )
    parser.add_argument(
        "--schrodinger-home",
        help="Writable HOME directory for Schrodinger subprocess state.",
    )
    parser.add_argument(
        "--schrodinger-tmpdir",
        help="Writable directory to use for SCHRODINGER_TMPDIR/TMPDIR/TMP/TEMP.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help=(
            "Return exit 0 even when some PrepWizard jobs fail. The failure "
            "list is still printed. Useful for large batches where a small "
            "fraction of structures are expected to fail (PrepWizard "
            "refinement errors, multi-conformer maegz that can't be reduced "
            "to one .cif, etc.) and the caller wants the orchestrator to "
            "continue to downstream stages."
        ),
    )
    parser.add_argument(
        "--max-failure-fraction",
        type=float,
        default=0.10,
        help=(
            "When --allow-failures is set, still exit nonzero if the failure "
            "fraction exceeds this threshold (default: 0.10 = 10%%). Guards "
            "against silently shipping a mostly-broken batch."
        ),
    )

    batch_args, prepwizard_args = parser.parse_known_args(list(argv) if argv is not None else None)
    _validate_batch_args(batch_args, prepwizard_args)
    return batch_args, prepwizard_args


def run_batch(argv: Iterable[str] | None = None) -> int:
    batch_args, prepwizard_args = parse_batch_args(argv)

    input_dir = Path(batch_args.input_dir).expanduser().resolve()
    output_dir = Path(batch_args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory was not found: {input_dir}")

    tools = linux.resolve_schrodinger_tools()
    linux.ensure_local_job_server(
        tools.root,
        schrodinger_home=batch_args.schrodinger_home,
        jobserver_dir=batch_args.jobserver_dir,
        schrodinger_tmpdir=batch_args.schrodinger_tmpdir,
    )

    input_paths = sorted(path.resolve() for path in input_dir.glob(batch_args.glob) if path.is_file())
    if batch_args.start_index:
        input_paths = input_paths[batch_args.start_index :]
    if batch_args.limit is not None:
        input_paths = input_paths[: batch_args.limit]

    if not input_paths:
        raise FileNotFoundError(
            f"No files matched {batch_args.glob!r} under {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    env = linux.build_schrodinger_runtime_env(
        tools.root,
        schrodinger_home=batch_args.schrodinger_home,
        jobserver_dir=batch_args.jobserver_dir,
        schrodinger_tmpdir=batch_args.schrodinger_tmpdir,
    )

    print(
        f"Running {len(input_paths)} PrepWizard jobs with concurrency={batch_args.concurrency} "
        f"and threads_per_job={batch_args.threads_per_job}"
    )
    print(f"Output root: {output_dir}")

    failures: list[tuple[Path, Exception]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=batch_args.concurrency) as executor:
        futures = {
            executor.submit(
                _run_one_structure,
                input_path,
                output_dir,
                prepwizard_args,
                batch_args.threads_per_job,
                batch_args.overwrite,
                tools,
                env,
                batch_args.schrodinger_home,
                batch_args.jobserver_dir,
                batch_args.schrodinger_tmpdir,
            ): input_path
            for input_path in input_paths
        }

        with tqdm(total=len(input_paths), desc="PrepWizard structures", unit="pdb") as pbar:
            for future in as_completed(futures):
                input_path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append((input_path, exc))
                    print(f"FAILED {input_path.name}: {exc}")
                else:
                    completed += 1
                    print(
                        f"[{completed}/{len(input_paths)}] Prepared {result.input_path.name} "
                        f"-> {result.maegz_path}, {result.pdb_path}, {result.cif_path}"
                    )
                pbar.update(1)
                pbar.set_postfix(completed=completed, failed=len(failures))

    if failures:
        print(f"{len(failures)} job(s) failed out of {len(input_paths)}.")
        for path, exc in failures:
            print(f"  FAILED {path.name}: {exc}")

        if not batch_args.allow_failures:
            return 1

        fraction = len(failures) / max(1, len(input_paths))
        if fraction > batch_args.max_failure_fraction:
            print(
                f"failure fraction {fraction:.1%} exceeds "
                f"--max-failure-fraction {batch_args.max_failure_fraction:.1%}; "
                f"returning nonzero so the orchestrator halts."
            )
            return 1
        print(
            f"--allow-failures is set and failure fraction {fraction:.1%} is "
            f"within --max-failure-fraction {batch_args.max_failure_fraction:.1%}; "
            f"returning 0 so downstream stages can proceed."
        )
        return 0

    print("All jobs completed successfully.")
    return 0


def _validate_batch_args(batch_args: argparse.Namespace, prepwizard_args: Sequence[str]) -> None:
    if batch_args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    if batch_args.threads_per_job < 1:
        raise ValueError("--threads-per-job must be at least 1.")
    if batch_args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if batch_args.limit is not None and batch_args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided.")

    disallowed_flags = {
        "--host": "The batch runner manages --host based on --threads-per-job.",
        "--no-wait": "The batch runner must wait for each PrepWizard job before converting formats.",
        "--use-mae-intermediate": "The batch runner already uses .maegz as the master output.",
        "--intermediate-output": "The batch runner already uses .maegz as the master output.",
        "--keep-intermediate": "The batch runner already uses .maegz as the master output.",
        "--dry-run": "Use --limit 1 for a smoke test; this runner executes real jobs.",
    }
    for flag, message in disallowed_flags.items():
        if flag in prepwizard_args:
            raise ValueError(f"{flag} is not supported here. {message}")


def _run_one_structure(
    input_path: Path,
    output_dir: Path,
    prepwizard_args: Sequence[str],
    threads_per_job: int,
    overwrite: bool,
    tools: base.SchrodingerTools,
    env: dict[str, str],
    schrodinger_home: str | None,
    jobserver_dir: str | None,
    schrodinger_tmpdir: str | None,
) -> BatchJobResult:
    stem = input_path.stem
    structure_dir = output_dir / stem
    structure_dir.mkdir(parents=True, exist_ok=True)
    maegz_path = structure_dir / f"{stem}.maegz"
    pdb_path = structure_dir / f"{stem}.pdb"
    cif_path = structure_dir / f"{stem}.cif"

    args = _build_prepwizard_args(
        prepwizard_args=prepwizard_args,
        input_path=input_path,
        maegz_path=maegz_path,
        threads_per_job=threads_per_job,
        overwrite=overwrite,
        schrodinger_root=tools.root,
        schrodinger_home=schrodinger_home,
        jobserver_dir=jobserver_dir,
        schrodinger_tmpdir=schrodinger_tmpdir,
    )

    linux.preprocess_structure(args)

    for output_path in (pdb_path, cif_path):
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing converted output without --overwrite: {output_path}"
            )

    base._run_command(
        [str(tools.structconvert), str(maegz_path), str(pdb_path)],
        env=env,
    )
    base._run_command(
        [str(tools.structconvert), "-PDBx", str(maegz_path), str(cif_path)],
        env=env,
    )

    return BatchJobResult(
        input_path=input_path,
        maegz_path=maegz_path,
        pdb_path=pdb_path,
        cif_path=cif_path,
    )


def _build_prepwizard_args(
    *,
    prepwizard_args: Sequence[str],
    input_path: Path,
    maegz_path: Path,
    threads_per_job: int,
    overwrite: bool,
    schrodinger_root: Path,
    schrodinger_home: str | None,
    jobserver_dir: str | None,
    schrodinger_tmpdir: str | None,
) -> argparse.Namespace:
    argv = list(prepwizard_args)
    argv.extend(["--schrodinger-root", str(schrodinger_root)])
    argv.extend(["--host", f"localhost:{threads_per_job}"])
    if schrodinger_home:
        argv.extend(["--schrodinger-home", str(schrodinger_home)])
    if jobserver_dir:
        argv.extend(["--jobserver-dir", str(jobserver_dir)])
    if schrodinger_tmpdir:
        argv.extend(["--schrodinger-tmpdir", str(schrodinger_tmpdir)])
    if overwrite:
        argv.append("--overwrite")
    argv.extend([str(input_path), str(maegz_path)])
    return base.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    return run_batch(argv)


if __name__ == "__main__":
    raise SystemExit(main())
