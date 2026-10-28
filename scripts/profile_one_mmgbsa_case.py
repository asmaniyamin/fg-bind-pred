#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Profile one LP-PDBBind MMGBSA case stepwise.

Measures wall time and peak RSS for:
1. receptor cleanup / pdb4amber
2. ligand parameterization (antechamber + parmchk2)
3. tleap topology build
4. baseline MMGBSA
5. one masked MMGBSA group

The script chooses a large mask-ready structure by default.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MMPBSA_DIR = REPO_ROOT / "temp_dirs" / "mmgbsa-temp"
RUN_MMPBSA_PATH = MMPBSA_DIR / "run_mmpbsa.py"
MASK_BATCH_PATH = REPO_ROOT / "scripts" / "run_per_fg_masked_gbsa_batch.py"

DEFAULT_SELECTED_CSV = (
    Path("/mnt/data/projects/cs273b/asmani-branch/PDBind")
    / "lp_pdbbind_1000_20260520_bound_pdbs"
    / "selected_1000_20260520_metadata.csv"
)
DEFAULT_PREPARED_ROOT = (
    Path("/mnt/data/projects/cs273b/asmani-branch/PDBind")
    / "PDB_1000_20260520"
)
DEFAULT_FG_ROOT = (
    Path("/mnt/data/projects/cs273b/asmani-branch/outputs")
    / "fg_masks_1000_20260520"
)
DEFAULT_WORKROOT = REPO_ROOT / "temp_dirs" / "mmgbsa_profile_cases"
TIME_BIN = shutil.which("time") or "/usr/bin/time"


def load_module(module_name: str, path: Path):
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_time_output(text: str) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    patterns = {
        "wall_clock_hms": r"Elapsed \(wall clock\) time .*: (.+)",
        "user_seconds": r"User time \(seconds\): ([0-9.]+)",
        "sys_seconds": r"System time \(seconds\): ([0-9.]+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\): ([0-9]+)",
        "exit_status": r"Exit status: ([0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            if key in {"user_seconds", "sys_seconds"}:
                stats[key] = float(value)
            elif key in {"max_rss_kb", "exit_status"}:
                stats[key] = int(value)
            else:
                stats[key] = value
    if "max_rss_kb" in stats:
        stats["max_rss_gb"] = round(stats["max_rss_kb"] / (1024 * 1024), 3)
    return stats


def run_profiled_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    wrapped = [TIME_BIN, "-v", *cmd]
    start = time.time()
    proc = subprocess.run(
        wrapped,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed_seconds = time.time() - start
    if log_path is not None:
        log_path.write_text(
            (proc.stdout or "") + "\n----STDERR----\n" + (proc.stderr or ""),
            encoding="utf-8",
        )
    stats = parse_time_output(proc.stderr or "")
    stats["elapsed_seconds"] = round(elapsed_seconds, 3)
    stats["returncode"] = proc.returncode
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout tail:\n{os.linesep.join((proc.stdout or '').splitlines()[-40:])}\n"
            f"stderr tail:\n{os.linesep.join((proc.stderr or '').splitlines()[-80:])}"
        )
    return stats


def parse_ss_pairs_from_cleaned(cleaned: Path) -> list[list[int]]:
    ss_file = cleaned.with_name(cleaned.stem + "_sslink")
    pairs: list[list[int]] = []
    if not ss_file.exists():
        return pairs
    for line in ss_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pairs.append([int(parts[0]), int(parts[1])])
        except ValueError:
            continue
    return pairs


def choose_large_mask_ready_case(selected_csv: Path, fg_root: Path, prepared_root: Path) -> str:
    rows: list[tuple[str, int]] = []
    with selected_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = (row.get("pdb_id") or row.get("") or "").strip().lower()
            if not pdb_id:
                continue
            cif_path = prepared_root / pdb_id / f"{pdb_id}.cif"
            if not cif_path.is_file():
                continue
            fg_dir = fg_root / pdb_id
            if not fg_dir.is_dir() or not any(fg_dir.glob("*.json")):
                continue
            seq = row.get("seq") or ""
            rows.append((pdb_id, len(seq)))
    if not rows:
        raise RuntimeError("Could not find any prepared, FG-ready structures")
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[0][0]


def prepare_workdir(pdb_id: str, workroot: Path) -> Path:
    workdir = workroot / pdb_id.lower()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    return workdir


def import_helpers():
    run_mmpbsa = load_module("run_mmpbsa", RUN_MMPBSA_PATH)
    mask_batch = load_module("profile_mask_batch", MASK_BATCH_PATH)
    return run_mmpbsa, mask_batch


def run_inner_mmpbsa(workdir: Path, mode: str | None = None) -> None:
    run_mmpbsa, _ = import_helpers()
    if mode is None:
        run_mmpbsa.run_mmpbsa(workdir)
    else:
        run_mmpbsa.run_mmpbsa_masked(workdir, mode=mode)


def inner_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-workdir", required=True)
    parser.add_argument("--inner-mode", default="")
    args = parser.parse_args(argv)
    run_inner_mmpbsa(Path(args.inner_workdir), args.inner_mode or None)
    return 0


def profile_case(
    pdb_id: str,
    *,
    selected_csv: Path,
    prepared_root: Path,
    fg_root: Path,
    workroot: Path,
    threads_per_complex: int,
    ligand_selection_mode: str,
) -> dict[str, Any]:
    cif_path = prepared_root / pdb_id / f"{pdb_id}.cif"
    if not cif_path.is_file():
        raise FileNotFoundError(f"Prepared CIF not found: {cif_path}")

    workdir = prepare_workdir(pdb_id, workroot)
    run_mmpbsa, mask_batch = import_helpers()
    run_mmpbsa.configure_thread_environment(threads_per_complex)

    metrics: dict[str, Any] = {
        "pdb_id": pdb_id,
        "cif_path": str(cif_path),
        "workdir": str(workdir),
        "threads_per_complex": threads_per_complex,
        "ligand_selection_mode": ligand_selection_mode,
        "steps": {},
    }

    rec_pdb = workdir / "receptor_raw.pdb"
    lig_pdb = workdir / "ligand_raw.pdb"
    lig_resname, lig_nc = run_mmpbsa.split_cif(
        cif_path,
        rec_pdb,
        lig_pdb,
        ligand_selection_mode=ligand_selection_mode,
    )
    metrics["selected_ligand"] = {"resname": lig_resname, "net_charge": lig_nc}

    rec_clean = workdir / "receptor_amber.pdb"
    metrics["steps"]["pdb4amber"] = run_profiled_command(
        [
            "pdb4amber",
            "-i",
            rec_pdb.name,
            "-o",
            rec_clean.name,
            "--no-conect",
            "-l",
            "pdb4amber.log",
        ],
        cwd=workdir,
        env=os.environ.copy(),
        log_path=workdir / "profile_pdb4amber.log",
    )
    ss_pairs = parse_ss_pairs_from_cleaned(rec_clean)
    metrics["disulfide_pairs"] = ss_pairs

    metrics["steps"]["parameterize_ligand"] = run_profiled_command(
        [
            sys.executable,
            str(__file__),
            "--inner-helper",
            "parameterize",
            "--inner-workdir",
            str(workdir),
            "--inner-resname",
            lig_resname,
            "--inner-net-charge",
            str(lig_nc),
        ],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        log_path=workdir / "profile_parameterize.log",
    )

    mol2 = workdir / "ligand.mol2"
    frcmod = workdir / "ligand.frcmod"
    metrics["steps"]["tleap_build"] = run_profiled_command(
        [
            sys.executable,
            str(__file__),
            "--inner-helper",
            "tleap",
            "--inner-workdir",
            str(workdir),
            "--inner-rec-clean",
            str(rec_clean),
            "--inner-mol2",
            str(mol2),
            "--inner-frcmod",
            str(frcmod),
            "--inner-ss-pairs-json",
            json.dumps(ss_pairs),
        ],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        log_path=workdir / "profile_tleap.log",
    )

    metrics["steps"]["baseline_mmpbsa"] = run_profiled_command(
        [
            sys.executable,
            str(__file__),
            "--inner-workdir",
            str(workdir),
        ],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        log_path=workdir / "profile_baseline_mmpbsa.log",
    )

    json_path, mask_data, notes = mask_batch.select_mask_payload_for_workdir(
        pdb_id=pdb_id,
        workdir=workdir,
        fg_masks_dir=fg_root,
        mmpbsa_dir=MMPBSA_DIR,
        pdbbind_root=prepared_root,
        allow_extract=False,
        allow_rescue=True,
    )
    group_specs = mask_batch.build_group_specs(mask_data)
    if not group_specs:
        raise RuntimeError(f"No maskable functional groups found for {pdb_id}")
    first_group = group_specs[0]
    metrics["mask_probe"] = {
        "fg_json": str(json_path),
        "notes": notes,
        "group_name": first_group["group_name"],
        "atom_names": first_group["atom_names"],
    }

    run_mmpbsa.apply_ligand_mask(workdir, first_group["atom_names"], mode="ghost")
    metrics["steps"]["masked_mmpbsa_ghost"] = run_profiled_command(
        [
            sys.executable,
            str(__file__),
            "--inner-workdir",
            str(workdir),
            "--inner-mode",
            "ghost",
        ],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        log_path=workdir / "profile_masked_mmpbsa_ghost.log",
    )

    summary_path = workdir / "profile_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["summary_path"] = str(summary_path)
    return metrics


def inner_helper_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-helper", required=True, choices=("parameterize", "tleap"))
    parser.add_argument("--inner-workdir", required=True)
    parser.add_argument("--inner-resname")
    parser.add_argument("--inner-net-charge", type=int)
    parser.add_argument("--inner-rec-clean")
    parser.add_argument("--inner-mol2")
    parser.add_argument("--inner-frcmod")
    parser.add_argument("--inner-ss-pairs-json")
    args = parser.parse_args(argv)

    run_mmpbsa, _ = import_helpers()
    workdir = Path(args.inner_workdir)

    if args.inner_helper == "parameterize":
        lig_pdb = workdir / "ligand_raw.pdb"
        run_mmpbsa.parameterize_ligand(
            lig_pdb,
            workdir,
            args.inner_resname,
            args.inner_net_charge,
        )
        return 0

    ss_pairs = json.loads(args.inner_ss_pairs_json or "[]")
    run_mmpbsa.build_topologies(
        workdir,
        Path(args.inner_rec_clean),
        Path(args.inner_mol2),
        Path(args.inner_frcmod),
        ss_pairs,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--inner-helper" in argv:
        return inner_helper_main(argv)
    if "--inner-workdir" in argv:
        return inner_main(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-id", help="PDB ID to profile. Defaults to the largest mask-ready case.")
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--fg-root", type=Path, default=DEFAULT_FG_ROOT)
    parser.add_argument("--workroot", type=Path, default=DEFAULT_WORKROOT)
    parser.add_argument("--threads-per-complex", type=int, default=2)
    parser.add_argument(
        "--ligand-selection-mode",
        default="lp_pdbbind_strict",
        choices=("heuristic", "lp_pdbbind", "lp_pdbbind_strict"),
    )
    args = parser.parse_args(argv)

    pdb_id = args.pdb_id.lower() if args.pdb_id else choose_large_mask_ready_case(
        args.selected_csv,
        args.fg_root,
        args.prepared_root,
    )
    metrics = profile_case(
        pdb_id,
        selected_csv=args.selected_csv,
        prepared_root=args.prepared_root,
        fg_root=args.fg_root,
        workroot=args.workroot,
        threads_per_complex=args.threads_per_complex,
        ligand_selection_mode=args.ligand_selection_mode,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
