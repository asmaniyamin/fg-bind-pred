#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_SELECTED_CSV = Path(
    "/mnt/data/projects/cs273b/asmani-branch/PDBind/lp_pdbbind_1000_20260520_bound_pdbs/selected_1000_20260520_metadata.csv"
)
DEFAULT_PREPARED_ROOT = Path(
    "/mnt/data/projects/cs273b/asmani-branch/PDBind/PDB_1000_20260520"
)
DEFAULT_BASELINE_ROOT = Path(
    "/mnt/data/projects/cs273b/asmani-branch/temp_dirs/mmgbsa_1000_20260520/mmpbsa_work"
)
DEFAULT_MASKED_ROOT = Path(
    "/mnt/data/projects/cs273b/asmani-branch/temp_dirs/mmgbsa_1000_20260520/per_fg_masked_gbsa"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/ayamin/projects/cs273b/asmani-branch/temp_dirs/mmgbsa_1000_status_reports"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write MMGBSA batch status CSV/JSON reports.")
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--prepared-root", default=str(DEFAULT_PREPARED_ROOT))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--masked-root", default=str(DEFAULT_MASKED_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_selected_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return ids
        for row in reader:
            if not row:
                continue
            pdb_id = str(row[0]).strip().lower()
            if pdb_id:
                ids.append(pdb_id)
    return ids


def discover_prepared(ids: list[str], prepared_root: Path) -> tuple[list[str], list[str]]:
    prepared: list[str] = []
    missing: list[str] = []
    for pdb_id in ids:
        cif_path = prepared_root / pdb_id / f"{pdb_id}.cif"
        if cif_path.exists():
            prepared.append(pdb_id)
        else:
            missing.append(pdb_id)
    return prepared, missing


def classify_incomplete_workdir(workdir: Path) -> tuple[str, list[str]]:
    artifacts: list[str] = []
    if (workdir / "complex.prmtop").exists():
        artifacts.extend(
            name
            for name in ["complex.prmtop", "receptor.prmtop", "ligand.prmtop", "complex.inpcrd"]
            if (workdir / name).exists()
        )
        return "topology_built_no_final", artifacts
    if (workdir / "ligand.mol2").exists() or (workdir / "antechamber.log").exists():
        artifacts.extend(
            name
            for name in ["ligand.mol2", "antechamber.log", "sqm.out", "parmchk2.log", "ATOMTYPE.INF"]
            if (workdir / name).exists()
        )
        return "parameterization_incomplete", artifacts
    artifacts.extend(
        name
        for name in ["pdb4amber.log", "pdb4amber.stdio", "receptor_amber.pdb", "leap.log"]
        if (workdir / name).exists()
    )
    return "early_or_other_incomplete", artifacts


def masked_group_modes(group_dir: Path) -> list[str]:
    modes: list[str] = []
    if (group_dir / "FINAL_RESULTS_MMGBSA_GHOST.dat").exists():
        modes.append("ghost")
    if (group_dir / "FINAL_RESULTS_MMGBSA_CHARGE_ONLY.dat").exists():
        modes.append("charge_only")
    if (group_dir / "FINAL_RESULTS_MMGBSA_LJ_ONLY.dat").exists():
        modes.append("lj_only")
    return modes


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    selected_csv = Path(args.selected_csv).expanduser().resolve()
    prepared_root = Path(args.prepared_root).expanduser().resolve()
    baseline_root = Path(args.baseline_root).expanduser().resolve()
    masked_root = Path(args.masked_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = read_selected_ids(selected_csv)
    prepared_ids, prepared_missing_ids = discover_prepared(selected_ids, prepared_root)
    prepared_set = set(prepared_ids)

    baseline_dirs = {d.name.lower(): d for d in baseline_root.iterdir() if d.is_dir()} if baseline_root.exists() else {}

    baseline_completed_rows: list[dict[str, object]] = []
    baseline_incomplete_rows: list[dict[str, object]] = []
    baseline_not_started_rows: list[dict[str, object]] = []
    prepared_missing_rows: list[dict[str, object]] = []
    masked_incomplete_rows: list[dict[str, object]] = []
    masked_incomplete_struct_rows: list[dict[str, object]] = []

    completed_ids: list[str] = []
    incomplete_ids: list[str] = []

    for pdb_id in prepared_missing_ids:
        prepared_missing_rows.append({"pdb_id": pdb_id, "reason": "missing_prepared_cif"})

    for pdb_id in selected_ids:
        if pdb_id not in prepared_set:
            continue
        workdir = baseline_dirs.get(pdb_id)
        if workdir is None:
            baseline_not_started_rows.append(
                {
                    "pdb_id": pdb_id,
                    "reason": "prepared_but_no_baseline_workdir",
                    "prepared_cif": str(prepared_root / pdb_id / f"{pdb_id}.cif"),
                }
            )
            continue

        final_results = workdir / "FINAL_RESULTS_MMGBSA.dat"
        if final_results.exists():
            completed_ids.append(pdb_id)
            baseline_completed_rows.append(
                {
                    "pdb_id": pdb_id,
                    "workdir": str(workdir),
                    "results_path": str(final_results),
                }
            )
        else:
            incomplete_ids.append(pdb_id)
            status_bucket, artifacts = classify_incomplete_workdir(workdir)
            baseline_incomplete_rows.append(
                {
                    "pdb_id": pdb_id,
                    "status_bucket": status_bucket,
                    "workdir": str(workdir),
                    "artifacts": " ".join(artifacts),
                }
            )

    for pdb_id in completed_ids:
        struct_dir = masked_root / pdb_id
        if not struct_dir.exists():
            continue
        bad_groups = 0
        for group_dir in sorted(d for d in struct_dir.iterdir() if d.is_dir()):
            modes = masked_group_modes(group_dir)
            if len(modes) == 3:
                continue
            bad_groups += 1
            masked_incomplete_rows.append(
                {
                    "pdb_id": pdb_id,
                    "group_slug": group_dir.name,
                    "group_dir": str(group_dir),
                    "modes_present": " ".join(modes),
                    "modes_missing": " ".join(mode for mode in ["ghost", "charge_only", "lj_only"] if mode not in modes),
                }
            )
        if bad_groups:
            masked_incomplete_struct_rows.append(
                {
                    "pdb_id": pdb_id,
                    "incomplete_group_count": bad_groups,
                    "masked_root": str(struct_dir),
                }
            )

    summary = {
        "selected_total": len(selected_ids),
        "prepared_total": len(prepared_ids),
        "prepared_missing_total": len(prepared_missing_rows),
        "baseline_workdirs_total": len(baseline_dirs),
        "baseline_completed_total": len(baseline_completed_rows),
        "baseline_incomplete_started_total": len(baseline_incomplete_rows),
        "baseline_not_started_total": len(baseline_not_started_rows),
        "masked_incomplete_group_total": len(masked_incomplete_rows),
        "masked_incomplete_structure_total": len(masked_incomplete_struct_rows),
    }

    write_csv(
        output_dir / "baseline_completed.csv",
        ["pdb_id", "workdir", "results_path"],
        baseline_completed_rows,
    )
    write_csv(
        output_dir / "baseline_incomplete_started.csv",
        ["pdb_id", "status_bucket", "workdir", "artifacts"],
        baseline_incomplete_rows,
    )
    write_csv(
        output_dir / "baseline_not_started.csv",
        ["pdb_id", "reason", "prepared_cif"],
        baseline_not_started_rows,
    )
    write_csv(
        output_dir / "prepared_missing.csv",
        ["pdb_id", "reason"],
        prepared_missing_rows,
    )
    write_csv(
        output_dir / "masked_incomplete.csv",
        ["pdb_id", "group_slug", "group_dir", "modes_present", "modes_missing"],
        masked_incomplete_rows,
    )
    write_csv(
        output_dir / "masked_incomplete_structures.csv",
        ["pdb_id", "incomplete_group_count", "masked_root"],
        masked_incomplete_struct_rows,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
