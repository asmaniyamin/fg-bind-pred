#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Read-only regression checks for the masked MM/GBSA pipeline.

This script verifies that the current FG-selection logic still agrees with the
masked group directories we have already produced, without regenerating or
rescuing any JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import parmed as pmd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_per_fg_masked_gbsa_batch as batch  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that existing per-FG mask outputs still match the current "
            "selection logic and ligand atom names."
        )
    )
    parser.add_argument(
        "--workroot",
        default=str(batch.DEFAULT_WORKROOT),
        help="Directory containing successful baseline work directories.",
    )
    parser.add_argument(
        "--fg-masks-dir",
        default=str(batch.DEFAULT_FG_MASKS_DIR),
        help="Directory containing FG JSON files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(batch.DEFAULT_OUTPUT_ROOT),
        help="Directory containing per-FG masked outputs.",
    )
    parser.add_argument(
        "--mmpbsa-dir",
        default=str(batch.DEFAULT_MMPBSA_DIR),
        help="Directory containing mmgbsa helper inputs.",
    )
    parser.add_argument(
        "--pdbbind-root",
        default=str(batch.DEFAULT_PDBBIND_ROOT),
        help="Prepared PDBbind dataset root used for CIF lookups.",
    )
    parser.add_argument(
        "--pdb-ids",
        nargs="+",
        default=None,
        help="Optional subset of PDB IDs to verify.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a machine-readable summary JSON.",
    )
    return parser.parse_args(argv)


def load_prmtop_atom_names(prmtop_path: Path) -> set[str]:
    return {atom.name for atom in pmd.load_file(str(prmtop_path)).atoms}


def existing_group_dirs(group_root: Path) -> list[Path]:
    if not group_root.is_dir():
        return []
    return sorted(path for path in group_root.iterdir() if path.is_dir())


def resolve_prmtop_names(workdir: Path, group_dirs: list[Path]) -> set[str]:
    candidates = [workdir / "ligand.prmtop", *(group_dir / "ligand.prmtop" for group_dir in group_dirs)]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return load_prmtop_atom_names(candidate)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"Could not load a usable ligand.prmtop for {workdir.name}")


def verify_one_structure(
    workdir: Path,
    *,
    fg_masks_dir: Path,
    output_root: Path,
    mmpbsa_dir: Path,
    pdbbind_root: Path,
) -> dict[str, Any]:
    pdb_id = workdir.name.lower()
    group_root = output_root / pdb_id
    group_dirs = existing_group_dirs(group_root)
    prmtop_names = resolve_prmtop_names(workdir, group_dirs)
    selected_json_path, selected_data, notes = batch.select_mask_payload_for_workdir(
        pdb_id,
        workdir,
        fg_masks_dir,
        mmpbsa_dir,
        pdbbind_root,
        allow_extract=False,
        allow_rescue=False,
    )
    selected_specs = batch.build_group_specs(selected_data)
    selected_by_slug = {
        spec["group_slug"]: tuple(spec["atom_names"])
        for spec in selected_specs
    }

    mismatches: list[str] = []
    verified_groups = 0
    partial_groups = 0
    unrun_existing_groups = 0

    for group_dir in group_dirs:
        metadata_path = group_dir / "mask_group.json"
        if not metadata_path.is_file():
            mismatches.append(f"{group_dir.name}: missing mask_group.json")
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        group = payload.get("group", {})
        group_slug = str(group.get("group_slug", ""))
        actual_atom_names = tuple(str(name) for name in group.get("atom_names", []))
        expected_atom_names = selected_by_slug.get(group_slug)
        if expected_atom_names is None:
            mismatches.append(
                f"{group_dir.name}: group slug {group_slug!r} is no longer selected"
            )
            continue
        if actual_atom_names != expected_atom_names:
            mismatches.append(
                f"{group_dir.name}: atom names changed from {list(actual_atom_names)} "
                f"to {list(expected_atom_names)}"
            )
            continue
        missing_atom_names = [
            atom_name for atom_name in actual_atom_names if atom_name not in prmtop_names
        ]
        if missing_atom_names:
            mismatches.append(
                f"{group_dir.name}: atom names not present in ligand.prmtop: {missing_atom_names}"
            )
            continue

        mode_count = sum(
            (group_dir / f"FINAL_RESULTS_MMGBSA_{mode}.dat").is_file()
            for mode in ("GHOST", "CHARGE_ONLY", "LJ_ONLY")
        )
        if mode_count == 3:
            verified_groups += 1
        elif mode_count > 0:
            partial_groups += 1
        else:
            unrun_existing_groups += 1

    existing_slugs = {group_dir.name for group_dir in group_dirs}
    unrun_specs = [
        spec["group_slug"] for spec in selected_specs
        if spec["group_slug"] not in existing_slugs
    ]

    return {
        "pdb_id": pdb_id,
        "selected_json": str(selected_json_path.relative_to(REPO_ROOT)),
        "selected_group_count": len(selected_specs),
        "existing_group_count": len(group_dirs),
        "verified_groups": verified_groups,
        "partial_groups": partial_groups,
        "unrun_existing_groups": unrun_existing_groups,
        "unrun_groups": unrun_specs,
        "notes": notes,
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workroot = Path(args.workroot).expanduser().resolve()
    fg_masks_dir = Path(args.fg_masks_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    mmpbsa_dir = Path(args.mmpbsa_dir).expanduser().resolve()
    pdbbind_root = Path(args.pdbbind_root).expanduser().resolve()

    selected_ids = {pdb_id.lower() for pdb_id in (args.pdb_ids or [])}
    workdirs = batch.discover_successful_workdirs(workroot)
    if selected_ids:
        workdirs = [workdir for workdir in workdirs if workdir.name.lower() in selected_ids]
    workdirs = [
        workdir for workdir in workdirs
        if existing_group_dirs(output_root / workdir.name.lower())
    ]

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for workdir in workdirs:
        try:
            results.append(
                verify_one_structure(
                    workdir,
                    fg_masks_dir=fg_masks_dir,
                    output_root=output_root,
                    mmpbsa_dir=mmpbsa_dir,
                    pdbbind_root=pdbbind_root,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"pdb_id": workdir.name.lower(), "error": f"{type(exc).__name__}: {exc}"})

    mismatch_total = sum(len(result["mismatches"]) for result in results)
    summary = {
        "structures_checked": len(results),
        "errors": errors,
        "error_count": len(errors),
        "mismatch_count": mismatch_total,
        "fully_verified_groups": sum(result["verified_groups"] for result in results),
        "partial_groups": sum(result["partial_groups"] for result in results),
        "unrun_existing_groups": sum(result["unrun_existing_groups"] for result in results),
        "results": results,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Checked {summary['structures_checked']} structures; "
        f"{summary['fully_verified_groups']} groups fully verified, "
        f"{summary['partial_groups']} partial groups, "
        f"{summary['unrun_existing_groups']} staged-but-unrun groups, "
        f"{summary['mismatch_count']} mismatches, "
        f"{summary['error_count']} errors."
    )
    for result in results:
        if result["mismatches"]:
            print(f"[{result['pdb_id']}] mismatches:")
            for mismatch in result["mismatches"]:
                print(f"  - {mismatch}")
        elif result["unrun_groups"]:
            print(
                f"[{result['pdb_id']}] consistent; "
                f"{len(result['unrun_groups'])} selected groups not yet run"
            )
    for error in errors:
        print(f"[{error['pdb_id']}] ERROR {error['error']}")

    return 1 if mismatch_total or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
