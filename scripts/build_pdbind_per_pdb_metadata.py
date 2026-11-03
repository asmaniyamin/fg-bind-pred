#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Build a consolidated per-PDB metadata table under PDBind/.

Sources:
  - PDBind/LP-PDBBind/dataset/LP_PDBBind.csv
  - PDBind/lp_pdbbind_100_bound_pdbs/selected_100_metadata.csv
  - PDBind/PDB_100_4292026/*/{pdb}.cif|pdb|maegz
  - PDBind/PDB_100_4292026/*/*_pockets/manifest.json
  - PDBind/PDB_100_4292026/_batch_pocket_log.txt

The output is a single CSV with one row per PDB ID.
"""

from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PDBIND_DIR = REPO_ROOT / "PDBind"
LP_PDBBIND_CSV = PDBIND_DIR / "LP-PDBBind" / "dataset" / "LP_PDBBind.csv"
SELECTED_100_CSV = PDBIND_DIR / "lp_pdbbind_100_bound_pdbs" / "selected_100_metadata.csv"
PDB_100_DIR = PDBIND_DIR / "PDB_100_4292026"
OUTPUT_CSV = PDBIND_DIR / "per_pdb_metadata.csv"
LP_ALL_COMPLEXES_CSV = PDBIND_DIR / "LP_PDBBind_all_complexes.csv"

LP_FIELDS = [
    "header",
    "smiles",
    "category",
    "seq",
    "resolution",
    "date",
    "type",
    "new_split",
    "CL1",
    "CL2",
    "CL3",
    "remove_for_balancing_val",
    "kd/ki",
    "value",
    "covalent",
]

OUTPUT_FIELDS = [
    "pdb_id",
    "in_lp_pdbbind",
    "in_selected_100_metadata",
    *LP_FIELDS,
    "selected_100_row_matches_lp",
    "local_structure_dir",
    "local_cif_path",
    "local_pdb_path",
    "local_maegz_path",
    "local_pockets_dir",
    "pocket_manifest_path",
    "pocket_manifest_exists",
    "pocket_ligand_count",
    "pocket_ligand_ids",
    "pocket_resnames",
    "pocket_batch_status",
    "pocket_batch_runtime_s",
]


def load_csv_by_pdb(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            pdb_id = row.get("") or row.get("pdb_id")
            if not pdb_id:
                continue
            rows[pdb_id.lower()] = row
        return rows


def parse_batch_pocket_log(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    pattern = re.compile(
        r"^\[\s*\d+/\d+\]\s+([0-9a-z]+):\s+([^,]+),\s+(\d+)\s+ligand\(s\),\s+([0-9.]+)s$",
        re.IGNORECASE,
    )
    out: dict[str, dict[str, str]] = {}
    for line in path.read_text().splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        pdb_id, status, ligand_count, runtime_s = match.groups()
        out[pdb_id.lower()] = {
            "pocket_batch_status": status,
            "pocket_batch_runtime_s": runtime_s,
            "pocket_batch_log_ligand_count": ligand_count,
        }
    return out


def write_lp_pdbbind_all_complexes(lp_rows: dict[str, dict[str, str]]) -> None:
    """Write a normalized top-level LP-PDBBind CSV with one row per PDB."""
    fieldnames = ["pdb_id", *LP_FIELDS]
    with LP_ALL_COMPLEXES_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for pdb_id in sorted(lp_rows):
            src = lp_rows[pdb_id]
            row = {"pdb_id": pdb_id}
            for field in LP_FIELDS:
                row[field] = src.get(field, "")
            writer.writerow(row)


def collect_local_structure_paths(pdb_dir: Path) -> dict[str, str]:
    pdb_id = pdb_dir.name
    cif_path = pdb_dir / f"{pdb_id}.cif"
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    maegz_path = pdb_dir / f"{pdb_id}.maegz"
    pockets_dir = pdb_dir / f"{pdb_id}_pockets"
    manifest_path = pockets_dir / "manifest.json"

    result = {
        "local_structure_dir": str(pdb_dir.relative_to(REPO_ROOT)),
        "local_cif_path": str(cif_path.relative_to(REPO_ROOT)) if cif_path.exists() else "",
        "local_pdb_path": str(pdb_path.relative_to(REPO_ROOT)) if pdb_path.exists() else "",
        "local_maegz_path": str(maegz_path.relative_to(REPO_ROOT)) if maegz_path.exists() else "",
        "local_pockets_dir": str(pockets_dir.relative_to(REPO_ROOT)) if pockets_dir.exists() else "",
        "pocket_manifest_path": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.exists() else "",
        "pocket_manifest_exists": "True" if manifest_path.exists() else "False",
        "pocket_ligand_count": "",
        "pocket_ligand_ids": "",
        "pocket_resnames": "",
    }

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        ligands = payload.get("ligands", [])
        ligand_ids = []
        resnames = []
        for ligand in ligands:
            ligand_id = ligand.get("ligand_id", "")
            if ligand_id:
                ligand_ids.append(ligand_id)
            for residue in ligand.get("residues", []):
                resname = residue.get("resname", "")
                if resname:
                    resnames.append(resname)
        result["pocket_ligand_count"] = str(payload.get("ligand_count", len(ligands)))
        result["pocket_ligand_ids"] = ";".join(ligand_ids)
        result["pocket_resnames"] = ";".join(OrderedDict.fromkeys(resnames))
    return result


def main() -> int:
    lp_rows = load_csv_by_pdb(LP_PDBBIND_CSV)
    selected_rows = load_csv_by_pdb(SELECTED_100_CSV)
    pocket_log = parse_batch_pocket_log(PDB_100_DIR / "_batch_pocket_log.txt")
    write_lp_pdbbind_all_complexes(lp_rows)

    local_dirs = {
        path.name.lower(): path
        for path in sorted(PDB_100_DIR.iterdir())
        if path.is_dir() and len(path.name) == 4
    }

    pdb_ids = sorted(set(lp_rows) | set(selected_rows) | set(local_dirs))
    rows = []
    for pdb_id in pdb_ids:
        row = {field: "" for field in OUTPUT_FIELDS}
        row["pdb_id"] = pdb_id

        lp_row = lp_rows.get(pdb_id)
        selected_row = selected_rows.get(pdb_id)
        local_dir = local_dirs.get(pdb_id)

        row["in_lp_pdbbind"] = "True" if lp_row else "False"
        row["in_selected_100_metadata"] = "True" if selected_row else "False"

        if lp_row:
            for field in LP_FIELDS:
                row[field] = lp_row.get(field, "")
        elif selected_row:
            for field in LP_FIELDS:
                row[field] = selected_row.get(field, "")

        if selected_row and lp_row:
            lp_slice = {field: lp_row.get(field, "") for field in LP_FIELDS}
            selected_slice = {field: selected_row.get(field, "") for field in LP_FIELDS}
            row["selected_100_row_matches_lp"] = "True" if lp_slice == selected_slice else "False"
        elif selected_row:
            row["selected_100_row_matches_lp"] = "False"

        if local_dir:
            row.update(collect_local_structure_paths(local_dir))
        if pdb_id in pocket_log:
            row["pocket_batch_status"] = pocket_log[pdb_id]["pocket_batch_status"]
            row["pocket_batch_runtime_s"] = pocket_log[pdb_id]["pocket_batch_runtime_s"]
            if not row["pocket_ligand_count"]:
                row["pocket_ligand_count"] = pocket_log[pdb_id]["pocket_batch_log_ligand_count"]

        rows.append(row)

    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(lp_rows)} rows to {LP_ALL_COMPLEXES_CSV}")
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
