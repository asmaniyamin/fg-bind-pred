# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
scripts/extract_fg_masks.py

Process a directory of mmCIF files, extract ligands, detect functional groups,
and write one rich metadata JSON per ligand.

Usage:
    python scripts/extract_fg_masks.py \
        --input-dir PDBind/PDB_100_4292026/ \
        --output-dir outputs/fg_masks/

Expected input layout:
    <input-dir>/{pdb_id}/{pdb_id}.cif

Output layout:
    <output-dir>/{pdb_id}/{ligand_id}.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

# Bootstrap repo root onto sys.path so sibling packages resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLKIT_DIR = _REPO_ROOT / "small-molecule-mod-toolkit"
if str(_TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_DIR))

from rdkit import Chem
from rdkit.Chem import rdFMCS

from protein.extract.find_pocket_unprocessed import (
    PocketExtractionError,
    build_component_metadata,
    build_struct_conn_links,
    build_template_bonds,
    detect_ligand_groups,
    build_ligand_molecule,
    index_model_residues,
    load_structure_and_cif,
)
from protein.extract._ligands import build_ligand_molecule_from_struct_conn_links
from chem.functional_groups.detect import detect_from_mol


LP_PDBBIND_CSV = _REPO_ROOT / "PDBind" / "LP-PDBBind" / "dataset" / "LP_PDBBind.csv"
LP_SELECTED_100_CSV = _REPO_ROOT / "PDBind" / "lp_pdbbind_100_bound_pdbs" / "selected_100_metadata.csv"
GAFF2_TYPE_TO_ELEMENT: dict[str, str] = {}
for _element, _prefix in (
    ("C", "c"),
    ("N", "n"),
    ("O", "o"),
    ("S", "s"),
    ("P", "p"),
    ("H", "h"),
    ("F", "f"),
    ("Br", "br"),
    ("Cl", "cl"),
    ("I", "i"),
):
    for _suffix in ["", *list("abcdefghijklmnopqrstuvwxyz0123456789")]:
        GAFF2_TYPE_TO_ELEMENT[_prefix + _suffix] = _element

SMILES_AROMATIC_ELEMENTS = {
    "b": "B",
    "c": "C",
    "n": "N",
    "o": "O",
    "p": "P",
    "s": "S",
}
SMILES_TWO_LETTER_ELEMENTS = {
    "Ag", "Al", "Ar", "As", "At", "Au", "Ba", "Be", "Bi", "Br", "Ca", "Cd",
    "Ce", "Cl", "Co", "Cr", "Cs", "Cu", "Dy", "Er", "Eu", "Fe", "Ga", "Gd",
    "Ge", "He", "Hg", "Ho", "In", "Ir", "Kr", "La", "Li", "Lu", "Mg", "Mn",
    "Mo", "Na", "Nb", "Nd", "Ne", "Ni", "Os", "Pb", "Pd", "Po", "Pr", "Pt",
    "Rb", "Re", "Rh", "Rn", "Ru", "Sb", "Sc", "Se", "Si", "Sm", "Sn", "Sr",
    "Ta", "Tb", "Tc", "Te", "Ti", "Tl", "Tm", "Xe", "Yb", "Zn", "Zr",
}


@lru_cache(maxsize=1)
def load_lp_pdbbind_smiles_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    for csv_path in (LP_SELECTED_100_CSV, LP_PDBBIND_CSV):
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pdb_id = str(row.get("pdb_id") or row.get("") or "").strip().lower()
                smiles = str(row.get("smiles") or "").strip()
                if pdb_id and smiles and pdb_id not in rows:
                    rows[pdb_id] = smiles
    return rows


def load_lp_pdbbind_smiles(pdb_id: str) -> str | None:
    return load_lp_pdbbind_smiles_rows().get(pdb_id.lower())


def _parse_smiles_bracket_element(token: str) -> str | None:
    i = 0
    while i < len(token) and token[i].isdigit():
        i += 1
    if i >= len(token):
        return None
    char = token[i]
    if char.isupper():
        if (
            i + 1 < len(token)
            and token[i + 1].islower()
            and char + token[i + 1] in SMILES_TWO_LETTER_ELEMENTS
        ):
            return char + token[i + 1]
        return char
    return SMILES_AROMATIC_ELEMENTS.get(char)


def smiles_heavy_atom_counts(smiles: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    i = 0
    while i < len(smiles):
        char = smiles[i]
        if char == "[":
            end = smiles.find("]", i + 1)
            if end == -1:
                break
            elem = _parse_smiles_bracket_element(smiles[i + 1 : end])
            if elem and elem != "H":
                elem = elem.upper()
                counts[elem] = counts.get(elem, 0) + 1
            i = end + 1
            continue
        if char.isupper():
            if (
                i + 1 < len(smiles)
                and smiles[i + 1].islower()
                and char + smiles[i + 1] in SMILES_TWO_LETTER_ELEMENTS
            ):
                elem = char + smiles[i + 1]
                i += 2
            else:
                elem = char
                i += 1
            if elem != "H":
                elem = elem.upper()
                counts[elem] = counts.get(elem, 0) + 1
            continue
        if char in SMILES_AROMATIC_ELEMENTS:
            elem = SMILES_AROMATIC_ELEMENTS[char].upper()
            counts[elem] = counts.get(elem, 0) + 1
        i += 1
    return counts


def ligand_group_heavy_atom_counts(group, residue_lookup: dict[Any, object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for residue_key in group.residue_keys:
        residue = residue_lookup[residue_key]
        for atom in residue.get_atoms():
            elem = str(getattr(atom, "element", "")).strip().upper()
            if elem in {"", "H", "D", "T"}:
                continue
            counts[elem] = counts.get(elem, 0) + 1
    return counts


def parse_pdb_atom_records(pdb_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name[:1]).strip()
        records.append(
            {
                "idx": len(records),
                "atom_name": atom_name,
                "element": element.capitalize(),
                "resname": line[17:20].strip(),
                "chain_id": line[21].strip() or "A",
                "resseq": line[22:26].strip(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
            }
        )
    return records


def patch_gaff2_mol2_atom_types(mol2_path: Path) -> str:
    lines = mol2_path.read_text(encoding="utf-8").splitlines()
    patched: list[str] = []
    in_atoms = False
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            patched.append(line)
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            in_atoms = False
        if in_atoms:
            parts = line.split()
            if len(parts) >= 6:
                atom_type = parts[5].lower()
                element = GAFF2_TYPE_TO_ELEMENT.get(atom_type, parts[5][:1].upper())
                suffix = f" {' '.join(parts[6:])}" if len(parts) > 6 else ""
                line = line[: line.rindex(parts[5])] + element + suffix
        patched.append(line)
    return "\n".join(patched)


def load_patched_mol2(mol2_path: Path) -> Chem.Mol:
    patched_text = patch_gaff2_mol2_atom_types(mol2_path)
    with tempfile.NamedTemporaryFile("w", suffix=".mol2", delete=False) as handle:
        handle.write(patched_text)
        temp_path = Path(handle.name)
    try:
        mol = Chem.MolFromMol2File(str(temp_path), removeHs=True)
    finally:
        os.unlink(temp_path)
    if mol is None:
        raise PocketExtractionError(f"RDKit could not load rescued mol2: {mol2_path}")
    return mol


def parse_mol2_heavy_atom_names(mol2_path: Path) -> list[str]:
    names: list[str] = []
    in_atoms = False
    for line in mol2_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        atom_name = parts[1]
        atom_type = parts[5].lower()
        element = GAFF2_TYPE_TO_ELEMENT.get(atom_type, parts[5][:1].upper()).upper()
        if element not in {"H", "D", "T"}:
            names.append(atom_name)
    return names


def build_smiles_to_mol2_mapping(smiles_mol: Chem.Mol, mol2_mol: Chem.Mol) -> dict[int, int]:
    exact_match = mol2_mol.GetSubstructMatch(smiles_mol)
    if len(exact_match) == smiles_mol.GetNumAtoms():
        return {smiles_idx: mol2_idx for smiles_idx, mol2_idx in enumerate(exact_match)}

    params = rdFMCS.MCSParameters()
    params.AtomCompareParameters.MatchFormalCharge = False
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondTyper = rdFMCS.BondCompare.CompareAny
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.MaximizeBonds = False
    mcs_result = rdFMCS.FindMCS([mol2_mol, smiles_mol], params)
    if not mcs_result.smartsString:
        raise PocketExtractionError("Could not build rescue atom mapping: no MCS SMARTS")

    query = Chem.MolFromSmarts(mcs_result.smartsString)
    if query is None:
        raise PocketExtractionError("Could not build rescue atom mapping: invalid MCS SMARTS")

    mol2_match = mol2_mol.GetSubstructMatch(query)
    smiles_match = smiles_mol.GetSubstructMatch(query)
    if len(smiles_match) != smiles_mol.GetNumAtoms():
        raise PocketExtractionError(
            "Could not build rescue atom mapping: MCS did not cover the full LP-PDBBind SMILES"
        )
    if len(mol2_match) != mol2_mol.GetNumAtoms():
        raise PocketExtractionError(
            "Could not build rescue atom mapping: MCS did not cover the full workdir mol2"
        )
    return {
        smiles_idx: mol2_idx
        for smiles_idx, mol2_idx in zip(smiles_match, mol2_match)
    }


def build_rescued_mask_payload(
    pdb_id: str,
    workdir: Path,
    smiles: str,
) -> dict[str, Any]:
    ligand_pdb = workdir / "ligand_raw.pdb"
    ligand_mol2 = workdir / "ligand.mol2"
    if not ligand_pdb.is_file():
        raise FileNotFoundError(f"Missing ligand_raw.pdb for rescue: {ligand_pdb}")
    if not ligand_mol2.is_file():
        raise FileNotFoundError(f"Missing ligand.mol2 for rescue: {ligand_mol2}")

    atom_records = parse_pdb_atom_records(ligand_pdb)
    heavy_atom_positions = [
        idx for idx, rec in enumerate(atom_records)
        if str(rec["element"]).upper() not in {"H", "D", "T"}
    ]
    heavy_pdb_names = [atom_records[idx]["atom_name"] for idx in heavy_atom_positions]
    heavy_mol2_names = parse_mol2_heavy_atom_names(ligand_mol2)
    if heavy_mol2_names != heavy_pdb_names:
        raise PocketExtractionError(
            "Rescue expects ligand.mol2 and ligand_raw.pdb heavy atoms in the same order"
        )

    mol2_mol = load_patched_mol2(ligand_mol2)
    smiles_mol = Chem.MolFromSmiles(smiles)
    if smiles_mol is None:
        raise PocketExtractionError(f"LP-PDBBind SMILES could not be parsed for {pdb_id}")
    smiles_mol = Chem.RemoveHs(smiles_mol)
    if smiles_mol.GetNumAtoms() != len(heavy_atom_positions):
        raise PocketExtractionError(
            f"LP-PDBBind SMILES heavy-atom count ({smiles_mol.GetNumAtoms()}) "
            f"does not match workdir ligand ({len(heavy_atom_positions)})"
        )
    if mol2_mol.GetNumAtoms() != len(heavy_atom_positions):
        raise PocketExtractionError(
            f"workdir mol2 heavy-atom count ({mol2_mol.GetNumAtoms()}) "
            f"does not match ligand_raw.pdb ({len(heavy_atom_positions)})"
        )

    smiles_to_mol2 = build_smiles_to_mol2_mapping(smiles_mol, mol2_mol)
    fg_result = detect_from_mol(smiles_mol)

    fg_matches_data: list[dict[str, Any]] = []
    atoms_to_groups: dict[str, list[str]] = {}
    fg_atom_set: set[int] = set()
    for match in fg_result.matches:
        full_atom_indices = [
            heavy_atom_positions[smiles_to_mol2[smiles_idx]]
            for smiles_idx in match.atom_indices
            if smiles_idx in smiles_to_mol2
        ]
        fg_matches_data.append(
            {
                "name": match.name,
                "atom_indices": full_atom_indices,
                "smarts": match.smarts,
                "parent": match.parent,
                "children": list(match.children),
            }
        )
        for full_idx in full_atom_indices:
            atoms_to_groups.setdefault(str(full_idx), []).append(match.name)
            fg_atom_set.add(full_idx)

    mask = [1 if idx in fg_atom_set else 0 for idx in range(len(atom_records))]
    non_fg_atoms = [idx for idx in range(len(atom_records)) if idx not in fg_atom_set]
    ligand_id = (
        f"ligand_001_{atom_records[0]['chain_id']}_{atom_records[0]['resname']}_{atom_records[0]['resseq']}"
    )
    return {
        "structure_id": pdb_id,
        "ligand_id": ligand_id,
        "n_atoms": len(atom_records),
        "smiles": smiles,
        "smiles_source": "lp_pdbbind",
        "functional_groups": fg_matches_data,
        "atoms_to_groups": atoms_to_groups,
        "functional_group_mask": mask,
        "non_fg_atoms": non_fg_atoms,
        "atom_records": atom_records,
        "atom_mapping_source": "ligand.mol2_exact_or_mcs",
    }


def rescue_mask_json_from_workdir(
    pdb_id: str,
    workdir: Path,
    output_dir: Path,
    smiles: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    resolved_smiles = smiles or load_lp_pdbbind_smiles(pdb_id)
    if not resolved_smiles:
        raise PocketExtractionError(f"No LP-PDBBind SMILES available for {pdb_id}")
    payload = build_rescued_mask_payload(pdb_id, workdir, resolved_smiles)
    structure_out = output_dir / pdb_id.lower()
    structure_out.mkdir(parents=True, exist_ok=True)
    out_path = structure_out / f"{payload['ligand_id']}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path, payload


# ---------------------------------------------------------------------------
# Per-CIF processing
# ---------------------------------------------------------------------------

def process_cif(
    cif_path: Path,
    output_dir: Path,
    min_heavy_atoms: int,
    lp_pdbbind_only: bool = False,
) -> dict[str, int]:
    """Process one CIF file and write JSON files for each valid ligand.

    Returns a dict with keys ``processed`` and ``skipped``.
    """
    structure_id = cif_path.stem
    structure_out = output_dir / structure_id
    structure_out.mkdir(parents=True, exist_ok=True)

    if lp_pdbbind_only:
        for existing_json in structure_out.glob("*.json"):
            existing_json.unlink()

    processed = 0
    skipped = 0

    try:
        structure, cif_dict = load_structure_and_cif(cif_path)
    except Exception as exc:
        print(f"  [ERROR] Could not load {cif_path.name}: {exc}")
        return {"processed": 0, "skipped": 0}

    try:
        model = structure[0]
        comp_meta = build_component_metadata(cif_dict)
        tmpl = build_template_bonds(cif_dict)
        sc = build_struct_conn_links(cif_dict)
        rl = index_model_residues(model)
    except Exception as exc:
        print(f"  [ERROR] Could not build CIF metadata for {structure_id}: {exc}")
        return {"processed": 0, "skipped": 0}

    try:
        groups = detect_ligand_groups(
            residue_lookup=rl,
            component_metadata=comp_meta,
            struct_conn_links=sc,
            include_resnames=set(),
            exclude_resnames=set(),
            min_heavy_atoms=min_heavy_atoms,
        )
    except Exception as exc:
        print(f"  [ERROR] Could not detect ligand groups for {structure_id}: {exc}")
        return {"processed": 0, "skipped": 0}

    if lp_pdbbind_only:
        smiles = load_lp_pdbbind_smiles(structure_id)
        if smiles:
            target_counts = smiles_heavy_atom_counts(smiles)
            matched_groups = [
                group for group in groups
                if ligand_group_heavy_atom_counts(group, rl) == target_counts
            ]
            if matched_groups:
                if len(matched_groups) != len(groups):
                    print(
                        f"  [INFO] {structure_id}: restricting FG extraction to "
                        f"{len(matched_groups)}/{len(groups)} LP-PDBBind-matched ligand group(s)"
                    )
                groups = matched_groups
            else:
                print(
                    f"  [WARN] {structure_id}: no exact LP-PDBBind ligand match found in detected groups; "
                    "skipping FG extraction for this structure"
                )
                groups = []
        else:
            print(
                f"  [WARN] {structure_id}: no LP-PDBBind SMILES available; "
                "skipping FG extraction for this structure"
            )
            groups = []

    if not groups:
        return {"processed": 0, "skipped": 0}

    for group in groups:
        lid = group.ligand_id

        # --- Build RDKit mol ---
        try:
            residues = [rl[k] for k in group.residue_keys]
            try:
                mol, atom_records = build_ligand_molecule_from_struct_conn_links(
                    ligand_id=lid,
                    ligand_residues=residues,
                    struct_conn_links=sc,
                )
            except PocketExtractionError:
                mol, atom_records = build_ligand_molecule(
                    ligand_id=lid,
                    ligand_residues=residues,
                    template_bonds=tmpl,
                    struct_conn_links=sc,
                )
        except PocketExtractionError as exc:
            print(f"  [SKIP] {structure_id}/{lid}: mol build failed: {exc}")
            skipped += 1
            continue
        except Exception as exc:
            print(f"  [SKIP] {structure_id}/{lid}: unexpected mol build error: {exc}")
            skipped += 1
            continue

        n_atoms = mol.GetNumAtoms()

        # --- Sanity: atom-count consistency ---
        if n_atoms != len(atom_records):
            print(
                f"  [WARN] {structure_id}/{lid}: atom count mismatch "
                f"mol={n_atoms} records={len(atom_records)}"
            )

        # --- Sanity: minimum heavy atoms ---
        n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        if n_heavy < min_heavy_atoms:
            print(
                f"  [SKIP] {structure_id}/{lid}: only {n_heavy} heavy atoms "
                f"(min={min_heavy_atoms})"
            )
            skipped += 1
            continue

        # --- SMILES ---
        try:
            smiles: str | None = Chem.MolToSmiles(mol)
        except Exception:
            smiles = None

        # --- Functional group detection ---
        fg_matches_data: list[dict[str, Any]] = []
        atoms_to_groups: dict[str, list[str]] = {}
        mask: list[int] = [0] * n_atoms
        non_fg_atoms: list[int] = list(range(n_atoms))

        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
            fg_result = detect_from_mol(mol)
        except Exception as exc:
            print(f"  [WARN] {structure_id}/{lid}: FG detection failed: {exc}")
            fg_result = None

        if fg_result is None or len(fg_result.matches) == 0:
            print(f"  [WARN] {structure_id}/{lid}: 0 functional groups detected")
        else:
            fg_matches_data = [
                {
                    "name": m.name,
                    "atom_indices": list(m.atom_indices),
                    "smarts": m.smarts,
                    "parent": m.parent,
                    "children": list(m.children),
                }
                for m in fg_result.matches
            ]

            a2g = fg_result.atoms_to_groups()
            atoms_to_groups = {str(k): v for k, v in a2g.items()}

            fg_atom_set: set[int] = set()
            for m in fg_result.matches:
                fg_atom_set.update(m.atom_indices)
            mask = [1 if i in fg_atom_set else 0 for i in range(n_atoms)]
            non_fg_atoms = fg_result.non_fg_atoms(n_atoms).tolist()

        # --- Sanity: mask extremes ---
        mask_sum = sum(mask)
        if mask_sum == n_atoms:
            print(f"  [WARN] {structure_id}/{lid}: mask is all 1s ({n_atoms} atoms)")
        elif mask_sum == 0:
            print(f"  [WARN] {structure_id}/{lid}: mask is all 0s")

        # --- Atom records payload ---
        atom_records_data = [
            {
                "idx": r.atom_index,
                "atom_name": r.atom_name,
                "element": r.element,
                "resname": r.residue_key.resname,
                "chain_id": r.residue_key.chain_id,
                "resseq": r.residue_key.resseq,
                "x": r.x,
                "y": r.y,
                "z": r.z,
            }
            for r in atom_records
        ]

        payload: dict[str, Any] = {
            "structure_id": structure_id,
            "ligand_id": lid,
            "n_atoms": n_atoms,
            "smiles": smiles,
            "functional_groups": fg_matches_data,
            "atoms_to_groups": atoms_to_groups,
            "functional_group_mask": mask,
            "non_fg_atoms": non_fg_atoms,
            "atom_records": atom_records_data,
        }

        out_path = structure_out / f"{lid}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        processed += 1

    return {"processed": processed, "skipped": skipped}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract functional group masks from mmCIF files and write "
            "one JSON per ligand."
        )
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing {pdb_id}/{pdb_id}.cif sub-directories.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output JSON files (created if absent).",
    )
    p.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=6,
        help="Skip ligands with fewer heavy atoms than this threshold (default: 6).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CIF files to process in parallel.",
    )
    p.add_argument(
        "--lp-pdbbind-only",
        action="store_true",
        help=(
            "When LP-PDBBind SMILES are available, only write JSONs for ligand groups "
            "whose heavy-atom composition exactly matches the LP-PDBBind ligand."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1
    if args.workers < 1:
        print("[ERROR] --workers must be at least 1")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(input_dir.glob("*/*.cif"))
    if not cif_files:
        print(f"[ERROR] No CIF files found under {input_dir}")
        return 1

    print(f"Found {len(cif_files)} CIF files under {input_dir}")
    print(f"Output directory: {output_dir}\n")

    total_processed = 0
    total_skipped = 0
    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(cif_files), desc="Extracting FG masks", unit="cif")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                process_cif,
                cif_path,
                output_dir,
                args.min_heavy_atoms,
                args.lp_pdbbind_only,
            ): (i, cif_path)
            for i, cif_path in enumerate(cif_files, 1)
        }
        for future in as_completed(future_map):
            i, cif_path = future_map[future]
            structure_id = cif_path.stem
            print(f"[{i:3d}/{len(cif_files)}] {structure_id}")
            stats = future.result()
            p = stats.get("processed", 0)
            s = stats.get("skipped", 0)
            total_processed += p
            total_skipped += s
            print(f"         processed={p}  skipped={s}")
            if progress is not None:
                progress.update(1)
                progress.set_postfix(processed=total_processed, skipped=total_skipped)

    if progress is not None:
        progress.close()

    print(f"\n{'='*50}")
    print(f"Total ligands processed : {total_processed}")
    print(f"Total ligands skipped   : {total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
