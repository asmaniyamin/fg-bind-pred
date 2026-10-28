# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterable, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLKIT_DIR = _REPO_ROOT / "small-molecule-mod-toolkit"
for _directory in (_REPO_ROOT, _TOOLKIT_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from chem.build.create_molgraph import mol_to_molgraph, smiles_to_molgraph
from rdkit import Chem


DEFAULT_BASELINE_ROOT = Path(
    "/mnt/data/projects/cs273b/asmani-branch/temp_dirs/mmgbsa_1000_20260520/mmpbsa_work"
)
DEFAULT_MASKED_ROOT = Path(
    "/mnt/data/projects/cs273b/asmani-branch/temp_dirs/mmgbsa_1000_20260520/per_fg_masked_gbsa"
)
DEFAULT_FG_ROOT = Path(
    "/mnt/data/projects/cs273b/asmani-branch/outputs/fg_masks_1000_20260520"
)
DEFAULT_PREP_ROOT = Path(
    "/mnt/data/projects/cs273b/asmani-branch/PDBind/PDB_1000_20260520"
)
WATER_RESNAMES = {"HOH", "WAT", "DOD"}


@dataclass
class DatasetRoots:
    """Resolves on-disk paths for a PDB id under either layout.

    Legacy layout (default): four separate roots, each containing
    ``{root}/{pdb_id}/...`` artifacts.

    Per-PDB layout: ``case_roots`` is a list of "case" directories whose
    immediate children are PDB-id directories, each with self-contained
    ``baseline/``, ``masked/``, ``fg_masks/``, ``pockets/``, ``prepared/``
    subdirs. When set, ``case_roots`` takes precedence over the four
    ``*_root`` fields. PDB id -> case-dir mapping is built lazily on first
    access; if a PDB appears in more than one case root, the first case
    listed wins.
    """

    baseline_root: Path = DEFAULT_BASELINE_ROOT
    masked_root: Path = DEFAULT_MASKED_ROOT
    fg_root: Path = DEFAULT_FG_ROOT
    prep_root: Path = DEFAULT_PREP_ROOT
    case_roots: Optional[list[Path]] = None
    _pdb_index: dict[str, Path] = field(default_factory=dict, init=False, repr=False)

    def _ensure_index(self) -> None:
        if not self.case_roots or self._pdb_index:
            return
        index: dict[str, Path] = {}
        for case_root in self.case_roots:
            case_root = Path(case_root)
            if not case_root.is_dir():
                continue
            for pdb_dir in case_root.iterdir():
                if pdb_dir.is_dir() and pdb_dir.name not in index:
                    index[pdb_dir.name] = pdb_dir
        self._pdb_index = index

    def _case_subdir(self, pdb_id: str, subdir: str) -> Path:
        if self.case_roots is not None:
            self._ensure_index()
            case_dir = self._pdb_index.get(pdb_id)
            if case_dir is not None:
                return case_dir / subdir
            # Fall back to first case root even if pdb missing, so the caller
            # can still get a non-existent Path to check .exists() on.
            return Path(self.case_roots[0]) / pdb_id / subdir
        raise RuntimeError("case_roots not configured")

    def baseline_dir(self, pdb_id: str) -> Path:
        if self.case_roots is not None:
            return self._case_subdir(pdb_id, "baseline")
        return self.baseline_root / pdb_id

    def masked_dir(self, pdb_id: str) -> Path:
        if self.case_roots is not None:
            return self._case_subdir(pdb_id, "masked")
        return self.masked_root / pdb_id

    def fg_dir(self, pdb_id: str) -> Path:
        if self.case_roots is not None:
            return self._case_subdir(pdb_id, "fg_masks")
        return self.fg_root / pdb_id

    def prep_dir(self, pdb_id: str) -> Path:
        if self.case_roots is not None:
            return self._case_subdir(pdb_id, "prepared")
        return self.prep_root / pdb_id

    def pockets_dir(self, pdb_id: str) -> Path:
        if self.case_roots is not None:
            return self._case_subdir(pdb_id, "pockets")
        # Legacy: pockets live as a "{pdb_id}_pockets" subdir under prep_root/{pdb_id}.
        return self.prep_root / pdb_id / f"{pdb_id}_pockets"

    def discover_pdb_ids(self, require_baseline: bool = True) -> list[str]:
        if self.case_roots is not None:
            self._ensure_index()
            result: list[str] = []
            for pdb_id, case_dir in self._pdb_index.items():
                if require_baseline and not (case_dir / "baseline" / "FINAL_RESULTS_MMGBSA.dat").exists():
                    continue
                result.append(pdb_id)
            return sorted(result)
        result = []
        if not self.baseline_root.exists():
            return result
        for workdir in sorted(p for p in self.baseline_root.iterdir() if p.is_dir()):
            if require_baseline and not (workdir / "FINAL_RESULTS_MMGBSA.dat").exists():
                continue
            result.append(workdir.name)
        return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_resname(resname: str | None) -> str:
    token = (resname or "").strip().upper()
    if token in {"HID", "HIE", "HIP"}:
        return "HIS"
    if token == "ASH":
        return "ASP"
    if token == "GLH":
        return "GLU"
    if token == "LYN":
        return "LYS"
    if token in {"CYM", "CYX", "CYN"}:
        return "CYS"
    if token == "MSE":
        return "MET"
    return token


def _resolve_manifest_asset_path(asset_path: str | None, pocket_dir: Path) -> Optional[Path]:
    if not asset_path:
        return None
    candidate = Path(asset_path)
    if candidate.exists():
        return candidate
    local_candidate = pocket_dir / candidate.name
    if local_candidate.exists():
        return local_candidate
    return None


def _iter_unique_pdb_residues(pdb_path: Path) -> list[dict[str, Any]]:
    residues: list[dict[str, Any]] = []
    last_key = None
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (
            line[21].strip() or "A",
            int(line[22:26].strip()),
            line[26].strip(),
            line[17:20].strip().upper(),
            line[:6].strip(),
        )
        if key == last_key:
            continue
        last_key = key
        residues.append(
            {
                "chain_id": key[0],
                "resseq": key[1],
                "insertion_code": key[2],
                "resname": key[3],
                "record_type": key[4],
            }
        )
    return residues


def _collect_manifest_ligand_keys(manifest: Optional[dict[str, Any]]) -> set[tuple[str, int, str, str]]:
    keys: set[tuple[str, int, str, str]] = set()
    if not manifest:
        return keys
    for ligand_entry in manifest.get("ligands", []) or []:
        for residue in ligand_entry.get("residues", []) or []:
            try:
                keys.add(
                    (
                        str(residue.get("chain_id") or "").strip() or "A",
                        int(residue.get("resseq")),
                        str(residue.get("insertion_code") or "").strip(),
                        str(residue.get("resname") or "").strip().upper(),
                    )
                )
            except (TypeError, ValueError):
                continue
    return keys


def _build_prepared_to_receptor_residue_map(
    prep_pdb_path: Path,
    receptor_raw_pdb_path: Path,
    manifest: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not prep_pdb_path.exists() or not receptor_raw_pdb_path.exists():
        return []

    ligand_keys = _collect_manifest_ligand_keys(manifest)
    prepared_residues = [
        residue
        for residue in _iter_unique_pdb_residues(prep_pdb_path)
        if (
            (
                residue["chain_id"],
                residue["resseq"],
                residue["insertion_code"],
                residue["resname"],
            )
            not in ligand_keys
            and residue["resname"] not in WATER_RESNAMES
        )
    ]
    receptor_residues = _iter_unique_pdb_residues(receptor_raw_pdb_path)

    prepared_by_chain: dict[str, list[dict[str, Any]]] = {}
    receptor_by_chain: dict[str, list[dict[str, Any]]] = {}
    for residue in prepared_residues:
        prepared_by_chain.setdefault(residue["chain_id"], []).append(residue)
    for residue in receptor_residues:
        receptor_by_chain.setdefault(residue["chain_id"], []).append(residue)

    mappings: list[dict[str, Any]] = []
    for chain_id, prepared_chain_residues in prepared_by_chain.items():
        receptor_chain_residues = receptor_by_chain.get(chain_id, [])
        prepared_tokens = [_normalize_resname(residue["resname"]) for residue in prepared_chain_residues]
        receptor_tokens = [_normalize_resname(residue["resname"]) for residue in receptor_chain_residues]
        matcher = SequenceMatcher(a=prepared_tokens, b=receptor_tokens, autojunk=False)
        for match in matcher.get_matching_blocks():
            for offset in range(match.size):
                prepared_residue = prepared_chain_residues[match.a + offset]
                receptor_residue = receptor_chain_residues[match.b + offset]
                mappings.append(
                    {
                        "chain_id": chain_id,
                        "prepared_resseq": prepared_residue["resseq"],
                        "prepared_insertion_code": prepared_residue["insertion_code"],
                        "prepared_resname": prepared_residue["resname"],
                        "receptor_resseq": receptor_residue["resseq"],
                        "receptor_resname": receptor_residue["resname"],
                    }
                )
    return mappings


def parse_result_totals(results_path: Path) -> dict[str, float]:
    """Return DELTA TOTAL values keyed by solvent model (typically GB)."""
    parsed = parse_results_file(results_path)
    totals: dict[str, float] = {}
    for model_name, model_payload in parsed.items():
        differences = model_payload.get("differences", {})
        delta_total = differences.get("DELTA TOTAL")
        if isinstance(delta_total, dict) and "average" in delta_total:
            totals[model_name] = float(delta_total["average"])
    return totals


def parse_results_file(results_path: Path) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Parse entity/component tables from FINAL_RESULTS_MMGBSA.dat-like files."""
    models: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    current_model: Optional[str] = None
    current_entity: Optional[str] = None

    entity_labels = {
        "Complex:": "complex",
        "Receptor:": "receptor",
        "Ligand:": "ligand",
        "Differences (Complex - Receptor - Ligand):": "differences",
    }

    totals: dict[str, float] = {}
    for line in results_path.read_text().splitlines():
        s = line.strip()
        if s == "GENERALIZED BORN:":
            current_model = "GB"
            models.setdefault(current_model, {})
            current_entity = None
            continue
        if s == "POISSON BOLTZMANN:":
            current_model = "PB"
            models.setdefault(current_model, {})
            current_entity = None
            continue
        if current_model and s in entity_labels:
            current_entity = entity_labels[s]
            models[current_model].setdefault(current_entity, {})
            continue
        if not current_model or not current_entity:
            continue
        if not s or s.startswith("Energy Component") or set(s) == {"-"}:
            continue

        parts = s.split()
        if len(parts) < 2:
            continue

        if parts[0] == "DELTA" and len(parts) >= 3:
            component = " ".join(parts[:3]) if parts[1] == "G" else " ".join(parts[:2])
            value_index = 3 if parts[1] == "G" else 2
        elif parts[0] == "G" and len(parts) >= 3:
            component = " ".join(parts[:2])
            value_index = 2
        else:
            component = parts[0]
            value_index = 1

        try:
            average = float(parts[value_index])
        except (IndexError, ValueError):
            continue

        std_dev = None
        std_err = None
        if len(parts) > value_index + 1:
            try:
                std_dev = float(parts[value_index + 1])
            except ValueError:
                std_dev = None
        if len(parts) > value_index + 2:
            try:
                std_err = float(parts[value_index + 2])
            except ValueError:
                std_err = None

        models[current_model][current_entity][component] = {
            "average": average,
            "std_dev": std_dev,
            "std_err": std_err,
        }

        if current_entity == "differences" and component == "DELTA TOTAL":
            totals[current_model] = average

    return models


def parse_decomp(decomp_path: Path) -> list[dict[str, Any]]:
    """Parse the first GB DELTAS block from FINAL_DECOMP_MMGBSA.dat."""
    lines = decomp_path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "DELTAS:"), None)
    if start is None:
        return []

    rows: list[dict[str, Any]] = []
    for line in lines[start + 4 :]:
        s = line.strip()
        if not s or s.startswith("DELTAS:") or s.startswith("Backbone"):
            break
        parts = [part.strip() for part in s.split(",")]
        if len(parts) < 18:
            continue
        try:
            rows.append(
                {
                    "residue": parts[0],
                    "location": parts[1],
                    "internal": float(parts[2]),
                    "vdw": float(parts[5]),
                    "elec": float(parts[8]),
                    "polar": float(parts[11]),
                    "nonpolar": float(parts[14]),
                    "total": float(parts[17]),
                }
            )
        except ValueError:
            continue

    rows.sort(key=lambda row: abs(row["total"]), reverse=True)
    return rows


def parse_receptor_amber_renum(renum_path: Path) -> list[dict[str, Any]]:
    """Parse receptor_amber_renum.txt into original->Amber residue mappings.

    File format is whitespace-delimited with columns:
      original_resname original_chain original_resseq amber_resname amber_resseq
    """
    rows: list[dict[str, Any]] = []
    for line in renum_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append(
                {
                    "original_resname": parts[0].strip().upper(),
                    "original_chain": parts[1].strip(),
                    "original_resseq": int(parts[2]),
                    "amber_resname": parts[3].strip().upper(),
                    "amber_resseq": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return rows


def _read_ligand_residue_from_pdb(ligand_raw_pdb: Path) -> tuple[str, str, str]:
    for line in ligand_raw_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            chain = line[21].strip() or "A"
            resname = line[17:20].strip()
            resseq = line[22:26].strip()
            return chain, resname, resseq
    raise ValueError(f"No ligand atoms found in {ligand_raw_pdb}")


def _ligand_id_suffix_matches(ligand_id: str, chain: str, resname: str, resseq: str) -> bool:
    return ligand_id.endswith(f"_{chain}_{resname}_{resseq}")


def select_fg_json_for_structure(
    pdb_id: str,
    roots: DatasetRoots,
) -> Optional[Path]:
    masked_dir = roots.masked_dir(pdb_id)
    for group_dir in sorted(masked_dir.iterdir()) if masked_dir.is_dir() else []:
        mask_group_json = group_dir / "mask_group.json"
        if mask_group_json.exists():
            payload = _load_json(mask_group_json)
            mask_json = payload.get("mask_json")
            if mask_json and Path(mask_json).exists():
                return Path(mask_json)
            # Fallback: the manifest's absolute mask_json path may point at
            # the prior layout. Try the local fg_dir by filename.
            if mask_json:
                local = roots.fg_dir(pdb_id) / Path(mask_json).name
                if local.exists():
                    return local

    fg_dir = roots.fg_dir(pdb_id)
    if not fg_dir.is_dir():
        return None

    ligand_raw_pdb = roots.baseline_dir(pdb_id) / "ligand_raw.pdb"
    if ligand_raw_pdb.exists():
        chain, resname, resseq = _read_ligand_residue_from_pdb(ligand_raw_pdb)
        for fg_json in sorted(fg_dir.glob("*.json")):
            payload = _load_json(fg_json)
            ligand_id = payload.get("ligand_id", "")
            if _ligand_id_suffix_matches(ligand_id, chain, resname, resseq):
                return fg_json

    jsons = sorted(fg_dir.glob("*.json"))
    if len(jsons) == 1:
        return jsons[0]
    return None


def build_ligand_molgraph(fg_payload: dict[str, Any]):
    """Build a MolGraph from the exact FG JSON SMILES and attach recorded coordinates.

    Parses the SMILES literally with ``sanitize=False`` so explicit ``[H]``
    tokens are preserved exactly (RDKit's standard sanitize step shifts H
    counts on amidinium / ``[nH]`` / tertiary-amine fragments and breaks
    atom-count alignment with ``atom_records``). We then sanitize properties
    *without* the H-adjustment step so aromaticity / hybridization features
    populate correctly.

    Atom-count resolution: ``atom_records`` always lists heavy atoms first as
    a contiguous prefix and FG ``atom_indices`` only reference heavy atoms,
    so when the SMILES is in canonical (no-explicit-H) form we slice
    ``atom_records`` / ``functional_group_mask`` to the heavy prefix. The
    sliced versions are stored on ``molgraph.meta`` and the caller's sample
    dict is built from those.
    """
    atom_records = fg_payload.get("atom_records", [])
    expected_n = len(atom_records)
    heavy_n = sum(1 for r in atom_records if (r.get("element") or "").strip() != "H")
    smiles = fg_payload["smiles"]

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Unable to parse SMILES (sanitize=False): {smiles[:80]}")
    raw_n = mol.GetNumAtoms()

    if raw_n == expected_n:
        records_to_use = atom_records
        is_heavy_only = False
    elif raw_n == heavy_n:
        records_to_use = atom_records[:heavy_n]
        is_heavy_only = True
    else:
        # Last-resort fallback: try the original sanitized + AddHs path. If
        # that also misses, raise so the driver logs it per-PDB.
        mg = smiles_to_molgraph(
            smiles, add_hs=True, compute_charges=False, keep_rdkit_mol=False, coords=None
        )
        if mg is None or len(mg.arrays.atomic_num) != expected_n:
            raise ValueError(
                f"MolGraph atom count {raw_n} matches neither expected={expected_n} "
                f"nor heavy={heavy_n} for SMILES: {smiles[:80]}"
            )
        return _attach_records_to_molgraph(mg, fg_payload, atom_records, is_heavy_only=False)

    # Sanitize properties without touching H counts. SANITIZE_ADJUSTHS is
    # the operation that drifts H counts on non-standard protonation; mask
    # it out. Aromaticity / hybridization perception still runs.
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS,
        )
    except Exception:
        # Some malformed SMILES still won't sanitize cleanly; try without
        # aromaticity perception too. If that also fails, fall through with
        # the un-sanitized mol (downstream featurization may be lossy).
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SANITIZE_ALL
                ^ Chem.SANITIZE_ADJUSTHS
                ^ Chem.SANITIZE_SETAROMATICITY,
            )
        except Exception:
            pass

    molgraph = mol_to_molgraph(
        mol, compute_charges=False, keep_rdkit_mol=False, coords=None
    )
    if molgraph is None:
        raise ValueError("mol_to_molgraph returned None")
    return _attach_records_to_molgraph(molgraph, fg_payload, records_to_use, is_heavy_only)


def _attach_records_to_molgraph(molgraph, fg_payload, records_to_use, is_heavy_only):
    expected_n = len(records_to_use)
    n_graph_atoms = len(molgraph.arrays.atomic_num)
    if n_graph_atoms != expected_n:
        raise ValueError(
            f"MolGraph atom count {n_graph_atoms} does not match atom_records {expected_n}"
        )
    coords = np.asarray(
        [[record["x"], record["y"], record["z"]] for record in records_to_use],
        dtype=np.float32,
    )
    molgraph.arrays.pos = coords
    molgraph.arrays.coord_frame = "user"
    molgraph.arrays.coord_valid = np.ones((expected_n,), dtype=bool)
    molgraph.meta["ligand_id"] = fg_payload.get("ligand_id")
    molgraph.meta["smiles"] = fg_payload.get("smiles")
    molgraph.meta["atom_records"] = records_to_use
    molgraph.meta["functional_groups"] = fg_payload.get("functional_groups", [])
    fg_mask = fg_payload.get("functional_group_mask") or []
    if is_heavy_only and len(fg_mask) > expected_n:
        fg_mask = list(fg_mask)[:expected_n]
    molgraph.meta["functional_group_mask"] = fg_mask
    molgraph.meta["heavy_only"] = bool(is_heavy_only)
    return molgraph


def _parse_mmgbsa_outputs(prefix: Path) -> Optional[dict[str, Any]]:
    if not prefix.exists():
        return None
    result_dat = prefix
    stem = result_dat.stem
    if stem.startswith("FINAL_RESULTS_MMGBSA"):
        decomp_name = result_dat.name.replace("FINAL_RESULTS_MMGBSA", "FINAL_DECOMP_MMGBSA")
    else:
        raise ValueError(f"Unexpected results filename: {result_dat}")
    decomp_dat = result_dat.with_name(decomp_name)
    return {
        "results_path": str(result_dat),
        "decomp_path": str(decomp_dat) if decomp_dat.exists() else None,
        "components": parse_results_file(result_dat),
        "totals": parse_result_totals(result_dat),
        "residue_decomposition": parse_decomp(decomp_dat) if decomp_dat.exists() else [],
    }


def _extract_pocket_atoms_from_residues(pocket_residues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for residue_entry in pocket_residues:
        residue = residue_entry.get("residue", {})
        for atom in residue_entry.get("atoms", []):
            atom_copy = dict(atom)
            atom_copy.setdefault("residue", residue)
            atom_copy["min_distance"] = residue_entry.get("min_distance")
            atoms.append(atom_copy)
    return atoms


def _match_pocket_ligand_entry(
    ligand_entries: list[dict[str, Any]],
    preferred_ligand_id: Optional[str],
) -> Optional[dict[str, Any]]:
    if not ligand_entries:
        return None
    if preferred_ligand_id:
        for entry in ligand_entries:
            if entry.get("ligand_id") == preferred_ligand_id:
                return entry
    return ligand_entries[0]


def _load_successful_pocket_entry(
    ligand_entry: dict[str, Any],
    pocket_dir: Path,
) -> dict[str, Any]:
    pocket_json_path = _resolve_manifest_asset_path(ligand_entry.get("pocket_json"), pocket_dir)
    ligand_sdf_path = _resolve_manifest_asset_path(ligand_entry.get("ligand_sdf"), pocket_dir)

    pocket_json = _load_json(pocket_json_path) if pocket_json_path and pocket_json_path.exists() else None
    pocket_residues = pocket_json.get("pocket_residues", []) if pocket_json else []
    pocket_atoms = (
        pocket_json.get("pocket_atoms")
        if pocket_json and pocket_json.get("pocket_atoms") is not None
        else _extract_pocket_atoms_from_residues(pocket_residues) if pocket_json else []
    )

    return {
        "selected_ligand": ligand_entry,
        "ligand_id": ligand_entry.get("ligand_id"),
        "ligand_sdf_path": str(ligand_sdf_path) if ligand_sdf_path else None,
        "pocket_json_path": str(pocket_json_path) if pocket_json_path else None,
        "pocket_json": pocket_json,
        "ligand_atoms": pocket_json.get("ligand_atoms") if pocket_json else None,
        "ligand_bonds": pocket_json.get("ligand_bonds") if pocket_json else None,
        "pocket_radius": pocket_json.get("pocket_radius") if pocket_json else ligand_entry.get("pocket_radius"),
        "pocket_residue_count": pocket_json.get("pocket_residue_count") if pocket_json else ligand_entry.get("pocket_residue_count"),
        "pocket_residues": pocket_residues,
        "pocket_atoms": pocket_atoms,
    }


def _load_pocket_payload(
    pdb_id: str,
    roots: DatasetRoots,
    preferred_ligand_id: Optional[str] = None,
) -> dict[str, Any]:
    pocket_dir = roots.pockets_dir(pdb_id)
    manifest_path = pocket_dir / "manifest.json"
    failed_path = pocket_dir / "failed_ligands.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else None
    failed_ligands = _load_json(failed_path) if failed_path.exists() else None

    selected_success = None
    if manifest:
        ligand_entries = manifest.get("ligands", []) or []
        matched_entry = _match_pocket_ligand_entry(ligand_entries, preferred_ligand_id)
        if matched_entry is not None:
            selected_success = _load_successful_pocket_entry(matched_entry, pocket_dir)

    return {
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "failed_ligands_path": str(failed_path) if failed_path.exists() else None,
        "manifest": manifest,
        "failed_ligands": failed_ligands,
        "selected_pocket": selected_success,
    }


def _load_baseline_payload(pdb_id: str, roots: DatasetRoots) -> dict[str, Any]:
    workdir = roots.baseline_dir(pdb_id)
    results_path = workdir / "FINAL_RESULTS_MMGBSA.dat"
    renum_path = workdir / "receptor_amber_renum.txt"
    payload = {
        "workdir": str(workdir),
        "completed": results_path.exists(),
        "ligand_raw_pdb": str(workdir / "ligand_raw.pdb") if (workdir / "ligand_raw.pdb").exists() else None,
        "ligand_mol2": str(workdir / "ligand.mol2") if (workdir / "ligand.mol2").exists() else None,
        "receptor_amber_renum_path": str(renum_path) if renum_path.exists() else None,
        "receptor_amber_renum": parse_receptor_amber_renum(renum_path) if renum_path.exists() else [],
    }
    if results_path.exists():
        payload.update(_parse_mmgbsa_outputs(results_path) or {})
    return payload


def _load_masked_payload(pdb_id: str, roots: DatasetRoots) -> dict[str, Any]:
    masked_dir = roots.masked_dir(pdb_id)
    groups: dict[str, Any] = {}
    if not masked_dir.is_dir():
        return {"root": str(masked_dir), "groups": groups}

    for group_dir in sorted(path for path in masked_dir.iterdir() if path.is_dir()):
        group_name = group_dir.name
        group_payload: dict[str, Any] = {
            "group_dir": str(group_dir),
            "group_metadata": None,
            "modes": {},
        }
        mask_group_path = group_dir / "mask_group.json"
        if mask_group_path.exists():
            group_payload["group_metadata"] = _load_json(mask_group_path)

        mode_to_filename = {
            "ghost": "FINAL_RESULTS_MMGBSA_GHOST.dat",
            "charge_only": "FINAL_RESULTS_MMGBSA_CHARGE_ONLY.dat",
            "lj_only": "FINAL_RESULTS_MMGBSA_LJ_ONLY.dat",
            "baseline_copy": "FINAL_RESULTS_MMGBSA.dat",
        }
        for mode, filename in mode_to_filename.items():
            result_path = group_dir / filename
            if result_path.exists():
                group_payload["modes"][mode] = _parse_mmgbsa_outputs(result_path)
        groups[group_name] = group_payload

    return {"root": str(masked_dir), "groups": groups}


def build_sample(pdb_id: str, roots: Optional[DatasetRoots] = None) -> dict[str, Any]:
    roots = roots or DatasetRoots()
    fg_json = select_fg_json_for_structure(pdb_id, roots)
    fg_payload = _load_json(fg_json) if fg_json else None
    ligand_graph = build_ligand_molgraph(fg_payload) if fg_payload else None
    preferred_ligand_id = fg_payload.get("ligand_id") if fg_payload else None
    pocket_payload = _load_pocket_payload(pdb_id, roots, preferred_ligand_id=preferred_ligand_id)
    baseline_payload = _load_baseline_payload(pdb_id, roots)
    prep_pdb_path = roots.prep_dir(pdb_id) / f"{pdb_id}.pdb"
    receptor_raw_pdb_path = roots.baseline_dir(pdb_id) / "receptor_raw.pdb"
    baseline_payload["prepared_to_receptor_residue_map"] = _build_prepared_to_receptor_residue_map(
        prep_pdb_path,
        receptor_raw_pdb_path,
        pocket_payload.get("manifest"),
    )

    sample = {
        "pdb_id": pdb_id,
        "paths": {
            "baseline_workdir": str(roots.baseline_dir(pdb_id)),
            "masked_root": str(roots.masked_dir(pdb_id)),
            "fg_json": str(fg_json) if fg_json else None,
            "prep_dir": str(roots.prep_dir(pdb_id)),
        },
        "ligand": {
            "ligand_id": fg_payload.get("ligand_id") if fg_payload else None,
            "smiles": fg_payload.get("smiles") if fg_payload else None,
            # Prefer the graph's stored copies: when build_ligand_molgraph
            # falls back to heavy-only, atom_records and functional_group_mask
            # are sliced to the heavy prefix to stay aligned with ligand_x.
            "atom_records": (
                ligand_graph.meta.get("atom_records") if ligand_graph is not None
                else (fg_payload.get("atom_records") if fg_payload else None)
            ),
            "functional_groups": fg_payload.get("functional_groups") if fg_payload else None,
            "functional_group_mask": (
                ligand_graph.meta.get("functional_group_mask") if ligand_graph is not None
                else (fg_payload.get("functional_group_mask") if fg_payload else None)
            ),
            "heavy_only": (
                bool(ligand_graph.meta.get("heavy_only")) if ligand_graph is not None else False
            ),
            "molgraph": ligand_graph,
        },
        "pocket": pocket_payload,
        "baseline_mmgbsa": baseline_payload,
        "masked_mmgbsa": _load_masked_payload(pdb_id, roots),
    }
    return sample


class MMGBSADataset:
    """Simple indexable dataset over completed or partially completed MMGBSA structures."""

    def __init__(
        self,
        pdb_ids: Optional[Iterable[str]] = None,
        roots: Optional[DatasetRoots] = None,
        require_baseline: bool = True,
    ) -> None:
        self.roots = roots or DatasetRoots()
        if pdb_ids is None:
            pdb_ids = self._discover_pdb_ids(require_baseline=require_baseline)
        self.pdb_ids = sorted(set(str(pdb_id) for pdb_id in pdb_ids))

    def _discover_pdb_ids(self, require_baseline: bool = True) -> list[str]:
        return self.roots.discover_pdb_ids(require_baseline=require_baseline)

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, index: int | str) -> dict[str, Any]:
        pdb_id = self.pdb_ids[index] if isinstance(index, int) else str(index)
        return build_sample(pdb_id, roots=self.roots)


def _summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    molgraph = sample["ligand"]["molgraph"]
    masked_groups = sample["masked_mmgbsa"]["groups"]
    return {
        "pdb_id": sample["pdb_id"],
        "ligand_id": sample["ligand"]["ligand_id"],
        "ligand_atoms": int(len(molgraph.arrays.atomic_num)) if molgraph is not None else None,
        "functional_groups": len(sample["ligand"]["functional_groups"] or []),
        "baseline_completed": sample["baseline_mmgbsa"]["completed"],
        "baseline_totals": sample["baseline_mmgbsa"].get("totals"),
        "masked_groups": len(masked_groups),
        "masked_group_modes": {
            name: sorted(group["modes"].keys()) for name, group in masked_groups.items()
        },
        "pocket_manifest_present": sample["pocket"]["manifest"] is not None,
        "pocket_selected_present": sample["pocket"]["selected_pocket"] is not None,
        "pocket_residue_count": (
            sample["pocket"]["selected_pocket"]["pocket_residue_count"]
            if sample["pocket"]["selected_pocket"] is not None
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect MMGBSA dataloader samples")
    parser.add_argument("--pdb-id", required=True, help="PDB ID to inspect")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    sample = build_sample(args.pdb_id.lower())
    summary = _summarize_sample(sample)
    if args.pretty:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
