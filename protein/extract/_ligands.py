# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/extract/_ligands.py

Ligand RDKit molecule construction and post-extraction cleanup.

The module has four public entry points:

    * ``build_ligand_molecule`` - unprocessed path: try mmCIF ``_chem_comp_bond``
      templates first, fall back to OpenBabel bond perception on failure.
    * ``build_ligand_molecule_with_openbabel`` - direct OpenBabel-only path,
      reused by the processed extractor when ``--allow-openbabel-fallback``
      fires.
    * ``build_ligand_molecule_from_struct_conn_links`` - processed path:
      construct the ligand directly from explicit ``_struct_conn`` records.
    * ``clean_and_validate_ligand`` - thin fail-open adapter around
      ``chem.clean.MoleculeCleaner`` + ``chem.clean.detect_tools`` validators.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Geometry import Point3D

from ._core import (
    AtomRecord,
    MISSING_CONN_VALUES,
    PocketExtractionError,
    ResidueKey,
    STRUCT_CONN_GROUPING_TYPES,
    StructConnLink,
    TemplateBond,
    add_rdkit_bond,
    is_bond_like_connection,
    map_openbabel_bond_order,
    normalize_conn_value,
    normalize_element_symbol,
    ob,
    pybel,
    residue_key_from_biopython,
    sanitize_molecule_best_effort,
)


# ---------------------------------------------------------------------------
# Shared atom-construction helper
# ---------------------------------------------------------------------------

def _populate_rdkit_mol_from_residues(
    rw_mol: Chem.RWMol,
    ligand_residues: Sequence[object],
) -> tuple[list[AtomRecord], dict[tuple[ResidueKey, str], int], Chem.Conformer]:
    """Add all atoms of ``ligand_residues`` to ``rw_mol`` and build a 3D conformer.

    Returns the atom-record list (in RDKit atom-index order), the
    ``(residue_key, atom_name) -> rd_index`` lookup used by bond builders,
    and a conformer sized exactly to the added atoms. Atom props mirror the
    per-atom tagging both extractors relied on: ``atom_name``, ``resname``,
    ``chain_id``, ``resseq``, and ``insertion_code`` (when present).
    """
    atom_records: list[AtomRecord] = []
    atom_index_lookup: dict[tuple[ResidueKey, str], int] = {}
    conformer = Chem.Conformer(sum(1 for residue in ligand_residues for _ in residue.get_atoms()))
    for residue in ligand_residues:
        residue_key = residue_key_from_biopython(residue.parent.id, residue)
        for atom in residue.get_atoms():
            atom_name = atom.get_name().strip()
            rd_atom = Chem.Atom(normalize_element_symbol(atom.element))
            rd_atom.SetProp("atom_name", atom_name)
            rd_atom.SetProp("resname", residue_key.resname)
            rd_atom.SetProp("chain_id", residue_key.chain_id)
            rd_atom.SetIntProp("resseq", residue_key.resseq)
            if residue_key.insertion_code:
                rd_atom.SetProp("insertion_code", residue_key.insertion_code)
            rd_index = rw_mol.AddAtom(rd_atom)
            atom_index_lookup[(residue_key, atom_name)] = rd_index
            x, y, z = atom.coord
            conformer.SetAtomPosition(rd_index, Point3D(float(x), float(y), float(z)))
            atom_records.append(
                AtomRecord(
                    atom_index=rd_index,
                    atom_name=atom_name,
                    element=atom.element,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    residue_key=residue_key,
                )
            )
    return atom_records, atom_index_lookup, conformer


# ---------------------------------------------------------------------------
# Ligand molecule construction
# ---------------------------------------------------------------------------

def build_ligand_molecule(
    *,
    ligand_id: str,
    ligand_residues: Sequence[object],
    template_bonds: dict[str, list[TemplateBond]],
    struct_conn_links: Sequence[StructConnLink],
) -> tuple[Chem.Mol, list[AtomRecord]]:
    """Build an RDKit ligand molecule with coordinates.

    Tries the mmCIF ``_chem_comp_bond`` template path first, falling back to
    OpenBabel bond perception if templates are missing or fail to sanitize.
    """
    try:
        return build_ligand_molecule_from_templates(
            ligand_id=ligand_id,
            ligand_residues=ligand_residues,
            template_bonds=template_bonds,
            struct_conn_links=struct_conn_links,
        )
    except PocketExtractionError:
        return build_ligand_molecule_with_openbabel(
            ligand_id=ligand_id,
            ligand_residues=ligand_residues,
        )


def build_ligand_molecule_from_templates(
    *,
    ligand_id: str,
    ligand_residues: Sequence[object],
    template_bonds: dict[str, list[TemplateBond]],
    struct_conn_links: Sequence[StructConnLink],
) -> tuple[Chem.Mol, list[AtomRecord]]:
    """Build an RDKit molecule from mmCIF chem-comp templates plus struct_conn."""
    for residue in ligand_residues:
        resname = residue_key_from_biopython(residue.parent.id, residue).resname
        if resname not in template_bonds:
            raise PocketExtractionError(
                f"No _chem_comp_bond template was found for ligand component {resname}."
            )

    rw_mol = Chem.RWMol()
    atom_records, atom_index_lookup, conformer = _populate_rdkit_mol_from_residues(
        rw_mol, ligand_residues
    )
    residue_keys = {record.residue_key for record in atom_records}

    for residue in ligand_residues:
        residue_key = residue_key_from_biopython(residue.parent.id, residue)
        atom_names_in_residue = {atom.get_name() for atom in residue.get_atoms()}
        for bond in template_bonds[residue_key.resname]:
            key_1 = (residue_key, bond.atom_name_1)
            key_2 = (residue_key, bond.atom_name_2)
            if key_1 not in atom_index_lookup or key_2 not in atom_index_lookup:
                if {bond.atom_name_1, bond.atom_name_2}.issubset(atom_names_in_residue):
                    raise PocketExtractionError(
                        f"Failed to map bond template atoms for {residue_key.compact_label()}."
                    )
                continue
            add_rdkit_bond(
                rw_mol,
                atom_index_lookup[key_1],
                atom_index_lookup[key_2],
                order=bond.order,
                aromatic=bond.aromatic,
            )

    for link in struct_conn_links:
        if link.conn_type not in STRUCT_CONN_GROUPING_TYPES:
            continue
        if link.residue_key_1 not in residue_keys or link.residue_key_2 not in residue_keys:
            continue
        key_1 = (link.residue_key_1, link.atom_name_1)
        key_2 = (link.residue_key_2, link.atom_name_2)
        if key_1 not in atom_index_lookup or key_2 not in atom_index_lookup:
            continue
        add_rdkit_bond(
            rw_mol,
            atom_index_lookup[key_1],
            atom_index_lookup[key_2],
            order="sing",
            aromatic=False,
        )

    mol = rw_mol.GetMol()
    conformer.SetId(0)
    mol.AddConformer(conformer, assignId=True)
    mol.SetProp("_Name", ligand_id)
    sanitize_molecule_best_effort(mol)
    return mol, atom_records


def build_ligand_molecule_with_openbabel(
    *,
    ligand_id: str,
    ligand_residues: Sequence[object],
) -> tuple[Chem.Mol, list[AtomRecord]]:
    """Fallback ligand builder using OpenBabel bond perception from coordinates."""
    if pybel is None or ob is None:
        raise PocketExtractionError(
            "OpenBabel is required for fallback bond perception, but it is not installed."
        )

    rw_mol = Chem.RWMol()
    atom_records, _, conformer = _populate_rdkit_mol_from_residues(rw_mol, ligand_residues)

    xyz_lines = [str(len(atom_records)), ligand_id]
    for atom_record in atom_records:
        xyz_lines.append(
            f"{atom_record.element} {atom_record.x:.6f} {atom_record.y:.6f} {atom_record.z:.6f}"
        )
    pybel_mol = pybel.readstring("xyz", "\n".join(xyz_lines))
    ob_mol = pybel_mol.OBMol

    for ob_bond in ob.OBMolBondIter(ob_mol):
        begin_idx = ob_bond.GetBeginAtomIdx() - 1
        end_idx = ob_bond.GetEndAtomIdx() - 1
        aromatic = bool(ob_bond.IsAromatic())
        bond_type = map_openbabel_bond_order(ob_bond.GetBondOrder(), aromatic=aromatic)
        if rw_mol.GetBondBetweenAtoms(begin_idx, end_idx) is not None:
            continue
        rw_mol.AddBond(begin_idx, end_idx, bond_type)
        bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
        if bond is not None and aromatic:
            bond.SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(begin_idx).SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(end_idx).SetIsAromatic(True)

    mol = rw_mol.GetMol()
    mol.AddConformer(conformer, assignId=True)
    mol.SetProp("_Name", ligand_id)
    sanitize_molecule_best_effort(mol)
    return mol, atom_records


# ---------------------------------------------------------------------------
# Ligand molecule construction - processed path (_struct_conn records)
# ---------------------------------------------------------------------------

def is_aromatic_connection(link: StructConnLink) -> bool:
    """Infer aromatic bonds from connection record order/details."""
    normalized = normalize_conn_value(link.value_order)
    return normalized in {"arom", "delo"} or "arom" in link.details.lower()


def has_disconnected_heavy_atom_fragments(mol: Chem.Mol) -> bool:
    """Return True when the molecule has more than one heavy-atom fragment."""
    heavy_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(heavy_atom_indices) <= 1:
        return False
    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    heavy_set = set(heavy_atom_indices)
    heavy_fragments = [
        [idx for idx in fragment if idx in heavy_set]
        for fragment in fragments
    ]
    heavy_fragments = [frag for frag in heavy_fragments if frag]
    return len(heavy_fragments) > 1


def _supplement_hydrogen_bonds_from_openbabel(
    rw_mol: Chem.RWMol,
    atom_records: Sequence[AtomRecord],
) -> None:
    """Fill in missing hydrogen bonds without overwriting heavy-atom connectivity.

    Processed mmCIFs often record explicit heavy-atom bonds but omit bonds
    to hydrogens added by PrepWizard / Epik. This routine runs OpenBabel
    bond perception once and only adds edges that touch an isolated H,
    leaving the heavy-atom graph alone.
    """
    if pybel is None or ob is None:
        return

    isolated_hydrogen_indices = [
        atom.GetIdx()
        for atom in rw_mol.GetAtoms()
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0
    ]
    if not isolated_hydrogen_indices:
        return

    xyz_lines = [str(len(atom_records)), "ligand"]
    for atom_record in atom_records:
        xyz_lines.append(
            f"{atom_record.element} {atom_record.x:.6f} {atom_record.y:.6f} {atom_record.z:.6f}"
        )
    pybel_mol = pybel.readstring("xyz", "\n".join(xyz_lines))
    ob_mol = pybel_mol.OBMol

    for ob_bond in ob.OBMolBondIter(ob_mol):
        begin_idx = ob_bond.GetBeginAtomIdx() - 1
        end_idx = ob_bond.GetEndAtomIdx() - 1
        if rw_mol.GetBondBetweenAtoms(begin_idx, end_idx) is not None:
            continue
        begin_atom = rw_mol.GetAtomWithIdx(begin_idx)
        end_atom = rw_mol.GetAtomWithIdx(end_idx)
        if begin_atom.GetAtomicNum() != 1 and end_atom.GetAtomicNum() != 1:
            continue
        aromatic = bool(ob_bond.IsAromatic())
        bond_type = map_openbabel_bond_order(ob_bond.GetBondOrder(), aromatic=aromatic)
        rw_mol.AddBond(begin_idx, end_idx, bond_type)
        bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
        if bond is not None and aromatic:
            bond.SetIsAromatic(True)
            begin_atom.SetIsAromatic(True)
            end_atom.SetIsAromatic(True)


def build_ligand_molecule_from_struct_conn_links(
    *,
    ligand_id: str,
    ligand_residues: Sequence[object],
    struct_conn_links: Sequence[StructConnLink],
) -> tuple[Chem.Mol, list[AtomRecord]]:
    """Build an RDKit ligand molecule from processed mmCIF ``_struct_conn`` records."""
    rw_mol = Chem.RWMol()
    atom_records, atom_index_lookup, conformer = _populate_rdkit_mol_from_residues(
        rw_mol, ligand_residues
    )
    residue_keys = {record.residue_key for record in atom_records}

    usable_links = [
        link
        for link in struct_conn_links
        if is_bond_like_connection(link)
        and link.residue_key_1 in residue_keys
        and link.residue_key_2 in residue_keys
    ]
    if not usable_links:
        raise PocketExtractionError(
            f"No usable intra-ligand _struct_conn records were found for {ligand_id}."
        )

    added_bonds = 0
    for link in usable_links:
        key_1 = (link.residue_key_1, link.atom_name_1)
        key_2 = (link.residue_key_2, link.atom_name_2)
        if key_1 not in atom_index_lookup or key_2 not in atom_index_lookup:
            continue
        order = normalize_conn_value(link.value_order)
        if order in MISSING_CONN_VALUES:
            order = "sing"
        add_rdkit_bond(
            rw_mol,
            atom_index_lookup[key_1],
            atom_index_lookup[key_2],
            order=order,
            aromatic=is_aromatic_connection(link),
        )
        added_bonds += 1

    if added_bonds == 0:
        raise PocketExtractionError(
            f"_struct_conn records were present for {ligand_id}, but none mapped to ligand atoms."
        )

    _supplement_hydrogen_bonds_from_openbabel(rw_mol, atom_records)

    mol = rw_mol.GetMol()
    conformer.SetId(0)
    mol.AddConformer(conformer, assignId=True)
    mol.SetProp("_Name", ligand_id)
    sanitize_molecule_best_effort(mol)

    if has_disconnected_heavy_atom_fragments(mol):
        raise PocketExtractionError(
            f"_struct_conn bond records appear incomplete for {ligand_id}; "
            "heavy-atom connectivity is fragmented."
        )

    return mol, atom_records


# ---------------------------------------------------------------------------
# Post-extraction ligand cleanup
# ---------------------------------------------------------------------------

def _ensure_repo_on_syspath() -> None:
    """Make ``chem.clean`` importable when this module is run as a script."""
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "chem").is_dir() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def clean_and_validate_ligand(mol: Chem.Mol) -> tuple[Chem.Mol, dict[str, object]]:
    """Run ``chem.clean.MoleculeCleaner`` and validators on an extracted ligand.

    The cleaner fixes valence / kekulization / hypervalence / fragment issues
    that ``sanitize_molecule_best_effort`` swallows silently. Returned tuple is
    ``(mol_to_write, cleanup_info)``.

    Fail-open contract:
        - If ``chem.clean`` can't be imported, if the cleaner raises, if the
          cleaner discards the molecule, or if cleanup would drop the 3D
          conformer, the *original* mol is returned unchanged so the SDF is
          still written. The reason is recorded in ``cleanup_info`` so the
          manifest surfaces every non-default outcome.
    """
    info: dict[str, object] = {
        "cleaned": False,
        "discarded_by_cleaner": False,
        "conformer_preserved": True,
        "log": [],
        "error": None,
        "pre": {"valence_errors": None, "disconnected_fragments": None, "overvalent_atoms": None},
        "post": {"valence_errors": None, "disconnected_fragments": None, "overvalent_atoms": None},
    }

    try:
        _ensure_repo_on_syspath()
        from chem.clean.clean_molecule import MoleculeCleaner
        from chem.clean.detect_tools import (
            detect_valence_errors,
            detect_unconnected_fragments,
            find_overvalent_atoms,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        info["error"] = f"chem.clean import failed: {exc}"
        return mol, info

    def _snapshot(m: Chem.Mol) -> dict[str, object]:
        return {
            "valence_errors": bool(detect_valence_errors(m)),
            "disconnected_fragments": bool(detect_unconnected_fragments(m)),
            "overvalent_atoms": len(find_overvalent_atoms(m)),
        }

    try:
        info["pre"] = _snapshot(mol)
    except Exception as exc:
        info["error"] = f"pre-cleanup validation raised: {exc}"
        return mol, info

    try:
        cleaner = MoleculeCleaner(verbose=False, return_log=True)
        result = cleaner.clean(mol)
    except Exception as exc:
        info["error"] = f"MoleculeCleaner raised: {exc}"
        return mol, info

    cleaned_mol, log = result if isinstance(result, tuple) else (result, [])
    info["log"] = list(log or [])

    if cleaned_mol is None:
        info["discarded_by_cleaner"] = True
        info["post"] = dict(info["pre"])
        return mol, info

    # Kekulization fallback inside the cleaner does SMILES round-trips that
    # strip 3D coordinates. An SDF without a conformer is useless for this
    # pipeline, so keep the original in that case.
    if cleaned_mol.GetNumConformers() < mol.GetNumConformers():
        info["conformer_preserved"] = False
        info["error"] = "cleaner returned molecule without 3D conformer; keeping original"
        info["post"] = dict(info["pre"])
        return mol, info

    try:
        info["post"] = _snapshot(cleaned_mol)
    except Exception as exc:
        info["error"] = f"post-cleanup validation raised: {exc}"
        return mol, info

    info["cleaned"] = True
    return cleaned_mol, info
