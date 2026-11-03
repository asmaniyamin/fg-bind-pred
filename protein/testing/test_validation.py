# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/testing/test_validation.py

Unit tests for the strict validity gate:

    * ``validate_ligand_sanitization`` - non-mutating sanitization probe
      that reports the failing ``SanitizeFlags`` via the manifest rather
      than swallowing them.
    * ``audit_atom_completeness`` - checks that every ``(residue_key,
      atom_name)`` the builder tagged onto the RWMol is still present in
      the post-cleanup mol that goes to SDF.
    * ``audit_coordinate_sanity`` - flags NaN/Inf coords and
      implausible ligand-protein geometry (clashes, orphan ligands).
    * ``--strict-validity`` CLI flag on both extractors.

These are fast and self-contained: no mmCIF, no RCSB fetch, no pipeline
run. Pipeline-integration coverage sits in test_pipeline_e2e.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

# Let this file run as a script from the repo root without requiring the
# ``protein`` package to be installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protein.testing.runner import run_suite, suite_main  # noqa: E402
from protein.extract import find_pocket_unprocessed as unprocessed  # noqa: E402
from protein.extract import find_pocket_processed as processed  # noqa: E402
from protein.extract._core import (  # noqa: E402
    AtomRecord,
    DEFAULT_COORD_CLASH_FLOOR,
    DEFAULT_COORD_ORPHAN_CEILING,
    ResidueKey,
    audit_atom_completeness,
    audit_coordinate_sanity,
    validate_ligand_sanitization,
)


# ---------------------------------------------------------------------------
# Mol builders (small, explicit - no mmCIF round-trip)
# ---------------------------------------------------------------------------

def _tagged_atom(
    element: str,
    atom_name: str,
    *,
    chain_id: str = "A",
    resseq: int = 1,
    resname: str = "LIG",
    insertion_code: str = "",
) -> Chem.Atom:
    """Build an RDKit atom with the same props the real builder tags."""
    atom = Chem.Atom(element)
    atom.SetProp("atom_name", atom_name)
    atom.SetProp("chain_id", chain_id)
    atom.SetIntProp("resseq", resseq)
    atom.SetProp("resname", resname)
    if insertion_code:
        atom.SetProp("insertion_code", insertion_code)
    return atom


def _atom_record(
    index: int,
    atom_name: str,
    element: str,
    *,
    chain_id: str = "A",
    resseq: int = 1,
    resname: str = "LIG",
    insertion_code: str = "",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> AtomRecord:
    residue_key = ResidueKey(
        chain_id=chain_id,
        resseq=resseq,
        insertion_code=insertion_code,
        resname=resname,
    )
    return AtomRecord(
        atom_index=index,
        atom_name=atom_name,
        element=element,
        x=x,
        y=y,
        z=z,
        residue_key=residue_key,
    )


def _build_methane_with_tags() -> tuple[Chem.Mol, list[AtomRecord]]:
    """Build a sanitizable methane (C + 4H) with tagged atoms + matching records."""
    rw = Chem.RWMol()
    atom_records: list[AtomRecord] = []
    records_data = [
        ("C", "C1"),
        ("H", "H1"),
        ("H", "H2"),
        ("H", "H3"),
        ("H", "H4"),
    ]
    for idx, (element, atom_name) in enumerate(records_data):
        rd_idx = rw.AddAtom(_tagged_atom(element, atom_name))
        atom_records.append(_atom_record(rd_idx, atom_name, element))
    for hydrogen_index in range(1, 5):
        rw.AddBond(0, hydrogen_index, Chem.BondType.SINGLE)
    return rw.GetMol(), atom_records


def _build_pentavalent_carbon() -> Chem.Mol:
    """Build a deliberately invalid mol (5 bonds on one carbon) for failure tests."""
    rw = Chem.RWMol()
    center = rw.AddAtom(Chem.Atom("C"))
    for _ in range(5):
        neighbor = rw.AddAtom(Chem.Atom("C"))
        rw.AddBond(center, neighbor, Chem.BondType.SINGLE)
    return rw.GetMol()


# ---------------------------------------------------------------------------
# validate_ligand_sanitization
# ---------------------------------------------------------------------------

def test_validate_clean_mol_reports_success() -> None:
    mol, _ = _build_methane_with_tags()
    info = validate_ligand_sanitization(mol)
    assert info["sanitized"] is True, info
    assert info["failed_operations"] == []
    assert info["error"] is None


def test_validate_broken_mol_reports_failure_without_raising() -> None:
    mol = _build_pentavalent_carbon()
    info = validate_ligand_sanitization(mol)
    assert info["sanitized"] is False, info
    # Exactly which SanitizeFlag RDKit picks depends on the build, but the
    # contract is: report at least one named failing op, don't raise, don't
    # leak the integer bitmask to the manifest.
    assert info["failed_operations"], info
    assert all(isinstance(name, str) for name in info["failed_operations"])


def test_validate_does_not_mutate_input_mol() -> None:
    """The probe runs on a copy; the caller's mol must be byte-identical after."""
    mol = _build_pentavalent_carbon()
    smiles_before = Chem.MolToSmiles(mol, canonical=False)
    num_atoms_before = mol.GetNumAtoms()
    num_bonds_before = mol.GetNumBonds()
    validate_ligand_sanitization(mol)
    assert Chem.MolToSmiles(mol, canonical=False) == smiles_before
    assert mol.GetNumAtoms() == num_atoms_before
    assert mol.GetNumBonds() == num_bonds_before


# ---------------------------------------------------------------------------
# audit_atom_completeness
# ---------------------------------------------------------------------------

def test_audit_complete_when_all_atoms_preserved() -> None:
    mol, atom_records = _build_methane_with_tags()
    info = audit_atom_completeness(mol, atom_records)
    assert info["complete"] is True, info
    assert info["missing_count"] == 0
    assert info["missing_atoms"] == []
    assert info["expected_atom_count"] == 5
    assert info["preserved_tagged_atom_count"] == 5
    assert info["untagged_atom_count"] == 0


def test_audit_flags_dropped_atom() -> None:
    """Remove one atom after construction → audit reports it as missing."""
    mol, atom_records = _build_methane_with_tags()
    rw = Chem.RWMol(mol)
    # Drop H4 (highest atom index - avoid reindexing earlier atoms in records).
    rw.RemoveAtom(4)
    mutated_mol = rw.GetMol()
    info = audit_atom_completeness(mutated_mol, atom_records)
    assert info["complete"] is False, info
    assert info["missing_count"] == 1
    assert len(info["missing_atoms"]) == 1
    assert info["missing_atoms"][0]["atom_name"] == "H4"


def test_audit_count_shortfall_flags_incomplete_even_without_tag_mismatch() -> None:
    """A mol whose atoms are untagged and count-short is flagged incomplete.

    If the cleaner rebuilt the mol via SMILES round-trip and stripped props,
    there's no per-atom ground truth to compare. The count check is the only
    signal available - and it must still flag a shortfall.
    """
    mol, atom_records = _build_methane_with_tags()
    # Build a replacement mol with one fewer atom and NO tags.
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom("C"))
    for _ in range(3):  # only 3 H instead of 4
        h_idx = rw.AddAtom(Chem.Atom("H"))
        rw.AddBond(0, h_idx, Chem.BondType.SINGLE)
    untagged_short = rw.GetMol()

    info = audit_atom_completeness(untagged_short, atom_records)
    assert info["complete"] is False, info
    # No tagged atoms, so preserved_tagged_atom_count is 0 and every source
    # atom appears in missing_atoms.
    assert info["preserved_tagged_atom_count"] == 0
    assert info["untagged_atom_count"] == untagged_short.GetNumAtoms()
    assert info["total_mol_atom_count"] == 4
    assert info["missing_count"] == 5


def test_audit_tolerates_extra_hydrogens_added_by_cleaner() -> None:
    """Cleaner-added atoms (e.g. explicit hydrogens) must not fail the audit."""
    mol, atom_records = _build_methane_with_tags()
    rw = Chem.RWMol(mol)
    # Add an extra untagged H atom bonded to C - simulates cleaner side effect.
    extra_h = rw.AddAtom(Chem.Atom("H"))
    rw.AddBond(0, extra_h, Chem.BondType.SINGLE)
    info = audit_atom_completeness(rw.GetMol(), atom_records)
    assert info["complete"] is True, info
    assert info["untagged_atom_count"] == 1
    assert info["total_mol_atom_count"] == 6


# ---------------------------------------------------------------------------
# JSON serializability
#
# Both validator outputs get dumped into pocket_payload["validation"] and the
# manifest entry via ``json.dumps`` in ``_pocket.run_pocket_extraction``. If
# any value isn't JSON-serializable (e.g. a ResidueKey that forgot to go
# through ``to_dict()``, or a numpy scalar sneaking in), the pipeline would
# crash at manifest-write time on the first affected ligand. These tests
# catch that class of failure without needing a real mmCIF.
# ---------------------------------------------------------------------------

def test_sanitization_info_json_serializable_on_success() -> None:
    mol, _ = _build_methane_with_tags()
    info = validate_ligand_sanitization(mol)
    # Round-trip: dumping must not raise, and re-loading must equal the input.
    restored = json.loads(json.dumps(info))
    assert restored == info


def test_sanitization_info_json_serializable_on_failure() -> None:
    mol = _build_pentavalent_carbon()
    info = validate_ligand_sanitization(mol)
    restored = json.loads(json.dumps(info))
    assert restored == info
    # Guard the specific failure-mode shape that the manifest will expose:
    # failed_operations is a list of strings, never an int bitmask.
    assert restored["sanitized"] is False
    assert isinstance(restored["failed_operations"], list)
    assert all(isinstance(name, str) for name in restored["failed_operations"])


def test_completeness_info_json_serializable_with_missing_atoms() -> None:
    """missing_atoms contains ResidueKey - must go through to_dict()."""
    mol, atom_records = _build_methane_with_tags()
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(4)
    info = audit_atom_completeness(rw.GetMol(), atom_records)
    restored = json.loads(json.dumps(info))
    assert restored == info
    # ResidueKey serialized as dict, not as a tuple or opaque object.
    assert restored["missing_count"] == 1
    assert len(restored["missing_atoms"]) == 1
    residue_field = restored["missing_atoms"][0]["residue"]
    assert isinstance(residue_field, dict)
    assert {"chain_id", "resseq", "insertion_code", "resname"} <= set(residue_field.keys())


def test_combined_validation_payload_json_serializable() -> None:
    """The exact shape _pocket.run_pocket_extraction writes to the manifest."""
    mol, atom_records = _build_methane_with_tags()
    validation_payload = {
        "sanitization": validate_ligand_sanitization(mol),
        "atom_completeness": audit_atom_completeness(mol, atom_records),
    }
    # Must not raise. Asserting equality after round-trip also catches any
    # JSON-lossy types (e.g. tuples collapsed to lists would fail equality).
    restored = json.loads(json.dumps(validation_payload))
    assert restored == validation_payload


# ---------------------------------------------------------------------------
# audit_coordinate_sanity
#
# Each test builds a tiny "protein" (a few atoms) wrapped in a real
# scipy cKDTree - same code path the orchestrator uses - and a tiny
# ligand AtomRecord list with whatever pathology the test wants to
# probe. No mmCIF, no pipeline run.
# ---------------------------------------------------------------------------

def _protein_kdtree(coords: list[tuple[float, float, float]]) -> cKDTree:
    """Build a cKDTree over ``coords`` matching ``_search`` build options."""
    array = np.asarray(coords, dtype=np.float64)
    return cKDTree(array, leafsize=16, balanced_tree=True, compact_nodes=True)


def _ligand_at(positions: list[tuple[float, float, float]]) -> list[AtomRecord]:
    """Build one AtomRecord per (x, y, z), all in the same dummy residue."""
    return [
        _atom_record(idx, f"C{idx}", "C", x=x, y=y, z=z)
        for idx, (x, y, z) in enumerate(positions)
    ]


def test_coord_sanity_clean_passes() -> None:
    """Ligand 3 Å from a protein atom: in-band, no clash, not orphan."""
    protein = _protein_kdtree([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    ligand = _ligand_at([(3.0, 0.0, 0.0), (3.5, 0.5, 0.0)])
    info = audit_coordinate_sanity(
        ligand_atoms=ligand,
        protein_kdtree=protein,
    )
    assert info["coordinate_sanity_ok"] is True, info
    assert info["nan_atom_count"] == 0
    assert info["inf_atom_count"] == 0
    assert info["clash_atom_count"] == 0
    assert info["orphan_ligand"] is False
    assert info["min_ligand_protein_distance"] is not None
    assert info["max_ligand_protein_distance"] is not None
    assert info["issues"] == []


def test_coord_sanity_nan_atom_flagged() -> None:
    """A NaN x-coord on one ligand atom must be reported in the manifest."""
    protein = _protein_kdtree([(0.0, 0.0, 0.0)])
    ligand = _ligand_at([(3.0, 0.0, 0.0), (float("nan"), 1.0, 0.0)])
    info = audit_coordinate_sanity(
        ligand_atoms=ligand,
        protein_kdtree=protein,
    )
    assert info["coordinate_sanity_ok"] is False, info
    assert info["nan_atom_count"] == 1
    assert info["nan_atom_examples"], info
    assert any("NaN" in issue for issue in info["issues"])
    # The remaining finite atom is still measured against the protein.
    assert info["min_ligand_protein_distance"] is not None


def test_coord_sanity_inf_atom_flagged() -> None:
    """An Inf z-coord on one ligand atom must be reported."""
    protein = _protein_kdtree([(0.0, 0.0, 0.0)])
    ligand = _ligand_at([(3.0, 0.0, 0.0), (1.0, 1.0, float("inf"))])
    info = audit_coordinate_sanity(
        ligand_atoms=ligand,
        protein_kdtree=protein,
    )
    assert info["coordinate_sanity_ok"] is False, info
    assert info["inf_atom_count"] == 1
    assert info["inf_atom_examples"], info
    assert any("Inf" in issue for issue in info["issues"])


def test_coord_sanity_clash_flagged() -> None:
    """A ligand atom <0.7 Å from a protein atom is a clash."""
    protein = _protein_kdtree([(0.0, 0.0, 0.0)])
    # 0.3 Å is well inside the 0.7 Å clash floor.
    ligand = _ligand_at([(0.3, 0.0, 0.0), (3.0, 0.0, 0.0)])
    info = audit_coordinate_sanity(
        ligand_atoms=ligand,
        protein_kdtree=protein,
    )
    assert info["coordinate_sanity_ok"] is False, info
    assert info["clash_atom_count"] == 1
    assert info["clash_atom_examples"], info
    # The example carries the offending distance back for triage.
    assert info["clash_atom_examples"][0]["distance_to_protein"] < 0.7
    assert info["min_contact_floor"] == DEFAULT_COORD_CLASH_FLOOR


def test_coord_sanity_orphan_ligand_flagged() -> None:
    """All ligand atoms farther than 8 Å from any protein atom = orphan."""
    protein = _protein_kdtree([(0.0, 0.0, 0.0)])
    # 20 Å away - well past the 8 Å orphan ceiling.
    ligand = _ligand_at([(20.0, 0.0, 0.0), (21.0, 0.0, 0.0)])
    info = audit_coordinate_sanity(
        ligand_atoms=ligand,
        protein_kdtree=protein,
    )
    assert info["coordinate_sanity_ok"] is False, info
    assert info["orphan_ligand"] is True
    assert info["clash_atom_count"] == 0
    assert info["max_contact_ceiling"] == DEFAULT_COORD_ORPHAN_CEILING
    assert any("orphan" in issue.lower() or "exceeds" in issue for issue in info["issues"])


def test_coord_sanity_handles_empty_protein() -> None:
    """No protein kdtree → audit flags it as missing-comparison, doesn't crash."""
    ligand = _ligand_at([(0.0, 0.0, 0.0)])
    info = audit_coordinate_sanity(
        ligand_atoms=ligand,
        protein_kdtree=None,
    )
    assert info["coordinate_sanity_ok"] is False, info
    assert info["min_ligand_protein_distance"] is None
    assert any("no protein" in issue.lower() for issue in info["issues"])


def test_coord_sanity_handles_all_nan_ligand() -> None:
    """When every ligand atom is NaN, the distance check has nothing to chew on."""
    protein = _protein_kdtree([(0.0, 0.0, 0.0)])
    ligand = _ligand_at([(float("nan"), 0.0, 0.0), (0.0, float("nan"), 0.0)])
    info = audit_coordinate_sanity(
        ligand_atoms=ligand,
        protein_kdtree=protein,
    )
    assert info["coordinate_sanity_ok"] is False, info
    assert info["nan_atom_count"] == 2
    assert info["min_ligand_protein_distance"] is None
    assert info["max_ligand_protein_distance"] is None


def test_coord_sanity_tunable_thresholds_respected() -> None:
    """A 1.5 Å contact passes the default 0.7 Å floor but fails a 2.0 Å override."""
    protein = _protein_kdtree([(0.0, 0.0, 0.0)])
    ligand = _ligand_at([(1.5, 0.0, 0.0)])
    info_default = audit_coordinate_sanity(
        ligand_atoms=ligand,
        protein_kdtree=protein,
    )
    assert info_default["coordinate_sanity_ok"] is True
    info_strict = audit_coordinate_sanity(
        ligand_atoms=ligand,
        protein_kdtree=protein,
        min_contact_floor=2.0,
    )
    assert info_strict["coordinate_sanity_ok"] is False
    assert info_strict["clash_atom_count"] == 1


def test_coord_sanity_does_not_mutate_ligand_records() -> None:
    """The audit reads from AtomRecord (frozen dataclass) but never reassigns it."""
    protein = _protein_kdtree([(0.0, 0.0, 0.0)])
    ligand = _ligand_at([(1.0, 0.0, 0.0)])
    snapshot = (ligand[0].x, ligand[0].y, ligand[0].z)
    audit_coordinate_sanity(ligand_atoms=ligand, protein_kdtree=protein)
    assert (ligand[0].x, ligand[0].y, ligand[0].z) == snapshot


def test_coord_sanity_json_serializable() -> None:
    """Audit output is dropped into pocket_payload['validation'] via json.dumps."""
    protein = _protein_kdtree([(0.0, 0.0, 0.0)])
    ligand = _ligand_at([(0.3, 0.0, 0.0), (float("nan"), 0.0, 0.0)])
    info = audit_coordinate_sanity(ligand_atoms=ligand, protein_kdtree=protein)
    restored = json.loads(json.dumps(info))
    assert restored == info
    # The example payloads carry ResidueKey-shaped dicts, not opaque keys.
    assert isinstance(restored["nan_atom_examples"][0]["residue"], dict)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_strict_validity_flag_unprocessed() -> None:
    args_on = unprocessed.parse_args(["/tmp/fake.cif", "--strict-validity"])
    assert args_on.strict_validity is True
    args_default = unprocessed.parse_args(["/tmp/fake.cif"])
    assert args_default.strict_validity is False


def test_strict_validity_flag_processed() -> None:
    args_on = processed.parse_args(["/tmp/fake.cif", "--strict-validity"])
    assert args_on.strict_validity is True
    args_default = processed.parse_args(["/tmp/fake.cif"])
    assert args_default.strict_validity is False


def test_force_resume_flag_unprocessed() -> None:
    args_on = unprocessed.parse_args(["/tmp/fake.cif", "--force-resume"])
    assert args_on.force_resume is True
    args_default = unprocessed.parse_args(["/tmp/fake.cif"])
    assert args_default.force_resume is False


def test_force_resume_flag_processed() -> None:
    args_on = processed.parse_args(["/tmp/fake.cif", "--force-resume"])
    assert args_on.force_resume is True
    args_default = processed.parse_args(["/tmp/fake.cif"])
    assert args_default.force_resume is False


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

ALL_TESTS = (
    test_validate_clean_mol_reports_success,
    test_validate_broken_mol_reports_failure_without_raising,
    test_validate_does_not_mutate_input_mol,
    test_audit_complete_when_all_atoms_preserved,
    test_audit_flags_dropped_atom,
    test_audit_count_shortfall_flags_incomplete_even_without_tag_mismatch,
    test_audit_tolerates_extra_hydrogens_added_by_cleaner,
    test_sanitization_info_json_serializable_on_success,
    test_sanitization_info_json_serializable_on_failure,
    test_completeness_info_json_serializable_with_missing_atoms,
    test_combined_validation_payload_json_serializable,
    test_coord_sanity_clean_passes,
    test_coord_sanity_nan_atom_flagged,
    test_coord_sanity_inf_atom_flagged,
    test_coord_sanity_clash_flagged,
    test_coord_sanity_orphan_ligand_flagged,
    test_coord_sanity_handles_empty_protein,
    test_coord_sanity_handles_all_nan_ligand,
    test_coord_sanity_tunable_thresholds_respected,
    test_coord_sanity_does_not_mutate_ligand_records,
    test_coord_sanity_json_serializable,
    test_strict_validity_flag_unprocessed,
    test_strict_validity_flag_processed,
    test_force_resume_flag_unprocessed,
    test_force_resume_flag_processed,
)


def run_all_tests() -> bool:
    """Entry point for the master :mod:`run_all_tests` runner."""
    return run_suite(ALL_TESTS)


if __name__ == "__main__":
    sys.exit(suite_main(ALL_TESTS))
