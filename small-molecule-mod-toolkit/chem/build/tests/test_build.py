# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/build/tests/test_build.py

Tests for:
  - chem.build.build_utils: extract_atom_arrays, extract_bond_arrays,
      extract_attachment_points, build_edge_index, build_bond_pair_to_index,
      prepare_molecule, compute_gasteiger_charges
  - chem.build.create_molgraph: smiles_to_molgraph, mol_to_molgraph

Run:
  python -m chem.build.tests.test_build
(or call run_all_tests()).
"""

from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import numpy as np
from rdkit import Chem

from core.structs import BondType, AtomHybridization, MolGraph

from chem.build.build_utils import (
    extract_atom_arrays,
    extract_bond_arrays,
    extract_attachment_points,
    build_edge_index,
    build_bond_pair_to_index,
    prepare_molecule,
)
from chem.build.create_molgraph import smiles_to_molgraph, mol_to_molgraph


# =============================================================================
# Helpers
# =============================================================================

def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Could not parse SMILES: {smiles}"
    return mol


# =============================================================================
# build_utils: extract_atom_arrays
# =============================================================================

def test_extract_atom_arrays_basic_counts() -> None:
    """extract_atom_arrays returns correct lengths for a simple molecule."""
    mol = _mol("CCO")  # ethanol: 3 heavy atoms
    data = extract_atom_arrays(mol, compute_charges=False)

    assert data["atomic_num"].shape[0] == 3
    assert data["formal_charge"].shape[0] == 3


def test_extract_atom_arrays_atomic_nums() -> None:
    """Check atomic numbers for ethanol C-C-O."""
    mol = _mol("CCO")
    data = extract_atom_arrays(mol, compute_charges=False)
    nums = sorted(data["atomic_num"].tolist())
    assert nums == sorted([6, 6, 8])


def test_extract_atom_arrays_hybridization_present() -> None:
    mol = _mol("C=C")  # ethylene: SP2
    data = extract_atom_arrays(mol, compute_charges=False)
    assert data["hybridization"] is not None
    # Both carbons should be SP2
    sp2 = AtomHybridization.SP2
    assert all(h == sp2 for h in data["hybridization"].tolist())


def test_extract_atom_arrays_aromatic_benzene() -> None:
    mol = _mol("c1ccccc1")  # benzene
    data = extract_atom_arrays(mol, compute_charges=False)
    assert data["is_aromatic"] is not None
    assert all(data["is_aromatic"].tolist())


def test_extract_atom_arrays_with_charges() -> None:
    """Gasteiger charges are floats and finite for a simple mol."""
    mol = _mol("CC")
    data = extract_atom_arrays(mol, compute_charges=True)
    pc = data["partial_charge"]
    assert pc is not None
    assert pc.shape[0] == 2
    assert all(np.isfinite(pc))


def test_extract_atom_arrays_dummy_atom() -> None:
    """Dummy atom (atomic_num 0) from a [*:1] SMILES gets attachment_label."""
    mol = _mol("[*:1]C")  # dummy + carbon
    data = extract_atom_arrays(mol, compute_charges=False)
    assert data["attachment_label"] is not None
    labels = list(data["attachment_label"])
    assert labels[0] == 1  # atom map num 1 becomes label


# =============================================================================
# build_utils: extract_bond_arrays
# =============================================================================

def test_extract_bond_arrays_basic_bond_count() -> None:
    mol = _mol("CC")  # ethane: 1 bond
    data = extract_bond_arrays(mol)
    assert data["bonds"].shape == (1, 2)
    assert data["bond_type"].shape == (1,)


def test_extract_bond_arrays_double_bond() -> None:
    mol = _mol("C=C")
    data = extract_bond_arrays(mol)
    assert int(data["bond_type"][0]) == BondType.DOUBLE


def test_extract_bond_arrays_aromatic_bond() -> None:
    mol = _mol("c1ccccc1")
    data = extract_bond_arrays(mol)
    assert all(int(bt) == BondType.AROMATIC for bt in data["bond_type"].tolist())


def test_extract_bond_arrays_normalized_pairs() -> None:
    """All bond pairs must be stored normalized (u < v)."""
    mol = _mol("CC(C)C")  # isobutane
    data = extract_bond_arrays(mol)
    for u, v in data["bonds"].tolist():
        assert u < v, f"Bond ({u},{v}) not normalized"


def test_extract_bond_arrays_empty_for_single_atom() -> None:
    mol = Chem.MolFromSmiles("[Xe]")
    data = extract_bond_arrays(mol)
    assert data["bonds"].shape == (0, 2)
    assert data["bond_type"].shape == (0,)


# =============================================================================
# build_utils: extract_attachment_points
# =============================================================================

def test_extract_attachment_points_none_for_no_dummies() -> None:
    mol = _mol("CCO")
    result = extract_attachment_points(mol)
    assert result is None


def test_extract_attachment_points_finds_dummy() -> None:
    mol = _mol("[*:1]C")
    result = extract_attachment_points(mol)
    assert result is not None
    assert result.idx.shape[0] == 1
    assert result.label_id is not None
    assert int(result.label_id[0]) == 1


def test_extract_attachment_points_two_dummies() -> None:
    mol = _mol("[*:1]C[*:2]")
    result = extract_attachment_points(mol)
    assert result is not None
    assert result.idx.shape[0] == 2


# =============================================================================
# build_utils: graph-level helpers
# =============================================================================

def test_build_edge_index_symmetric() -> None:
    """edge_index should have shape (2, 2M) for M bonds."""
    from core.structs import MolArrays
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    bond_type = np.zeros(2, dtype=np.int8)
    arrays = MolArrays(
        atomic_num=np.array([6, 6, 6], dtype=np.int16),
        formal_charge=np.zeros(3, dtype=np.int8),
        bonds=bonds,
        bond_type=bond_type,
        total_charge=0,
    )
    ei = build_edge_index(arrays)
    assert ei.shape == (2, 4)  # 2 bonds * 2 directions


def test_build_bond_pair_to_index_lookup() -> None:
    from core.structs import MolArrays
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    bond_type = np.zeros(2, dtype=np.int8)
    arrays = MolArrays(
        atomic_num=np.array([6, 6, 6], dtype=np.int16),
        formal_charge=np.zeros(3, dtype=np.int8),
        bonds=bonds,
        bond_type=bond_type,
        total_charge=0,
    )
    lookup = build_bond_pair_to_index(arrays)
    # both directions should be in the lookup
    assert (0, 1) in lookup or (1, 0) in lookup
    assert lookup.get((0, 1), lookup.get((1, 0))) == 0
    assert lookup.get((1, 2), lookup.get((2, 1))) == 1


# =============================================================================
# smiles_to_molgraph
# =============================================================================

def test_smiles_to_molgraph_returns_molgraph() -> None:
    mg = smiles_to_molgraph("CCO")
    assert isinstance(mg, MolGraph)


def test_smiles_to_molgraph_atom_count_ethanol() -> None:
    mg = smiles_to_molgraph("CCO")
    assert int(mg.arrays.atomic_num.shape[0]) == 3


def test_smiles_to_molgraph_atom_count_benzene() -> None:
    mg = smiles_to_molgraph("c1ccccc1")
    assert int(mg.arrays.atomic_num.shape[0]) == 6


def test_smiles_to_molgraph_bond_arrays_not_empty() -> None:
    mg = smiles_to_molgraph("CC")
    assert mg.arrays.bonds.shape[0] == 1
    assert mg.arrays.bond_type.shape[0] == 1


def test_smiles_to_molgraph_has_smiles_in_meta() -> None:
    mg = smiles_to_molgraph("CCO")
    assert "smiles" in mg.meta


def test_smiles_to_molgraph_has_repair_log_in_meta() -> None:
    mg = smiles_to_molgraph("CCO")
    assert "repair_log" in mg.meta


def test_smiles_to_molgraph_input_smiles_stored() -> None:
    mg = smiles_to_molgraph("CCO")
    assert mg.meta.get("input_smiles") == "CCO"


def test_smiles_to_molgraph_partial_charges_present() -> None:
    mg = smiles_to_molgraph("CCO", compute_charges=True)
    assert mg.arrays.partial_charge is not None
    assert mg.arrays.partial_charge.shape[0] == int(mg.arrays.atomic_num.shape[0])


def test_smiles_to_molgraph_dummy_attachment() -> None:
    mg = smiles_to_molgraph("[*:1]CC")
    assert mg.arrays.atomic_num[0] == 0 or any(int(z) == 0 for z in mg.arrays.atomic_num)


def test_smiles_to_molgraph_invalid_returns_none() -> None:
    mg = smiles_to_molgraph("INVALID_SMILES_XXXXXX")
    assert mg is None


def test_smiles_to_molgraph_no_coords_by_default() -> None:
    mg = smiles_to_molgraph("CCO")
    assert mg.arrays.pos is None


def test_smiles_to_molgraph_canonical_coords() -> None:
    mg = smiles_to_molgraph("CCO", coords="canonical")
    if mg.arrays.pos is not None:
        N = int(mg.arrays.atomic_num.shape[0])
        assert mg.arrays.pos.ndim == 2
        assert mg.arrays.pos.shape == (N, 3)
        assert mg.arrays.coord_valid is not None
        assert mg.arrays.coord_valid.shape == (N,)


def test_smiles_to_molgraph_ensemble_coords() -> None:
    mg = smiles_to_molgraph("CCCCO", coords="ensemble", num_confs=3)
    if mg.arrays.pos is not None:
        N = int(mg.arrays.atomic_num.shape[0])
        assert mg.arrays.pos.ndim == 3
        K = mg.arrays.pos.shape[0]
        assert mg.arrays.pos.shape == (K, N, 3)
        assert mg.arrays.coord_valid.shape == (K, N)


# =============================================================================
# mol_to_molgraph
# =============================================================================

def test_mol_to_molgraph_basic() -> None:
    mol = _mol("CO")  # methanol
    mg = mol_to_molgraph(mol)
    assert isinstance(mg, MolGraph)
    assert int(mg.arrays.atomic_num.shape[0]) == 2


def test_mol_to_molgraph_keeps_rdkit_mol_when_requested() -> None:
    mol = _mol("CC")
    mg = mol_to_molgraph(mol, keep_rdkit_mol=True)
    assert mg.rdkit_mol is not None


def test_mol_to_molgraph_drops_rdkit_mol_by_default() -> None:
    mol = _mol("CC")
    mg = mol_to_molgraph(mol, keep_rdkit_mol=False)
    assert mg.rdkit_mol is None


# =============================================================================
# Runner
# =============================================================================

def run_all_tests() -> bool:
    tests = [
        # extract_atom_arrays
        test_extract_atom_arrays_basic_counts,
        test_extract_atom_arrays_atomic_nums,
        test_extract_atom_arrays_hybridization_present,
        test_extract_atom_arrays_aromatic_benzene,
        test_extract_atom_arrays_with_charges,
        test_extract_atom_arrays_dummy_atom,
        # extract_bond_arrays
        test_extract_bond_arrays_basic_bond_count,
        test_extract_bond_arrays_double_bond,
        test_extract_bond_arrays_aromatic_bond,
        test_extract_bond_arrays_normalized_pairs,
        test_extract_bond_arrays_empty_for_single_atom,
        # extract_attachment_points
        test_extract_attachment_points_none_for_no_dummies,
        test_extract_attachment_points_finds_dummy,
        test_extract_attachment_points_two_dummies,
        # graph helpers
        test_build_edge_index_symmetric,
        test_build_bond_pair_to_index_lookup,
        # smiles_to_molgraph
        test_smiles_to_molgraph_returns_molgraph,
        test_smiles_to_molgraph_atom_count_ethanol,
        test_smiles_to_molgraph_atom_count_benzene,
        test_smiles_to_molgraph_bond_arrays_not_empty,
        test_smiles_to_molgraph_has_smiles_in_meta,
        test_smiles_to_molgraph_has_repair_log_in_meta,
        test_smiles_to_molgraph_input_smiles_stored,
        test_smiles_to_molgraph_partial_charges_present,
        test_smiles_to_molgraph_dummy_attachment,
        test_smiles_to_molgraph_invalid_returns_none,
        test_smiles_to_molgraph_no_coords_by_default,
        test_smiles_to_molgraph_canonical_coords,
        test_smiles_to_molgraph_ensemble_coords,
        # mol_to_molgraph
        test_mol_to_molgraph_basic,
        test_mol_to_molgraph_keeps_rdkit_mol_when_requested,
        test_mol_to_molgraph_drops_rdkit_mol_by_default,
    ]

    print("\n" + "#" * 60)
    print("# RUNNING ALL BUILD TESTS")
    print("#" * 60)

    passed = 0
    for t in tests:
        name = t.__name__
        print(f"\n{'=' * 60}")
        print(f"TEST: {name}")
        print("=" * 60)
        try:
            t()
            print(f"✓ {name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "#" * 60)
    print(f"# RESULTS: {passed} passed, {len(tests) - passed} failed")
    print("#" * 60)
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
