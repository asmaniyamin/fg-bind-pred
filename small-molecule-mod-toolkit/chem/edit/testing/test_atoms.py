# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/edit/testing/test_atoms.py

Unit tests for chem.edit.atoms against the current MolGraph/MolArrays schema.

Run:
  python -m chem.edit.testing.test_atoms
(or call run_all_tests()).

These tests avoid RDKit and construct MolGraph objects directly.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import numpy as np

from core.structs import (
    MolArrays,
    MolGraph,
    AtomHybridization,
    AtomChiralTag,
    AtomCIPCode,
    BondType,
    BondDir,
    BondStereo,
    BondResonanceType,
)

from chem.edit.atoms import (
    delete_atom,
    delete_atoms,
    add_atom,
    replace_atom,
    set_dummy_label,
    permute_atoms,
)


def _make_graph_with_optionals() -> MolGraph:
    """
    Build a small graph:
      0: dummy (Z=0) label=1
      1: carbon
      2: oxygen
      3: carbon

    Bonds (undirected, normalized):
      0-1 (single)
      1-2 (double)
      1-3 (single)

    Includes a broad set of optional atom & bond arrays to ensure slicing/masking is consistent.
    """
    atomic_num = np.array([0, 6, 8, 6], dtype=np.int16)
    formal_charge = np.array([0, 0, 0, 0], dtype=np.int8)

    bonds = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.DOUBLE, BondType.SINGLE], dtype=np.int8)

    # Optional atoms
    isotope = np.array([0, 0, 0, 0], dtype=np.int16)
    is_aromatic = np.array([0, 0, 0, 0], dtype=np.int8)
    hybridization = np.array(
        [AtomHybridization.OTHER, AtomHybridization.SP3, AtomHybridization.SP2, AtomHybridization.SP3],
        dtype=np.int8,
    )
    chiral_tag = np.array([AtomChiralTag.UNSPECIFIED] * 4, dtype=np.int8)
    cip_code = np.array([AtomCIPCode.NONE] * 4, dtype=np.int8)
    atom_map = np.array([1, 0, 0, 0], dtype=np.int32)
    attachment_label = np.array([1, None, None, None], dtype=object)
    explicit_h = np.array([0, 0, 0, 0], dtype=np.int8)
    implicit_h = np.array([0, 3, 0, 3], dtype=np.int8)
    partial_charge = np.array([0.0, 0.0, -0.2, 0.0], dtype=np.float32)
    pos = np.zeros((4, 3), dtype=np.float32)

    # Optional bonds
    is_conjugated = np.array([0, 1, 0], dtype=np.int8)
    is_in_ring = np.array([0, 0, 0], dtype=np.int8)
    bond_dir = np.array([BondDir.NONE, BondDir.NONE, BondDir.NONE], dtype=np.int8)
    bond_stereo = np.array([BondStereo.NONE, BondStereo.NONE, BondStereo.NONE], dtype=np.int8)
    bond_resonance_type = np.array(
        [BondResonanceType.NONE, BondResonanceType.LOCALIZED, BondResonanceType.NONE],
        dtype=np.int8,
    )

    arr = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        isotope=isotope,
        is_aromatic=is_aromatic,
        hybridization=hybridization,
        chiral_tag=chiral_tag,
        cip_code=cip_code,
        atom_map=atom_map,
        attachment_label=attachment_label,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        partial_charge=partial_charge,
        pos=pos,
        is_conjugated=is_conjugated,
        is_in_ring=is_in_ring,
        bond_dir=bond_dir,
        bond_stereo=bond_stereo,
        bond_resonance_type=bond_resonance_type,
        total_charge=0,
        multiplicity=None,
    )
    mg = MolGraph(arrays=arr)
    mg.meta["smiles"] = "[*:1]C(=O)C"  # just a tag for debugging
    return mg


def test_delete_atom_reindexes_and_masks():
    mg = _make_graph_with_optionals()

    # delete atom 0 (dummy). This should remove bond (0-1) and reindex atoms:
    # old indices 1,2,3 -> new 0,1,2
    mg2 = delete_atom(mg, 0)

    arr = mg2.arrays
    assert arr.atomic_num.tolist() == [6, 8, 6]
    assert arr.bonds.shape == (2, 2)
    assert arr.bonds.tolist() == [[0, 1], [0, 2]]  # old (1-2)->(0-1), old (1-3)->(0-2)
    assert arr.bond_type.tolist() == [BondType.DOUBLE, BondType.SINGLE]

    # Optional arrays sliced/masked consistently
    assert arr.hybridization.tolist() == [AtomHybridization.SP3, AtomHybridization.SP2, AtomHybridization.SP3]
    assert arr.atom_map.tolist() == [0, 0, 0]  # old map[1:]=[0,0,0]
    assert arr.attachment_label.tolist() == [None, None, None]  # dummy removed

    assert arr.is_conjugated.tolist() == [1, 0]
    assert arr.bond_resonance_type.tolist() == [BondResonanceType.LOCALIZED, BondResonanceType.NONE]

    # Derived caches should be invalidated
    assert mg2.attachments is None
    assert mg2.cache.edge_index is None
    assert mg2.dirty.structure_dirty is True


def test_delete_atoms_multiple():
    mg = _make_graph_with_optionals()

    # delete atoms 2 and 3 (O and terminal C)
    mg2 = delete_atoms(mg, [2, 3])
    arr = mg2.arrays

    # Remaining atoms: 0(dummy),1(C)
    assert arr.atomic_num.tolist() == [0, 6]
    # Remaining bonds: only 0-1
    assert arr.bonds.tolist() == [[0, 1]]
    assert arr.bond_type.tolist() == [BondType.SINGLE]
    # attachment label for dummy preserved
    assert arr.attachment_label.tolist() == [1, None]


def test_add_atom_appends_required_and_respects_optional_presence():
    mg = _make_graph_with_optionals()

    # Add a non-dummy atom and (incorrectly) try to pass attachment_label; should be forced None.
    mg2, idx = add_atom(mg, atomic_num=7, formal_charge=1, attachment_label="should_drop")
    arr = mg2.arrays

    assert idx == 4
    assert arr.atomic_num.tolist() == [0, 6, 8, 6, 7]
    assert arr.formal_charge.tolist() == [0, 0, 0, 0, 1]
    assert arr.attachment_label.tolist()[-1] is None

    # Optional arrays should have been extended (since they existed)
    assert arr.hybridization.shape[0] == 5
    assert arr.pos.shape == (5, 3)


def test_replace_atom_attachment_invariant():
    mg = _make_graph_with_optionals()

    # Replace dummy (idx 0) with carbon -> must clear attachment_label at 0
    mg2 = replace_atom(mg, 0, atomic_num=6)
    assert mg2.arrays.atomic_num[0] == 6
    assert mg2.arrays.attachment_label[0] is None

    # Replace atom 1 with dummy; unless label explicitly set, attachment_label should be None
    mg3 = replace_atom(mg2, 1, atomic_num=0)
    assert int(mg3.arrays.atomic_num[1]) == 0
    assert mg3.arrays.attachment_label[1] is None

    # Now explicitly set attachment label on that dummy via replace_atom
    mg4 = replace_atom(mg3, 1, attachment_label=99)
    assert mg4.arrays.attachment_label[1] == 99


def test_set_dummy_label_guards():
    mg = _make_graph_with_optionals()

    # Non-dummy should raise
    try:
        set_dummy_label(mg, 1, "X")
        raise AssertionError("Expected ValueError for non-dummy atom")
    except ValueError:
        pass

    # Dummy should set
    mg2 = set_dummy_label(mg, 0, "A")
    assert mg2.arrays.attachment_label[0] == "A"

def _make_graph_with_ensemble_coords() -> MolGraph:
    """
    Same as _make_graph_with_optionals(), but pos is (K, N, 3) and coord_valid is (K, N).
    """
    mg = _make_graph_with_optionals()
    arr = mg.arrays

    K = 2
    N = arr.atomic_num.shape[0]

    # Make ensemble coords deterministic and checkable
    # conf 0: x = atom_index
    # conf 1: x = atom_index + 10
    pos = np.zeros((K, N, 3), dtype=np.float32)
    for a in range(N):
        pos[0, a, 0] = float(a)
        pos[1, a, 0] = float(a + 10)

    coord_valid = np.ones((K, N), dtype=bool)

    # replace arrays (MolArrays is dataclass-like; simplest: construct new MolArrays)
    mg.arrays = MolArrays(
        atomic_num=arr.atomic_num,
        formal_charge=arr.formal_charge,
        bonds=arr.bonds,
        bond_type=arr.bond_type,
        isotope=arr.isotope,
        is_aromatic=arr.is_aromatic,
        hybridization=arr.hybridization,
        chiral_tag=arr.chiral_tag,
        cip_code=arr.cip_code,
        atom_map=arr.atom_map,
        attachment_label=arr.attachment_label,
        explicit_h=arr.explicit_h,
        implicit_h=arr.implicit_h,
        partial_charge=arr.partial_charge,
        pos=pos,
        coord_frame="etkdg",
        coord_valid=coord_valid,
        is_conjugated=arr.is_conjugated,
        is_in_ring=arr.is_in_ring,
        bond_dir=arr.bond_dir,
        bond_stereo=arr.bond_stereo,
        bond_resonance_type=arr.bond_resonance_type,
        total_charge=arr.total_charge,
        multiplicity=arr.multiplicity,
    )
    return mg


def test_delete_atoms_slices_ensemble_pos_correctly():
    mg = _make_graph_with_ensemble_coords()

    # delete atoms 2 and 3 (as in the test fixture)
    mg2 = delete_atoms(mg, [2, 3])
    arr = mg2.arrays

    # Remaining atoms: 0(dummy),1(C) => N=2
    assert arr.atomic_num.tolist() == [0, 6]
    assert arr.pos is not None
    assert arr.pos.shape == (2, 2, 3), "ensemble pos should be (K,N,3) after deletion"
    assert arr.coord_valid is not None
    assert arr.coord_valid.shape == (2, 2)

    # Check that the correct atoms were kept in both conformers:
    # kept old atoms [0,1]
    # conf 0 x coords should be [0,1]
    assert arr.pos[0, :, 0].tolist() == [0.0, 1.0]
    # conf 1 x coords should be [10,11]
    assert arr.pos[1, :, 0].tolist() == [10.0, 11.0]


def test_add_atom_appends_ensemble_pos_correctly():
    mg = _make_graph_with_ensemble_coords()

    # Add one atom (N->N+1) with no pos provided; should append zeros for each conformer
    mg2, idx = add_atom(mg, atomic_num=7, formal_charge=0)
    arr = mg2.arrays
    assert idx == 4
    assert arr.pos is not None
    assert arr.pos.shape == (2, 5, 3), "ensemble pos should be (K,N+1,3) after add_atom"

    # New atom should have x=0 in both conformers because we appended zeros
    assert float(arr.pos[0, 4, 0]) == 0.0
    assert float(arr.pos[1, 4, 0]) == 0.0

    # Existing atoms unchanged
    assert arr.pos[0, 0, 0] == 0.0
    assert arr.pos[0, 1, 0] == 1.0
    assert arr.pos[1, 0, 0] == 10.0
    assert arr.pos[1, 1, 0] == 11.0


def test_permute_atoms_reorders_atoms_and_bonds_and_preserves_alignment():
    mg = _make_graph_with_optionals()
    arr = mg.arrays

    # New order: [3,1,2,0] (new idx 0 is old 3, new idx 3 is old 0)
    new_to_old = np.array([3, 1, 2, 0], dtype=np.int64)

    mg2, old_to_new, bond_perm = permute_atoms(mg, new_to_old)
    arr2 = mg2.arrays

    # Mapping sanity
    assert old_to_new.tolist() == [3, 1, 2, 0]

    # Atom arrays permuted
    assert arr2.atomic_num.tolist() == arr.atomic_num[new_to_old].tolist()
    assert arr2.formal_charge.tolist() == arr.formal_charge[new_to_old].tolist()
    assert arr2.atom_map.tolist() == arr.atom_map[new_to_old].tolist()
    assert arr2.attachment_label.tolist() == arr.attachment_label[new_to_old].tolist()

    # Coords permuted for canonical pos (N,3)
    assert arr2.pos is not None and arr2.pos.shape == (4, 3)
    assert np.allclose(arr2.pos, arr.pos[new_to_old])

    # Bonds should be remapped through old_to_new, then normalized and sorted.
    # Original bonds: (0-1), (1-2), (1-3)
    # old_to_new: 0->3, 1->1, 2->2, 3->0
    # remapped:
    # 0-1 -> 3-1 -> (1,3)
    # 1-2 -> 1-2 -> (1,2)
    # 1-3 -> 1-0 -> (0,1)
    expected_bonds_sorted = [[0, 1], [1, 2], [1, 3]]
    assert arr2.bonds.tolist() == expected_bonds_sorted

    # Bond arrays must follow the same new bond order.
    # Build expected bond_perm by reproducing the key sort.
    remapped = np.array([[1, 3], [1, 2], [0, 1]], dtype=np.int64)
    keys = remapped[:, 0] * (4 + 1) + remapped[:, 1]
    expected_perm = np.argsort(keys, kind="mergesort").astype(np.int64)
    assert bond_perm.tolist() == expected_perm.tolist()

    # bond_type / bond optionals aligned
    assert arr2.bond_type.tolist() == arr.bond_type[bond_perm].tolist()
    assert arr2.is_conjugated.tolist() == arr.is_conjugated[bond_perm].tolist()
    assert arr2.bond_resonance_type.tolist() == arr.bond_resonance_type[bond_perm].tolist()


def test_permute_atoms_handles_ensemble_pos_and_coord_valid():
    mg = _make_graph_with_ensemble_coords()
    arr = mg.arrays

    new_to_old = np.array([3, 1, 2, 0], dtype=np.int64)
    mg2, old_to_new, bond_perm = permute_atoms(mg, new_to_old)
    arr2 = mg2.arrays

    assert arr2.pos is not None
    assert arr2.pos.shape == (2, 4, 3)
    assert arr2.coord_valid is not None
    assert arr2.coord_valid.shape == (2, 4)

    # pos should be permuted along atom axis (axis=1) for ensemble
    assert np.allclose(arr2.pos[:, :, :], arr.pos[:, new_to_old, :])

    # coord_valid permuted along atom axis too
    assert np.array_equal(arr2.coord_valid, arr.coord_valid[:, new_to_old])


def run_all_tests() -> bool:
    tests = [
        test_delete_atom_reindexes_and_masks,
        test_delete_atoms_multiple,
        test_add_atom_appends_required_and_respects_optional_presence,
        test_replace_atom_attachment_invariant,
        test_set_dummy_label_guards,
        test_delete_atoms_slices_ensemble_pos_correctly,
        test_add_atom_appends_ensemble_pos_correctly,
        test_permute_atoms_reorders_atoms_and_bonds_and_preserves_alignment,
        test_permute_atoms_handles_ensemble_pos_and_coord_valid,

    ]

    print("\n" + "#" * 60)
    print("# RUNNING ALL EDIT/ATOMS TESTS")
    print("#" * 60)

    passed = 0
    for t in tests:
        name = t.__name__
        print("\n" + "=" * 60)
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
