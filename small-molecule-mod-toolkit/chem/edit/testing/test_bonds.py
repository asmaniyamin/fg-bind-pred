# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/edit/testing/test_bonds.py

Unit tests for chem.edit.bonds against the current MolGraph/MolArrays schema.

Run:
  python -m chem.edit.testing.test_bonds
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
    BondType,
    BondDir,
    BondStereo,
    BondResonanceType,
)

from chem.edit.bonds import (
    delete_bond,
    delete_bonds,
    add_bond,
)

# If you added edit_bond, we'll test it; otherwise skip gracefully.
try:
    from chem.edit.bonds import edit_bond
    _HAS_EDIT_BOND = True
except Exception:
    _HAS_EDIT_BOND = False


def _make_graph_with_bond_optionals() -> MolGraph:
    # 0-1 single, 1-2 double, 1-3 single
    atomic_num = np.array([6, 6, 8, 6], dtype=np.int16)
    formal_charge = np.array([0, 0, 0, 0], dtype=np.int8)

    bonds = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.DOUBLE, BondType.SINGLE], dtype=np.int8)

    is_conjugated = np.array([0, 1, 0], dtype=np.int8)
    is_in_ring = np.array([0, 0, 0], dtype=np.int8)
    bond_dir = np.array([BondDir.NONE, BondDir.NONE, BondDir.NONE], dtype=np.int8)
    bond_stereo = np.array([BondStereo.NONE, BondStereo.E, BondStereo.NONE], dtype=np.int8)
    bond_resonance_type = np.array([BondResonanceType.NONE, BondResonanceType.LOCALIZED, BondResonanceType.NONE], dtype=np.int8)

    arr = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        is_conjugated=is_conjugated,
        is_in_ring=is_in_ring,
        bond_dir=bond_dir,
        bond_stereo=bond_stereo,
        bond_resonance_type=bond_resonance_type,
        total_charge=0,
        multiplicity=None,
    )
    return MolGraph(arrays=arr)


def test_delete_bond_by_pair_masks_all_bond_arrays():
    mg = _make_graph_with_bond_optionals()
    mg2 = delete_bond(mg, 1, 2)

    arr = mg2.arrays
    assert arr.bonds.tolist() == [[0, 1], [1, 3]]
    assert arr.bond_type.tolist() == [BondType.SINGLE, BondType.SINGLE]
    assert arr.is_conjugated.tolist() == [0, 0]
    assert arr.bond_stereo.tolist() == [BondStereo.NONE, BondStereo.NONE]
    assert arr.bond_resonance_type.tolist() == [BondResonanceType.NONE, BondResonanceType.NONE]

    assert mg2.cache.bond_pair_to_index is None
    assert mg2.dirty.structure_dirty is True


def test_delete_bonds_by_index():
    mg = _make_graph_with_bond_optionals()
    # remove bonds 0 and 2 -> keep only [1,2] double
    mg2 = delete_bonds(mg, [0, 2])

    arr = mg2.arrays
    assert arr.bonds.tolist() == [[1, 2]]
    assert arr.bond_type.tolist() == [BondType.DOUBLE]
    assert arr.is_conjugated.tolist() == [1]


def test_add_bond_normalizes_and_prevents_duplicates():
    mg = _make_graph_with_bond_optionals()

    # Add bond 0-2 (should store [0,2])
    mg2 = add_bond(mg, 2, 0, bond_type=BondType.SINGLE, bond_stereo=BondStereo.Z)
    assert mg2.arrays.bonds.tolist()[-1] == [0, 2]
    assert mg2.arrays.bond_type.tolist()[-1] == BondType.SINGLE

    # Duplicate should raise
    try:
        add_bond(mg2, 0, 2, bond_type=BondType.SINGLE)
        raise AssertionError("Expected ValueError for duplicate bond")
    except ValueError:
        pass


def test_edit_bond_updates_only_requested_fields():
    if not _HAS_EDIT_BOND:
        print("! edit_bond not present; skipping test_edit_bond_updates_only_requested_fields")
        return

    mg = _make_graph_with_bond_optionals()
    mg2 = edit_bond(mg, 1, 2, bond_type=BondType.AROMATIC, bond_stereo=BondStereo.Z)

    # bond (1,2) is index 1 in our fixture
    assert mg2.arrays.bond_type[1] == BondType.AROMATIC
    assert mg2.arrays.bond_stereo[1] == BondStereo.Z
    # unchanged field
    assert mg2.arrays.is_conjugated[1] == 1


def run_all_tests() -> bool:
    tests = [
        test_delete_bond_by_pair_masks_all_bond_arrays,
        test_delete_bonds_by_index,
        test_add_bond_normalizes_and_prevents_duplicates,
        test_edit_bond_updates_only_requested_fields,
    ]

    print("\n" + "#" * 60)
    print("# RUNNING ALL EDIT/BONDS TESTS")
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
