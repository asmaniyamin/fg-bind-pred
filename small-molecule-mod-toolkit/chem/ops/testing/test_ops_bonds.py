# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/ops/testing/test_bonds.py

Tests for chem.ops.bonds wrappers (Transform layer):
  - AddBond
  - DeleteBond
  - DeleteBonds
  - EditBond

Run:
  python -m chem.ops.testing.test_bonds
(or call run_all_tests()).
"""

from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import numpy as np

from core.structs import MolArrays, MolGraph, BondType
from chem.ops.base import apply_transform
from chem.ops.bonds import AddBond, DeleteBond, DeleteBonds, EditBond


def _make_graph_for_bond_ops(include_optional_bond_arrays: bool = False) -> MolGraph:
    """
    Build a simple 3-atom chain: C-C-C
      bonds: (0,1) single, (1,2) single

    Bonds are stored normalized (u < v), matching the edit layer conventions.
    """
    atomic_num = np.array([6, 6, 6], dtype=np.int16)
    formal_charge = np.array([0, 0, 0], dtype=np.int8)

    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    bond_type = np.array([int(BondType.SINGLE), int(BondType.SINGLE)], dtype=np.int8)

    # Optional bond arrays (only created when requested; edit_bond should error if you try to set
    # an optional field while the corresponding array is None).
    if include_optional_bond_arrays:
        is_conjugated = np.array([False, False], dtype=bool)
        is_in_ring = np.array([False, False], dtype=bool)
        bond_dir = np.array([0, 0], dtype=np.int8)
        bond_stereo = np.array([0, 0], dtype=np.int8)
        bond_resonance_type = np.array([0, 0], dtype=np.int8)
    else:
        is_conjugated = None
        is_in_ring = None
        bond_dir = None
        bond_stereo = None
        bond_resonance_type = None

    arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        is_conjugated=is_conjugated,
        is_in_ring=is_in_ring,
        bond_dir=bond_dir,
        bond_stereo=bond_stereo,
        bond_resonance_type=bond_resonance_type,
        pos=None,
        coord_frame=None,
        coord_valid=None,
        total_charge=0,
        multiplicity=1,
    )
    return MolGraph(arrays=arrays, meta={"smiles": "CCC"})


def test_ops_add_bond_adds_and_logs() -> None:
    g = _make_graph_for_bond_ops()
    n0 = int(g.arrays.atomic_num.shape[0])
    m0 = int(g.arrays.bonds.shape[0])

    g2, info = apply_transform(g, AddBond(u=0, v=2, bond_type=int(BondType.SINGLE)))

    assert int(g2.arrays.atomic_num.shape[0]) == n0
    assert int(g2.arrays.bonds.shape[0]) == m0 + 1

    # bond should be normalized (0,2) in the stored array.
    assert any((int(u) == 0 and int(v) == 2) for u, v in g2.arrays.bonds.tolist())

    # op_log should include this op
    assert "op_log" in g2.meta
    last = g2.meta["op_log"][-1]
    assert last["op"] == "add_bond"
    assert last["params"]["u"] == 0
    assert last["params"]["v"] == 2

    # identity atom maps
    assert info.old_to_new is not None
    assert info.new_to_old is not None
    assert np.array_equal(info.old_to_new, np.arange(n0, dtype=np.int64))
    assert np.array_equal(info.new_to_old, np.arange(n0, dtype=np.int64))


def test_ops_edit_bond_type_updates_and_logs() -> None:
    g = _make_graph_for_bond_ops()
    # bond (0,1) exists initially; edit_bond finds it and edits in place.

    g2, info = apply_transform(g, EditBond(u=0, v=1, bond_type=int(BondType.DOUBLE)))

    # bond count unchanged
    assert int(g2.arrays.bonds.shape[0]) == 2

    # find bond index for (0,1) and verify type changed
    bonds = g2.arrays.bonds.tolist()
    idx01 = next(i for i, (u, v) in enumerate(bonds) if int(u) == 0 and int(v) == 1)
    assert int(g2.arrays.bond_type[idx01]) == int(BondType.DOUBLE)

    last = g2.meta["op_log"][-1]
    assert last["op"] == "edit_bond"
    assert last["params"]["u"] == 0
    assert last["params"]["v"] == 1
    assert last["params"]["bond_type"] == int(BondType.DOUBLE)


def test_ops_edit_bond_optional_field_requires_array_present() -> None:
    # optional arrays absent -> edit_bond should raise if you try to set is_conjugated.
    g = _make_graph_for_bond_ops(include_optional_bond_arrays=False)

    threw = False
    try:
        apply_transform(g, EditBond(u=0, v=1, is_conjugated=True))
    except ValueError as e:
        threw = True
        assert "arrays.is_conjugated is None" in str(e)
    assert threw, "Expected ValueError when setting optional field with missing array"


def test_ops_add_bond_optional_fields_only_apply_if_arrays_present() -> None:
    # add_bond only appends optional arrays if present; if absent, it should keep None.
    g = _make_graph_for_bond_ops(include_optional_bond_arrays=False)

    g2, info = apply_transform(g, AddBond(u=0, v=2, bond_type=int(BondType.SINGLE), is_conjugated=True))

    # Because arrays.is_conjugated is None, it should remain None after add_bond.
    assert g2.arrays.is_conjugated is None
    assert g2.arrays.bond_dir is None
    assert g2.arrays.bond_stereo is None


def test_ops_delete_bond_by_pair_deletes_and_logs() -> None:
    g = _make_graph_for_bond_ops()
    m0 = int(g.arrays.bonds.shape[0])

    g2, info = apply_transform(g, DeleteBond(u=1, v=2, undirected=True))

    assert int(g2.arrays.bonds.shape[0]) == m0 - 1
    assert not any((int(u) == 1 and int(v) == 2) for u, v in g2.arrays.bonds.tolist())

    last = g2.meta["op_log"][-1]
    assert last["op"] == "delete_bond"
    assert last["params"]["u"] == 1
    assert last["params"]["v"] == 2
    assert last["params"]["undirected"] is True


def test_ops_delete_bonds_by_index_deletes_and_logs() -> None:
    g = _make_graph_for_bond_ops()
    m0 = int(g.arrays.bonds.shape[0])

    # delete bond index 0 (which is (0,1) in our construction)
    g2, info = apply_transform(g, DeleteBonds(bond_indices=[0]))

    assert int(g2.arrays.bonds.shape[0]) == m0 - 1
    assert not any((int(u) == 0 and int(v) == 1) for u, v in g2.arrays.bonds.tolist())

    last = g2.meta["op_log"][-1]
    assert last["op"] == "delete_bonds"
    assert last["params"]["bond_indices"] == [0]


def run_all_tests() -> bool:
    tests = [
        test_ops_add_bond_adds_and_logs,
        test_ops_edit_bond_type_updates_and_logs,
        test_ops_edit_bond_optional_field_requires_array_present,
        test_ops_add_bond_optional_fields_only_apply_if_arrays_present,
        test_ops_delete_bond_by_pair_deletes_and_logs,
        test_ops_delete_bonds_by_index_deletes_and_logs,
    ]

    print("\n" + "#" * 60)
    print("# RUNNING ALL OPS BONDS TESTS")
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
