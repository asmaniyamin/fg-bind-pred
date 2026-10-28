# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/ops/testing/test_atoms.py

Unit tests for chem.ops.atoms wrappers (Transform layer).
These tests avoid RDKit and construct MolGraph objects directly.

Run:
  python -m chem.ops.testing.test_atoms
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
from chem.ops.atoms import DeleteAtom, PermuteAtoms, SetDummyLabel


def _make_graph_for_atom_ops() -> MolGraph:
    """
    Build a small graph:
      0: dummy (Z=0) label=1
      1: carbon
      2: oxygen
      3: carbon

    Bonds:
      0-1 single
      1-2 double
      1-3 single
    """
    atomic_num = np.array([0, 6, 8, 6], dtype=np.int16)
    formal_charge = np.array([0, 0, 0, 0], dtype=np.int8)

    bonds = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.DOUBLE, BondType.SINGLE], dtype=np.int8)

    attachment_label = np.array([1, 0, 0, 0], dtype=np.int16)

    # include optional arrays to make sure ops don't break them
    implicit_h = np.array([0, 3, 0, 3], dtype=np.int8)
    explicit_h = np.array([0, 0, 0, 0], dtype=np.int8)

    arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        attachment_label=attachment_label,
        implicit_h=implicit_h,
        explicit_h=explicit_h,
        pos=None,
        coord_frame=None,
        coord_valid=None,
        total_charge=int(np.sum(formal_charge).item()),
        multiplicity=1,
    )
    return MolGraph(arrays=arrays, meta={"smiles": "dummy-C(=O)C"})


def test_ops_delete_atom_logs_and_maps() -> None:
    g = _make_graph_for_atom_ops()

    g2, info = apply_transform(g, DeleteAtom(idx=2))

    # log exists
    assert "op_log" in g2.meta, "apply_transform should create op_log"
    assert len(g2.meta["op_log"]) >= 1
    last = g2.meta["op_log"][-1]
    assert last["op"] == "delete_atom"
    assert last["params"]["idx"] == 2

    # mapping exists for DeleteAtom wrapper
    assert info.old_to_new is not None
    assert info.new_to_old is not None
    assert info.old_to_new.shape == (4,)
    assert int(info.old_to_new[2]) == -1, "deleted atom should map to -1"

    # atom count decreased by 1
    assert int(g2.arrays.atomic_num.shape[0]) == 3


def test_ops_permute_atoms_identity_vs_swap() -> None:
    g = _make_graph_for_atom_ops()
    n = int(g.arrays.atomic_num.shape[0])

    # swap atoms 1 and 3 (keep 0,2 fixed)
    new_to_old = np.array([0, 3, 2, 1], dtype=np.int64)
    g2, info = apply_transform(g, PermuteAtoms(new_to_old=new_to_old))

    assert info.old_to_new is not None
    assert info.new_to_old is not None
    assert info.bond_perm is not None

    # new_to_old should be exactly what we passed
    assert np.array_equal(info.new_to_old, new_to_old)

    # old_to_new should be inverse permutation
    inv = np.full(n, -1, dtype=np.int64)
    inv[new_to_old] = np.arange(n, dtype=np.int64)
    assert np.array_equal(info.old_to_new, inv)


def test_ops_set_dummy_label_logs() -> None:
    g = _make_graph_for_atom_ops()

    g2, info = apply_transform(g, SetDummyLabel(idx=0, label=7))

    assert len(g2.meta["op_log"]) >= 1
    last = g2.meta["op_log"][-1]
    assert last["op"] == "set_dummy_label"
    assert last["params"]["idx"] == 0
    assert last["params"]["label"] == 7

    assert g2.arrays.attachment_label is not None
    assert int(g2.arrays.attachment_label[0]) == 7


def run_all_tests() -> bool:
    tests = [
        test_ops_delete_atom_logs_and_maps,
        test_ops_permute_atoms_identity_vs_swap,
        test_ops_set_dummy_label_logs,
    ]

    print("\n" + "#" * 60)
    print("# RUNNING ALL OPS ATOMS TESTS")
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
