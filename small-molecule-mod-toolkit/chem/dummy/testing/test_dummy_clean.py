# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Tests for chem.dummy.clean
Run: python -m test_dummy_clean
"""


if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import numpy as np
from core.structs import MolArrays, MolGraph, BondType

def make_graph_with_dummies():
    atomic_num = np.array([0, 6, 6, 0], dtype=np.int16)
    formal_charge = np.zeros(4, dtype=np.int8)
    bonds = np.array([[0,1],[1,2],[2,3]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.SINGLE, BondType.SINGLE], dtype=np.int8)
    attachment_label = np.array(["A", None, None, "B"], dtype=object)
    arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        attachment_label=attachment_label,
    )
    return MolGraph(arrays=arrays)

def make_graph_no_dummies():
    atomic_num = np.array([6,6,8], dtype=np.int16)
    formal_charge = np.zeros(3, dtype=np.int8)
    bonds = np.array([[0,1],[1,2]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.DOUBLE], dtype=np.int8)
    arrays = MolArrays(atomic_num=atomic_num, formal_charge=formal_charge, bonds=bonds, bond_type=bond_type)
    return MolGraph(arrays=arrays)


from chem.dummy import clean

def test_validate_dummy_invariants_leakage():
    g = make_graph_with_dummies()
    g.arrays.attachment_label[1] = "BAD"
    warnings = clean.validate_dummy_invariants(g)
    assert any("non-dummy" in w for w in warnings), warnings

def test_remove_orphan_dummies_mark_only():
    atomic_num = np.array([0,6], dtype=np.int16)
    formal_charge = np.zeros(2, dtype=np.int8)
    bonds = np.zeros((0,2), dtype=np.int32)
    bond_type = np.zeros((0,), dtype=np.int8)
    attachment_label = np.array(["X", None], dtype=object)
    arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        attachment_label=attachment_label,
    )
    g = MolGraph(arrays=arrays)
    g = clean.remove_orphan_dummies_mark_only(g)
    assert g.meta.get("orphan_dummies") == [0]

def run_all_tests():
    tests = [
        test_validate_dummy_invariants_leakage,
        test_remove_orphan_dummies_mark_only,
    ]
    for t in tests:
        t()
    print("✓ test_dummy_clean: all tests passed")

if __name__ == "__main__":
    run_all_tests()
