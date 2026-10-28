# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Tests for chem.dummy.edit
Run: python -m test_dummy_edit
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


from chem.dummy import edit
from chem.dummy import query

def test_set_dummy_label_autocreate_array():
    g = make_graph_with_dummies()
    g.arrays.attachment_label = None
    g = edit.set_dummy_label(g, 0, "X")
    assert g.arrays.attachment_label is not None
    assert g.arrays.attachment_label[0] == "X"
    assert g.arrays.attachment_label[1] is None

def test_set_dummy_label_requires_dummy():
    g = make_graph_with_dummies()
    try:
        edit.set_dummy_label(g, 1, "X")
        raise AssertionError("Expected ValueError for non-dummy atom")
    except ValueError:
        pass

def test_clear_and_relabel():
    g = make_graph_with_dummies()
    g = edit.clear_dummy_label(g, 0)
    assert g.arrays.attachment_label[0] is None

    g = edit.relabel_dummy(g, "B", "C")
    assert query.dummy_by_label(g, "C") == 3
    assert query.dummy_by_label(g, "B") is None

def test_relabel_errors():
    g = make_graph_with_dummies()
    try:
        edit.relabel_dummy(g, "missing", "X")
        raise AssertionError("Expected ValueError for missing label")
    except ValueError:
        pass

    try:
        edit.relabel_dummy(g, "A", "B")
        raise AssertionError("Expected ValueError for duplicate label")
    except ValueError:
        pass

def test_enforce_label_invariant_clears_non_dummy():
    g = make_graph_with_dummies()
    g.arrays.attachment_label[1] = "BAD"
    g = edit.enforce_label_invariant(g)
    assert g.arrays.attachment_label[1] is None
    assert g.arrays.attachment_label[0] == "A"

def run_all_tests():
    tests = [
        test_set_dummy_label_autocreate_array,
        test_set_dummy_label_requires_dummy,
        test_clear_and_relabel,
        test_relabel_errors,
        test_enforce_label_invariant_clears_non_dummy,
    ]
    for t in tests:
        t()
    print("✓ test_dummy_edit: all tests passed")

if __name__ == "__main__":
    run_all_tests()
