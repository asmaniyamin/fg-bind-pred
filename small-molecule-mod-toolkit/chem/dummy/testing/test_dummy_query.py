# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Tests for chem.dummy.query
Run: python -m test_dummy_query
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


from chem.dummy import query

def test_dummy_indices_and_labels():
    g = make_graph_with_dummies()
    idx = query.dummy_indices(g)
    assert idx.tolist() == [0, 3], f"Expected [0,3], got {idx.tolist()}"

    labs = query.dummy_labels(g)
    assert labs.tolist() == ["A", "B"], f"Expected ['A','B'], got {labs.tolist()}"

def test_dummy_labels_when_array_missing():
    g = make_graph_with_dummies()
    g.arrays.attachment_label = None
    labs = query.dummy_labels(g)
    assert labs.dtype == object
    assert labs.tolist() == [None, None]

def test_dummy_by_label_str_int_equivalence():
    g = make_graph_with_dummies()
    g.arrays.attachment_label[0] = 5
    assert query.dummy_by_label(g, "5") == 0
    assert query.dummy_by_label(g, 5) == 0
    assert query.dummy_by_label(g, "nope") is None

def test_neighbors_and_targets():
    g = make_graph_with_dummies()
    n1 = query.neighbors(g, 1)
    assert sorted(n1.tolist()) == [0,2], f"neighbors(1) wrong: {n1}"

    assert query.dummy_target(g, 0) == 1
    assert query.dummy_target(g, 3) == 2

def test_insertion_dummy_heuristic():
    g = make_graph_with_dummies()
    g.arrays.bonds = np.array([[0,1],[0,2],[2,3]], dtype=np.int32)
    assert query.is_insertion_dummy(g, 0) is True
    assert query.is_insertion_dummy(g, 3) is False

def run_all_tests():
    tests = [
        test_dummy_indices_and_labels,
        test_dummy_labels_when_array_missing,
        test_dummy_by_label_str_int_equivalence,
        test_neighbors_and_targets,
        test_insertion_dummy_heuristic,
    ]
    for t in tests:
        t()
    print("✓ test_dummy_query: all tests passed")

if __name__ == "__main__":
    run_all_tests()
