# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Tests for chem.dummy.derive
Run: python -m test_dummy_derive
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


from core.structs import AttachmentKind
from chem.dummy import derive

def test_build_attachment_points_none_when_no_dummies():
    g = make_graph_no_dummies()
    ap = derive.build_attachment_points(g)
    assert ap is None

def test_build_attachment_points_basic():
    g = make_graph_with_dummies()
    ap = derive.build_attachment_points(g)
    assert ap is not None
    assert ap.idx.tolist() == [0,3]
    assert ap.kind.tolist() == [AttachmentKind.DUMMY, AttachmentKind.DUMMY]
    assert ap.target.tolist() == [1,2]
    assert ap.label_id.tolist() == ["A","B"]
    assert ap.is_insertion.tolist() == [False, False]

def test_build_attachment_points_ambiguous_target():
    g = make_graph_with_dummies()
    g.arrays.bonds = np.array([[0,1],[0,2],[2,3]], dtype=np.int32)
    ap = derive.build_attachment_points(g)
    assert ap is not None
    assert ap.idx.tolist() == [0,3]
    assert int(ap.target[0]) == -1
    assert int(ap.target[1]) == 2
    assert bool(ap.is_insertion[0]) is True

def run_all_tests():
    tests = [
        test_build_attachment_points_none_when_no_dummies,
        test_build_attachment_points_basic,
        test_build_attachment_points_ambiguous_target,
    ]
    for t in tests:
        t()
    print("✓ test_dummy_derive: all tests passed")

if __name__ == "__main__":
    run_all_tests()
