# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/ops/testing/test_ops_add_replace_atom.py

Unit tests for AddAtom and ReplaceAtom Transform wrappers in chem.ops.atoms.

These cover the bugs that were fixed (Bug 3: aromatic kwarg, Bug A: None kwargs crash).

Run:
  python -m chem.ops.testing.test_ops_add_replace_atom
(or call run_all_tests()).
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
    AtomHybridization,
    AtomChiralTag,
    AtomCIPCode,
    BondDir,
    BondStereo,
    BondResonanceType,
)
from chem.ops.base import apply_transform
from chem.ops.atoms import AddAtom, ReplaceAtom


def _make_graph() -> MolGraph:
    """Small graph: dummy(0)-C(1)-O(2)=C(3), with optional arrays present."""
    atomic_num = np.array([0, 6, 8, 6], dtype=np.int16)
    formal_charge = np.array([0, 0, 0, 0], dtype=np.int8)

    bonds = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE], dtype=np.int8)

    isotope = np.array([0, 12, 0, 12], dtype=np.int16)
    is_aromatic = np.array([0, 0, 0, 0], dtype=np.int8)
    hybridization = np.array(
        [AtomHybridization.OTHER, AtomHybridization.SP3, AtomHybridization.SP2, AtomHybridization.SP2],
        dtype=np.int8,
    )
    chiral_tag = np.array([AtomChiralTag.UNSPECIFIED] * 4, dtype=np.int8)
    cip_code = np.array([AtomCIPCode.NONE] * 4, dtype=np.int8)
    atom_map = np.array([1, 0, 0, 0], dtype=np.int32)
    attachment_label = np.array([1, None, None, None], dtype=object)
    explicit_h = np.array([0, 0, 0, 0], dtype=np.int8)
    implicit_h = np.array([0, 3, 0, 3], dtype=np.int8)
    partial_charge = np.array([0.0, 0.0, -0.2, 0.0], dtype=np.float32)

    is_conjugated = np.array([0, 0, 1], dtype=np.int8)
    is_in_ring = np.array([0, 0, 0], dtype=np.int8)
    bond_dir = np.array([BondDir.NONE] * 3, dtype=np.int8)
    bond_stereo = np.array([BondStereo.NONE] * 3, dtype=np.int8)
    bond_resonance_type = np.array([BondResonanceType.NONE] * 3, dtype=np.int8)

    arrays = MolArrays(
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
        pos=None,
        coord_frame=None,
        coord_valid=None,
        is_conjugated=is_conjugated,
        is_in_ring=is_in_ring,
        bond_dir=bond_dir,
        bond_stereo=bond_stereo,
        bond_resonance_type=bond_resonance_type,
        total_charge=0,
        multiplicity=1,
    )
    return MolGraph(arrays=arrays, meta={"smiles": "[*:1]COC"})


# --------------------------------------------------------------------------
# AddAtom tests
# --------------------------------------------------------------------------

def test_add_atom_basic_appends_and_logs() -> None:
    g = _make_graph()
    n0 = int(g.arrays.atomic_num.shape[0])

    g2, info = apply_transform(g, AddAtom(atomic_num=7, formal_charge=0))

    assert int(g2.arrays.atomic_num.shape[0]) == n0 + 1
    assert int(g2.arrays.atomic_num[-1]) == 7
    assert int(g2.arrays.formal_charge[-1]) == 0

    assert "op_log" in g2.meta
    last = g2.meta["op_log"][-1]
    assert last["op"] == "add_atom"
    assert last["params"]["atomic_num"] == 7
    assert last["params"]["formal_charge"] == 0

    assert info.changes is not None
    assert info.changes.n_atoms_before == n0
    assert info.changes.n_atoms_after == n0 + 1


def test_add_atom_reports_new_index_in_changes() -> None:
    g = _make_graph()
    n0 = int(g.arrays.atomic_num.shape[0])

    g2, info = apply_transform(g, AddAtom(atomic_num=16))

    assert info.changes is not None
    assert info.changes.atoms_added_new is not None
    assert int(info.changes.atoms_added_new[0]) == n0


def test_add_atom_with_formal_charge() -> None:
    g = _make_graph()

    g2, info = apply_transform(g, AddAtom(atomic_num=7, formal_charge=1))

    assert int(g2.arrays.formal_charge[-1]) == 1
    # total_charge should update
    assert g2.arrays.total_charge == 1


def test_add_atom_extends_optional_arrays() -> None:
    """When optional arrays exist they must be extended."""
    g = _make_graph()
    n0 = int(g.arrays.atomic_num.shape[0])

    g2, _ = apply_transform(g, AddAtom(atomic_num=6))

    arr = g2.arrays
    assert arr.isotope is not None and arr.isotope.shape[0] == n0 + 1
    assert arr.is_aromatic is not None and arr.is_aromatic.shape[0] == n0 + 1
    assert arr.hybridization is not None and arr.hybridization.shape[0] == n0 + 1
    assert arr.explicit_h is not None and arr.explicit_h.shape[0] == n0 + 1
    assert arr.implicit_h is not None and arr.implicit_h.shape[0] == n0 + 1
    assert arr.partial_charge is not None and arr.partial_charge.shape[0] == n0 + 1
    assert arr.attachment_label is not None and arr.attachment_label.shape[0] == n0 + 1
    # New non-dummy atom: attachment_label[-1] must be None
    assert arr.attachment_label[-1] is None


def test_add_atom_is_aromatic_kwarg_accepted() -> None:
    """AddAtom forwards the aromatic flag to the atom array."""
    g = _make_graph()

    # is_aromatic=True is accepted.
    g2, info = apply_transform(g, AddAtom(atomic_num=6, aromatic=True))
    arr = g2.arrays
    # The last atom should have is_aromatic=True (1)
    assert int(arr.is_aromatic[-1]) == 1


def test_add_atom_isotope_accepted() -> None:
    g = _make_graph()
    g2, _ = apply_transform(g, AddAtom(atomic_num=6, isotope=13))
    assert int(g2.arrays.isotope[-1]) == 13


# --------------------------------------------------------------------------
# ReplaceAtom tests
# --------------------------------------------------------------------------

def test_replace_atom_changes_atomic_num() -> None:
    g = _make_graph()

    g2, info = apply_transform(g, ReplaceAtom(idx=1, atomic_num=7))

    assert int(g2.arrays.atomic_num[1]) == 7
    assert int(g2.arrays.atomic_num.shape[0]) == 4  # no change in count

    assert "op_log" in g2.meta
    last = g2.meta["op_log"][-1]
    assert last["op"] == "replace_atom"
    assert last["params"]["idx"] == 1
    assert last["params"]["atomic_num"] == 7


def test_replace_atom_optional_none_does_not_crash() -> None:
    """ReplaceAtom omits optional fields whose value is None."""
    g = _make_graph()

    # All optional fields default to None - this must NOT crash
    g2, info = apply_transform(g, ReplaceAtom(idx=1, atomic_num=7))
    assert int(g2.arrays.atomic_num[1]) == 7


def test_replace_atom_with_formal_charge() -> None:
    g = _make_graph()

    g2, _ = apply_transform(g, ReplaceAtom(idx=1, atomic_num=7, formal_charge=1))

    assert int(g2.arrays.atomic_num[1]) == 7
    assert int(g2.arrays.formal_charge[1]) == 1
    assert g2.arrays.total_charge == 1


def test_replace_atom_with_isotope() -> None:
    g = _make_graph()

    g2, _ = apply_transform(g, ReplaceAtom(idx=1, atomic_num=6, isotope=13))
    assert int(g2.arrays.isotope[1]) == 13


def test_replace_atom_is_aromatic_kwarg_accepted() -> None:
    """ReplaceAtom forwards the aromatic flag to the atom array."""
    g = _make_graph()

    # aromatic=True triggers is_aromatic= field forwarding (no crash)
    g2, _ = apply_transform(g, ReplaceAtom(idx=1, atomic_num=6, aromatic=True))
    assert int(g2.arrays.is_aromatic[1]) == 1


def test_replace_atom_dummy_clears_attachment_label() -> None:
    """Replacing a dummy atom with a heavy atom must clear its attachment_label."""
    g = _make_graph()
    assert g.arrays.attachment_label[0] == 1  # was dummy with label 1

    g2, _ = apply_transform(g, ReplaceAtom(idx=0, atomic_num=6))
    assert g2.arrays.atomic_num[0] == 6
    assert g2.arrays.attachment_label[0] is None


def test_replace_atom_logs_only_provided_fields() -> None:
    """params in op_log should only contain fields that were provided, not None ones."""
    g = _make_graph()

    g2, info = apply_transform(g, ReplaceAtom(idx=2, atomic_num=8, formal_charge=-1))

    last = g2.meta["op_log"][-1]
    params = last["params"]
    assert "atomic_num" in params
    assert "formal_charge" in params
    # isotope was not provided; it should not appear as None in params
    assert params.get("isotope") is None or "isotope" not in params


def test_replace_atom_bond_count_unchanged() -> None:
    """ReplaceAtom does not touch bonds."""
    g = _make_graph()
    m0 = int(g.arrays.bonds.shape[0])

    g2, info = apply_transform(g, ReplaceAtom(idx=2, atomic_num=16))

    assert int(g2.arrays.bonds.shape[0]) == m0
    assert info.changes.n_bonds_before == m0
    assert info.changes.n_bonds_after == m0


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_all_tests() -> bool:
    tests = [
        test_add_atom_basic_appends_and_logs,
        test_add_atom_reports_new_index_in_changes,
        test_add_atom_with_formal_charge,
        test_add_atom_extends_optional_arrays,
        test_add_atom_is_aromatic_kwarg_accepted,
        test_add_atom_isotope_accepted,
        test_replace_atom_changes_atomic_num,
        test_replace_atom_optional_none_does_not_crash,
        test_replace_atom_with_formal_charge,
        test_replace_atom_with_isotope,
        test_replace_atom_is_aromatic_kwarg_accepted,
        test_replace_atom_dummy_clears_attachment_label,
        test_replace_atom_logs_only_provided_fields,
        test_replace_atom_bond_count_unchanged,
    ]

    print("\n" + "#" * 60)
    print("# RUNNING ALL OPS ADD/REPLACE ATOM TESTS")
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
