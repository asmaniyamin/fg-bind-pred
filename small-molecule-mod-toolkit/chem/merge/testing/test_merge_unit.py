# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/merge/testing/test_merge_unit.py

Unit tests for chem.merge.merge:
  - resolve_site() (substituent and insertion sites)
  - concat_graphs()
  - merge_by_labels() (substituent and insertion merge)
  - H-removal during merge (explicit H node removal)
  - Index map consistency after merge
  - Error paths (bad label, wrong site, both-insertion)

Constructs graphs directly from SMILES using smiles_to_molgraph where needed,
or builds minimal MolGraph objects by hand for isolated unit tests.

Run:
  python -m chem.merge.testing.test_merge_unit
(or call run_all_tests()).
"""

from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import numpy as np

from core.structs import MolArrays, MolGraph, BondType, AtomHybridization

from chem.merge.merge import resolve_site, concat_graphs, merge_by_labels

try:
    from chem.build.create_molgraph import smiles_to_molgraph
    _HAS_BUILD = True
except Exception:
    smiles_to_molgraph = None  # type: ignore
    _HAS_BUILD = False


# =============================================================================
# Minimal graph builder (no RDKit required)
# =============================================================================

def _make_substituent_graph(label: int = 1) -> MolGraph:
    """
    Graph with one dummy (label=<label>) attached to carbon:
      0: dummy (Z=0, attachment_label=label)
      1: carbon
    Bond: 0-1 single
    """
    atomic_num = np.array([0, 6], dtype=np.int16)
    formal_charge = np.zeros(2, dtype=np.int8)
    bonds = np.array([[0, 1]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE], dtype=np.int8)
    attachment_label = np.array([label, None], dtype=object)

    arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        attachment_label=attachment_label,
        total_charge=0,
    )
    return MolGraph(arrays=arrays, meta={"smiles": f"[*:{label}]C"})


def _make_substituent_graph_with_h(label: int = 1) -> MolGraph:
    """
    Graph with dummy + carbon + one explicit H atom on the carbon:
      0: dummy (Z=0, label=label)
      1: carbon
      2: H (Z=1)
    Bonds: 0-1, 1-2 single
    """
    atomic_num = np.array([0, 6, 1], dtype=np.int16)
    formal_charge = np.zeros(3, dtype=np.int8)
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.SINGLE], dtype=np.int8)
    attachment_label = np.array([label, None, None], dtype=object)

    arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        attachment_label=attachment_label,
        total_charge=0,
    )
    return MolGraph(arrays=arrays, meta={"smiles": f"[*:{label}][CH3]"})


def _make_insertion_graph(label: int = 1) -> MolGraph:
    """
    Graph with dummy (label=label) between two carbons (insertion site):
      0: carbon-a
      1: dummy (Z=0, label=label)
      2: carbon-b
    Bonds: 0-1, 1-2 single
    """
    atomic_num = np.array([6, 0, 6], dtype=np.int16)
    formal_charge = np.zeros(3, dtype=np.int8)
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    bond_type = np.array([BondType.SINGLE, BondType.SINGLE], dtype=np.int8)
    attachment_label = np.array([None, label, None], dtype=object)

    arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        attachment_label=attachment_label,
        total_charge=0,
    )
    return MolGraph(arrays=arrays, meta={"smiles": f"C[*:{label}]C"})


# =============================================================================
# resolve_site tests
# =============================================================================

def test_resolve_site_substituent_finds_correct_anchor() -> None:
    g = _make_substituent_graph(label=1)
    site = resolve_site(g, 1)

    assert site.label == 1
    assert site.dummy_idx == 0
    assert site.mode == "substituent"
    assert len(site.heavy_neighbors) == 1
    assert int(site.heavy_neighbors[0]) == 1


def test_resolve_site_insertion_finds_two_heavy_neighbors() -> None:
    g = _make_insertion_graph(label=1)
    site = resolve_site(g, 1)

    assert site.label == 1
    assert site.dummy_idx == 1
    assert site.mode == "insertion"
    assert len(site.heavy_neighbors) == 2
    anchors = sorted(site.heavy_neighbors.tolist())
    assert anchors == [0, 2]


def test_resolve_site_raises_for_missing_label() -> None:
    g = _make_substituent_graph(label=1)
    raised = False
    try:
        resolve_site(g, 99)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for missing label"


def test_resolve_site_h_neighbors_is_array() -> None:
    """h_neighbors is always an ndarray (may be empty for typical SMILES graphs)."""
    g = _make_substituent_graph(label=1)
    site = resolve_site(g, 1)
    assert isinstance(site.h_neighbors, np.ndarray)


# =============================================================================
# concat_graphs tests
# =============================================================================

def test_concat_graphs_atom_count() -> None:
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    n_a = int(gA.arrays.atomic_num.shape[0])
    n_b = int(gB.arrays.atomic_num.shape[0])

    g_ab, offset = concat_graphs(gA, gB)

    assert offset == n_a
    assert int(g_ab.arrays.atomic_num.shape[0]) == n_a + n_b


def test_concat_graphs_bond_count() -> None:
    gA = _make_substituent_graph(label=1)  # 1 bond
    gB = _make_substituent_graph(label=2)  # 1 bond

    g_ab, offset = concat_graphs(gA, gB)
    assert int(g_ab.arrays.bonds.shape[0]) == 2


def test_concat_graphs_bonds_offset_correctly() -> None:
    """B's bond indices must be shifted by n_a."""
    gA = _make_substituent_graph(label=1)  # atoms 0,1; bond (0,1)
    gB = _make_substituent_graph(label=2)  # atoms 0,1; bond (0,1) -> should become (2,3)

    n_a = int(gA.arrays.atomic_num.shape[0])
    g_ab, _ = concat_graphs(gA, gB)

    bonds = g_ab.arrays.bonds.tolist()
    # A's bond (0,1) should remain
    assert [0, 1] in bonds
    # B's bond should be offset: (0+2, 1+2) = (2,3)
    assert [2, 3] in bonds


def test_concat_graphs_optional_union_fill() -> None:
    """If one graph has optional arrays and the other doesn't, the result should have them."""
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    # Add is_aromatic to gA only
    n_a = int(gA.arrays.atomic_num.shape[0])
    gA.arrays.is_aromatic = np.zeros(n_a, dtype=np.int8)

    g_ab, _ = concat_graphs(gA, gB)
    # is_aromatic should be present and filled for B (with default 0)
    assert g_ab.arrays.is_aromatic is not None
    assert g_ab.arrays.is_aromatic.shape[0] == n_a + int(gB.arrays.atomic_num.shape[0])


# =============================================================================
# merge_by_labels: substituent + substituent
# =============================================================================

def test_merge_substituent_substituent_atom_count() -> None:
    """Merging two substituent graphs removes 2 dummies, adds 1 bond."""
    gA = _make_substituent_graph(label=1)  # 2 atoms
    gB = _make_substituent_graph(label=2)  # 2 atoms

    # After merge: 2+2 - 2 (dummies) = 2 atoms, 2 bonds (A's + new bridge)
    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    n = int(g_merged.arrays.atomic_num.shape[0])
    assert n == 2, f"Expected 2 atoms after merge, got {n}"


def test_merge_substituent_substituent_no_dummy_in_result() -> None:
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    # No dummy atoms should remain
    assert all(int(z) != 0 for z in g_merged.arrays.atomic_num.tolist())


def test_merge_substituent_substituent_bond_created() -> None:
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    # Should have exactly 1 bond (C-C)
    assert int(g_merged.arrays.bonds.shape[0]) == 1


def test_merge_logs_to_meta() -> None:
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    assert "merge_log" in g_merged.meta
    assert len(g_merged.meta["merge_log"]) >= 1


def test_merge_substituent_substituent_bond_type_single() -> None:
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2, bond_type_code=BondType.SINGLE)

    assert int(g_merged.arrays.bond_type[0]) == BondType.SINGLE


def test_merge_substituent_substituent_bond_type_double() -> None:
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2, bond_type_code=BondType.DOUBLE)

    assert int(g_merged.arrays.bond_type[0]) == BondType.DOUBLE


def test_merge_attachments_invalidated_after_merge() -> None:
    """attachments and derived caches must be None/dirty after merge."""
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    assert g_merged.attachments is None
    assert g_merged.dirty.structure_dirty is True


# =============================================================================
# merge_by_labels: explicit H removal
# =============================================================================

def test_merge_removes_explicit_h_on_anchor() -> None:
    """When anchor has an explicit H node, one should be removed during merge."""
    gA = _make_substituent_graph_with_h(label=1)  # 3 atoms: dummy, C, H
    gB = _make_substituent_graph(label=2)           # 2 atoms: dummy, C

    # After merge: 3+2 - 2(dummies) - 1(H removed) = 2 atoms
    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    n = int(g_merged.arrays.atomic_num.shape[0])
    assert n == 2, f"Expected 2 atoms after merge with H removal, got {n}"
    assert all(int(z) != 0 for z in g_merged.arrays.atomic_num)


# =============================================================================
# merge_by_labels: insertion merge
# =============================================================================

def test_merge_insertion_substituent_atom_count() -> None:
    """Insertion on A (2 heavy neighbors), substituent on B (1 heavy neighbor)."""
    gA = _make_insertion_graph(label=1)   # 3 atoms: C, dummy, C
    gB = _make_substituent_graph(label=2) # 2 atoms: dummy, C

    # After merge: 3 + 2 - 2 (dummies) = 3 atoms; 2 new bonds (A-anchor1 + A-anchor2)
    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    n = int(g_merged.arrays.atomic_num.shape[0])
    assert n == 3, f"Expected 3 atoms for insertion+substituent merge, got {n}"


def test_merge_insertion_substituent_bonds() -> None:
    gA = _make_insertion_graph(label=1)   # bonds: C0-d, d-C2 => 2 bonds
    gB = _make_substituent_graph(label=2) # bond: d-C => 1 bond

    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    # Should have 2 bonds: C0-Cb and Cb-C2 (insertion)
    m = int(g_merged.arrays.bonds.shape[0])
    assert m == 2, f"Expected 2 bonds for insertion merge, got {m}"


# =============================================================================
# merge_by_labels: error cases
# =============================================================================

def test_merge_raises_for_missing_label_a() -> None:
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    raised = False
    try:
        merge_by_labels(gA, gB, label_a=99, label_b=2)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for missing label_a"


def test_merge_raises_for_missing_label_b() -> None:
    gA = _make_substituent_graph(label=1)
    gB = _make_substituent_graph(label=2)

    raised = False
    try:
        merge_by_labels(gA, gB, label_a=1, label_b=99)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for missing label_b"


def test_merge_both_insertion_raises() -> None:
    """Both-insertion is not supported."""
    gA = _make_insertion_graph(label=1)
    gB = _make_insertion_graph(label=2)

    raised = False
    try:
        merge_by_labels(gA, gB, label_a=1, label_b=2)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for both-insertion merge"


# =============================================================================
# merge_by_labels with SMILES (integration, requires build)
# =============================================================================

def test_merge_smiles_substituent_produces_correct_atoms() -> None:
    if not _HAS_BUILD:
        print("  SKIP: smiles_to_molgraph not available")
        return

    gA = smiles_to_molgraph("[*:1]CC")   # 3 heavy atoms (dummy + 2C)
    gB = smiles_to_molgraph("F[*:2]")    # 2 heavy atoms (F + dummy)

    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    # Merged: 3 + 2 - 2 dummies = 3 atoms
    n = int(g_merged.arrays.atomic_num.shape[0])
    assert n == 3, f"Expected 3, got {n}"


def test_merge_smiles_total_charge_consistent() -> None:
    if not _HAS_BUILD:
        print("  SKIP: smiles_to_molgraph not available")
        return

    gA = smiles_to_molgraph("[NH3+][*:1]")   # charge +1
    gB = smiles_to_molgraph("[O-][*:2]")     # charge -1

    g_merged = merge_by_labels(gA, gB, label_a=1, label_b=2)

    expected_total = int(np.sum(g_merged.arrays.formal_charge))
    assert g_merged.arrays.total_charge == expected_total


# =============================================================================
# Runner
# =============================================================================

def run_all_tests() -> bool:
    tests = [
        # resolve_site
        test_resolve_site_substituent_finds_correct_anchor,
        test_resolve_site_insertion_finds_two_heavy_neighbors,
        test_resolve_site_raises_for_missing_label,
        test_resolve_site_h_neighbors_is_array,
        # concat_graphs
        test_concat_graphs_atom_count,
        test_concat_graphs_bond_count,
        test_concat_graphs_bonds_offset_correctly,
        test_concat_graphs_optional_union_fill,
        # merge: substituent+substituent
        test_merge_substituent_substituent_atom_count,
        test_merge_substituent_substituent_no_dummy_in_result,
        test_merge_substituent_substituent_bond_created,
        test_merge_logs_to_meta,
        test_merge_substituent_substituent_bond_type_single,
        test_merge_substituent_substituent_bond_type_double,
        test_merge_attachments_invalidated_after_merge,
        # H removal
        test_merge_removes_explicit_h_on_anchor,
        # insertion
        test_merge_insertion_substituent_atom_count,
        test_merge_insertion_substituent_bonds,
        # errors
        test_merge_raises_for_missing_label_a,
        test_merge_raises_for_missing_label_b,
        test_merge_both_insertion_raises,
        # SMILES integration
        test_merge_smiles_substituent_produces_correct_atoms,
        test_merge_smiles_total_charge_consistent,
    ]

    print("\n" + "#" * 60)
    print("# RUNNING ALL MERGE UNIT TESTS")
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
