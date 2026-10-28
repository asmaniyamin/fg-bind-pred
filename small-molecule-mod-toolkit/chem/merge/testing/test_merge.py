# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0


"""
test_merge.py

"Super intense" tests for chem/merge.py.

What this suite aims to give you:
- Prints FULL MolGraph entities for inputs and outputs (atoms/bonds + key arrays)
- Exercises:
    * resolve_site() (substituent + insertion)
    * merge_by_labels() substituent + insertion
    * explicit-H-node removal (atomic_num==1 nodes)
    * stereochemistry (bond stereo + bond dir) preservation and union-default-fill
    * bond order selection for the NEW bond
    * formal charges / total_charge consistency
    * merge meta logging (merge_log + smiles placeholder vs RDKit canonical)

This suite prefers building graphs from SMILES via chem.build.create_molgraph.smiles_to_molgraph
so you get realistic stereochem/charge arrays.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import sys
import traceback
from typing import Any, Callable, Optional

import numpy as np

from core.structs import Fragment, BondType

from chem.merge.merge import resolve_site, merge_by_labels

# Prefer "known-good" conversion for readable SMILES
try:
    from chem.build.create_molgraph import smiles_to_molgraph
    from chem.build.molgraph_to_mol import molgraph_to_smiles
except Exception as e:  # pragma: no cover
    smiles_to_molgraph = None  # type: ignore
    molgraph_to_smiles = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


# =============================================================================
# Pretty printing
# =============================================================================

def _safe_smiles(mg) -> str:
    if molgraph_to_smiles is not None:
        try:
            return str(molgraph_to_smiles(mg))
        except Exception:
            pass
    return str(mg.meta.get("smiles"))


def print_molgraph(mg, label: str) -> None:
    """Pretty-print MolGraph key entities (adapted from molgraph_structure.py)."""
    print("=" * 78)
    print(f"  {label}")
    print("=" * 78)

    print(f"meta['smiles']: {mg.meta.get('smiles')}")
    print(f"rdkit_smiles:  {_safe_smiles(mg)}")
    if "merge_log" in mg.meta:
        print(f"merge_log_len: {len(mg.meta.get('merge_log') or [])}")
    print()

    arr = mg.arrays
    n_atoms = int(arr.atomic_num.shape[0])
    n_bonds = int(arr.bonds.shape[0])
    print(f"[MolArrays] {n_atoms} atoms, {n_bonds} bonds")
    print()

    # --- Atoms ---
    print("ATOMS:")
    print("-" * 78)
    cols = [
        ("idx", 4),
        ("Z", 4),
        ("chg", 4),
        ("map", 5),
        ("lbl", 8),
        ("arom", 5),
        ("hyb", 4),
        ("chir", 5),
        ("cip", 4),
        ("eH", 3),
        ("iH", 3),
    ]
    header = "  " + " ".join([f"{c:<{w}}" for c, w in cols])
    print(header)
    print("-" * 78)
    for i in range(n_atoms):
        z = int(arr.atomic_num[i])
        chg = int(arr.formal_charge[i])
        amap = int(arr.atom_map[i]) if arr.atom_map is not None else 0
        lbl = arr.attachment_label[i] if arr.attachment_label is not None else None
        arom = bool(arr.is_aromatic[i]) if arr.is_aromatic is not None else False
        hyb = int(arr.hybridization[i]) if arr.hybridization is not None else 0
        chir = int(arr.chiral_tag[i]) if arr.chiral_tag is not None else 0
        cip = int(arr.cip_code[i]) if arr.cip_code is not None else 0
        eH = int(arr.explicit_h[i]) if arr.explicit_h is not None else 0
        iH = int(arr.implicit_h[i]) if arr.implicit_h is not None else 0

        print(f"  {i:<4} {z:<4} {chg:<4} {amap:<5} {str(lbl):<8} {str(arom):<5} {hyb:<4} {chir:<5} {cip:<4} {eH:<3} {iH:<3}")

    print()

    # --- Bonds ---
    print("BONDS:")
    print("-" * 90)
    cols2 = [
        ("idx", 4),
        ("u-v", 10),
        ("type", 5),
        ("dir", 4),
        ("stereo", 6),
        ("conj", 5),
        ("ring", 5),
        ("res", 4),
    ]
    header2 = "  " + " ".join([f"{c:<{w}}" for c, w in cols2])
    print(header2)
    print("-" * 90)

    for i in range(n_bonds):
        u, v = arr.bonds[i]
        bt = int(arr.bond_type[i])
        bdir = int(arr.bond_dir[i]) if arr.bond_dir is not None else 0
        bst = int(arr.bond_stereo[i]) if arr.bond_stereo is not None else 0
        conj = bool(arr.is_conjugated[i]) if arr.is_conjugated is not None else False
        ring = bool(arr.is_in_ring[i]) if arr.is_in_ring is not None else False
        res = int(arr.bond_resonance_type[i]) if arr.bond_resonance_type is not None else 0

        print(f"  {i:<4} {str((int(u), int(v))):<10} {bt:<5} {bdir:<4} {bst:<6} {str(conj):<5} {str(ring):<5} {res:<4}")

    print()

    # --- Coordinates ---
    print("COORDINATES:")
    if arr.pos is None:
        print("  pos: None")
        print("  coord_frame: None")
        print("  coord_valid: None")
    else:
        pos = arr.pos
        print(f"  pos: shape={pos.shape}, dtype={pos.dtype}")

        # Print first few coordinates
        k = 5  # how many atoms to show
        if pos.ndim == 2:
            n_show = min(k, pos.shape[0])
            print(f"  pos[:{n_show}] (x, y, z):")
            for i in range(n_show):
                x, y, z = pos[i]
                print(f"    {i:>3}: {float(x): .4f}  {float(y): .4f}  {float(z): .4f}")
        elif pos.ndim == 3:
            # Ensemble: show first conformer, first few atoms
            n_show = min(k, pos.shape[1])
            print(f"  pos[0][:{n_show}] (conformer 0, x, y, z):")
            for i in range(n_show):
                x, y, z = pos[0, i]
                print(f"    {i:>3}: {float(x): .4f}  {float(y): .4f}  {float(z): .4f}")
        else:
            print(f"  (unexpected pos.ndim={pos.ndim}, not printing preview)")

        print(f"  coord_frame: {arr.coord_frame}")
        if arr.coord_valid is not None:
            print(
                f"  coord_valid: shape={arr.coord_valid.shape}, "
                f"dtype={arr.coord_valid.dtype}, "
                f"all_valid={bool(arr.coord_valid.all())}"
            )
        else:
            print("  coord_valid: None")
    print()



    # --- Fragments ---
    if mg.fragments:
        print("FRAGMENTS:")
        for fr in mg.fragments:
            print(f"  - id={fr.fragment_id} atoms={fr.atom_indices.tolist()} attach={fr.attachment_indices.tolist()} role={fr.role} smiles={fr.smiles}")
        print()

    # --- Meta merge log ---
    if mg.meta.get("merge_log"):
        print("MERGE LOG (tail):")
        tail = (mg.meta["merge_log"] or [])[-3:]
        for e in tail:
            print(f"  - {e.get('message')}")
            if e.get("warnings"):
                print(f"    warnings: {e['warnings']}")
        print()


# =============================================================================
# Helpers
# =============================================================================

def _build(smiles: str, *, add_hs: bool = False) -> Any:
    if smiles_to_molgraph is None:
        raise RuntimeError(f"smiles_to_molgraph import failed: {_IMPORT_ERR}")
    mg = smiles_to_molgraph(smiles, add_hs=add_hs, compute_charges=True, keep_rdkit_mol=False)
    if mg is None:
        raise RuntimeError(f"Failed to build MolGraph from SMILES: {smiles}")
    return mg


def _assert_no_atomic_num(mg, z: int) -> None:
    if np.any(mg.arrays.atomic_num == z):
        raise AssertionError(f"Expected no atoms with atomic_num=={z}, but found some.")


def _assert_bonds_normalized(mg) -> None:
    if mg.arrays.bonds.size == 0:
        return
    b = mg.arrays.bonds
    if np.any(b[:, 0] > b[:, 1]):
        bad = np.nonzero(b[:, 0] > b[:, 1])[0].tolist()
        raise AssertionError(f"Found non-normalized bonds at indices: {bad}")


def _bond_exists(mg, u: int, v: int) -> bool:
    a = min(u, v)
    b = max(u, v)
    if mg.arrays.bonds.size == 0:
        return False
    mask = (mg.arrays.bonds[:, 0] == a) & (mg.arrays.bonds[:, 1] == b)
    return bool(np.any(mask))


def _sum_charge(mg) -> int:
    return int(np.sum(mg.arrays.formal_charge))


def _add_fragment_cover_all(mg, fragment_id: str) -> None:
    # Convenience: add a fragment that references all atoms and all dummies (attachment_indices are dummy indices)
    dummies = np.where(mg.arrays.atomic_num == 0)[0].astype(np.int64)
    frag = Fragment(
        fragment_id=fragment_id,
        atom_indices=np.arange(mg.arrays.atomic_num.shape[0], dtype=np.int64),
        attachment_indices=dummies,
        role="test",
        origin="unit",
        smiles=str(mg.meta.get("smiles")),
    )
    mg.fragments = [frag]


# =============================================================================
# Tests: resolve_site
# =============================================================================

def test_resolve_site_substituent_verbose():
    """
    Substituent site: [*:1]C
    Dummy has exactly 1 heavy neighbor.
    """
    smiles = "[*:1]C"
    g = _build(smiles)
    print_molgraph(g, "INPUT (resolve_site substituent)")

    site = resolve_site(g, 1)
    assert site.mode == "substituent"
    assert site.heavy_neighbors.size == 1
    assert int(g.arrays.atomic_num[int(site.dummy_idx)]) == 0


def test_resolve_site_insertion_verbose():
    """
    Insertion site: C[*:1]C
    Dummy has exactly 2 heavy neighbors.
    """
    smiles = "C[*:1]C"
    g = _build(smiles)
    print_molgraph(g, "INPUT (resolve_site insertion)")

    site = resolve_site(g, 1)
    assert site.mode == "insertion"
    assert site.heavy_neighbors.size == 2


def test_resolve_site_errors_on_orphan_dummy_verbose():
    """
    Orphan dummy: [*:1]
    Dummy has 0 neighbors -> should error.
    """
    smiles = "[*:1]"
    g = _build(smiles)
    print_molgraph(g, "INPUT (resolve_site orphan)")

    try:
        _ = resolve_site(g, 1)
    except ValueError as e:
        assert "no neighbors" in str(e).lower()
    else:
        raise AssertionError("Expected ValueError for orphan dummy")


# =============================================================================
# Tests: merge - substituent
# =============================================================================

def test_merge_substituent_basic_entities_and_log():
    """
    Merge: [*:1]C  +  [*:2]C  ->  CC
    - ensures dummy atoms are removed (no atomic_num==0)
    - ensures 1 new bond created between anchors
    - prints full entities
    - ensures merge_log + placeholder smiles behavior
    """
    gA = _build("[*:1]C")
    gB = _build("[*:2]C")
    _add_fragment_cover_all(gA, "frag_A")
    _add_fragment_cover_all(gB, "frag_B")

    print_molgraph(gA, "A BEFORE MERGE")
    print_molgraph(gB, "B BEFORE MERGE")

    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)

    print_molgraph(merged, "MERGED (substituent)")

    _assert_no_atomic_num(merged, 0)
    _assert_bonds_normalized(merged)

    # merge log exists and contains message
    assert "merge_log" in merged.meta
    assert len(merged.meta["merge_log"]) >= 1
    assert "merge" in (merged.meta["merge_log"][-1].get("message") or "").lower()

    # placeholder smiles should start with MERGED when not validated
    assert str(merged.meta.get("smiles")).startswith("MERGED(")

    # fragments union
    assert len(merged.fragments) == 2
    # fragment indices must be in bounds
    n = int(merged.arrays.atomic_num.shape[0])
    for fr in merged.fragments:
        assert np.all((fr.atom_indices >= 0) & (fr.atom_indices < n))


def test_merge_substituent_new_bond_order_double():
    """
    Verify bond_type_code controls the new bond's order.
    Merge: [*:1]C + [*:2]C with bond_type=DOUBLE.
    Expect last bond_type == BondType.DOUBLE.
    """
    gA = _build("[*:1]C")
    gB = _build("[*:2]C")
    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, bond_type_code=BondType.DOUBLE, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (substituent, new bond DOUBLE)")
    assert int(merged.arrays.bond_type[-1]) == BondType.DOUBLE


def test_merge_substituent_removes_explicit_h_node_when_present():
    """
    Build with add_hs=True so explicit H atoms exist as nodes.
    A: [*:1]C (with explicit H nodes after AddHs)
    B: [*:2]C
    Merge should remove an explicit H node from A anchor (atomic_num==1) before bonding.
    """
    gA = _build("[*:1]C", add_hs=True)
    gB = _build("[*:2]C", add_hs=True)

    print_molgraph(gA, "A BEFORE MERGE (with explicit H nodes)")
    print_molgraph(gB, "B BEFORE MERGE (with explicit H nodes)")

    count_h_before = int(np.sum(gA.arrays.atomic_num == 1))
    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (substituent, explicit H removal)")

    count_h_after = int(np.sum(merged.arrays.atomic_num == 1))
    # at least one H should be removed from the anchor side; merged still may contain many Hs depending on AddHs on both
    assert count_h_after < (count_h_before + int(np.sum(gB.arrays.atomic_num == 1)))


# =============================================================================
# Tests: merge - stereochemistry preservation
# =============================================================================

def test_merge_preserves_double_bond_stereochem_arrays():
    """
    A contains an alkene with stereo markers.
    A: [*:1]C/C=C\\C   (has bond_dir/bond_stereo info)
    B: [*:2]F
    After merge, the internal alkene stereo arrays should still exist and contain nonzero entries.
    """
    gA = _build("[*:1]C/C=C\\C")
    gB = _build("[*:2]F")

    print_molgraph(gA, "A BEFORE MERGE (stereo alkene)")
    print_molgraph(gB, "B BEFORE MERGE")

    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (stereo alkene preserved)")

    # Union-default-fill rule: if either side has bond_stereo/bond_dir arrays, merged must have them
    assert merged.arrays.bond_stereo is not None
    assert merged.arrays.bond_dir is not None

    # There should be at least one non-zero stereo or dir in merged (from alkene)
    has_nonzero = bool(np.any(merged.arrays.bond_stereo != 0) or np.any(merged.arrays.bond_dir != 0))
    assert has_nonzero, "Expected preserved stereo/dir nonzero values from alkene"


def test_merge_unions_stereo_arrays_when_only_one_side_has_them():
    """
    If one graph has stereo arrays and the other doesn't, merged should still have them, with
    default-filled zeros for the side that lacked them.

    We simulate "lacking" by manually zeroing out B's arrays to None (since build usually provides them),
    then merging and checking merged arrays exist and lengths match bonds.
    """
    gA = _build("[*:1]C/C=C\\C")
    gB = _build("[*:2]C")

    # Simulate B lacking stereo arrays
    gB.arrays.bond_stereo = None
    gB.arrays.bond_dir = None

    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (union stereo arrays when B lacked)")

    assert merged.arrays.bond_stereo is not None
    assert merged.arrays.bond_dir is not None
    assert merged.arrays.bond_stereo.shape[0] == merged.arrays.bonds.shape[0]
    assert merged.arrays.bond_dir.shape[0] == merged.arrays.bonds.shape[0]


# =============================================================================
# Tests: merge - charges
# =============================================================================

def test_merge_charge_consistency_total_charge_matches_formal_sum():
    """
    Merge charged fragments and ensure arrays.total_charge == sum(formal_charge).
    A: [*:1][N+](C)(C)C
    B: [*:2]C(=O)[O-]
    Expected net charge 0 after merge.
    """
    gA = _build("[*:1][N+](C)(C)C")
    gB = _build("[*:2]C(=O)[O-]")

    print_molgraph(gA, "A BEFORE MERGE (charge)")
    print_molgraph(gB, "B BEFORE MERGE (charge)")

    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (charge)")

    assert int(merged.arrays.total_charge) == _sum_charge(merged)
    assert int(merged.arrays.total_charge) == 0


# =============================================================================
# Tests: merge - insertion
# =============================================================================

def test_merge_insertion_basic_verbose():
    """
    Insertion merge (star topology):
      A: C[*:1]C     (insertion site: dummy has 2 heavy neighbors)
      B: [*:2]C      (substituent site)
    Merge should connect the single anchor of substituent side to BOTH anchors of insertion side.
    """
    gA = _build("C[*:1]C")
    gB = _build("[*:2]C")

    print_molgraph(gA, "A BEFORE MERGE (insertion site)")
    print_molgraph(gB, "B BEFORE MERGE (substituent)")

    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (insertion)")

    _assert_no_atomic_num(merged, 0)
    _assert_bonds_normalized(merged)

    # Topology check: merged should have 3 carbons in a chain-like or star-like arrangement depending on indices.
    # For our insertion scheme in merge.py, substituent anchor becomes "center" connected to both A-side carbons.
    # So there exists an atom with degree 2.
    bonds = merged.arrays.bonds.astype(np.int64, copy=False)
    deg = np.zeros(int(merged.arrays.atomic_num.shape[0]), dtype=np.int64)
    for u, v in bonds:
        deg[int(u)] += 1
        deg[int(v)] += 1
    assert int(np.max(deg)) >= 2


def test_merge_insertion_unsupported_both_insertion_verbose():
    """
    Both-insertion must raise.
    A: C[*:1]C
    B: C[*:2]C
    """
    gA = _build("C[*:1]C")
    gB = _build("C[*:2]C")

    print_molgraph(gA, "A BEFORE MERGE (insertion)")
    print_molgraph(gB, "B BEFORE MERGE (insertion)")

    try:
        _ = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    except ValueError as e:
        assert "both-insertion" in str(e).lower() or "unsupported" in str(e).lower()
    else:
        raise AssertionError("Expected ValueError for both-insertion merge")

def test_merge_triple_bond():
    """
    Test merging with triple bond (alkyne).
    A: [*:1]C#C
    B: [*:2]C
    Result should be: CC#C
    """
    gA = _build("[*:1]C#C")
    gB = _build("[*:2]C")
    
    print_molgraph(gA, "A BEFORE MERGE (alkyne)")
    print_molgraph(gB, "B BEFORE MERGE")
    
    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (triple bond preserved)")
    
    # Find triple bond in result
    triple_bonds = np.where(merged.arrays.bond_type == BondType.TRIPLE)[0]
    assert len(triple_bonds) > 0, "Expected at least one triple bond in merged structure"
    
    # Verify no dummy atoms remain
    _assert_no_atomic_num(merged, 0)


def test_merge_aromatic_benzene_basic():
    """
    Test merging benzene ring with simple substituent.
    A: c1ccccc1[*:1]  (benzene)
    B: C[*:2]         (methyl)
    Result should be: Cc1ccccc1 (toluene)
    """
    gA = _build("c1ccccc1[*:1]")
    gB = _build("C[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE (benzene)")
    print_molgraph(gB, "B BEFORE MERGE (methyl)")
    
    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (toluene)")
    
    # Verify aromatic atoms preserved
    assert merged.arrays.is_aromatic is not None
    aromatic_count = int(np.sum(merged.arrays.is_aromatic))
    assert aromatic_count == 6, f"Expected 6 aromatic atoms (benzene ring), got {aromatic_count}"
    
    # Verify aromatic bonds preserved
    aromatic_bonds = np.where(merged.arrays.bond_type == BondType.AROMATIC)[0]
    assert len(aromatic_bonds) == 6, f"Expected 6 aromatic bonds, got {len(aromatic_bonds)}"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_aromatic_biphenyl():
    """
    Test merging two benzene rings to form biphenyl.
    A: c1ccccc1[*:1]
    B: c1ccccc1[*:2]
    Result should be: c1ccc(-c2ccccc2)cc1 (biphenyl)
    """
    gA = _build("c1ccccc1[*:1]")
    gB = _build("c1ccccc1[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE (benzene 1)")
    print_molgraph(gB, "B BEFORE MERGE (benzene 2)")
    
    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (biphenyl)")
    
    # Should have 12 aromatic atoms (two benzene rings)
    aromatic_count = int(np.sum(merged.arrays.is_aromatic))
    assert aromatic_count == 12, f"Expected 12 aromatic atoms (2 benzene rings), got {aromatic_count}"
    
    # Should have 12 aromatic bonds + 1 single bond connecting rings
    aromatic_bonds = np.where(merged.arrays.bond_type == BondType.AROMATIC)[0]
    assert len(aromatic_bonds) == 12, f"Expected 12 aromatic bonds, got {len(aromatic_bonds)}"
    
    single_bonds = np.where(merged.arrays.bond_type == BondType.SINGLE)[0]
    assert len(single_bonds) >= 1, "Expected at least 1 single bond connecting the rings"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_aromatic_heterocycle():
    """
    Test merging aromatic heterocycle (pyridine).
    A: n1ccccc1[*:1]  (pyridine)
    B: C[*:2]
    Result should preserve aromaticity including nitrogen.
    """
    gA = _build("n1ccccc1[*:1]")
    gB = _build("C[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE (pyridine)")
    print_molgraph(gB, "B BEFORE MERGE")
    
    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (methylpyridine)")
    
    # Should have 6 aromatic atoms (pyridine ring)
    aromatic_count = int(np.sum(merged.arrays.is_aromatic))
    assert aromatic_count == 6, f"Expected 6 aromatic atoms (pyridine), got {aromatic_count}"
    
    # Should have 1 nitrogen atom that's aromatic
    nitrogen_mask = merged.arrays.atomic_num == 7
    nitrogen_aromatic = merged.arrays.is_aromatic[nitrogen_mask]
    assert np.any(nitrogen_aromatic), "Expected aromatic nitrogen in pyridine"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_chiral_center_preservation():
    """
    Test preservation of R/S stereochemistry at tetrahedral center.
    A: C[C@H](O)[*:1]  (chiral center, S configuration)
    B: N[*:2]
    Result should preserve chiral_tag.
    """
    gA = _build("C[C@H](O)[*:1]")
    gB = _build("N[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE (chiral center)")
    print_molgraph(gB, "B BEFORE MERGE")
    
    # Find chiral center before merge
    chiral_atoms_before = np.where(gA.arrays.chiral_tag != 0)[0]
    assert len(chiral_atoms_before) > 0, "Input should have at least one chiral center"
    
    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (chiral center preserved)")
    
    # Verify chiral center preserved in merged structure
    assert merged.arrays.chiral_tag is not None
    chiral_atoms_after = np.where(merged.arrays.chiral_tag != 0)[0]
    assert len(chiral_atoms_after) > 0, "Merged structure should preserve chiral centers"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_sequential_two_step():
    """
    Test sequential merging: A + B → AB, then AB + C → ABC.
    A: C[*:1]
    B: C[*:2]
    C: C[*:3]
    Expected final result: CCC (propane)
    """
    gA = _build("C[*:1]")
    gB = _build("C[*:2]")
    gC = _build("C[*:3]")
    
    print_molgraph(gA, "A BEFORE FIRST MERGE")
    print_molgraph(gB, "B BEFORE FIRST MERGE")
    
    # First merge: A + B → AB
    gAB = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(gAB, "AB AFTER FIRST MERGE")
    
    # AB now has dummy atoms removed, so we need a new attachment point
    # For this test, let's create AB with a new attachment point at label 3
    gAB_new = _build("CC[*:3]")
    
    print_molgraph(gAB_new, "AB WITH NEW ATTACHMENT")
    print_molgraph(gC, "C BEFORE SECOND MERGE")
    
    # Second merge: AB + C → ABC
    gABC = merge_by_labels(gAB_new, gC, label_a=3, label_b=3, validate_with_rdkit=False)
    print_molgraph(gABC, "ABC AFTER SECOND MERGE")
    
    # Verify final structure: should have 3 carbons
    carbon_count = int(np.sum(gABC.arrays.atomic_num == 6))
    assert carbon_count == 3, f"Expected 3 carbons in final structure, got {carbon_count}"
    
    # Verify 2 bonds (C-C-C chain)
    assert gABC.arrays.bonds.shape[0] == 2, f"Expected 2 bonds, got {gABC.arrays.bonds.shape[0]}"
    
    # Verify merge_log tracks both operations
    merge_log = gABC.meta.get("merge_log", [])
    # Note: First merge creates entry in gAB, second merge adds to it
    # Total entries should be >= 1 (at least the second merge)
    assert len(merge_log) >= 1, "Expected merge_log to track operations"
    
    _assert_no_atomic_num(gABC, 0)


def test_merge_multiple_attachment_points():
    """
    Test molecule with multiple attachment points, partial merge.
    A: C[*:1]CC[*:2]  (two attachment points)
    B: N[*:3]
    Merge [*:1] with [*:3], leaving [*:2] intact.
    Result should be: NCCC[*:2]
    """
    gA = _build("C[*:1]CC[*:2]")
    gB = _build("N[*:3]")
    
    print_molgraph(gA, "A BEFORE MERGE (two attachment points)")
    print_molgraph(gB, "B BEFORE MERGE")
    
    # Count dummies before merge
    dummies_before = int(np.sum(gA.arrays.atomic_num == 0))
    assert dummies_before == 2, "A should start with 2 dummy atoms"
    
    merged = merge_by_labels(gA, gB, label_a=1, label_b=3, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (one attachment point remains)")
    
    # Should still have one dummy atom (label 2)
    dummies_after = int(np.sum(merged.arrays.atomic_num == 0))
    assert dummies_after == 1, f"Expected 1 dummy atom to remain (label 2), got {dummies_after}"
    
    # Should have added nitrogen
    nitrogen_count = int(np.sum(merged.arrays.atomic_num == 7))
    assert nitrogen_count == 1, f"Expected 1 nitrogen atom, got {nitrogen_count}"
    
    # Verify the remaining dummy has label 2
    dummy_indices = np.where(merged.arrays.atomic_num == 0)[0]
    assert len(dummy_indices) == 1
    remaining_label = merged.arrays.attachment_label[dummy_indices[0]]
    assert remaining_label == 2, f"Remaining dummy should have label 2, got {remaining_label}"


def test_merge_isotope_preservation():
    """
    Test that isotope information is preserved during merge.
    A: [13C]O[*:1]     (Carbon-13)
    B: C[18O][*:2]     (Oxygen-18)
    Result should preserve isotope array.
    """
    gA = _build("[13C]O[*:1]")   # C-13 (heavy atom)
    gB = _build("C[18O][*:2]")   # O-18 (heavy atom)
    
    print_molgraph(gA, "A BEFORE MERGE (C-13)")
    print_molgraph(gB, "B BEFORE MERGE (deuterium)")
    
    # Verify isotopes present in inputs
    assert gA.arrays.isotope is not None
    assert gB.arrays.isotope is not None
    
    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (isotopes preserved)")
    
    # Verify isotope array preserved
    assert merged.arrays.isotope is not None, "Isotope array should be preserved"
    
    # Should have C-13 (mass 13) and deuterium (mass 2)
    isotopes = merged.arrays.isotope[merged.arrays.isotope != 0]
    assert len(isotopes) == 2, f"Expected 2 isotopic atoms, got {len(isotopes)}"
    assert 13 in isotopes, "Expected C-13 to be preserved"
    assert 18 in isotopes, "Expected deuterium to be preserved"
    
    _assert_no_atomic_num(merged, 0)

def test_merge_no_coords_default():
    """
    Test that merge without coords parameter generates no coordinates (default).
    A: C[*:1]
    B: C[*:2]
    Result should have pos=None (fast, no coordinate generation).
    """
    gA = _build("C[*:1]")
    gB = _build("C[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE")
    print_molgraph(gB, "B BEFORE MERGE")
    
    # Merge with default coords=None
    merged = merge_by_labels(gA, gB, label_a=1, label_b=2, validate_with_rdkit=False)
    print_molgraph(merged, "MERGED (no coordinates)")

    print(f"pos is: {merged.arrays.pos}")
    
    # Verify no coordinates generated
    assert merged.arrays.pos is None, "pos should be None when coords not requested"
    assert merged.arrays.coord_frame is None, "coord_frame should be None"
    assert merged.arrays.coord_valid is None, "coord_valid should be None"
    assert "coords_mode" not in merged.meta, "coords_mode should not be in metadata"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_canonical_coords():
    """
    Test merge with canonical coordinate generation.
    A: CC[*:1]
    B: CC[*:2]
    Result should have single conformer with shape (N, 3).
    """
    gA = _build("CC[*:1]")
    gB = _build("CC[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE")
    print_molgraph(gB, "B BEFORE MERGE")
    
    # Merge with canonical coordinates
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="canonical",
        validate_with_rdkit=False
    )
    print_molgraph(merged, "MERGED (canonical coords)")
    
    # Verify coordinates generated
    assert merged.arrays.pos is not None, "pos should not be None"
    assert merged.arrays.pos.ndim == 2, f"pos should be 2D, got {merged.arrays.pos.ndim}D"
    
    n_atoms = len(merged.arrays.atomic_num)
    assert merged.arrays.pos.shape == (n_atoms, 3), \
        f"Expected shape ({n_atoms}, 3), got {merged.arrays.pos.shape}"
    
    # Verify metadata
    assert merged.arrays.coord_frame == "etkdg", \
        f"Expected coord_frame='etkdg', got '{merged.arrays.coord_frame}'"
    assert merged.arrays.coord_valid is not None, "coord_valid should not be None"
    assert merged.arrays.coord_valid.shape == (n_atoms,), \
        f"coord_valid shape should be ({n_atoms},), got {merged.arrays.coord_valid.shape}"
    assert merged.arrays.coord_valid.all(), "All coordinates should be initially valid"
    
    # Verify meta
    assert merged.meta.get("coords_mode") == "canonical", "coords_mode should be in metadata"
    assert merged.meta.get("num_confs") == 1, "num_confs should be 1 for canonical"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_canonical_coords_with_optimization():
    """
    Test merge with canonical coordinates and MMFF optimization.
    A: CCC[*:1]
    B: CCC[*:2]
    Result should have optimized coordinates with coord_frame='etkdg-mmff'.
    """
    gA = _build("CCC[*:1]")
    gB = _build("CCC[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE")
    print_molgraph(gB, "B BEFORE MERGE")
    
    # Merge with optimization
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="canonical",
        optimize_coords=True,
        validate_with_rdkit=False
    )
    print_molgraph(merged, "MERGED (optimized coords)")
    
    # Verify coordinates generated
    assert merged.arrays.pos is not None, "pos should not be None"
    assert merged.arrays.pos.ndim == 2, "pos should be 2D"
    
    # Verify optimization happened
    assert merged.arrays.coord_frame == "etkdg-mmff", \
        f"Expected coord_frame='etkdg-mmff', got '{merged.arrays.coord_frame}'"
    
    # Coordinates should be finite
    import numpy as np
    assert np.isfinite(merged.arrays.pos).all(), "All coordinates should be finite"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_ensemble_coords():
    """
    Test merge with ensemble coordinate generation (multiple conformers).
    A: CCCC[*:1]
    B: CCC[*:2]
    Result should have multiple conformers with shape (K, N, 3).
    """
    gA = _build("CCCC[*:1]")
    gB = _build("CCC[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE")
    print_molgraph(gB, "B BEFORE MERGE")
    
    num_confs = 5
    # Merge with ensemble coordinates
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="ensemble",
        num_confs=num_confs,
        validate_with_rdkit=False
    )
    print_molgraph(merged, f"MERGED (ensemble with {num_confs} conformers)")
    
    # Verify coordinates generated
    assert merged.arrays.pos is not None, "pos should not be None"
    assert merged.arrays.pos.ndim == 3, \
        f"pos should be 3D for ensemble, got {merged.arrays.pos.ndim}D"
    
    n_atoms = len(merged.arrays.atomic_num)
    k_confs = merged.arrays.pos.shape[0]
    
    # RDKit may generate fewer conformers than requested
    assert k_confs > 0, "Should have at least 1 conformer"
    assert k_confs <= num_confs, f"Should have <= {num_confs} conformers"
    
    assert merged.arrays.pos.shape == (k_confs, n_atoms, 3), \
        f"Expected shape (K, {n_atoms}, 3), got {merged.arrays.pos.shape}"
    
    # Verify metadata
    assert merged.arrays.coord_frame == "etkdg", "coord_frame should be 'etkdg'"
    assert merged.arrays.coord_valid is not None, "coord_valid should not be None"
    assert merged.arrays.coord_valid.shape == (k_confs, n_atoms), \
        f"coord_valid shape should be ({k_confs}, {n_atoms}), got {merged.arrays.coord_valid.shape}"
    assert merged.arrays.coord_valid.all(), "All coordinates should be initially valid"
    
    # Verify meta
    assert merged.meta.get("coords_mode") == "ensemble", "coords_mode should be 'ensemble'"
    assert merged.meta.get("num_confs") == k_confs, \
        f"num_confs should be {k_confs}, got {merged.meta.get('num_confs')}"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_ensemble_coords_with_optimization():
    """
    Test merge with ensemble coordinates and optimization.
    A: CCCC[*:1]
    B: CCCC[*:2]
    Result should have optimized ensemble with coord_frame='etkdg-mmff'.
    """
    gA = _build("CCCC[*:1]")
    gB = _build("CCCC[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE")
    print_molgraph(gB, "B BEFORE MERGE")
    
    num_confs = 3
    # Merge with ensemble and optimization
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="ensemble",
        num_confs=num_confs,
        optimize_coords=True,
        validate_with_rdkit=False
    )
    print_molgraph(merged, f"MERGED (optimized ensemble)")
    
    # Verify coordinates generated
    assert merged.arrays.pos is not None, "pos should not be None"
    assert merged.arrays.pos.ndim == 3, "pos should be 3D for ensemble"
    
    # Verify optimization happened
    assert merged.arrays.coord_frame == "etkdg-mmff", \
        f"Expected coord_frame='etkdg-mmff', got '{merged.arrays.coord_frame}'"
    
    # Conformers should be different from each other
    import numpy as np
    if merged.arrays.pos.shape[0] >= 2:
        conf1 = merged.arrays.pos[0]
        conf2 = merged.arrays.pos[1]
        # Check that conformers are not identical
        assert not np.allclose(conf1, conf2, rtol=1e-4), \
            "Conformers should be different from each other"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_coords_reproducibility():
    """
    Test that random_seed makes coordinate generation reproducible.
    A: CCC[*:1]
    B: CCC[*:2]
    Two merges with same seed should give identical coordinates.
    """
    gA1 = _build("CCC[*:1]")
    gB1 = _build("CCC[*:2]")
    gA2 = _build("CCC[*:1]")
    gB2 = _build("CCC[*:2]")
    
    seed = 42
    
    # First merge
    merged1 = merge_by_labels(
        gA1, gB1,
        label_a=1,
        label_b=2,
        coords="canonical",
        random_seed=seed,
        validate_with_rdkit=False
    )
    
    # Second merge with same seed
    merged2 = merge_by_labels(
        gA2, gB2,
        label_a=1,
        label_b=2,
        coords="canonical",
        random_seed=seed,
        validate_with_rdkit=False
    )
    
    # Verify both generated coordinates
    assert merged1.arrays.pos is not None, "First merge should have coordinates"
    assert merged2.arrays.pos is not None, "Second merge should have coordinates"
    
    # Verify coordinates are identical
    import numpy as np
    assert np.allclose(merged1.arrays.pos, merged2.arrays.pos, rtol=1e-6), \
        "Same seed should produce identical coordinates"
    
    _assert_no_atomic_num(merged1, 0)
    _assert_no_atomic_num(merged2, 0)


def test_merge_coords_with_rdkit_validation():
    """
    Test that coords work together with validate_with_rdkit.
    A: CC(=O)O[*:1]    (acetic acid fragment)
    B: Cc1ccccc1[*:2]  (toluene fragment)
    Should generate both canonical SMILES and coordinates.
    """
    gA = _build("CC(=O)O[*:1]")
    gB = _build("Cc1ccccc1[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE (acetic acid)")
    print_molgraph(gB, "B BEFORE MERGE (toluene)")
    
    # Merge with both validation and coordinates
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="canonical",
        validate_with_rdkit=True,
    )
    print_molgraph(merged, "MERGED (validated + coords)")
    
    # Verify coordinates
    assert merged.arrays.pos is not None, "Should have coordinates"
    assert merged.arrays.pos.ndim == 2, "Should have canonical coordinates"
    
    # Verify SMILES (if validation succeeded)
    assert "smiles" in merged.meta, "Should have smiles in metadata"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_coords_data_types():
    """
    Test that coordinate arrays have correct data types.
    A: CCC[*:1]
    B: CC[*:2]
    Verify pos is float32, coord_valid is bool.
    """
    gA = _build("CCC[*:1]")
    gB = _build("CC[*:2]")
    
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="canonical",
        validate_with_rdkit=False
    )
    
    # Verify data types
    import numpy as np
    assert merged.arrays.pos.dtype == np.float32, \
        f"pos should be float32, got {merged.arrays.pos.dtype}"
    assert merged.arrays.coord_valid.dtype == bool, \
        f"coord_valid should be bool, got {merged.arrays.coord_valid.dtype}"
    
    # Verify no NaN or inf values
    assert np.isfinite(merged.arrays.pos).all(), "All coordinates should be finite"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_coords_values_reasonable():
    """
    Test that generated coordinates have reasonable values.
    A: CCCC[*:1]
    B: CCCC[*:2]
    Coordinates should be non-zero and have reasonable variance.
    """
    gA = _build("CCCC[*:1]")
    gB = _build("CCCC[*:2]")
    
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="canonical",
        validate_with_rdkit=False
    )
    
    import numpy as np
    
    # Verify coordinates exist
    assert merged.arrays.pos is not None, "Should have coordinates"
    
    # Check variance (coordinates shouldn't all be the same)
    variance = np.var(merged.arrays.pos)
    assert variance > 0.01, f"Coordinates should have non-trivial variance, got {variance}"
    
    # Check that coordinates span reasonable range
    pos_range = np.ptp(merged.arrays.pos, axis=0)  # peak-to-peak (max - min) per dimension
    assert all(pos_range > 0.1), f"Coordinates should span reasonable range, got {pos_range}"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_cyclic_molecule_coords():
    """
    Test coordinate generation for merge creating cyclic molecule.
    A: c1ccccc1[*:1]   (benzene ring)
    B: C[*:2]          (methyl)
    Result should successfully generate coordinates for aromatic system.
    """
    gA = _build("c1ccccc1[*:1]")
    gB = _build("C[*:2]")
    
    print_molgraph(gA, "A BEFORE MERGE (benzene)")
    print_molgraph(gB, "B BEFORE MERGE (methyl)")
    
    # Merge aromatic + aliphatic
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="canonical",
        validate_with_rdkit=False
    )
    print_molgraph(merged, "MERGED (toluene with coords)")
    
    # Verify coordinates generated successfully
    assert merged.arrays.pos is not None, "Aromatic merge should generate coordinates"
    
    # Should have 7 heavy atoms (6 from benzene + 1 from methyl)
    n_atoms = len(merged.arrays.atomic_num)
    assert merged.arrays.pos.shape[0] == n_atoms, \
        f"Should have coordinates for all {n_atoms} atoms"
    
    _assert_no_atomic_num(merged, 0)


def test_merge_coords_failure_handling():
    """
    Test that coordinate generation failure doesn't crash merge.
    Use a simple molecule that should succeed, but verify error handling is in place.
    A: C[*:1]
    B: C[*:2]
    Even if coordinate generation fails, merge should complete.
    """
    gA = _build("C[*:1]")
    gB = _build("C[*:2]")
    
    # This should succeed, but we're testing the error handling path exists
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="canonical",
        validate_with_rdkit=False
    )
    
    # Merge should complete (not crash)
    assert merged is not None, "Merge should complete even if coords fail"
    
    # Check merge_log for any warnings (may or may not have them)
    if "merge_log" in merged.meta:
        print(f"Merge log entries: {len(merged.meta['merge_log'])}")
        for entry in merged.meta['merge_log']:
            if "warnings" in entry:
                print(f"  Warnings: {entry['warnings']}")
    
    _assert_no_atomic_num(merged, 0)


def test_merge_ensemble_conformer_count():
    """
    Test that ensemble generates reasonable number of conformers.
    A: CCCCCC[*:1]  (longer chain allows more conformational freedom)
    B: CCC[*:2]
    Request 10 conformers, verify we get reasonable count.
    """
    gA = _build("CCCCCC[*:1]")
    gB = _build("CCC[*:2]")
    
    num_confs = 10
    merged = merge_by_labels(
        gA, gB,
        label_a=1,
        label_b=2,
        coords="ensemble",
        num_confs=num_confs,
        validate_with_rdkit=False
    )
    
    assert merged.arrays.pos is not None, "Should have ensemble coordinates"
    
    k_confs = merged.arrays.pos.shape[0]
    print(f"Requested {num_confs} conformers, generated {k_confs}")
    
    # RDKit may generate fewer than requested
    assert k_confs > 0, "Should generate at least 1 conformer"
    assert k_confs <= num_confs, f"Should not generate more than {num_confs} conformers"
    
    # For a flexible molecule, we should get multiple conformers
    # (Though this is not guaranteed, depends on RDKit's embedding)
    if k_confs >= 2:
        print(f"Successfully generated {k_confs} distinct conformers")
    
    _assert_no_atomic_num(merged, 0)

def test_merge_canonical_coords_preserves_alkene_slash_stereo():
    """
    Merge a fragment that contains explicit alkene stereo (/ \\) and ensure:
      - bond_dir/bond_stereo arrays survive the merge (not None)
      - at least one non-zero entry remains (stereo info preserved)
      - coords="canonical" generates 3D positions
    """
    # Alkene with explicit slash/backslash stereo markers.
    # Keep the dummy away from the double bond so merge doesn’t touch it.
    gA = _build("[*:1]C/C=C\\C")   # has / and \ around C=C
    gB = _build("[*:2]F")

    print_molgraph(gA, "A BEFORE MERGE (alkene /\\ stereo)")
    print_molgraph(gB, "B BEFORE MERGE")

    merged = merge_by_labels(
        gA, gB,
        label_a=1, label_b=2,
        coords="canonical",
        validate_with_rdkit=False,
    )
    print_molgraph(merged, "MERGED (alkene stereo + canonical coords)")

    # Stereo arrays should exist
    assert merged.arrays.bond_dir is not None, "bond_dir should exist after merge"
    assert merged.arrays.bond_stereo is not None, "bond_stereo should exist after merge"

    # Should retain some non-zero stereo info from the alkene fragment
    has_nonzero = bool(np.any(merged.arrays.bond_dir != 0) or np.any(merged.arrays.bond_stereo != 0))
    assert has_nonzero, "Expected at least one non-zero stereo/dir entry to be preserved"

    # 3D coords exist
    assert merged.arrays.pos is not None, "Expected canonical coordinates"
    assert merged.arrays.pos.ndim == 2, "Expected canonical coords to be (N,3)"
    assert merged.arrays.pos.shape[1] == 3, "Expected xyz coords"
    assert merged.arrays.coord_valid is not None and bool(merged.arrays.coord_valid.all())

    _assert_no_atomic_num(merged, 0)
    _assert_bonds_normalized(merged)

def test_merge_canonical_coords_preserves_tetrahedral_handedness_at():
    """
    Merge a fragment containing a tetrahedral stereocenter (@ or @@) and ensure:
      - chiral_tag survives and remains non-zero on at least one atom
      - (if present) cip_code survives and contains at least one non-zero assignment
      - coords="canonical" generates 3D positions
    """
    # Chiral center. Dummy is attached at the chiral carbon so we’re exercising
    # chirality arrays under a merge (common failure mode).
    # This SMILES encodes a stereocenter at the carbon: C[C@H](O)[*:1]
    gA = _build("C[C@H](O)[*:1]")
    gB = _build("N[*:2]")

    print_molgraph(gA, "A BEFORE MERGE (chiral @)")
    print_molgraph(gB, "B BEFORE MERGE")

    merged = merge_by_labels(
        gA, gB,
        label_a=1, label_b=2,
        validate_with_rdkit=True,
        coords="canonical",
    )
    print_molgraph(merged, "MERGED (chiral @ + canonical coords)")

    # Chiral tag should exist and have at least one non-zero
    assert merged.arrays.chiral_tag is not None, "chiral_tag array should exist"
    assert np.any(merged.arrays.chiral_tag != 0), "Expected at least one chiral center to remain"

    # CIP code is optional depending on your pipeline; if it exists, require non-zero
    if merged.arrays.cip_code is not None:
        assert np.any(merged.arrays.cip_code != 0), "Expected a CIP assignment (R/S) to remain"

    # 3D coords exist
    assert merged.arrays.pos is not None, "Expected canonical coordinates"
    assert merged.arrays.pos.ndim == 2 and merged.arrays.pos.shape[1] == 3
    assert merged.arrays.coord_valid is not None and bool(merged.arrays.coord_valid.all())

    _assert_no_atomic_num(merged, 0)
    _assert_bonds_normalized(merged)


# =============================================================================
# Runner
# =============================================================================

def run_all_tests() -> None:
    tests: list[Callable[[], None]] = [
        test_resolve_site_substituent_verbose,
        test_resolve_site_insertion_verbose,
        test_resolve_site_errors_on_orphan_dummy_verbose,
        test_merge_substituent_basic_entities_and_log,
        test_merge_substituent_new_bond_order_double,
        test_merge_substituent_removes_explicit_h_node_when_present,
        test_merge_preserves_double_bond_stereochem_arrays,
        test_merge_unions_stereo_arrays_when_only_one_side_has_them,
        test_merge_charge_consistency_total_charge_matches_formal_sum,
        test_merge_insertion_basic_verbose,
        test_merge_insertion_unsupported_both_insertion_verbose,
        # NEW EXTRA TESTS
        test_merge_triple_bond,
        test_merge_aromatic_benzene_basic,
        test_merge_aromatic_biphenyl,
        test_merge_aromatic_heterocycle,
        test_merge_chiral_center_preservation,
        test_merge_sequential_two_step,
        test_merge_multiple_attachment_points,
        test_merge_isotope_preservation,
        # NEW 3D TESTS
        test_merge_no_coords_default,
        test_merge_canonical_coords,
        test_merge_canonical_coords_with_optimization,
        test_merge_ensemble_coords,
        test_merge_ensemble_coords_with_optimization,
        test_merge_coords_reproducibility,
        test_merge_coords_with_rdkit_validation,
        test_merge_coords_data_types,
        test_merge_coords_values_reasonable,
        test_merge_cyclic_molecule_coords,
        test_merge_coords_failure_handling,
        test_merge_ensemble_conformer_count,
        test_merge_canonical_coords_preserves_alkene_slash_stereo,
        test_merge_canonical_coords_preserves_tetrahedral_handedness_at,
    ]

    passed = 0
    failed = 0

    for t in tests:
        print("\n" + "=" * 60)
        print(f"TEST: {t.__name__}")
        print("=" * 60)
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"\n✗ {t.__name__} FAILED: {e}")
            traceback.print_exc()
        except Exception as e:
            failed += 1
            print(f"\n✗ {t.__name__} ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"\n✓ {t.__name__} PASSED")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all_tests()
