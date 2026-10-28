# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

############################################################
# OPS MERGE TESTS
############################################################

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

from chem.build.create_molgraph import smiles_to_molgraph
from chem.build.molgraph_to_mol import molgraph_to_smiles
from chem.ops.base import apply_transform
from chem.ops.merge import MergeByLabels


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _merge(gA, gB, la, lb, coords=None):
    op = MergeByLabels(
        g_b=gB,
        label_a=la,
        label_b=lb,
        validate_with_rdkit=True,
        coords=coords,
    )
    g2, info = apply_transform(gA, op)
    return g2, info


# ---------------------------------------------------------
# Aromatic substitution
# ---------------------------------------------------------

def test_merge_aromatic_substitution():
    gA = smiles_to_molgraph("c1ccc([*:1])cc1")  # benzene attachment
    gB = smiles_to_molgraph("C[*:2]")           # methyl

    g2, info = _merge(gA, gB, 1, 2)

    smi = molgraph_to_smiles(g2)
    assert smi == "Cc1ccccc1", f"Expected toluene, got {smi}"


# ---------------------------------------------------------
# Alkene stereo preserved
# ---------------------------------------------------------

def test_merge_alkene_stereo_preserved():
    gA = smiles_to_molgraph("C/C=C\\C[*:1]")
    gB = smiles_to_molgraph("F[*:2]")

    g2, _ = _merge(gA, gB, 1, 2)

    smi = molgraph_to_smiles(g2)
    assert "/" in smi or "\\" in smi, "Alkene stereo lost"


# ---------------------------------------------------------
# Chiral center preserved
# ---------------------------------------------------------

def test_merge_chiral_center_preserved():
    gA = smiles_to_molgraph("C[C@H](O)[*:1]")
    gB = smiles_to_molgraph("N[*:2]")

    g2, _ = _merge(gA, gB, 1, 2)

    smi = molgraph_to_smiles(g2)
    assert "@" in smi, f"Chirality lost: {smi}"


# ---------------------------------------------------------
# Oxygen insertion-style merge
# ---------------------------------------------------------

def test_merge_insertion_with_oxygen_anchor():
    gA = smiles_to_molgraph("CC([*:1])C")
    gB = smiles_to_molgraph("O([*:2])")

    g2, _ = _merge(gA, gB, 1, 2)

    smi = molgraph_to_smiles(g2)
    assert "O" in smi, "Oxygen insertion failed"


# ---------------------------------------------------------
# Charged merge
# ---------------------------------------------------------

def test_merge_charged_fragments():
    gA = smiles_to_molgraph("[NH3+][*:1]")
    gB = smiles_to_molgraph("[O-][*:2]")

    g2, _ = _merge(gA, gB, 1, 2)

    smi = molgraph_to_smiles(g2)
    assert "+" in smi or "-" in smi, "Charge lost"


# ---------------------------------------------------------
# Polyaromatic merge
# ---------------------------------------------------------

def test_merge_polyaromatic_system():
    gA = smiles_to_molgraph("c1ccc([*:1])cc1")
    gB = smiles_to_molgraph("c1cc([*:2])ccc1")

    g2, _ = _merge(gA, gB, 1, 2)

    smi = molgraph_to_smiles(g2)
    assert "c1" in smi, "Aromatic merge failed"


# ---------------------------------------------------------
# Enantiomer difference test
# ---------------------------------------------------------

def test_merge_chiral_vs_enantiomer_difference():
    gA = smiles_to_molgraph("C[C@H](O)[*:1]")
    gB = smiles_to_molgraph("C[C@@H](O)[*:1]")
    gAttach = smiles_to_molgraph("N[*:2]")

    g2a, _ = _merge(gA, gAttach, 1, 2)
    g2b, _ = _merge(gB, gAttach, 1, 2)

    smiA = molgraph_to_smiles(g2a)
    smiB = molgraph_to_smiles(g2b)

    assert smiA != smiB, "Enantiomers collapsed after merge"


# ---------------------------------------------------------
# 3D coords test
# ---------------------------------------------------------

def test_merge_generates_coords():
    gA = smiles_to_molgraph("c1ccc([*:1])cc1")
    gB = smiles_to_molgraph("C[*:2]")

    g2, _ = _merge(gA, gB, 1, 2, coords="canonical")

    assert g2.arrays.pos is not None
    assert g2.arrays.coord_frame is not None
    assert g2.arrays.coord_valid is not None


# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------

def run_all_tests():
    tests = [
        test_merge_aromatic_substitution,
        test_merge_alkene_stereo_preserved,
        test_merge_chiral_center_preserved,
        test_merge_insertion_with_oxygen_anchor,
        test_merge_charged_fragments,
        test_merge_polyaromatic_system,
        test_merge_chiral_vs_enantiomer_difference,
        test_merge_generates_coords,
    ]

    passed = 0
    failed = 0

    print("\n############################################################")
    print("# RUNNING ALL MERGE TESTS")
    print("############################################################\n")

    for t in tests:
        print(f"TEST: {t.__name__}")
        try:
            t()
            print(f"✓ {t.__name__} PASSED\n")
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__} FAILED: {e}\n")
            failed += 1

    print("############################################################")
    print(f"# RESULTS: {passed} passed, {failed} failed")
    print("############################################################")


if __name__ == "__main__":
    run_all_tests()
