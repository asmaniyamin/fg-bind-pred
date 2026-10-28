# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/ops/testing/test_ops_canonicalize.py
from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import traceback
import numpy as np

from chem.build.create_molgraph import smiles_to_molgraph
from chem.build.molgraph_to_mol import molgraph_to_smiles

from chem.ops.base import apply_transform
from chem.ops.atoms import PermuteAtoms
from core.molgraph.ops.canonicalize import Canonicalize


def _smi(mg) -> str:
    s = molgraph_to_smiles(mg, canonical=True, isomeric=True)
    return str(s) if s is not None else "<SMILES:None>"


def test_ops_canonicalize_roundtrip_invariant() -> None:
    # include aromatic + alkene stereo + chiral
    smi = "C/C=C\\C[C@H](O)c1ccccc1"
    g = smiles_to_molgraph(smi, coords="canonical", optimize_coords=False)
    assert g is not None

    # permute atoms (reverse) to break any implicit ordering
    n = int(g.arrays.atomic_num.shape[0])
    new_to_old = np.arange(n, dtype=np.int64)[::-1]
    g_perm, info_p = apply_transform(g, PermuteAtoms(new_to_old=new_to_old))
    assert g_perm is not None
    assert info_p.old_to_new is not None and info_p.new_to_old is not None
    assert int(info_p.old_to_new.shape[0]) == n

    # canonicalize (should produce deterministic order)
    g_can, info = apply_transform(g_perm, Canonicalize(iters=4, stereo=True))
    assert g_can is not None
    assert info.old_to_new is not None and info.new_to_old is not None
    assert int(info.old_to_new.shape[0]) == int(g_perm.arrays.atomic_num.shape[0])
    assert int(info.new_to_old.shape[0]) == int(g_perm.arrays.atomic_num.shape[0])
    assert info.bond_perm is not None
    assert int(info.bond_perm.shape[0]) == int(g_perm.arrays.bonds.shape[0])

    # canonicalize original too; canonical SMILES should match
    g_can0, info0 = apply_transform(g, Canonicalize(iters=4, stereo=True))
    assert _smi(g_can) == _smi(g_can0), "Canonicalize should erase atom-order differences"


def test_ops_canonicalize_preserves_optional_coords_none() -> None:
    smi = "CCO"
    g = smiles_to_molgraph(smi, coords=None)  # intentionally no coords
    assert g is not None
    assert g.arrays.pos is None

    g2, info = apply_transform(g, Canonicalize(iters=4, stereo=True))
    assert g2 is not None
    # canonicalization should NOT create coords
    assert g2.arrays.pos is None, "Canonicalize should not generate coords if missing"
    assert _smi(g2) == _smi(g), "Canonicalize should not change chemistry"


def run_all_tests() -> None:
    tests = [
        test_ops_canonicalize_roundtrip_invariant,
        test_ops_canonicalize_preserves_optional_coords_none,
    ]
    passed = 0
    failed = 0
    print("\n" + "#" * 60)
    print("# RUNNING OPS CANONICALIZE TESTS")
    print("#" * 60 + "\n")
    for t in tests:
        print("=" * 60)
        print(f"TEST: {t.__name__}")
        print("=" * 60)
        try:
            t()
            print(f"✓ {t.__name__} PASSED\n")
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__} FAILED: {e}")
            traceback.print_exc()
            print()
            failed += 1
    print("#" * 60)
    print(f"# RESULTS: {passed} passed, {failed} failed")
    print("#" * 60)


if __name__ == "__main__":
    run_all_tests()
