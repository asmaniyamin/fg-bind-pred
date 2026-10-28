# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/ops/testing/test_ops_hash.py
from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import traceback
import numpy as np

from chem.build.create_molgraph import smiles_to_molgraph

from chem.ops.base import apply_transform
from core.molgraph.ops.canonicalize import Canonicalize
from core.molgraph.ops.hash import HashMolGraph


def _get_hash_from_meta(mg, key: str) -> str:
    assert mg.meta is not None
    assert key in mg.meta, f"Expected {key} in meta, found keys={list(mg.meta.keys())}"
    return str(mg.meta[key])


def test_ops_hash_ignores_coordinates_in_topology_mode() -> None:
    smi = "CC(=O)O"  # acetic acid
    g1 = smiles_to_molgraph(smi, coords="canonical", optimize_coords=False)
    g2 = smiles_to_molgraph(smi, coords=None)
    assert g1 is not None and g2 is not None

    # Canonicalize with stereo=False for topology-like behavior
    g1c, _ = apply_transform(g1, Canonicalize(stereo=False))
    g2c, _ = apply_transform(g2, Canonicalize(stereo=False))

    g1h, _ = apply_transform(g1c, HashMolGraph(mode="topology", write_to_meta=True))
    g2h, _ = apply_transform(g2c, HashMolGraph(mode="topology", write_to_meta=True))

    h1 = _get_hash_from_meta(g1h, "hash_topology")
    h2 = _get_hash_from_meta(g2h, "hash_topology")
    assert h1 == h2, "Topology hash should ignore coordinates"


def test_ops_stereo_hash_distinguishes_enantiomers() -> None:
    # Simple chiral center: CH(F)(Cl)CH3
    a = "C[C@H](F)Cl"
    b = "C[C@@H](F)Cl"
    gA = smiles_to_molgraph(a, coords="canonical", optimize_coords=False)
    gB = smiles_to_molgraph(b, coords="canonical", optimize_coords=False)
    assert gA is not None and gB is not None

    # canonicalize with stereo=True (keep handedness meaningful in ordering)
    gAc, _ = apply_transform(gA, Canonicalize(stereo=True))
    gBc, _ = apply_transform(gB, Canonicalize(stereo=True))

    gAh, _ = apply_transform(gAc, HashMolGraph(mode="stereo", write_to_meta=True))
    gBh, _ = apply_transform(gBc, HashMolGraph(mode="stereo", write_to_meta=True))

    ha = _get_hash_from_meta(gAh, "hash_stereo")
    hb = _get_hash_from_meta(gBh, "hash_stereo")
    assert ha != hb, "Stereo hash should distinguish enantiomers"


def test_ops_topology_hash_treats_enantiomers_equivalent() -> None:
    a = "C[C@H](F)Cl"
    b = "C[C@@H](F)Cl"
    gA = smiles_to_molgraph(a, coords="canonical", optimize_coords=False)
    gB = smiles_to_molgraph(b, coords="canonical", optimize_coords=False)
    assert gA is not None and gB is not None

    # canonicalize with stereo=False to intentionally ignore handedness in canonicalization
    gAc, _ = apply_transform(gA, Canonicalize(stereo=False))
    gBc, _ = apply_transform(gB, Canonicalize(stereo=False))

    gAh, _ = apply_transform(gAc, HashMolGraph(mode="topology", write_to_meta=True))
    gBh, _ = apply_transform(gBc, HashMolGraph(mode="topology", write_to_meta=True))

    ha = _get_hash_from_meta(gAh, "hash_topology")
    hb = _get_hash_from_meta(gBh, "hash_topology")
    assert ha == hb, "Topology hash should ignore handedness and treat enantiomers as equivalent"


def run_all_tests() -> None:
    tests = [
        test_ops_hash_ignores_coordinates_in_topology_mode,
        test_ops_stereo_hash_distinguishes_enantiomers,
        test_ops_topology_hash_treats_enantiomers_equivalent,
    ]
    passed = 0
    failed = 0
    print("\n" + "#" * 60)
    print("# RUNNING OPS HASH TESTS")
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
