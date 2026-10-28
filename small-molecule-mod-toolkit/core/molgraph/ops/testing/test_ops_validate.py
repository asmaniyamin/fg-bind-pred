# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/ops/testing/test_ops_validate.py
from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import copy
import traceback
import numpy as np

from chem.build.create_molgraph import smiles_to_molgraph
from chem.ops.base import apply_transform
from core.molgraph.ops.validate import Validate

# core exception type (you used it in validate.py)
from core.molgraph.ops.validate import MolGraphValidationError


def test_ops_validate_strict_passes_on_valid_graph() -> None:
    g = smiles_to_molgraph("C[C@H](O)F", coords="canonical", optimize_coords=False)
    assert g is not None

    g2, info = apply_transform(g, Validate(strict=True))
    assert g2 is g, "Validate(strict=True) should be identity transform"
    assert info.warnings == [] or info.warnings is None


def test_ops_validate_non_strict_collects_warnings_on_invalid_bond_index() -> None:
    g = smiles_to_molgraph("CCO", coords="canonical", optimize_coords=False)
    assert g is not None

    # Make an invalid graph: set a bond endpoint to an out-of-range atom index.
    bad = copy.deepcopy(g)
    a = bad.arrays
    n = int(a.atomic_num.shape[0])
    assert int(a.bonds.shape[0]) > 0

    a.bonds = a.bonds.copy()
    a.bonds[0, 1] = n + 123  # out of range
    bad.arrays = a

    bad2, info = apply_transform(bad, Validate(strict=False))
    assert bad2 is bad, "Validate(strict=False) should not modify the graph"
    assert info.warnings is not None
    assert len(info.warnings) >= 1, "Expected at least one validation warning for out-of-range bond"


def test_ops_validate_strict_raises_on_invalid_bond_index() -> None:
    g = smiles_to_molgraph("CCO", coords="canonical", optimize_coords=False)
    assert g is not None

    bad = copy.deepcopy(g)
    a = bad.arrays
    n = int(a.atomic_num.shape[0])
    a.bonds = a.bonds.copy()
    a.bonds[0, 1] = n + 1
    bad.arrays = a

    try:
        apply_transform(bad, Validate(strict=True))
    except MolGraphValidationError:
        return
    raise AssertionError("Expected MolGraphValidationError for strict validation on invalid graph")


def run_all_tests() -> None:
    tests = [
        test_ops_validate_strict_passes_on_valid_graph,
        test_ops_validate_non_strict_collects_warnings_on_invalid_bond_index,
        test_ops_validate_strict_raises_on_invalid_bond_index,
    ]
    passed = 0
    failed = 0
    print("\n" + "#" * 60)
    print("# RUNNING OPS VALIDATE TESTS")
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
