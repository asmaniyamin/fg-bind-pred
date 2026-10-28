# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
run_all_tests.py

Master test runner for small-molecule-mod-toolkit.

Discovers and runs all test suites in:
  - chem/edit/testing/
  - chem/ops/testing/
  - chem/merge/testing/
  - chem/build/tests/

Run:
  python run_all_tests.py
"""

from __future__ import annotations

import sys
import traceback
import importlib
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# Registry: (module_path, runner_fn_name)
# ---------------------------------------------------------------------------

TEST_SUITES: List[Tuple[str, str]] = [
    # edit layer
    ("chem.edit.testing.test_atoms",              "run_all_tests"),
    ("chem.edit.testing.test_bonds",              "run_all_tests"),
    # ops layer
    ("chem.ops.testing.test_ops_atoms",           "run_all_tests"),
    ("chem.ops.testing.test_ops_bonds",           "run_all_tests"),
    ("chem.ops.testing.test_ops_add_replace_atom","run_all_tests"),
    ("chem.ops.testing.test_ops_merge",           "run_all_tests"),
    # merge
    ("chem.merge.testing.test_merge_unit",        "run_all_tests"),
    ("chem.merge.testing.test_merge",             "run_all_tests"),
    # build
    ("chem.build.tests.test_build",               "run_all_tests"),
    ("chem.build.tests.test_create_molgraph_3d",  "main"),
    # protein fetch
    ("protein.testing.test_fetch_metadata", "run_all_tests"),
    # protein extract
    ("protein.testing.test_cofactors",            "run_all_tests"),
    ("protein.testing.test_biological_assembly",  "run_all_tests"),
    ("protein.testing.test_validation",           "run_all_tests"),
    ("protein.testing.test_batch_robustness",     "run_all_tests"),
    ("protein.testing.test_pocket_search",        "run_all_tests"),
    ("protein.testing.test_detect_resnames",      "run_all_tests"),
]


def _run_suite(module_path: str, fn_name: str) -> Tuple[bool, str]:
    """Import module and call its runner. Returns (ok, error_msg)."""
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        return False, f"Import error: {e}"

    runner = getattr(mod, fn_name, None)
    if runner is None:
        return False, f"No function '{fn_name}' in {module_path}"

    try:
        result = runner()
        # runner returns True on full pass, False otherwise
        if isinstance(result, bool):
            return result, ""
        return True, ""
    except SystemExit as e:
        code = e.code
        if code == 0:
            return True, ""
        return False, f"sys.exit({code})"
    except Exception as e:
        return False, f"Runner raised: {e}\n{traceback.format_exc()}"


def main() -> int:
    width = 70
    print("\n" + "=" * width)
    print("  MASTER TEST RUNNER")
    print("=" * width)

    suite_results: List[Tuple[str, bool, str]] = []

    for module_path, fn_name in TEST_SUITES:
        print(f"\n{'-' * width}")
        print(f"  Suite: {module_path}")
        print(f"{'-' * width}")
        ok, msg = _run_suite(module_path, fn_name)
        suite_results.append((module_path, ok, msg))
        if not ok and msg:
            print(f"  ERROR: {msg}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * width)
    print("  MASTER SUMMARY")
    print("=" * width)
    passed_suites = [r for r in suite_results if r[1]]
    failed_suites = [r for r in suite_results if not r[1]]

    for path, ok, _ in suite_results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {path}")

    print(f"\n  Suites passed: {len(passed_suites)} / {len(suite_results)}")
    if failed_suites:
        print("\n  FAILED suites:")
        for path, _, msg in failed_suites:
            print(f"    - {path}" + (f": {msg}" if msg else ""))
    print("=" * width)

    return 0 if not failed_suites else 1


if __name__ == "__main__":
    sys.exit(main())
