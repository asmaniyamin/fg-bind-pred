# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/testing/runner.py

Shared runner scaffolding for the per-suite test files under
``protein/testing/`` (and any other suite that follows the same
pattern). Replaces ~20 lines of identical boilerplate per file with two
small helpers.

Pattern each test file follows::

    # 3-line bootstrap so this file can be run as a script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from protein.testing.runner import run_suite, suite_main
    # ... test functions ...

    ALL_TESTS = (
        test_one,
        test_two,
        ...
    )

    def run_all_tests() -> bool:
        return run_suite(ALL_TESTS)

    if __name__ == "__main__":
        sys.exit(suite_main(ALL_TESTS))

The 3-line ``sys.path`` bootstrap stays inline because importing this
helper itself requires the repo root to already be on ``sys.path``;
abstracting it would just push the chicken-and-egg around.

The master :mod:`run_all_tests` script picks each suite up by importing
its ``run_all_tests`` callable, which adopters define as a one-line
delegation to :func:`run_suite`.
"""

from __future__ import annotations

from typing import Callable, Sequence


TestFunc = Callable[[], None]


def run_suite(test_funcs: Sequence[TestFunc]) -> bool:
    """Run every test in ``test_funcs``. Returns True iff all passed.

    Each function is called with no args and is expected to raise on
    failure. Output is one ``PASS`` / ``FAIL`` line per test, plus a
    final summary line. Catches ``BaseException`` so a deliberately
    invalid mol that triggers a C-level RDKit assert still produces a
    readable diagnostic instead of aborting the whole run.
    """
    failures: list[tuple[str, BaseException]] = []
    for test in test_funcs:
        try:
            test()
            print(f"PASS {test.__name__}")
        except BaseException as exc:  # pragma: no cover - diagnostic path
            print(f"FAIL {test.__name__}: {exc}")
            failures.append((test.__name__, exc))
    total = len(test_funcs)
    if failures:
        print(f"\n{len(failures)} / {total} tests failed.")
        return False
    print(f"\n{total} / {total} tests passed.")
    return True


def suite_main(test_funcs: Sequence[TestFunc]) -> int:
    """Convenience wrapper for ``if __name__ == "__main__"`` blocks.

    Returns 0 on full pass, 1 on any failure - pass straight to
    ``sys.exit``.
    """
    return 0 if run_suite(test_funcs) else 1
