# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
test_fetch_metadata.py

Unit tests for protein.fetch.fetch metadata parsing and validation.

Covers input acquisition:
  - Verify resolution / experimental method / completeness metadata
  - Detect and warn on cryo-EM / NMR / computational models

Run:
  python -m protein.testing.test_fetch_metadata
"""

from __future__ import annotations

import sys
import traceback
import warnings
from typing import Callable, List, Tuple

from protein.fetch.fetch import (
    CompletenessError,
    ExperimentalMethodError,
    NonXrayMethodWarning,
    ResolutionError,
    StructureMetadata,
    parse_structure_metadata,
    validate_structure_metadata,
    _extract_cif_float,
    _extract_cif_value,
    _scan_cif_values,
)


# ---------------------------------------------------------------------------
# mmCIF fixtures
# ---------------------------------------------------------------------------

XRAY_SCALAR_CIF = """\
data_4HHB
#
_entry.id    4HHB
_exptl.entry_id  4HHB
_exptl.method  'X-RAY DIFFRACTION'
#
_refine.ls_d_res_high          1.74
_refine.ls_percent_reflns_obs  98.5
#
"""

XRAY_LOOP_CIF = """\
data_2XYZ
#
loop_
_exptl.entry_id
_exptl.method
2XYZ 'X-RAY DIFFRACTION'
#
_refine.ls_d_res_high  2.35
#
"""

XRAY_MULTILINE_METHOD_CIF = """\
data_3ABC
#
_exptl.entry_id  3ABC
_exptl.method
;X-RAY DIFFRACTION
;
#
_refine.ls_d_res_high          1.90
_refine.ls_percent_reflns_obs  99.1
#
"""

EM_CIF = """\
data_6EM1
#
_exptl.entry_id  6EM1
_exptl.method  'ELECTRON MICROSCOPY'
#
_em_3d_reconstruction.resolution  3.40
#
"""

NMR_CIF = """\
data_1NMR
#
_exptl.entry_id  1NMR
_exptl.method  'SOLUTION NMR'
#
"""

THEORETICAL_CIF = """\
data_9MOD
#
_exptl.entry_id  9MOD
_exptl.method  'THEORETICAL MODEL'
#
"""

MISSING_METADATA_CIF = """\
data_0000
#
_entry.id 0000
#
"""

RESOLUTION_FALLBACK_CIF = """\
data_5FAL
#
_exptl.entry_id  5FAL
_exptl.method  'X-RAY DIFFRACTION'
#
_reflns.d_resolution_high   2.60
_reflns.percent_possible_obs 94.2
#
"""

MISSING_SENTINEL_CIF = """\
data_5MIS
#
_exptl.entry_id  5MIS
_exptl.method  'X-RAY DIFFRACTION'
_refine.ls_d_res_high  ?
_refine.ls_percent_reflns_obs  .
#
"""


# ---------------------------------------------------------------------------
# Lightweight test runner (matches run_all_tests.py convention)
# ---------------------------------------------------------------------------

_tests: List[Tuple[str, Callable[[], None]]] = []


def register(fn: Callable[[], None]) -> Callable[[], None]:
    _tests.append((fn.__name__, fn))
    return fn


def _assert(cond: bool, msg: str = "assertion failed") -> None:
    if not cond:
        raise AssertionError(msg)


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Low-level extraction
# ---------------------------------------------------------------------------

@register
def test_extract_scalar_inline():
    v = _extract_cif_value(XRAY_SCALAR_CIF, "_exptl.method")
    _assert(v == "X-RAY DIFFRACTION", f"expected X-RAY DIFFRACTION, got {v!r}")


@register
def test_extract_scalar_float():
    v = _extract_cif_float(XRAY_SCALAR_CIF, "_refine.ls_d_res_high")
    _assert(v is not None and _approx(v, 1.74), f"expected 1.74, got {v}")


@register
def test_extract_loop_first_row():
    v = _extract_cif_value(XRAY_LOOP_CIF, "_exptl.method")
    _assert(v == "X-RAY DIFFRACTION", f"expected X-RAY DIFFRACTION, got {v!r}")


@register
def test_extract_multiline_semicolon_block():
    v = _extract_cif_value(XRAY_MULTILINE_METHOD_CIF, "_exptl.method")
    _assert(v == "X-RAY DIFFRACTION", f"expected X-RAY DIFFRACTION, got {v!r}")


@register
def test_extract_missing_returns_none():
    v = _extract_cif_value(MISSING_METADATA_CIF, "_exptl.method")
    _assert(v is None, f"expected None, got {v!r}")


@register
def test_extract_sentinel_returns_none():
    v = _extract_cif_float(MISSING_SENTINEL_CIF, "_refine.ls_d_res_high")
    _assert(v is None, f"expected None, got {v!r}")
    v2 = _extract_cif_float(MISSING_SENTINEL_CIF, "_refine.ls_percent_reflns_obs")
    _assert(v2 is None, f"expected None, got {v2!r}")


@register
def test_key_word_boundary():
    """`_refine.ls_d_res_high` must not match `_refine.ls_d_res_high_esd`."""
    cif = (
        "data_FAKE\n"
        "#\n"
        "_refine.ls_d_res_high_esd  0.05\n"
        "_refine.ls_d_res_high      1.80\n"
        "#\n"
    )
    v = _extract_cif_float(cif, "_refine.ls_d_res_high")
    _assert(v is not None and _approx(v, 1.80), f"expected 1.80, got {v}")
    v2 = _extract_cif_float(cif, "_refine.ls_d_res_high_esd")
    _assert(v2 is not None and _approx(v2, 0.05), f"expected 0.05, got {v2}")


# ---------------------------------------------------------------------------
# parse_structure_metadata
# ---------------------------------------------------------------------------

@register
def test_parse_xray_full():
    meta = parse_structure_metadata(XRAY_SCALAR_CIF)
    _assert(meta.is_x_ray, f"expected is_x_ray, got method={meta.method!r}")
    _assert(meta.resolution_angstroms == 1.74, f"res={meta.resolution_angstroms}")
    _assert(meta.completeness_percent == 98.5, f"comp={meta.completeness_percent}")
    _assert(meta.has_reliable_b_factors, "X-ray should have reliable B-factors")


@register
def test_parse_em():
    meta = parse_structure_metadata(EM_CIF)
    _assert(meta.is_electron_microscopy, f"expected EM, got {meta.method!r}")
    _assert(not meta.is_x_ray, "EM should not be flagged as X-ray")
    _assert(not meta.has_reliable_b_factors, "EM B-factors are sharpening-derived")
    _assert(
        meta.resolution_angstroms == 3.40,
        f"EM resolution fallback failed: {meta.resolution_angstroms}",
    )


@register
def test_parse_nmr():
    meta = parse_structure_metadata(NMR_CIF)
    _assert(meta.is_nmr, f"expected NMR, got {meta.method!r}")
    _assert(not meta.has_reliable_b_factors, "NMR has no meaningful B-factors")
    _assert(meta.resolution_angstroms is None, "NMR has no resolution")


@register
def test_parse_theoretical():
    meta = parse_structure_metadata(THEORETICAL_CIF)
    _assert(meta.is_theoretical, f"expected theoretical, got {meta.method!r}")
    _assert(not meta.is_x_ray)


@register
def test_parse_missing_returns_all_none():
    meta = parse_structure_metadata(MISSING_METADATA_CIF)
    _assert(meta.method is None, f"method={meta.method!r}")
    _assert(meta.resolution_angstroms is None)
    _assert(meta.completeness_percent is None)


@register
def test_parse_resolution_fallback_to_reflns():
    meta = parse_structure_metadata(RESOLUTION_FALLBACK_CIF)
    _assert(meta.resolution_angstroms == 2.60, f"res={meta.resolution_angstroms}")
    _assert(meta.completeness_percent == 94.2, f"comp={meta.completeness_percent}")


@register
def test_scan_cif_values_is_single_pass():
    """_scan_cif_values must collect every requested key in one walk.

    Regression guard for the refactor that eliminated the per-key full-file
    rescan. If someone reintroduces repeated scans, this test still passes
    on correctness, so we assert the stronger property that all keys come
    back from the same call - which is only interesting if the scanner
    actually supports multi-key lookup.
    """
    combined = XRAY_SCALAR_CIF + "\n" + RESOLUTION_FALLBACK_CIF
    keys = (
        "_exptl.method",
        "_refine.ls_d_res_high",
        "_reflns.d_resolution_high",
        "_refine.ls_percent_reflns_obs",
        "_reflns.percent_possible_obs",
    )
    found = _scan_cif_values(combined, keys)
    _assert(found.get("_exptl.method") == "X-RAY DIFFRACTION", f"{found=}")
    _assert(found.get("_refine.ls_d_res_high") == "1.74", f"{found=}")
    _assert(found.get("_reflns.d_resolution_high") == "2.60", f"{found=}")
    _assert(found.get("_refine.ls_percent_reflns_obs") == "98.5", f"{found=}")


@register
def test_scan_short_circuits_when_all_found():
    """If every wanted key is resolved, the scanner stops walking."""
    header = "_exptl.method 'X-RAY DIFFRACTION'\n_refine.ls_d_res_high 1.80\n"
    # Trailing content is malformed on purpose: if the scanner kept walking,
    # `_tokenize_mmcif_line` on the unclosed quote would raise ValueError.
    trailing = "_exptl.method  'unterminated\n"
    found = _scan_cif_values(
        header + trailing,
        ("_exptl.method", "_refine.ls_d_res_high"),
    )
    _assert(found["_exptl.method"] == "X-RAY DIFFRACTION", f"{found=}")
    _assert(found["_refine.ls_d_res_high"] == "1.80", f"{found=}")


# ---------------------------------------------------------------------------
# validate_structure_metadata
# ---------------------------------------------------------------------------

@register
def test_validate_xray_no_warnings():
    meta = parse_structure_metadata(XRAY_SCALAR_CIF)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_structure_metadata(meta)
    non_xray = [w for w in caught if issubclass(w.category, NonXrayMethodWarning)]
    _assert(len(non_xray) == 0, f"expected no NonXrayMethodWarning, got {non_xray}")


@register
def test_validate_em_emits_warning():
    meta = parse_structure_metadata(EM_CIF)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_structure_metadata(meta)
    non_xray = [w for w in caught if issubclass(w.category, NonXrayMethodWarning)]
    _assert(len(non_xray) == 1, f"expected 1 warning, got {len(non_xray)}")
    _assert(
        "cryo-EM" in str(non_xray[0].message)
        or "electron microscopy" in str(non_xray[0].message).lower(),
        f"warning text unexpected: {non_xray[0].message}",
    )


@register
def test_validate_nmr_emits_warning():
    meta = parse_structure_metadata(NMR_CIF)
    sink: List[str] = []
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        validate_structure_metadata(meta, on_warning=sink.append)
    _assert(len(sink) == 1, f"expected 1 warning via sink, got {sink}")
    _assert("NMR" in sink[0], f"expected NMR in warning: {sink[0]}")


@register
def test_validate_require_xray_raises_on_em():
    meta = parse_structure_metadata(EM_CIF)
    try:
        validate_structure_metadata(meta, require_xray=True)
    except ExperimentalMethodError:
        return
    raise AssertionError("Expected ExperimentalMethodError for EM under require_xray")


@register
def test_validate_resolution_gate_rejects_bad():
    meta = StructureMetadata(
        method="X-RAY DIFFRACTION", resolution_angstroms=3.50, completeness_percent=95.0
    )
    try:
        validate_structure_metadata(meta, min_resolution=3.00)
    except ResolutionError as e:
        _assert("3.50" in str(e), f"unexpected message: {e}")
        return
    raise AssertionError("Expected ResolutionError for 3.5 Å vs 3.0 Å cutoff")


@register
def test_validate_resolution_gate_accepts_good():
    meta = StructureMetadata(
        method="X-RAY DIFFRACTION", resolution_angstroms=1.80, completeness_percent=99.0
    )
    validate_structure_metadata(meta, min_resolution=3.00)  # no raise


@register
def test_validate_resolution_gate_silent_when_missing():
    meta = StructureMetadata(method="SOLUTION NMR", resolution_angstroms=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validate_structure_metadata(meta, min_resolution=3.00)  # no raise


@register
def test_validate_completeness_gate_rejects_low():
    meta = StructureMetadata(
        method="X-RAY DIFFRACTION", resolution_angstroms=2.0, completeness_percent=70.0
    )
    try:
        validate_structure_metadata(meta, min_completeness=90.0)
    except CompletenessError as e:
        _assert("70.0" in str(e), f"unexpected: {e}")
        return
    raise AssertionError("Expected CompletenessError")


@register
def test_validate_completeness_gate_accepts_high():
    meta = StructureMetadata(
        method="X-RAY DIFFRACTION", resolution_angstroms=2.0, completeness_percent=99.5
    )
    validate_structure_metadata(meta, min_completeness=90.0)  # no raise


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_tests() -> bool:
    print(f"  Running {len(_tests)} metadata tests")
    failed: List[Tuple[str, str]] = []
    for name, fn in _tests:
        try:
            fn()
            print(f"    ✓ {name}")
        except Exception as e:
            print(f"    ✗ {name}: {e}")
            failed.append((name, traceback.format_exc()))

    if failed:
        print(f"\n  {len(failed)}/{len(_tests)} failed:")
        for name, tb in failed:
            print(f"\n    {name}:\n{tb}")
        return False

    print(f"\n  All {len(_tests)} metadata tests passed.")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
