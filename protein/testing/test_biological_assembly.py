# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/testing/test_biological_assembly.py

Unit tests for the ``expand_to_biological_assembly`` helper.

Covers three behaviours:

    * Happy path: 1STP (AU = 1 chain, bio-unit = tetramer) expands to four
      chains with distinct IDs (``A1``/``A2``/``A3``/``A4``), and the
      expanded mmCIF retains the ``_chem_comp_bond`` table gemmi drops.

    * Invalid assembly_id surfaces a ``PocketExtractionError`` naming the
      available assembly identifiers.

    * Missing gemmi module surfaces a clear install-instructing error.

The 1STP fixture at ``data/1stp/proteins/1stp.cif`` is required; tests
skip gracefully (returning True) if it is absent.

This module is plain-function based rather than pytest-fixture based so
that ``run_all_tests.py`` can call ``run_all_tests()`` directly. It is
still runnable under ``pytest`` because the tests are ordinary callables.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protein.testing.runner import run_suite, suite_main  # noqa: E402
from protein.extract import _core  # noqa: E402


_FIXTURE_1STP = _REPO_ROOT / "data" / "1stp" / "proteins" / "1stp.cif"


def _fixture_or_skip() -> Path | None:
    """Return the 1STP mmCIF path, or None if the fixture is missing."""
    if not _FIXTURE_1STP.is_file():
        print(f"SKIP: 1STP fixture not available at {_FIXTURE_1STP}")
        return None
    return _FIXTURE_1STP


def test_expand_1stp_yields_tetramer() -> None:
    """AU has chain A; assembly 1 is the tetramer A1..A4."""
    if _core.gemmi is None:
        print("SKIP test_expand_1stp_yields_tetramer: gemmi not installed")
        return
    fixture = _fixture_or_skip()
    if fixture is None:
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_path = tmp_path / "1stp_assembly_1.cif"
        result_path = _core.expand_to_biological_assembly(
            fixture, assembly_id="1", output_path=out_path
        )
        assert result_path == out_path, f"Unexpected return path: {result_path}"
        assert result_path.is_file(), "Expanded mmCIF was not written"

        # gemmi drops _chem_comp_bond by default; helper must splice it back.
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict

        expanded = MMCIF2Dict(str(result_path))
        assert "_chem_comp_bond.comp_id" in expanded, (
            "_chem_comp_bond was not preserved in the expanded mmCIF"
        )
        assert "BTN" in set(expanded["_chem_comp_bond.comp_id"]), (
            "Expected BTN rows in _chem_comp_bond (splice from source CIF failed)"
        )

        # Chain IDs should be A1..A4 after AddNumber renaming.
        structure, _ = _core.load_structure_and_cif(result_path)
        chain_ids = sorted(c.id for c in structure[0])
        assert chain_ids == ["A1", "A2", "A3", "A4"], (
            f"Unexpected chain IDs after assembly expansion: {chain_ids}"
        )


def test_invalid_assembly_id_raises() -> None:
    """Unknown assembly_id surfaces an error naming the available IDs."""
    if _core.gemmi is None:
        print("SKIP test_invalid_assembly_id_raises: gemmi not installed")
        return
    fixture = _fixture_or_skip()
    if fixture is None:
        return

    with tempfile.TemporaryDirectory() as tmp:
        try:
            _core.expand_to_biological_assembly(
                fixture,
                assembly_id="99",
                output_path=Path(tmp) / "bogus.cif",
            )
        except _core.PocketExtractionError as exc:
            message = str(exc)
            assert "99" in message, f"Error did not name bad id: {message}"
            assert "Available" in message, (
                f"Error did not list available assemblies: {message}"
            )
            return
        raise AssertionError(
            "Expected PocketExtractionError for invalid assembly_id"
        )


def test_describe_reports_1stp_tetramer_expands() -> None:
    """describe_biological_assemblies flags 1STP assembly 1 as != AU."""
    if _core.gemmi is None:
        print("SKIP test_describe_reports_1stp_tetramer_expands: gemmi not installed")
        return
    fixture = _fixture_or_skip()
    if fixture is None:
        return

    report = _core.describe_biological_assemblies(fixture)
    assert report["au_chain_count"] == 1, (
        f"Expected 1STP AU to have 1 chain, got {report['au_chain_count']}"
    )
    assemblies = report["assemblies"]
    assert len(assemblies) == 1, f"Expected one declared assembly, got {assemblies}"
    a1 = assemblies[0]
    assert a1["id"] == "1"
    assert a1["bio_chain_count"] == 4
    assert a1["differs_from_au"] is True
    assert _core.should_expand_to_biological_assembly(fixture, assembly_id="1") is True


def test_describe_reports_4hhb_au_equals_bio() -> None:
    """4HHB's AU already is the tetramer -> no expansion recommended."""
    if _core.gemmi is None:
        print("SKIP test_describe_reports_4hhb_au_equals_bio: gemmi not installed")
        return
    fixture_4hhb = _REPO_ROOT / "data" / "4hhb" / "proteins" / "4hhb.cif"
    if not fixture_4hhb.is_file():
        print(f"SKIP: 4HHB fixture not available at {fixture_4hhb}")
        return

    report = _core.describe_biological_assemblies(fixture_4hhb)
    assert report["au_chain_count"] == 4
    a1 = report["assemblies"][0]
    assert a1["bio_chain_count"] == 4
    assert a1["differs_from_au"] is False
    assert _core.should_expand_to_biological_assembly(fixture_4hhb, assembly_id="1") is False


def test_missing_gemmi_is_reported() -> None:
    """With gemmi patched to None, the helper gives a clean install hint."""
    fixture = _fixture_or_skip()
    if fixture is None:
        return

    saved = _core.gemmi
    _core.gemmi = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                _core.expand_to_biological_assembly(
                    fixture,
                    assembly_id="1",
                    output_path=Path(tmp) / "should_not_be_written.cif",
                )
            except _core.PocketExtractionError as exc:
                assert "gemmi" in str(exc).lower(), (
                    f"Error did not mention gemmi: {exc}"
                )
                return
            raise AssertionError(
                "Expected PocketExtractionError when gemmi is unavailable"
            )
    finally:
        _core.gemmi = saved


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

ALL_TESTS = (
    test_expand_1stp_yields_tetramer,
    test_invalid_assembly_id_raises,
    test_describe_reports_1stp_tetramer_expands,
    test_describe_reports_4hhb_au_equals_bio,
    test_missing_gemmi_is_reported,
)


def run_all_tests() -> bool:
    """Entry point for the master :mod:`run_all_tests` runner."""
    return run_suite(ALL_TESTS)


if __name__ == "__main__":
    raise SystemExit(suite_main(ALL_TESTS))
