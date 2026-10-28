# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/testing/test_detect_resnames.py

Unit tests for :func:`protein.extract.detect_resnames.detect_ligand_resnames_from_raw_cif`.

Synthetic mmCIF inputs exercise the following behavior:

    * HETATM ligand large enough → kept
    * HETATM ligand below the heavy-atom floor → dropped
    * Resname in :data:`EXCLUDED_RESNAMES` → dropped regardless of size
    * ATOM-only file → empty list
    * File without ``group_PDB`` (e.g. Schrödinger output) → empty list
    * Multiple instances, only one large enough → kept (max-of-instances)
    * Resnames returned sorted lexicographically
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Bootstrap so this file can run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein.testing.runner import run_suite, suite_main  # noqa: E402
from protein.extract.detect_resnames import (  # noqa: E402
    DEFAULT_MIN_HEAVY_ATOMS,
    EXCLUDED_RESNAMES,
    detect_ligand_resnames_from_raw_cif,
)


# ---------------------------------------------------------------------------
# Synthetic mmCIF builder
# ---------------------------------------------------------------------------

_HEADER = """\
data_TEST
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.auth_comp_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""


def _row(
    *,
    group: str,
    atom_id: int,
    element: str,
    atom_name: str,
    resname: str,
    chain: str = "A",
    seq: int = 1,
) -> str:
    """One ``_atom_site`` row matching the column order in ``_HEADER``."""
    return (
        f"{group} {atom_id} {element} {atom_name} {resname} {chain} "
        f"{seq} {seq} {chain} {resname} 0.0 0.0 0.0\n"
    )


def _write_cif(rows: str) -> Path:
    """Write a temp CIF, return the path. Caller manages cleanup."""
    cif = _HEADER + rows + "#\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", delete=False, encoding="utf-8"
    )
    handle.write(cif)
    handle.close()
    return Path(handle.name)


def _hetatm_residue(
    resname: str,
    *,
    n_heavy: int,
    chain: str = "A",
    seq: int = 1,
    n_hydrogens: int = 0,
    start_id: int = 1,
) -> str:
    """Build ``n_heavy`` HETATM rows of carbon plus ``n_hydrogens`` H rows."""
    rows: list[str] = []
    atom_id = start_id
    for k in range(n_heavy):
        rows.append(
            _row(
                group="HETATM",
                atom_id=atom_id,
                element="C",
                atom_name=f"C{k}",
                resname=resname,
                chain=chain,
                seq=seq,
            )
        )
        atom_id += 1
    for k in range(n_hydrogens):
        rows.append(
            _row(
                group="HETATM",
                atom_id=atom_id,
                element="H",
                atom_name=f"H{k}",
                resname=resname,
                chain=chain,
                seq=seq,
            )
        )
        atom_id += 1
    return "".join(rows)


# ---------------------------------------------------------------------------
# Core contract
# ---------------------------------------------------------------------------

def test_real_ligand_kept() -> None:
    """A HETATM resname with ≥ min_heavy_atoms heavy atoms survives."""
    cif = _write_cif(_hetatm_residue("BTN", n_heavy=10))
    try:
        result = detect_ligand_resnames_from_raw_cif(cif)
    finally:
        cif.unlink(missing_ok=True)
    assert result == ["BTN"], result


def test_too_small_ligand_dropped() -> None:
    """A HETATM resname below the heavy-atom floor is discarded."""
    cif = _write_cif(_hetatm_residue("XYZ", n_heavy=DEFAULT_MIN_HEAVY_ATOMS - 1))
    try:
        result = detect_ligand_resnames_from_raw_cif(cif)
    finally:
        cif.unlink(missing_ok=True)
    assert result == [], result


def test_water_excluded_regardless_of_size() -> None:
    """``HOH`` would never be a ligand even at any size - exclude list wins."""
    # 10 heavy atoms is well past the floor, but HOH is in EXCLUDED_RESNAMES.
    assert "HOH" in EXCLUDED_RESNAMES
    cif = _write_cif(_hetatm_residue("HOH", n_heavy=10))
    try:
        result = detect_ligand_resnames_from_raw_cif(cif)
    finally:
        cif.unlink(missing_ok=True)
    assert result == [], result


def test_atom_only_file_returns_empty() -> None:
    """No HETATM rows → no ligands."""
    rows = "".join(
        _row(
            group="ATOM",
            atom_id=k + 1,
            element="C",
            atom_name=f"C{k}",
            resname="ALA",
            seq=k + 1,
        )
        for k in range(20)
    )
    cif = _write_cif(rows)
    try:
        result = detect_ligand_resnames_from_raw_cif(cif)
    finally:
        cif.unlink(missing_ok=True)
    assert result == [], result


def test_hydrogens_dont_count_toward_heavy_atom_threshold() -> None:
    """A resname with 5 heavy + 10 H atoms must NOT cross the threshold."""
    cif = _write_cif(_hetatm_residue("XYZ", n_heavy=5, n_hydrogens=10))
    try:
        result = detect_ligand_resnames_from_raw_cif(cif)
    finally:
        cif.unlink(missing_ok=True)
    assert result == [], result


def test_max_of_instances_decides() -> None:
    """One large instance carries the resname even when others are tiny.

    Real-world case: a bio-unit run might have one full ATP and several
    truncated copies; the resname should still survive.
    """
    rows = (
        _hetatm_residue("ATP", n_heavy=20, chain="A", seq=1, start_id=1)
        + _hetatm_residue("ATP", n_heavy=2, chain="B", seq=1, start_id=100)
    )
    cif = _write_cif(rows)
    try:
        result = detect_ligand_resnames_from_raw_cif(cif)
    finally:
        cif.unlink(missing_ok=True)
    assert result == ["ATP"], result


def test_results_sorted_lexicographically() -> None:
    """Output ordering is deterministic - required so the shell pipeline is reproducible."""
    rows = (
        _hetatm_residue("ZZZ", n_heavy=10, chain="A", seq=1, start_id=1)
        + _hetatm_residue("AAA", n_heavy=10, chain="B", seq=1, start_id=100)
        + _hetatm_residue("MMM", n_heavy=10, chain="C", seq=1, start_id=200)
    )
    cif = _write_cif(rows)
    try:
        result = detect_ligand_resnames_from_raw_cif(cif)
    finally:
        cif.unlink(missing_ok=True)
    assert result == ["AAA", "MMM", "ZZZ"], result


def test_missing_group_pdb_column_returns_empty() -> None:
    """Schrödinger-prepped CIFs lack ``group_PDB`` - must produce no ligands.

    This is the exact failure mode that motivated the module: a CIF that
    can't be classified as HETATM/ATOM should not emit anything, so the
    shell pipeline correctly reports "no candidate ligands" instead of
    misclassifying ATOM rows as ligands.
    """
    no_group_pdb_header = """\
data_TEST
#
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.auth_comp_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""
    rows = ""
    for k in range(10):
        rows += f"{k + 1} C BTN A 1 1 A BTN 0.0 0.0 0.0\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", delete=False, encoding="utf-8"
    )
    handle.write(no_group_pdb_header + rows + "#\n")
    handle.close()
    cif = Path(handle.name)
    try:
        result = detect_ligand_resnames_from_raw_cif(cif)
    finally:
        cif.unlink(missing_ok=True)
    assert result == [], result


def test_mixed_real_and_excluded_ligands() -> None:
    """Real ligands survive, excluded ones drop - same file."""
    rows = (
        _hetatm_residue("BTN", n_heavy=10, chain="A", seq=1, start_id=1)
        + _hetatm_residue("HOH", n_heavy=10, chain="B", seq=1, start_id=100)  # excluded
        + _hetatm_residue("ZN", n_heavy=10, chain="C", seq=1, start_id=200)   # excluded
        + _hetatm_residue("HEM", n_heavy=43, chain="D", seq=1, start_id=300)
    )
    cif = _write_cif(rows)
    try:
        result = detect_ligand_resnames_from_raw_cif(cif)
    finally:
        cif.unlink(missing_ok=True)
    assert result == ["BTN", "HEM"], result


def test_custom_min_heavy_atoms_threshold() -> None:
    """A higher floor can be passed to filter more aggressively."""
    cif = _write_cif(_hetatm_residue("BTN", n_heavy=10))
    try:
        # default floor 6 keeps BTN; floor 20 drops it.
        assert detect_ligand_resnames_from_raw_cif(cif) == ["BTN"]
        assert detect_ligand_resnames_from_raw_cif(cif, min_heavy_atoms=20) == []
    finally:
        cif.unlink(missing_ok=True)


def test_custom_excluded_resnames() -> None:
    """Caller can override the exclusion set when needed."""
    cif = _write_cif(_hetatm_residue("BTN", n_heavy=10))
    try:
        # Default excludes don't touch BTN - kept.
        assert detect_ligand_resnames_from_raw_cif(cif) == ["BTN"]
        # Force-exclude BTN - dropped.
        assert (
            detect_ligand_resnames_from_raw_cif(cif, excluded_resnames={"BTN"}) == []
        )
    finally:
        cif.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

ALL_TESTS = (
    test_real_ligand_kept,
    test_too_small_ligand_dropped,
    test_water_excluded_regardless_of_size,
    test_atom_only_file_returns_empty,
    test_hydrogens_dont_count_toward_heavy_atom_threshold,
    test_max_of_instances_decides,
    test_results_sorted_lexicographically,
    test_missing_group_pdb_column_returns_empty,
    test_mixed_real_and_excluded_ligands,
    test_custom_min_heavy_atoms_threshold,
    test_custom_excluded_resnames,
)


def run_all_tests() -> bool:
    """Entry point for the master :mod:`run_all_tests` runner."""
    return run_suite(ALL_TESTS)


if __name__ == "__main__":
    sys.exit(suite_main(ALL_TESTS))
