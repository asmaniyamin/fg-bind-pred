# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/testing/test_pocket_search.py

Unit tests for pocket-residue spatial search:

    * :class:`PocketSearchIndex` - the per-pipeline-run cache of the
      sorted residue-key list, residue → atom-record map, residue
      hetflag, flat protein-atom coord array, and parallel per-atom
      residue-index array.
    * :func:`find_pocket_residues` - finds nearby residues using the spatial index.

Tests build a synthetic ``PocketSearchIndex`` directly (no mmCIF parse,
no Bio.PDB residue mocks) so they're fast and deterministic. The
builder ``build_pocket_search_index`` is exercised end-to-end by
``test_pipeline_e2e.py``; the unit pass here focuses on the search
contract itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Let this file run as a script from the repo root without requiring the
# ``protein`` package to be installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protein.testing.runner import run_suite, suite_main  # noqa: E402
from protein.extract._core import (  # noqa: E402
    AtomRecord,
    LigandGroup,
    ResidueKey,
    ligand_sort_key,
)
from protein.extract._search import (  # noqa: E402
    PocketSearchIndex,
    find_pocket_residues,
)


# ---------------------------------------------------------------------------
# Fixtures: hand-built PocketSearchIndex
# ---------------------------------------------------------------------------

def _make_atom(
    residue_key: ResidueKey,
    atom_name: str,
    element: str,
    xyz: tuple[float, float, float],
    *,
    atom_index: int = 0,
) -> AtomRecord:
    return AtomRecord(
        atom_index=atom_index,
        atom_name=atom_name,
        element=element,
        x=float(xyz[0]),
        y=float(xyz[1]),
        z=float(xyz[2]),
        residue_key=residue_key,
    )


def _make_index(
    residue_specs: list[tuple[ResidueKey, list[tuple[str, str, tuple[float, float, float]]], str]],
) -> PocketSearchIndex:
    """Build a PocketSearchIndex from explicit residue specs.

    Each spec is ``(residue_key, atoms, hetflag)`` where ``atoms`` is a
    list of ``(atom_name, element, (x, y, z))``.
    """
    sorted_keys = tuple(sorted([k for k, _, _ in residue_specs], key=ligand_sort_key))
    atoms_by_key: dict[ResidueKey, list[AtomRecord]] = {}
    hetflag_by_key: dict[ResidueKey, str] = {}

    spec_lookup = {key: (atoms, hetflag) for key, atoms, hetflag in residue_specs}
    flat_coords: list[tuple[float, float, float]] = []
    flat_residue_indices: list[int] = []

    for residue_idx, key in enumerate(sorted_keys):
        atoms_data, hetflag = spec_lookup[key]
        atom_records = [
            _make_atom(key, name, element, xyz, atom_index=i)
            for i, (name, element, xyz) in enumerate(atoms_data)
        ]
        atoms_by_key[key] = atom_records
        hetflag_by_key[key] = hetflag
        for atom in atom_records:
            flat_coords.append((atom.x, atom.y, atom.z))
            flat_residue_indices.append(residue_idx)

    if flat_coords:
        coords = np.asarray(flat_coords, dtype=np.float32)
        residue_index = np.asarray(flat_residue_indices, dtype=np.int32)
        tree: cKDTree | None = cKDTree(coords.astype(np.float64))
    else:
        coords = np.zeros((0, 3), dtype=np.float32)
        residue_index = np.zeros(0, dtype=np.int32)
        tree = None

    return PocketSearchIndex(
        sorted_residue_keys=sorted_keys,
        residue_atoms=atoms_by_key,
        residue_hetflag=hetflag_by_key,
        protein_atom_coords=coords,
        protein_atom_residue_index=residue_index,
        protein_kdtree=tree,
    )


def _residue_key(
    chain_id: str = "A",
    resseq: int = 100,
    insertion_code: str = "",
    resname: str = "ALA",
) -> ResidueKey:
    return ResidueKey(
        chain_id=chain_id,
        resseq=resseq,
        insertion_code=insertion_code,
        resname=resname,
    )


def _ligand_atom(xyz: tuple[float, float, float]) -> AtomRecord:
    return _make_atom(
        _residue_key(resname="LIG", resseq=999),
        atom_name="C1",
        element="C",
        xyz=xyz,
    )


def _ligand_group(resname: str = "LIG", resseq: int = 999) -> LigandGroup:
    return LigandGroup(
        ligand_id=f"{resname}_A{resseq}",
        residue_keys=(_residue_key(resname=resname, resseq=resseq),),
    )


# ---------------------------------------------------------------------------
# In-range / out-of-range
# ---------------------------------------------------------------------------

def test_residue_within_radius_appears_in_output() -> None:
    res_key = _residue_key(resseq=10, resname="ALA")
    index = _make_index(
        [(res_key, [("CA", "C", (5.0, 0.0, 0.0))], "")]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=False,
        include_water=False,
    )
    assert len(pocket) == 1, pocket
    assert pocket[0]["residue"]["resseq"] == 10
    assert pocket[0]["residue"]["resname"] == "ALA"
    assert pocket[0]["min_distance"] == 5.0


def test_residue_outside_radius_excluded() -> None:
    res_key = _residue_key(resseq=10, resname="ALA")
    index = _make_index(
        [(res_key, [("CA", "C", (15.0, 0.0, 0.0))], "")]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=False,
        include_water=False,
    )
    assert pocket == []


def test_min_distance_picks_closest_atom_in_residue() -> None:
    """Multi-atom residue → reported min_distance is the smallest pair."""
    res_key = _residue_key(resseq=10, resname="GLU")
    # Residue with three atoms at increasing distances. min must be 3.0.
    index = _make_index(
        [(
            res_key,
            [
                ("CA", "C", (8.0, 0.0, 0.0)),
                ("CB", "C", (5.0, 0.0, 0.0)),
                ("OE1", "O", (3.0, 0.0, 0.0)),
            ],
            "",
        )]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=False,
        include_water=False,
    )
    assert len(pocket) == 1
    assert abs(pocket[0]["min_distance"] - 3.0) < 1e-5


def test_min_distance_picks_closest_ligand_atom() -> None:
    """Multi-atom ligand → reported min_distance is across all ligand atoms."""
    res_key = _residue_key(resseq=10, resname="ALA")
    index = _make_index(
        [(res_key, [("CA", "C", (0.0, 0.0, 0.0))], "")]
    )
    # Two ligand atoms - the second is closer.
    ligand_atoms = [
        _ligand_atom((10.0, 0.0, 0.0)),
        _ligand_atom((2.0, 0.0, 0.0)),
    ]
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=ligand_atoms,
        radius=15.0,
        include_hetero=False,
        include_water=False,
    )
    assert len(pocket) == 1
    assert abs(pocket[0]["min_distance"] - 2.0) < 1e-5


# ---------------------------------------------------------------------------
# Filters: water, hetero, self
# ---------------------------------------------------------------------------

def test_water_excluded_by_default() -> None:
    """Water residues at any distance are dropped unless include_water=True."""
    res_key = _residue_key(resseq=999, resname="HOH")
    index = _make_index(
        [(res_key, [("O", "O", (3.0, 0.0, 0.0))], "W")]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=False,
        include_water=False,
    )
    assert pocket == []


def test_water_included_when_flag_set() -> None:
    res_key = _residue_key(resseq=999, resname="HOH")
    index = _make_index(
        [(res_key, [("O", "O", (3.0, 0.0, 0.0))], "W")]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=False,
        include_water=True,
    )
    assert len(pocket) == 1
    assert pocket[0]["residue"]["resname"] == "HOH"


def test_hetero_excluded_by_default() -> None:
    """Non-water hetero residues (hetflag=H_*) are dropped unless include_hetero=True."""
    res_key = _residue_key(resseq=300, resname="GOL")  # glycerol - common cryo
    index = _make_index(
        [(res_key, [("C1", "C", (3.0, 0.0, 0.0))], "H_GOL")]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=False,
        include_water=False,
    )
    assert pocket == []


def test_hetero_included_when_flag_set() -> None:
    res_key = _residue_key(resseq=300, resname="GOL")
    index = _make_index(
        [(res_key, [("C1", "C", (3.0, 0.0, 0.0))], "H_GOL")]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=True,
        include_water=False,
    )
    assert len(pocket) == 1
    assert pocket[0]["residue"]["resname"] == "GOL"


def test_ligand_residues_excluded_from_their_own_pocket() -> None:
    """A ligand's own residue keys never appear in its pocket."""
    ligand_residue = _residue_key(resseq=999, resname="ATP")
    nearby_residue = _residue_key(resseq=10, resname="ALA")
    index = _make_index(
        [
            (ligand_residue, [("PA", "P", (0.0, 0.0, 0.0))], "H_ATP"),
            (nearby_residue, [("CA", "C", (3.0, 0.0, 0.0))], ""),
        ]
    )
    ligand_group = LigandGroup(
        ligand_id="ATP_A999",
        residue_keys=(ligand_residue,),
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=ligand_group,
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=True,
        include_water=False,
    )
    assert len(pocket) == 1
    assert pocket[0]["residue"]["resname"] == "ALA"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_index_returns_empty_pocket() -> None:
    index = _make_index([])
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=False,
        include_water=False,
    )
    assert pocket == []


def test_empty_ligand_returns_empty_pocket() -> None:
    res_key = _residue_key(resseq=10, resname="ALA")
    index = _make_index(
        [(res_key, [("CA", "C", (3.0, 0.0, 0.0))], "")]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[],
        radius=10.0,
        include_hetero=False,
        include_water=False,
    )
    assert pocket == []


def test_residues_returned_in_canonical_sort_order() -> None:
    """Output ordering must follow ``ligand_sort_key`` so existing pocket JSONs are stable."""
    # Two residues, same chain, different resseq - sort order is by resseq ascending.
    res_a = _residue_key(chain_id="A", resseq=20, resname="ALA")
    res_b = _residue_key(chain_id="A", resseq=10, resname="GLY")
    res_c = _residue_key(chain_id="A", resseq=15, resname="VAL")
    index = _make_index(
        [
            (res_a, [("CA", "C", (3.0, 0.0, 0.0))], ""),
            (res_b, [("CA", "C", (3.0, 1.0, 0.0))], ""),
            (res_c, [("CA", "C", (3.0, 2.0, 0.0))], ""),
        ]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=False,
        include_water=False,
    )
    resseqs = [entry["residue"]["resseq"] for entry in pocket]
    assert resseqs == [10, 15, 20], resseqs


def test_min_distance_is_python_float_not_numpy_scalar() -> None:
    """``json.dumps`` chokes on ``np.float32``; min_distance must be a Python float."""
    res_key = _residue_key(resseq=10, resname="ALA")
    index = _make_index(
        [(res_key, [("CA", "C", (3.0, 0.0, 0.0))], "")]
    )
    pocket = find_pocket_residues(
        search_index=index,
        ligand_group=_ligand_group(),
        ligand_atoms=[_ligand_atom((0.0, 0.0, 0.0))],
        radius=10.0,
        include_hetero=False,
        include_water=False,
    )
    assert isinstance(pocket[0]["min_distance"], float)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

ALL_TESTS = (
    test_residue_within_radius_appears_in_output,
    test_residue_outside_radius_excluded,
    test_min_distance_picks_closest_atom_in_residue,
    test_min_distance_picks_closest_ligand_atom,
    test_water_excluded_by_default,
    test_water_included_when_flag_set,
    test_hetero_excluded_by_default,
    test_hetero_included_when_flag_set,
    test_ligand_residues_excluded_from_their_own_pocket,
    test_empty_index_returns_empty_pocket,
    test_empty_ligand_returns_empty_pocket,
    test_residues_returned_in_canonical_sort_order,
    test_min_distance_is_python_float_not_numpy_scalar,
)


def run_all_tests() -> bool:
    """Entry point for the master :mod:`run_all_tests` runner."""
    return run_suite(ALL_TESTS)


if __name__ == "__main__":
    sys.exit(suite_main(ALL_TESTS))
