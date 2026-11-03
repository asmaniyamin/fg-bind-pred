# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/extract/_search.py

Pocket residue spatial search.

Two-phase split exists for cost reasons: :func:`build_pocket_search_index`
does the one-time work (sorting, per-residue coord/atom caches, the flat
protein-atom array, the protein-side ``cKDTree``) once per pipeline run;
:func:`find_pocket_residues` then runs per-ligand against the prebuilt
index.

Shared types are imported from ``_core``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from ._core import (
    AtomRecord,
    LigandGroup,
    ResidueKey,
    WATER_RESNAMES,
    collect_residue_atoms,
    ligand_sort_key,
)


@dataclass(frozen=True, slots=True)
class PocketSearchIndex:
    """One-time spatial index for :func:`find_pocket_residues`.

    Built once per pipeline run by :func:`build_pocket_search_index` and
    threaded through every per-ligand call. Carries everything the search
    needs that is *not* a function of the specific ligand: a stable
    sorted view of residue keys, per-residue caches of the atom record
    list and the cif hetflag, a flat ``(N_atoms, 3)`` numpy array of
    every protein atom's coordinates with a parallel int array mapping
    each atom back to its residue's index in ``sorted_residue_keys``,
    and a :class:`scipy.spatial.cKDTree` over the same coordinates.

    The kd-tree is what makes the per-ligand search asymptotically cheap:
    a query on a 30-atom ligand against a 50 K-atom protein visits only
    the ~600 atoms within range, regardless of ``N``. Build cost is
    O(N log N), paid once per pipeline run; per-ligand cost is
    O(L · log N + result_size). The flat coord array and residue-index
    array remain so result aggregation can stay vectorized.

    ``protein_kdtree`` is ``None`` only for the degenerate empty-protein
    case; :func:`find_pocket_residues` short-circuits before touching it.
    """

    sorted_residue_keys: tuple[ResidueKey, ...]
    residue_atoms: dict[ResidueKey, list[AtomRecord]]
    residue_hetflag: dict[ResidueKey, str]
    protein_atom_coords: np.ndarray  # shape (N_total_atoms, 3), float32
    protein_atom_residue_index: np.ndarray  # shape (N_total_atoms,), int32
    protein_kdtree: cKDTree | None


def build_pocket_search_index(
    residue_lookup: dict[ResidueKey, object],
) -> PocketSearchIndex:
    """Build the per-pipeline-run spatial index.

    Cost is one pass over every residue: extracting its atoms, recording
    the hetflag, and appending the atoms onto the flat coord array. The
    sort over residue keys happens once here instead of once per ligand
    inside :func:`find_pocket_residues`.
    """
    sorted_keys = tuple(sorted(residue_lookup.keys(), key=ligand_sort_key))
    residue_atoms: dict[ResidueKey, list[AtomRecord]] = {}
    residue_hetflag: dict[ResidueKey, str] = {}
    flat_coords: list[tuple[float, float, float]] = []
    flat_residue_indices: list[int] = []

    for residue_idx, key in enumerate(sorted_keys):
        residue = residue_lookup[key]
        atoms = collect_residue_atoms([residue])
        residue_atoms[key] = atoms
        residue_hetflag[key] = str(residue.id[0]).strip()
        for atom in atoms:
            flat_coords.append((atom.x, atom.y, atom.z))
            flat_residue_indices.append(residue_idx)

    if flat_coords:
        protein_atom_coords = np.asarray(flat_coords, dtype=np.float32)
        protein_atom_residue_index = np.asarray(flat_residue_indices, dtype=np.int32)
        # cKDTree internally upcasts to float64 once at build time; passing
        # float64 saves a per-query conversion. balanced_tree=True and
        # compact_nodes=True are the scipy defaults, restated here for
        # clarity - both improve query locality and matter for large N.
        protein_kdtree: cKDTree | None = cKDTree(
            protein_atom_coords.astype(np.float64),
            leafsize=16,
            balanced_tree=True,
            compact_nodes=True,
        )
    else:
        protein_atom_coords = np.zeros((0, 3), dtype=np.float32)
        protein_atom_residue_index = np.zeros(0, dtype=np.int32)
        protein_kdtree = None

    return PocketSearchIndex(
        sorted_residue_keys=sorted_keys,
        residue_atoms=residue_atoms,
        residue_hetflag=residue_hetflag,
        protein_atom_coords=protein_atom_coords,
        protein_atom_residue_index=protein_atom_residue_index,
        protein_kdtree=protein_kdtree,
    )


def find_pocket_residues(
    *,
    search_index: PocketSearchIndex,
    ligand_group: LigandGroup,
    ligand_atoms: Sequence[AtomRecord],
    radius: float,
    include_hetero: bool,
    include_water: bool,
) -> list[dict[str, object]]:
    """Find protein residues with any atom within ``radius`` Å of a ligand atom.

    Asymptotic cost is O(L · log N + result_size): the prebuilt protein
    :class:`scipy.spatial.cKDTree` on the index visits only the atoms
    within range, and the per-residue aggregation walks the result set
    (typically a few thousand pairs at most) once.

    Implementation notes:

    * Build a small ligand-side ``cKDTree`` per call. For ``L ≈ 30`` this
      build is a few hundred microseconds - cheaper than computing
      flat ``(lig_idx, prot_idx)`` arrays from ``query_ball_point``.
    * ``ligand_tree.sparse_distance_matrix(protein_tree, max_distance=r,
      output_type='coo_matrix')`` returns the in-range pairs as a sparse
      COO matrix with rows = ligand atom, cols = protein atom, and data
      = Euclidean distance. One scipy call gives indices *and* distances.
    * Per-residue min distance is aggregated in one Python pass over the
      sparse data - typically only thousands of entries, sub-millisecond.

    """
    if (
        len(ligand_atoms) == 0
        or len(search_index.protein_atom_coords) == 0
        or search_index.protein_kdtree is None
    ):
        return []

    ligand_coords = np.asarray(
        [[a.x, a.y, a.z] for a in ligand_atoms], dtype=np.float64
    )
    ligand_tree = cKDTree(
        ligand_coords,
        leafsize=16,
        balanced_tree=True,
        compact_nodes=True,
    )
    sparse_dm = ligand_tree.sparse_distance_matrix(
        search_index.protein_kdtree,
        max_distance=float(radius),
        output_type="coo_matrix",
    )
    if sparse_dm.nnz == 0:
        return []

    prot_indices = sparse_dm.col
    distances = sparse_dm.data
    residue_indices = search_index.protein_atom_residue_index[prot_indices]

    # Per-residue min: one pass over the (typically small) sparse result.
    per_residue_min: dict[int, float] = {}
    for r_idx, dist in zip(residue_indices.tolist(), distances.tolist()):
        prev = per_residue_min.get(r_idx)
        if prev is None or dist < prev:
            per_residue_min[r_idx] = dist

    ligand_key_set = set(ligand_group.residue_keys)
    pocket_residues: list[dict[str, object]] = []
    # Sorted residue indices preserve the canonical ligand_sort_key ordering
    # because sorted_residue_keys was sorted that way at index-build time.
    for r_idx in sorted(per_residue_min.keys()):
        residue_key = search_index.sorted_residue_keys[r_idx]
        if residue_key in ligand_key_set:
            continue
        hetflag = search_index.residue_hetflag[residue_key]
        if not include_water and residue_key.resname in WATER_RESNAMES:
            continue
        if not include_hetero and bool(hetflag) and not hetflag.startswith("W"):
            continue
        atom_records = search_index.residue_atoms[residue_key]
        if not atom_records:
            continue
        pocket_residues.append(
            {
                "residue": residue_key.to_dict(),
                "min_distance": float(per_residue_min[r_idx]),
                "atom_count": len(atom_records),
                "atoms": [record.to_dict() for record in atom_records],
            }
        )

    return pocket_residues


def minimum_distance_between_sets(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute the minimum inter-atomic distance between two coordinate sets.

    """
    deltas = coords_a[:, None, :] - coords_b[None, :, :]
    squared = np.sum(deltas * deltas, axis=2)
    return float(math.sqrt(float(np.min(squared))))
