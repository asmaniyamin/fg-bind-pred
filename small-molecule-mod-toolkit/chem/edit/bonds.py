# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/edit/bonds.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Optional, Union

import numpy as np

from core.structs import MolGraph, MolArrays, Editability, ResonanceSystem


BondIndexList = Union[list[int], np.ndarray]


# =========================
# Internal helpers
# =========================

def _invalidate_derived(g: MolGraph) -> MolGraph:
    """Invalidate any derived scaffolding/caches after a structural mutation."""
    g.attachments = None
    g.cache.edge_index = None
    g.cache.x = None
    g.cache.edge_attr = None
    g.cache.bond_pair_to_index = None
    g.dirty.structure_dirty = True
    g.dirty.features_dirty = True
    g.rdkit_mol = None
    g.meta.pop("label_to_index", None)
    return g


def _as_unique_sorted_bond_indices(bond_indices: BondIndexList, m_bonds: int) -> np.ndarray:
    idx = np.asarray(bond_indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return idx
    if np.any(idx < 0) or np.any(idx >= m_bonds):
        bad = idx[(idx < 0) | (idx >= m_bonds)]
        raise IndexError(f"Bond indices out of bounds (0..{m_bonds - 1}): {bad.tolist()}")
    idx = np.unique(idx)
    idx.sort()
    return idx


def _slice_optional(x: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
    if x is None:
        return None
    return x[mask]


def _bond_scores_len_ok(editability: Editability, m_bonds: int) -> bool:
    try:
        return editability.bond_scores.shape[0] == m_bonds
    except Exception:
        return False


def _remap_resonance_system_bonds(
    rs: ResonanceSystem,
    new_index_of_old_bonds: np.ndarray,
) -> ResonanceSystem:
    bond_indices = rs.bond_indices
    if bond_indices is None:
        return rs
    mapped = new_index_of_old_bonds[bond_indices]
    mapped = mapped[mapped >= 0]
    return replace(rs, bond_indices=mapped.astype(np.int64, copy=False))


def _normalize_pair(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _find_bond_indices_by_pair(bonds: np.ndarray, u: int, v: int) -> np.ndarray:
    """
    Return ALL bond indices matching (min(u,v), max(u,v)).
    Bonds are assumed to be stored normalized, but we still match robustly.
    """
    a, b = _normalize_pair(u, v)
    if bonds.size == 0:
        return np.zeros((0,), dtype=np.int64)
    # robust match: either already normalized or not
    m1 = (bonds[:, 0] == a) & (bonds[:, 1] == b)
    m2 = (bonds[:, 0] == b) & (bonds[:, 1] == a)
    return np.nonzero(m1 | m2)[0].astype(np.int64)


def _append_optional(x: Optional[np.ndarray], value, default) -> Optional[np.ndarray]:
    """
    Append one element to a bond-aligned optional array.
    If x is None -> keep None (do not auto-create optional arrays).
    """
    if x is None:
        return None
    if x.dtype == object:
        val = value if value is not None else default
        return np.concatenate([x, np.asarray([val], dtype=object)])
    val = default if value is None else value
    return np.concatenate([x, np.asarray([val], dtype=x.dtype)])


# =========================
# Public API
# =========================

def delete_bond(g: MolGraph, u: int, v: int, *, undirected: bool = True) -> MolGraph:
    """
    Delete a bond between atoms u and v.
    If multiple bonds match (shouldn't happen in typical simple graphs), deletes all matches.

    Raises:
        IndexError if u/v out of bounds
        ValueError if no such bond exists
    """
    n = int(g.arrays.atomic_num.shape[0])
    if not (0 <= u < n) or not (0 <= v < n):
        raise IndexError(f"Atom indices out of bounds (0..{n-1}): u={u}, v={v}")
    if u == v:
        raise ValueError("Cannot delete a self-bond (u == v).")

    bonds = g.arrays.bonds
    idxs = _find_bond_indices_by_pair(bonds, u, v) if undirected else np.nonzero((bonds[:, 0] == u) & (bonds[:, 1] == v))[0]
    idxs = np.asarray(idxs, dtype=np.int64).reshape(-1)

    if idxs.size == 0:
        a, b = _normalize_pair(u, v)
        raise ValueError(f"No bond found between atoms {a} and {b}.")

    return delete_bonds(g, idxs)


def delete_bonds(g: MolGraph, bond_indices: BondIndexList) -> MolGraph:
    """
    Delete multiple bonds by bond index (0..M-1) and keep all bond-aligned arrays consistent.

    Updates:
      - arrays.bonds, arrays.bond_type
      - optional bond arrays: is_conjugated, is_in_ring, bond_dir, bond_stereo, bond_resonance_type
      - editability.bond_scores (if present and aligned)
      - resonance_systems bond indices remapped (dropping deleted bonds)
      - invalidates derived caches/attachments/meta
    """
    arr = g.arrays
    m = int(arr.bonds.shape[0])

    delete_idx = _as_unique_sorted_bond_indices(bond_indices, m)
    if delete_idx.size == 0:
        return g  # no-op

    keep_mask = np.ones(m, dtype=bool)
    keep_mask[delete_idx] = False

    # old->new bond index map
    new_index_of_old_bonds = np.full(m, -1, dtype=np.int64)
    kept = np.nonzero(keep_mask)[0]
    new_index_of_old_bonds[kept] = np.arange(kept.size, dtype=np.int64)

    # Required bond arrays
    bonds = arr.bonds[keep_mask]
    bond_type = arr.bond_type[keep_mask]

    # Optional bond arrays
    is_conjugated = _slice_optional(arr.is_conjugated, keep_mask)
    is_in_ring = _slice_optional(arr.is_in_ring, keep_mask)
    bond_dir = _slice_optional(arr.bond_dir, keep_mask)
    bond_stereo = _slice_optional(arr.bond_stereo, keep_mask)
    bond_resonance_type = _slice_optional(arr.bond_resonance_type, keep_mask)

    new_arrays = MolArrays(
        atomic_num=arr.atomic_num,
        formal_charge=arr.formal_charge,
        bonds=bonds,
        bond_type=bond_type,
        isotope=arr.isotope,
        is_aromatic=arr.is_aromatic,
        hybridization=arr.hybridization,
        chiral_tag=arr.chiral_tag,
        cip_code=arr.cip_code,
        atom_map=arr.atom_map,
        attachment_label=arr.attachment_label,
        explicit_h=arr.explicit_h,
        implicit_h=arr.implicit_h,
        partial_charge=arr.partial_charge,
        pos=arr.pos,
        is_conjugated=is_conjugated,
        is_in_ring=is_in_ring,
        bond_dir=bond_dir,
        bond_stereo=bond_stereo,
        bond_resonance_type=bond_resonance_type,
        total_charge=arr.total_charge,
        multiplicity=arr.multiplicity,
    )

    g.arrays = new_arrays

    # Editability: keep aligned if it was aligned
    if g.editability is not None:
        atom_scores = g.editability.atom_scores
        if _bond_scores_len_ok(g.editability, m):
            bond_scores = g.editability.bond_scores[keep_mask]
        else:
            bond_scores = g.editability.bond_scores
        g.editability = Editability(atom_scores=atom_scores, bond_scores=bond_scores)

    # Resonance systems: remap bond indices
    if g.resonance_systems:
        g.resonance_systems = [
            _remap_resonance_system_bonds(rs, new_index_of_old_bonds)
            for rs in g.resonance_systems
        ]

    return _invalidate_derived(g)


def add_bond(
    g: MolGraph,
    u: int,
    v: int,
    *,
    bond_type: int,
    **fields: Any,
) -> MolGraph:
    """
    Add a single undirected bond (u, v) to the graph.

    Behavior:
    - Normalizes the stored pair to (min(u,v), max(u,v)), matching the build/extract convention.
    - Does NOT auto-create optional bond arrays if they are None.
    - Raises if the bond already exists.

    Optional per-bond fields you MAY pass (if the corresponding array exists):
      - is_conjugated: bool
      - is_in_ring: bool
      - bond_dir: int
      - bond_stereo: int
      - bond_resonance_type: int
    """
    arr = g.arrays
    n = int(arr.atomic_num.shape[0])
    if not (0 <= u < n) or not (0 <= v < n):
        raise IndexError(f"Atom indices out of bounds (0..{n-1}): u={u}, v={v}")
    if u == v:
        raise ValueError("Cannot add a self-bond (u == v).")

    a, b = _normalize_pair(u, v)

    # Check existence (fast path via cache if present)
    if g.cache.bond_pair_to_index is not None:
        if (a, b) in g.cache.bond_pair_to_index:
            raise ValueError(f"Bond already exists between atoms {a} and {b}.")
    else:
        if _find_bond_indices_by_pair(arr.bonds, a, b).size > 0:
            raise ValueError(f"Bond already exists between atoms {a} and {b}.")

    # Append required bond arrays
    arr.bonds = np.concatenate([arr.bonds, np.asarray([[a, b]], dtype=arr.bonds.dtype)], axis=0)
    arr.bond_type = np.concatenate([arr.bond_type, np.asarray([bond_type], dtype=arr.bond_type.dtype)], axis=0)

    # Append optional bond arrays only if present
    arr.is_conjugated = _append_optional(arr.is_conjugated, fields.get("is_conjugated"), False)
    arr.is_in_ring = _append_optional(arr.is_in_ring, fields.get("is_in_ring"), False)
    arr.bond_dir = _append_optional(arr.bond_dir, fields.get("bond_dir"), 0)
    arr.bond_stereo = _append_optional(arr.bond_stereo, fields.get("bond_stereo"), 0)
    arr.bond_resonance_type = _append_optional(arr.bond_resonance_type, fields.get("bond_resonance_type"), 0)

    g.arrays = arr

    # Editability: if aligned, append a neutral score
    if g.editability is not None and _bond_scores_len_ok(g.editability, int(arr.bond_type.shape[0]) - 1):
        bs = g.editability.bond_scores
        neutral = np.asarray([0], dtype=bs.dtype) if bs.dtype != object else np.asarray([0], dtype=object)
        g.editability = Editability(atom_scores=g.editability.atom_scores, bond_scores=np.concatenate([bs, neutral]))

    return _invalidate_derived(g)

def edit_bond(
    g: MolGraph,
    u: int,
    v: int,
    *,
    bond_type: Optional[int] = None,
    is_conjugated: Optional[bool] = None,
    is_in_ring: Optional[bool] = None,
    bond_dir: Optional[int] = None,
    bond_stereo: Optional[int] = None,
    bond_resonance_type: Optional[int] = None,
) -> MolGraph:
    """
    Edit attributes of an existing undirected bond (u, v) in-place.
    Updates only provided fields; does not change atom indices or bond count.
    """
    arr = g.arrays
    n = int(arr.atomic_num.shape[0])
    if not (0 <= u < n) or not (0 <= v < n):
        raise IndexError(f"Atom indices out of bounds (0..{n-1}): u={u}, v={v}")
    if u == v:
        raise ValueError("Cannot edit a self-bond (u == v).")

    a, b = _normalize_pair(u, v)

    # Find bond index (prefer cache)
    if g.cache.bond_pair_to_index is not None and (a, b) in g.cache.bond_pair_to_index:
        bond_idx = int(g.cache.bond_pair_to_index[(a, b)])
    else:
        idxs = _find_bond_indices_by_pair(arr.bonds, a, b)
        if idxs.size == 0:
            raise ValueError(f"No bond found between atoms {a} and {b}.")
        # If duplicates exist, we edit the first (duplicates should not happen normally)
        bond_idx = int(idxs[0])

    # Required field
    if bond_type is not None:
        arr.bond_type[bond_idx] = np.asarray(bond_type, dtype=arr.bond_type.dtype)

    # Optional fields: only if arrays exist
    if is_conjugated is not None:
        if arr.is_conjugated is None:
            raise ValueError("Cannot set is_conjugated: arrays.is_conjugated is None")
        arr.is_conjugated[bond_idx] = np.asarray(bool(is_conjugated), dtype=arr.is_conjugated.dtype)

    if is_in_ring is not None:
        if arr.is_in_ring is None:
            raise ValueError("Cannot set is_in_ring: arrays.is_in_ring is None")
        arr.is_in_ring[bond_idx] = np.asarray(bool(is_in_ring), dtype=arr.is_in_ring.dtype)

    if bond_dir is not None:
        if arr.bond_dir is None:
            raise ValueError("Cannot set bond_dir: arrays.bond_dir is None")
        arr.bond_dir[bond_idx] = np.asarray(bond_dir, dtype=arr.bond_dir.dtype)

    if bond_stereo is not None:
        if arr.bond_stereo is None:
            raise ValueError("Cannot set bond_stereo: arrays.bond_stereo is None")
        arr.bond_stereo[bond_idx] = np.asarray(bond_stereo, dtype=arr.bond_stereo.dtype)

    if bond_resonance_type is not None:
        if arr.bond_resonance_type is None:
            raise ValueError("Cannot set bond_resonance_type: arrays.bond_resonance_type is None")
        arr.bond_resonance_type[bond_idx] = np.asarray(bond_resonance_type, dtype=arr.bond_resonance_type.dtype)

    g.arrays = arr
    return _invalidate_derived(g)
