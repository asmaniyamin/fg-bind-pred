# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/edit/atoms.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

from core.structs import MolGraph, MolArrays, Editability, Fragment, ResonanceSystem


AtomIndexList = Union[list[int], np.ndarray]


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
    # meta caches that depend on indices (if present)
    g.meta.pop("label_to_index", None)
    return g


def _slice_optional(x: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
    if x is None:
        return None
    return x[mask]

def _slice_pos(pos: Optional[np.ndarray], atom_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Slice pos by atom mask.
    Supports pos shape (N,3) or (K,N,3).
    """
    if pos is None:
        return None
    if pos.ndim == 2:
        return pos[atom_mask]
    if pos.ndim == 3:
        return pos[:, atom_mask, :]
    raise ValueError(f"pos must have ndim 2 or 3, got {pos.ndim} with shape {pos.shape}")


def _slice_coord_valid(coord_valid: Optional[np.ndarray], atom_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Slice coord_valid by atom mask.
    Supports (N,) or (K,N).
    """
    if coord_valid is None:
        return None
    if coord_valid.ndim == 1:
        return coord_valid[atom_mask]
    if coord_valid.ndim == 2:
        return coord_valid[:, atom_mask]
    raise ValueError(f"coord_valid must have ndim 1 or 2, got {coord_valid.ndim} with shape {coord_valid.shape}")


def _permute_pos(pos: Optional[np.ndarray], new_to_old: np.ndarray) -> Optional[np.ndarray]:
    """
    Permute pos by new_to_old (new index -> old index).
    Supports (N,3) or (K,N,3).
    """
    if pos is None:
        return None
    if pos.ndim == 2:
        return pos[new_to_old]
    if pos.ndim == 3:
        return pos[:, new_to_old, :]
    raise ValueError(f"pos must have ndim 2 or 3, got {pos.ndim} with shape {pos.shape}")


def _permute_coord_valid(coord_valid: Optional[np.ndarray], new_to_old: np.ndarray) -> Optional[np.ndarray]:
    """
    Permute coord_valid by new_to_old.
    Supports (N,) or (K,N).
    """
    if coord_valid is None:
        return None
    if coord_valid.ndim == 1:
        return coord_valid[new_to_old]
    if coord_valid.ndim == 2:
        return coord_valid[:, new_to_old]
    raise ValueError(f"coord_valid must have ndim 1 or 2, got {coord_valid.ndim} with shape {coord_valid.shape}")



def _as_unique_sorted_atom_indices(atom_indices: AtomIndexList, n_atoms: int) -> np.ndarray:
    idx = np.asarray(atom_indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return idx
    if np.any(idx < 0) or np.any(idx >= n_atoms):
        bad = idx[(idx < 0) | (idx >= n_atoms)]
        raise IndexError(f"Atom indices out of bounds (0..{n_atoms - 1}): {bad.tolist()}")
    idx = np.unique(idx)
    idx.sort()
    return idx


def _compute_old_to_new_index_map(n_atoms: int, delete_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep_mask = np.ones(n_atoms, dtype=bool)
    keep_mask[delete_idx] = False

    new_index_of_old = np.full(n_atoms, -1, dtype=np.int64)
    kept = np.nonzero(keep_mask)[0]
    new_index_of_old[kept] = np.arange(kept.size, dtype=np.int64)
    return keep_mask, new_index_of_old


def _remap_fragment(f: Fragment, new_index_of_old: np.ndarray) -> Fragment:
    new_atoms = new_index_of_old[f.atom_indices]
    new_atoms = new_atoms[new_atoms >= 0]

    new_atts = new_index_of_old[f.attachment_indices]
    new_atts = new_atts[new_atts >= 0]

    return replace(
        f,
        atom_indices=new_atoms.astype(np.int64, copy=False),
        attachment_indices=new_atts.astype(np.int64, copy=False),
    )


def _remap_resonance_system(
    rs: ResonanceSystem,
    new_index_of_old_atoms: np.ndarray,
    new_index_of_old_bonds: Optional[np.ndarray],
) -> ResonanceSystem:
    atom_indices = rs.atom_indices
    if atom_indices is not None:
        mapped = new_index_of_old_atoms[atom_indices]
        mapped = mapped[mapped >= 0]
        atom_indices = mapped.astype(np.int64, copy=False)

    bond_indices = rs.bond_indices
    if bond_indices is not None and new_index_of_old_bonds is not None:
        mapped_b = new_index_of_old_bonds[bond_indices]
        mapped_b = mapped_b[mapped_b >= 0]
        bond_indices = mapped_b.astype(np.int64, copy=False)

    return replace(rs, atom_indices=atom_indices, bond_indices=bond_indices)


def _ensure_atom_aligned_length(x: np.ndarray, n: int, name: str) -> None:
    if x.shape[0] != n:
        raise ValueError(f"{name} must be length {n}, got shape {x.shape}")


def _ensure_bond_aligned_length(x: np.ndarray, m: int, name: str) -> None:
    if x.shape[0] != m:
        raise ValueError(f"{name} must be length {m}, got shape {x.shape}")


# =========================
# Public API
# =========================

"""
def delete_atom(g: MolGraph, atom_idx: int) -> MolGraph:
    # Delete a single atom (and all incident bonds).
    return delete_atoms(g, [atom_idx])
"""

def delete_atom(g: MolGraph, atom_index: int, *, return_maps: bool = False):
    """Delete a single atom (and all incident bonds)."""
    # just delegate
    return delete_atoms(g, [atom_index], return_maps=return_maps)


def delete_atoms(g: MolGraph, atom_indices: AtomIndexList, *, return_maps: bool = False):
    """
    Delete atoms from the graph and reindex all remaining atoms/bonds.

    Notes:
    - Does NOT attempt to remove orphan dummies; run dummy_cleaning.remove_orphan_dummies()
      explicitly if desired.
    - Treats g.attachments as derived cache and invalidates it.
    """
    arr = g.arrays
    n_atoms = int(arr.atomic_num.shape[0])

    delete_idx = _as_unique_sorted_atom_indices(atom_indices, n_atoms)
    if delete_idx.size == 0:
        return (g, np.arange(n_atoms, dtype=np.int64), np.arange(n_atoms, dtype=np.int64)) if return_maps else g

    keep_mask, new_index_of_old_atoms = _compute_old_to_new_index_map(n_atoms, delete_idx)

    # ---------- slice atom-aligned arrays ----------
    atomic_num = arr.atomic_num[keep_mask]
    formal_charge = arr.formal_charge[keep_mask]

    isotope = _slice_optional(arr.isotope, keep_mask)
    is_aromatic = _slice_optional(arr.is_aromatic, keep_mask)
    hybridization = _slice_optional(arr.hybridization, keep_mask)
    chiral_tag = _slice_optional(arr.chiral_tag, keep_mask)
    cip_code = _slice_optional(arr.cip_code, keep_mask)
    atom_map = _slice_optional(arr.atom_map, keep_mask)
    attachment_label = _slice_optional(arr.attachment_label, keep_mask)
    explicit_h = _slice_optional(arr.explicit_h, keep_mask)
    implicit_h = _slice_optional(arr.implicit_h, keep_mask)
    partial_charge = _slice_optional(arr.partial_charge, keep_mask)
    pos = _slice_pos(arr.pos, keep_mask)
    coord_frame = arr.coord_frame
    coord_valid = _slice_coord_valid(arr.coord_valid, keep_mask)

    # ---------- filter + remap bonds ----------
    bonds = arr.bonds
    m_bonds = int(bonds.shape[0])

    if m_bonds == 0:
        bond_keep = np.zeros((0,), dtype=bool)
        new_bonds = bonds.reshape((0, 2)).astype(arr.bonds.dtype, copy=False)
        bond_type = arr.bond_type.reshape((0,)).astype(arr.bond_type.dtype, copy=False)
        new_index_of_old_bonds = np.zeros((0,), dtype=np.int64)
    else:
        u = bonds[:, 0].astype(np.int64, copy=False)
        v = bonds[:, 1].astype(np.int64, copy=False)
        bond_keep = keep_mask[u] & keep_mask[v]

        kept_bonds = bonds[bond_keep]
        new_u = new_index_of_old_atoms[kept_bonds[:, 0].astype(np.int64, copy=False)]
        new_v = new_index_of_old_atoms[kept_bonds[:, 1].astype(np.int64, copy=False)]
        new_bonds = np.stack([new_u, new_v], axis=1).astype(arr.bonds.dtype, copy=False)

        bond_type = arr.bond_type[bond_keep]

        # bond old->new map for resonance systems
        new_index_of_old_bonds = np.full(m_bonds, -1, dtype=np.int64)
        kept_b = np.nonzero(bond_keep)[0]
        new_index_of_old_bonds[kept_b] = np.arange(kept_b.size, dtype=np.int64)

    # ---------- filter optional bond-aligned arrays ----------
    is_conjugated = arr.is_conjugated[bond_keep] if arr.is_conjugated is not None else None
    is_in_ring = arr.is_in_ring[bond_keep] if arr.is_in_ring is not None else None
    bond_dir = arr.bond_dir[bond_keep] if arr.bond_dir is not None else None
    bond_stereo = arr.bond_stereo[bond_keep] if arr.bond_stereo is not None else None
    bond_resonance_type = arr.bond_resonance_type[bond_keep] if arr.bond_resonance_type is not None else None

    # ---------- globals ----------
    total_charge = int(np.sum(formal_charge).item())
    multiplicity = arr.multiplicity

    new_arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=new_bonds,
        bond_type=bond_type,
        isotope=isotope,
        is_aromatic=is_aromatic,
        hybridization=hybridization,
        chiral_tag=chiral_tag,
        cip_code=cip_code,
        atom_map=atom_map,
        attachment_label=attachment_label,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        partial_charge=partial_charge,
        pos=pos,
        coord_frame=coord_frame,
        coord_valid=coord_valid,
        is_conjugated=is_conjugated,
        is_in_ring=is_in_ring,
        bond_dir=bond_dir,
        bond_stereo=bond_stereo,
        bond_resonance_type=bond_resonance_type,
        total_charge=total_charge,
        multiplicity=multiplicity,
    )

    # ---------- update optional non-canonical containers ----------
    g.arrays = new_arrays

    if g.editability is not None:
        # keep editability aligned (best-effort)
        atom_scores = g.editability.atom_scores[keep_mask]
        bond_scores = g.editability.bond_scores[bond_keep] if bond_scores_len_ok(g.editability, m_bonds) else g.editability.bond_scores
        g.editability = Editability(atom_scores=atom_scores, bond_scores=bond_scores)

    if g.fragments:
        g.fragments = [_remap_fragment(f, new_index_of_old_atoms) for f in g.fragments]

    if g.resonance_systems:
        g.resonance_systems = [
            _remap_resonance_system(rs, new_index_of_old_atoms, new_index_of_old_bonds)
            for rs in g.resonance_systems
        ]

    # return _invalidate_derived(g)

    # Return map if requested

    g2 = _invalidate_derived(g)

    if return_maps:
        old_to_new = new_index_of_old_atoms
        kept_old = np.nonzero(old_to_new >= 0)[0]
        order = np.argsort(old_to_new[kept_old], kind="mergesort")
        new_to_old = kept_old[order].astype(np.int64, copy=False)
        return g2, old_to_new, new_to_old
    
    return g2



def bond_scores_len_ok(editability: Editability, m_bonds: int) -> bool:
    """Guard: some pipelines may not set bond_scores length exactly M."""
    try:
        return editability.bond_scores.shape[0] == m_bonds
    except Exception:
        return False

def permute_atoms(g: MolGraph, new_to_old: np.ndarray) -> tuple[MolGraph, np.ndarray, np.ndarray]:
    """
    Reorder atoms by a permutation. Remaps all atom-indexed data and bonds.
    Returns: (g2, old_to_new, bond_perm)
    """
    arr = g.arrays
    n_atoms = int(arr.atomic_num.shape[0])

    new_to_old = np.asarray(new_to_old, dtype=np.int64).reshape(-1)
    if new_to_old.shape[0] != n_atoms:
        raise ValueError(f"new_to_old must have length {n_atoms}, got {new_to_old.shape[0]}")

    # Validate permutation
    if np.any(new_to_old < 0) or np.any(new_to_old >= n_atoms):
        raise IndexError("new_to_old contains out-of-range indices")
    if np.unique(new_to_old).shape[0] != n_atoms:
        raise ValueError("new_to_old must be a permutation (contains duplicates)")

    # old -> new map
    old_to_new = np.empty(n_atoms, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(n_atoms, dtype=np.int64)

    # ---------- permute atom-aligned arrays ----------
    atomic_num = arr.atomic_num[new_to_old]
    formal_charge = arr.formal_charge[new_to_old]

    isotope = arr.isotope[new_to_old] if arr.isotope is not None else None
    is_aromatic = arr.is_aromatic[new_to_old] if arr.is_aromatic is not None else None
    hybridization = arr.hybridization[new_to_old] if arr.hybridization is not None else None
    chiral_tag = arr.chiral_tag[new_to_old] if arr.chiral_tag is not None else None
    cip_code = arr.cip_code[new_to_old] if arr.cip_code is not None else None
    atom_map = arr.atom_map[new_to_old] if arr.atom_map is not None else None
    attachment_label = arr.attachment_label[new_to_old] if arr.attachment_label is not None else None
    explicit_h = arr.explicit_h[new_to_old] if arr.explicit_h is not None else None
    implicit_h = arr.implicit_h[new_to_old] if arr.implicit_h is not None else None
    partial_charge = arr.partial_charge[new_to_old] if arr.partial_charge is not None else None

    pos = _permute_pos(arr.pos, new_to_old)
    coord_frame = arr.coord_frame
    coord_valid = _permute_coord_valid(arr.coord_valid, new_to_old)

    # ---------- remap + normalize + sort bonds ----------
    bonds = arr.bonds.astype(np.int64, copy=False)
    m_bonds = int(bonds.shape[0])

    if m_bonds == 0:
        bond_perm = np.zeros((0,), dtype=np.int64)
        new_bonds = bonds.reshape((0, 2)).astype(arr.bonds.dtype, copy=False)
        bond_type = arr.bond_type.reshape((0,)).astype(arr.bond_type.dtype, copy=False)
    else:
        u = old_to_new[bonds[:, 0]]
        v = old_to_new[bonds[:, 1]]
        a = np.minimum(u, v)
        b = np.maximum(u, v)
        remapped = np.stack([a, b], axis=1)

        # stable sort by (u, v)
        keys = remapped[:, 0] * (n_atoms + 1) + remapped[:, 1]
        bond_perm = np.argsort(keys, kind="mergesort").astype(np.int64)
        new_bonds = remapped[bond_perm].astype(arr.bonds.dtype, copy=False)
        bond_type = arr.bond_type[bond_perm]

    # old bond idx -> new bond idx mapping (for resonance systems)
    old_to_new_bonds = np.empty(m_bonds, dtype=np.int64)
    old_to_new_bonds[bond_perm] = np.arange(m_bonds, dtype=np.int64)

    # ---------- permute optional bond arrays ----------
    is_conjugated = arr.is_conjugated[bond_perm] if arr.is_conjugated is not None else None
    is_in_ring = arr.is_in_ring[bond_perm] if arr.is_in_ring is not None else None
    bond_dir = arr.bond_dir[bond_perm] if arr.bond_dir is not None else None
    bond_stereo = arr.bond_stereo[bond_perm] if arr.bond_stereo is not None else None
    bond_resonance_type = arr.bond_resonance_type[bond_perm] if arr.bond_resonance_type is not None else None

    # ---------- globals ----------
    total_charge = int(np.sum(formal_charge).item())
    multiplicity = arr.multiplicity

    new_arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=new_bonds,
        bond_type=bond_type,
        isotope=isotope,
        is_aromatic=is_aromatic,
        hybridization=hybridization,
        chiral_tag=chiral_tag,
        cip_code=cip_code,
        atom_map=atom_map,
        attachment_label=attachment_label,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        partial_charge=partial_charge,
        pos=pos,
        coord_frame=coord_frame,
        coord_valid=coord_valid,
        is_conjugated=is_conjugated,
        is_in_ring=is_in_ring,
        bond_dir=bond_dir,
        bond_stereo=bond_stereo,
        bond_resonance_type=bond_resonance_type,
        total_charge=total_charge,
        multiplicity=multiplicity,
    )

    g.arrays = new_arrays

    # ---------- remap editability if aligned ----------
    if g.editability is not None:
        atom_scores = g.editability.atom_scores[new_to_old]
        if bond_scores_len_ok(g.editability, m_bonds):
            bond_scores = g.editability.bond_scores[bond_perm]
        else:
            bond_scores = g.editability.bond_scores
        g.editability = Editability(atom_scores=atom_scores, bond_scores=bond_scores)

    # ---------- remap fragments ----------
    if g.fragments:
        g.fragments = [_remap_fragment(f, old_to_new) for f in g.fragments]

    # ---------- remap resonance systems (atoms + bonds) ----------
    if g.resonance_systems:
        g.resonance_systems = [
            _remap_resonance_system(rs, old_to_new, old_to_new_bonds)
            for rs in g.resonance_systems
        ]

    return _invalidate_derived(g), old_to_new, bond_perm


def add_atom(
    g: MolGraph,
    *,
    atomic_num: int,
    formal_charge: int = 0,
    **fields: Any,
) -> tuple[MolGraph, int]:
    """
    Append an atom to the graph and return (new_graph, new_atom_index).

    Required fields always appended:
      - atomic_num
      - formal_charge

    Optional atom-aligned arrays are appended ONLY if they already exist in g.arrays
    (to avoid implicitly creating new optional arrays with guessed defaults).
    """
    arr = g.arrays
    n_old = int(arr.atomic_num.shape[0])
    new_idx = n_old

    # Required
    arr.atomic_num = np.concatenate([arr.atomic_num, np.asarray([atomic_num], dtype=arr.atomic_num.dtype)])
    arr.formal_charge = np.concatenate([arr.formal_charge, np.asarray([formal_charge], dtype=arr.formal_charge.dtype)])

    # Optional atom arrays: append only if present
    def append_opt(name: str, value, default):
        x = getattr(arr, name)
        if x is None:
            return
        # dtype/object handling
        if x.dtype == object:
            val = value if value is not None else default
            setattr(arr, name, np.concatenate([x, np.asarray([val], dtype=object)]))
        else:
            val = default if value is None else value
            setattr(arr, name, np.concatenate([x, np.asarray([val], dtype=x.dtype)]))

    append_opt("isotope", fields.get("isotope"), 0)
    append_opt("is_aromatic", fields.get("is_aromatic"), 0)
    append_opt("hybridization", fields.get("hybridization"), 0)
    append_opt("chiral_tag", fields.get("chiral_tag"), 0)
    append_opt("cip_code", fields.get("cip_code"), 0)
    append_opt("atom_map", fields.get("atom_map"), 0)

    # attachment_label is canonical for dummies; if appending a non-dummy atom, default is None
    if arr.attachment_label is not None:
        lab = fields.get("attachment_label", None)
        if atomic_num != 0:
            lab = None
        arr.attachment_label = np.concatenate([arr.attachment_label, np.asarray([lab], dtype=object)])

    append_opt("explicit_h", fields.get("explicit_h"), 0)
    append_opt("implicit_h", fields.get("implicit_h"), 0)
    append_opt("partial_charge", fields.get("partial_charge"), 0.0)

    # pos may be (N,3) or (K,N,3)
    if arr.pos is not None:
        pos = fields.get("pos", None)

        if arr.pos.ndim == 2:
            tail = tuple(arr.pos.shape[1:])  # (3,)
            if pos is None:
                row = np.zeros(tail, dtype=arr.pos.dtype)
            else:
                row = np.asarray(pos, dtype=arr.pos.dtype)
                if tuple(row.shape) != tail:
                    raise ValueError(f"pos must have shape {tail}, got {row.shape}")
            arr.pos = np.concatenate([arr.pos, row.reshape((1,) + tail)], axis=0)

        elif arr.pos.ndim == 3:
            k, _, d = arr.pos.shape  # (K,N,3)
            if d != 3:
                raise ValueError(f"pos last dim must be 3, got {d}")
            if pos is None:
                rows = np.zeros((k, 1, 3), dtype=arr.pos.dtype)
            else:
                rows = np.asarray(pos, dtype=arr.pos.dtype)
                if rows.shape != (k, 3):
                    raise ValueError(f"pos for ensemble must have shape (K,3) with K={k}, got {rows.shape}")
                rows = rows.reshape((k, 1, 3))
            arr.pos = np.concatenate([arr.pos, rows], axis=1)

        else:
            raise ValueError(f"pos must have ndim 2 or 3, got {arr.pos.ndim}")


    # Recompute globals
    arr.total_charge = int(np.sum(arr.formal_charge).item())

    g.arrays = arr
    g = _invalidate_derived(g)
    return g, new_idx


def replace_atom(
    g: MolGraph,
    idx: int,
    *,
    preserve_attachment_label: bool = True,
    **fields: Any,
) -> MolGraph:
    """
    Replace atom properties at index `idx` (no reindexing, no bond changes).

    Only fields provided in **fields are updated.

    Attachment behavior:
    - Dummy atoms are defined as atomic_num == 0.
    - If the resulting atomic_num != 0, we clear attachment_label at idx (unless you explicitly
      passed attachment_label, which will still be cleared because it's non-dummy).
    - If resulting atomic_num == 0, attachment_label is preserved by default unless explicitly
      provided.
    """
    arr = g.arrays
    n = int(arr.atomic_num.shape[0])
    if idx < 0 or idx >= n:
        raise IndexError(f"Atom index out of bounds (0..{n-1}): {idx}")

    # Validate and apply required-ish updates
    if "atomic_num" in fields:
        arr.atomic_num[idx] = np.asarray(fields["atomic_num"], dtype=arr.atomic_num.dtype)
    if "formal_charge" in fields:
        arr.formal_charge[idx] = np.asarray(fields["formal_charge"], dtype=arr.formal_charge.dtype)

    # Optional atom arrays: update only if present and field provided
    def set_opt(name: str):
        x = getattr(arr, name)
        if x is None or name not in fields:
            return
        if x.dtype == object:
            x[idx] = fields[name]
        else:
            x[idx] = np.asarray(fields[name], dtype=x.dtype)

    for name in [
        "isotope", "is_aromatic", "hybridization", "chiral_tag", "cip_code", "atom_map",
        "explicit_h", "implicit_h", "partial_charge",
    ]:
        set_opt(name)

    if arr.pos is not None and "pos" in fields:
        row = np.asarray(fields["pos"], dtype=arr.pos.dtype)
        if arr.pos.ndim == 2:
            # canonical (N, 3): replacement must be (3,)
            expected = tuple(arr.pos.shape[1:])  # (3,)
            if tuple(row.shape) != expected:
                raise ValueError(f"pos must have shape {expected}, got {row.shape}")
            arr.pos[idx] = row
        elif arr.pos.ndim == 3:
            # ensemble (K, N, 3): replacement must be (K, 3)
            k = arr.pos.shape[0]
            expected = (k, 3)
            if tuple(row.shape) != expected:
                raise ValueError(f"pos must have shape {expected} for ensemble, got {row.shape}")
            arr.pos[:, idx, :] = row
        else:
            raise ValueError(f"pos must have ndim 2 or 3, got {arr.pos.ndim}")

    # Attachment label rule
    new_z = int(arr.atomic_num[idx])
    if arr.attachment_label is not None:
        if "attachment_label" in fields:
            # even if user passes it, enforce canonical rule: only dummies carry labels
            arr.attachment_label[idx] = fields["attachment_label"] if new_z == 0 else None
        else:
            # preserve existing only if staying dummy and preserve flag is set
            if not preserve_attachment_label or new_z != 0:
                arr.attachment_label[idx] = None

    # Recompute globals
    arr.total_charge = int(np.sum(arr.formal_charge).item())

    g.arrays = arr
    return _invalidate_derived(g)


def set_dummy_label(g: MolGraph, idx: int, label: Any) -> MolGraph:
    """
    Set the canonical dummy attachment label at atom index idx.

    Requires atomic_num[idx] == 0 (dummy).
    """
    arr = g.arrays
    n = int(arr.atomic_num.shape[0])
    if idx < 0 or idx >= n:
        raise IndexError(f"Atom index out of bounds (0..{n-1}): {idx}")

    if int(arr.atomic_num[idx]) != 0:
        raise ValueError(f"Atom {idx} is not a dummy (atomic_num != 0); cannot set dummy label.")

    if arr.attachment_label is None:
        # If you want attachment labels in this project, you should initialize the array
        # at graph creation time. We do not auto-create it here to avoid implicit defaults.
        raise ValueError("arrays.attachment_label is None; initialize it before setting labels.")

    arr.attachment_label[idx] = label
    g.arrays = arr
    return _invalidate_derived(g)
