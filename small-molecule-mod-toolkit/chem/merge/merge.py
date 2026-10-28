# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/merge.py
"""
chem/merge.py

Careful, NumPy-first merging for MolGraph/MolArrays (core.structs).

Design goals (new project):
- Canonical dummy definition: atomic_num == 0
- Canonical dummy label: arrays.attachment_label[idx]
- No meta label maps (no label_to_index syncing)
- Attachments are DERIVED (MolGraph.attachments set to None after merge)
- Vectorized concat of MolArrays
- Preserve stereochemistry arrays; if one graph lacks stereo arrays, default-fill with NONE (0)
- If explicit H atoms exist as *nodes* (atomic_num==1) on an anchor, remove one when needed.
  If there are no explicit H nodes, ignore H "room-making" for now.

Supports two merge modes:
- Substituent merge: A(site with 1 heavy neighbor) + B(site with 1 heavy neighbor)
- Insertion merge: one side has insertion site (2 heavy neighbors), the other is substituent (1 heavy neighbor)
  (Both-insertion is not supported; raise.)

Optional RDKit validation/canonical SMILES:
- If validate_with_rdkit=True, we attempt to construct an RDKit Mol from the merged graph
  and compute a canonical SMILES. On failure, we record a warning in merge_log.

Meta:
- meta["merge_log"] is append-only and contains structured entries.
- meta["smiles"]:
    - If RDKit validation succeeds (validate_with_rdkit=True): set to canonical smiles.
    - Otherwise: set to placeholder "MERGED(<a_smiles>)+(<b_smiles>)" (non-authoritative).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from core.structs import (
    MolArrays,
    MolGraph,
    Fragment,
    BondType,
    BondDir,
    BondStereo,
    BondResonanceType,
    AtomHybridization,
    AtomChiralTag,
    AtomCIPCode,
)

from chem.dummy.query import dummy_by_label, neighbors

# We do NOT import chem.edit.atoms.delete_atoms here because we need the old->new index map.
# Instead, we implement a local batch deletion helper returning (g_new, old_to_new_map).


MergeMode = Literal["substituent", "insertion"]


# =============================================================================
# Internal: cache invalidation (derived-only)
# =============================================================================

def _invalidate_derived(g: MolGraph) -> MolGraph:
    g.attachments = None
    g.editability = None
    g.cache.edge_index = None
    g.cache.x = None
    g.cache.edge_attr = None
    g.cache.bond_pair_to_index = None
    g.dirty.structure_dirty = True
    g.dirty.features_dirty = True
    g.rdkit_mol = None
    return g


# =============================================================================
# Internal: site resolution
# =============================================================================

class _SiteInfo:
    __slots__ = ("label", "dummy_idx", "heavy_neighbors", "h_neighbors", "mode")

    def __init__(
        self,
        *,
        label: Any,
        dummy_idx: int,
        heavy_neighbors: np.ndarray,
        h_neighbors: np.ndarray,
        mode: Literal["substituent", "insertion"],
    ) -> None:
        self.label = label
        self.dummy_idx = int(dummy_idx)
        self.heavy_neighbors = heavy_neighbors.astype(np.int64, copy=False)
        self.h_neighbors = h_neighbors.astype(np.int64, copy=False)
        self.mode = mode


def resolve_site(g: MolGraph, label: Any) -> _SiteInfo:
    """
    Resolve a labeled dummy attachment site using canonical arrays.

    Rules:
      - dummy atom: atomic_num == 0
      - label: arrays.attachment_label[dummy_idx]
      - heavy neighbors: atomic_num > 1
      - H neighbors: atomic_num == 1 (explicit H nodes)

    Classification:
      - 1 heavy neighbor => substituent site (anchor = that atom)
      - 2 heavy neighbors => insertion site (anchors = those two atoms)
      - else => error
    """
    d_idx = dummy_by_label(g, label)
    if d_idx is None:
        raise ValueError(f"resolve_site: no dummy found with label '{label}'")

    d_idx = int(d_idx)
    if int(g.arrays.atomic_num[d_idx]) != 0:
        # Should not happen given dummy_by_label scans dummy_indices, but be defensive.
        raise ValueError(f"resolve_site: label '{label}' points to non-dummy atom {d_idx}")

    nbrs = neighbors(g, d_idx)
    if nbrs.size == 0:
        raise ValueError(f"resolve_site: dummy {d_idx} (label '{label}') has no neighbors")

    z = g.arrays.atomic_num.astype(np.int64, copy=False)
    heavy = nbrs[z[nbrs] > 1]
    hs = nbrs[z[nbrs] == 1]

    # Sort for determinism
    heavy = np.unique(heavy)
    heavy.sort()
    hs = np.unique(hs)
    hs.sort()

    if heavy.size == 1:
        mode: Literal["substituent", "insertion"] = "substituent"
    elif heavy.size == 2:
        mode = "insertion"
    else:
        raise ValueError(
            f"resolve_site: unsupported site at dummy {d_idx} (label '{label}'): "
            f"{heavy.size} heavy neighbors (need 1=substituent or 2=insertion)"
        )

    return _SiteInfo(label=label, dummy_idx=d_idx, heavy_neighbors=heavy, h_neighbors=hs, mode=mode)


# =============================================================================
# Internal: batch delete atoms with mapping
# =============================================================================

def _as_unique_sorted_ints(idxs: Iterable[int]) -> np.ndarray:
    a = np.asarray(list(idxs), dtype=np.int64).reshape(-1)
    if a.size == 0:
        return a
    a = np.unique(a)
    a.sort()
    return a


def _slice_optional(x: Optional[np.ndarray], keep_mask: np.ndarray) -> Optional[np.ndarray]:
    if x is None:
        return None
    return x[keep_mask]


def _delete_atoms_with_map(g: MolGraph, atom_indices: Sequence[int]) -> Tuple[MolGraph, np.ndarray]:
    """
    Delete atoms and all incident bonds. Returns:
      - new graph
      - old_to_new map (length old_N), where deleted atoms map to -1
    """
    arr = g.arrays
    n = int(arr.atomic_num.shape[0])

    del_idx = _as_unique_sorted_ints(atom_indices)
    if del_idx.size == 0:
        old_to_new = np.arange(n, dtype=np.int64)
        return g, old_to_new

    if np.any(del_idx < 0) or np.any(del_idx >= n):
        bad = del_idx[(del_idx < 0) | (del_idx >= n)]
        raise IndexError(f"_delete_atoms_with_map: atom indices out of bounds (0..{n-1}): {bad.tolist()}")

    keep_atoms = np.ones(n, dtype=bool)
    keep_atoms[del_idx] = False

    old_to_new = np.full(n, -1, dtype=np.int64)
    kept = np.nonzero(keep_atoms)[0]
    old_to_new[kept] = np.arange(kept.size, dtype=np.int64)

    # Filter bonds that touch deleted atoms
    if arr.bonds.size == 0:
        keep_bonds = np.zeros((0,), dtype=bool)
    else:
        b0 = arr.bonds[:, 0].astype(np.int64, copy=False)
        b1 = arr.bonds[:, 1].astype(np.int64, copy=False)
        keep_bonds = keep_atoms[b0] & keep_atoms[b1]

    # Remap bond endpoints
    if arr.bonds.size == 0:
        new_bonds = arr.bonds.copy()
        new_bond_type = arr.bond_type.copy()
    else:
        b = arr.bonds[keep_bonds].astype(np.int64, copy=False)
        new_bonds = np.stack([old_to_new[b[:, 0]], old_to_new[b[:, 1]]], axis=1).astype(np.int32, copy=False)

        # Normalize (min, max) to match the build convention
        lo = np.minimum(new_bonds[:, 0], new_bonds[:, 1])
        hi = np.maximum(new_bonds[:, 0], new_bonds[:, 1])
        new_bonds = np.stack([lo, hi], axis=1).astype(np.int32, copy=False)

        new_bond_type = arr.bond_type[keep_bonds]

    # Build new MolArrays, slicing atom-aligned arrays with keep_atoms and bond-aligned with keep_bonds
    new_arrays = MolArrays(
        atomic_num=arr.atomic_num[keep_atoms],
        formal_charge=arr.formal_charge[keep_atoms],
        bonds=new_bonds,
        bond_type=new_bond_type,
        isotope=_slice_optional(arr.isotope, keep_atoms),
        is_aromatic=_slice_optional(arr.is_aromatic, keep_atoms),
        hybridization=_slice_optional(arr.hybridization, keep_atoms),
        chiral_tag=_slice_optional(arr.chiral_tag, keep_atoms),
        cip_code=_slice_optional(arr.cip_code, keep_atoms),
        atom_map=_slice_optional(arr.atom_map, keep_atoms),
        attachment_label=_slice_optional(arr.attachment_label, keep_atoms),
        explicit_h=_slice_optional(arr.explicit_h, keep_atoms),
        implicit_h=_slice_optional(arr.implicit_h, keep_atoms),
        partial_charge=_slice_optional(arr.partial_charge, keep_atoms),
        # pos=_slice_optional(arr.pos, keep_atoms), [NOTE 7] decided to set to None for relieving difficulty of geometric merging
        pos=None,
        coord_frame=None,
        coord_valid=None,
        is_conjugated=_slice_optional(arr.is_conjugated, keep_bonds),
        is_in_ring=_slice_optional(arr.is_in_ring, keep_bonds),
        bond_dir=_slice_optional(arr.bond_dir, keep_bonds),
        bond_stereo=_slice_optional(arr.bond_stereo, keep_bonds),
        bond_resonance_type=_slice_optional(arr.bond_resonance_type, keep_bonds),
        total_charge=int(np.sum(arr.formal_charge[keep_atoms])) if arr.formal_charge is not None else 0,
        multiplicity=arr.multiplicity,
    )

    # Remap fragments (preserve, but drop deleted atoms)
    new_fragments: List[Fragment] = []
    for fr in g.fragments:
        ai = fr.atom_indices.astype(np.int64, copy=False)
        ni = old_to_new[ai]
        ni = ni[ni >= 0]
        at = fr.attachment_indices.astype(np.int64, copy=False)
        nt = old_to_new[at]
        nt = nt[nt >= 0]
        new_fragments.append(replace(fr, atom_indices=ni.astype(np.int64, copy=False), attachment_indices=nt.astype(np.int64, copy=False)))

    g2 = MolGraph(
        arrays=new_arrays,
        attachments=None,
        editability=None,  # drop editability
        fragments=new_fragments,
        resonance_systems=[],  # per updated policy: do not carry resonance systems
        cache=g.cache,
        dirty=g.dirty,
        meta=dict(g.meta),
        rdkit_mol=None,
    )
    _invalidate_derived(g2)
    return g2, old_to_new


# =============================================================================
# Internal: explicit H removal (node-based)
# =============================================================================

def _maybe_remove_one_explicit_h(g: MolGraph, anchor_idx: int) -> Tuple[MolGraph, np.ndarray]:
    """
    If anchor has an explicit H atom neighbor (atomic_num==1), delete one such H.
    Returns (g_new, old_to_new_map). If no H deleted, map is identity.
    """
    anchor_idx = int(anchor_idx)
    nbrs = neighbors(g, anchor_idx)
    if nbrs.size == 0:
        old_to_new = np.arange(int(g.arrays.atomic_num.shape[0]), dtype=np.int64)
        return g, old_to_new

    z = g.arrays.atomic_num.astype(np.int64, copy=False)
    h_nbrs = nbrs[z[nbrs] == 1]
    if h_nbrs.size == 0:
        old_to_new = np.arange(int(g.arrays.atomic_num.shape[0]), dtype=np.int64)
        return g, old_to_new

    h_idx = int(h_nbrs[0])
    return _delete_atoms_with_map(g, [h_idx])


# =============================================================================
# Internal: concat graphs (vectorized)
# =============================================================================

def _default_fill_atom(n: int, dtype, fill_value):
    a = np.empty(n, dtype=dtype)
    a[:] = fill_value
    return a


def _default_fill_obj(n: int, fill_value=None):
    a = np.empty(n, dtype=object)
    a[:] = fill_value
    return a


def _union_atom_optional(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    n_a: int,
    n_b: int,
    *,
    default_value,
    dtype_if_missing,
    object_array: bool = False,
) -> Optional[np.ndarray]:
    if a is None and b is None:
        return None
    if object_array:
        a2 = a if a is not None else _default_fill_obj(n_a, default_value)
        b2 = b if b is not None else _default_fill_obj(n_b, default_value)
        return np.concatenate([a2, b2], axis=0)
    a2 = a if a is not None else _default_fill_atom(n_a, dtype_if_missing, default_value)
    b2 = b if b is not None else _default_fill_atom(n_b, dtype_if_missing, default_value)
    return np.concatenate([a2, b2], axis=0)


def _union_bond_optional(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    m_a: int,
    m_b: int,
    *,
    default_value,
    dtype_if_missing,
) -> Optional[np.ndarray]:
    if a is None and b is None:
        return None
    a2 = a if a is not None else _default_fill_atom(m_a, dtype_if_missing, default_value)
    b2 = b if b is not None else _default_fill_atom(m_b, dtype_if_missing, default_value)
    return np.concatenate([a2, b2], axis=0)


def concat_graphs(g_a: MolGraph, g_b: MolGraph) -> Tuple[MolGraph, int]:
    """
    Concatenate two graphs at the array level (no bonding yet).
    - Offsets B atom indices by N_A.
    - Preserves existing optional arrays; if either side has a given optional, default-fill the other.
    - Drops editability, drops attachments, drops resonance systems (per policy).
    - Unions fragments (B fragment indices offset).
    Returns (g_ab, offset_atoms).
    """
    a = g_a.arrays
    b = g_b.arrays
    n_a = int(a.atomic_num.shape[0])
    n_b = int(b.atomic_num.shape[0])
    m_a = int(a.bonds.shape[0])
    m_b = int(b.bonds.shape[0])
    offset = n_a

    # Required atom arrays
    atomic_num = np.concatenate([a.atomic_num, b.atomic_num], axis=0)
    formal_charge = np.concatenate([a.formal_charge, b.formal_charge], axis=0)

    # Bonds (offset B endpoints)
    if m_b > 0:
        b_bonds = b.bonds.astype(np.int64, copy=False) + offset
        b_bonds = b_bonds.astype(np.int32, copy=False)
    else:
        b_bonds = b.bonds

    bonds = np.concatenate([a.bonds, b_bonds], axis=0) if (m_a + m_b) > 0 else np.zeros((0, 2), dtype=np.int32)
    bond_type = np.concatenate([a.bond_type, b.bond_type], axis=0) if (m_a + m_b) > 0 else np.zeros((0,), dtype=np.int8)

    # Normalize bonds (min,max) for safety
    if bonds.size > 0:
        lo = np.minimum(bonds[:, 0], bonds[:, 1])
        hi = np.maximum(bonds[:, 0], bonds[:, 1])
        bonds = np.stack([lo, hi], axis=1).astype(np.int32, copy=False)

    # Atom optional union + default fill
    isotope = _union_atom_optional(a.isotope, b.isotope, n_a, n_b, default_value=0, dtype_if_missing=np.int16)
    is_aromatic = _union_atom_optional(a.is_aromatic, b.is_aromatic, n_a, n_b, default_value=False, dtype_if_missing=bool)
    hybridization = _union_atom_optional(a.hybridization, b.hybridization, n_a, n_b, default_value=AtomHybridization.OTHER, dtype_if_missing=np.int8)
    chiral_tag = _union_atom_optional(a.chiral_tag, b.chiral_tag, n_a, n_b, default_value=AtomChiralTag.UNSPECIFIED, dtype_if_missing=np.int8)
    cip_code = _union_atom_optional(a.cip_code, b.cip_code, n_a, n_b, default_value=AtomCIPCode.NONE, dtype_if_missing=np.int8)
    atom_map = _union_atom_optional(a.atom_map, b.atom_map, n_a, n_b, default_value=0, dtype_if_missing=np.int32)
    attachment_label = _union_atom_optional(
        a.attachment_label, b.attachment_label, n_a, n_b,
        default_value=None, dtype_if_missing=object, object_array=True
    )
    explicit_h = _union_atom_optional(a.explicit_h, b.explicit_h, n_a, n_b, default_value=0, dtype_if_missing=np.int8)
    implicit_h = _union_atom_optional(a.implicit_h, b.implicit_h, n_a, n_b, default_value=0, dtype_if_missing=np.int8)
    partial_charge = _union_atom_optional(a.partial_charge, b.partial_charge, n_a, n_b, default_value=0.0, dtype_if_missing=np.float32)
    # pos = _union_atom_optional(a.pos, b.pos, n_a, n_b, default_value=0.0, dtype_if_missing=np.float32)  # assumes shape compatibility if present [NOTE 7] Set to None to relieve geometry errors
    pos=None
    coord_frame=None
    coord_valid=None
    # Bond optional union + default fill (preserve stereo; fill missing with NONE=0)
    is_conjugated = _union_bond_optional(a.is_conjugated, b.is_conjugated, m_a, m_b, default_value=False, dtype_if_missing=bool)
    is_in_ring = _union_bond_optional(a.is_in_ring, b.is_in_ring, m_a, m_b, default_value=False, dtype_if_missing=bool)
    bond_dir = _union_bond_optional(a.bond_dir, b.bond_dir, m_a, m_b, default_value=BondDir.NONE, dtype_if_missing=np.int8)
    bond_stereo = _union_bond_optional(a.bond_stereo, b.bond_stereo, m_a, m_b, default_value=BondStereo.NONE, dtype_if_missing=np.int8)
    bond_resonance_type = _union_bond_optional(a.bond_resonance_type, b.bond_resonance_type, m_a, m_b, default_value=BondResonanceType.NONE, dtype_if_missing=np.int8)

    total_charge = int(np.sum(formal_charge)) if formal_charge is not None else 0

    arrays = MolArrays(
        atomic_num=atomic_num,
        formal_charge=formal_charge,
        bonds=bonds,
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
        multiplicity=a.multiplicity if a.multiplicity is not None else b.multiplicity,
    )

    # Fragments union (offset B indices)
    frags: List[Fragment] = []
    frags.extend(g_a.fragments)

    for fr in g_b.fragments:
        ai = (fr.atom_indices.astype(np.int64, copy=False) + offset).astype(np.int64, copy=False)
        at = (fr.attachment_indices.astype(np.int64, copy=False) + offset).astype(np.int64, copy=False)
        frags.append(replace(fr, atom_indices=ai, attachment_indices=at))

    meta = dict(g_a.meta)  # base on A
    g_ab = MolGraph(
        arrays=arrays,
        attachments=None,
        editability=None,
        fragments=frags,
        resonance_systems=[],  # per your updated policy
        meta=meta,
    )
    _invalidate_derived(g_ab)
    return g_ab, offset


# =============================================================================
# Internal: add bond (append to arrays, preserve optional bond arrays)
# =============================================================================

def _append_bond(g: MolGraph, u: int, v: int, bond_type_code: int = BondType.SINGLE) -> MolGraph:
    """
    Append an undirected bond (u,v) and keep bond-aligned arrays consistent.
    Does not auto-create optional arrays; but if they exist, appends default values.
    """
    arr = g.arrays
    u = int(u); v = int(v)
    if u == v:
        raise ValueError("Cannot add self-bond (u==v).")

    a = min(u, v)
    b = max(u, v)

    # Append bonds / bond_type
    arr.bonds = np.concatenate([arr.bonds, np.asarray([[a, b]], dtype=np.int32)], axis=0)
    arr.bond_type = np.concatenate([arr.bond_type, np.asarray([bond_type_code], dtype=arr.bond_type.dtype)], axis=0)

    # Optional bond arrays: if present, append defaults
    if arr.is_conjugated is not None:
        arr.is_conjugated = np.concatenate([arr.is_conjugated, np.asarray([False], dtype=arr.is_conjugated.dtype)], axis=0)
    if arr.is_in_ring is not None:
        arr.is_in_ring = np.concatenate([arr.is_in_ring, np.asarray([False], dtype=arr.is_in_ring.dtype)], axis=0)
    if arr.bond_dir is not None:
        arr.bond_dir = np.concatenate([arr.bond_dir, np.asarray([BondDir.NONE], dtype=arr.bond_dir.dtype)], axis=0)
    if arr.bond_stereo is not None:
        arr.bond_stereo = np.concatenate([arr.bond_stereo, np.asarray([BondStereo.NONE], dtype=arr.bond_stereo.dtype)], axis=0)
    if arr.bond_resonance_type is not None:
        arr.bond_resonance_type = np.concatenate([arr.bond_resonance_type, np.asarray([BondResonanceType.NONE], dtype=arr.bond_resonance_type.dtype)], axis=0)

    g.arrays = arr
    _invalidate_derived(g)
    return g


# =============================================================================
# Meta merge logging
# =============================================================================

def _get_merge_log(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    log = meta.get("merge_log")
    if log is None:
        return []
    if isinstance(log, list):
        return list(log)
    # If corrupted, overwrite with empty list
    return []


def _next_merge_id(merge_log: List[Dict[str, Any]]) -> int:
    best = 0
    for e in merge_log:
        try:
            best = max(best, int(e.get("id", 0)))
        except Exception:
            continue
    return best + 1


def _append_merge_log(
    merged: MolGraph,
    *,
    mode: MergeMode,
    a_smiles: Optional[str],
    b_smiles: Optional[str],
    a_label: Any,
    b_label: Any,
    warnings: Optional[List[str]] = None,
) -> None:
    merge_log = _get_merge_log(merged.meta)
    mid = _next_merge_id(merge_log)
    ts = datetime.now().astimezone().isoformat()

    msg = f"merge {mid}: [{a_smiles}] with [{b_smiles}]"
    entry: Dict[str, Any] = {
        "type": "merge",
        "id": mid,
        "timestamp": ts,
        "mode": mode,
        "a_smiles": a_smiles,
        "b_smiles": b_smiles,
        "a_label": str(a_label),
        "b_label": str(b_label),
        "message": msg,
    }
    if warnings:
        entry["warnings"] = list(warnings)

    merge_log.append(entry)
    merged.meta["merge_log"] = merge_log


# =============================================================================
# Public API: merge by labels
# =============================================================================

def merge_by_labels(
    g_a: MolGraph,
    g_b: MolGraph,
    *,
    label_a: Any,
    label_b: Any,
    bond_type_code: int = BondType.SINGLE,
    validate_with_rdkit: bool = False,
    on_rdkit_fail: Literal["warn", "raise"] = "warn",
    coords: Optional[Literal["canonical", "ensemble"]] = None,  # NEW
    num_confs: int = 10,                                        # NEW
    optimize_coords: bool = False,                              # NEW
    random_seed: int = 42,                                      # NEW
) -> MolGraph:
    """
    Merge g_b into g_a using labeled dummy attachment points.

    Rules:
      - Resolve each label to a dummy atom (atomic_num==0).
      - Substituent merge: both sites have exactly 1 heavy neighbor.
      - Insertion merge: one site has exactly 2 heavy neighbors, the other has 1 heavy neighbor.
      - Explicit H removal: if an anchor has a neighbor atom with atomic_num==1, remove one such H node.
      - Delete the dummy atoms involved (always).
      - Concatenate graphs (vectorized).
      - Add required bond(s).
      - Drop editability, attachments, resonance systems.
      - Append merge history to meta["merge_log"].
      - meta["smiles"]:
          - If validate_with_rdkit succeeds: set to canonical smiles (from RDKit).
          - Else: set to placeholder MERGED(a)+ (b).

    Returns:
      MolGraph (merged).
    """
    site_a = resolve_site(g_a, label_a)
    site_b = resolve_site(g_b, label_b)

    # Determine merge mode and anchors
    if site_a.mode == "substituent" and site_b.mode == "substituent":
        mode: MergeMode = "substituent"
        anchors_a = [int(site_a.heavy_neighbors[0])]
        anchors_b = [int(site_b.heavy_neighbors[0])]
    elif site_a.mode == "insertion" and site_b.mode == "substituent":
        mode = "insertion"
        anchors_a = [int(site_a.heavy_neighbors[0]), int(site_a.heavy_neighbors[1])]  # two anchors
        anchors_b = [int(site_b.heavy_neighbors[0])]  # one anchor
    elif site_a.mode == "substituent" and site_b.mode == "insertion":
        mode = "insertion"
        anchors_a = [int(site_a.heavy_neighbors[0])]  # one anchor
        anchors_b = [int(site_b.heavy_neighbors[0]), int(site_b.heavy_neighbors[1])]  # two anchors
    else:
        raise ValueError(
            f"merge_by_labels: unsupported combination: site_a={site_a.mode} (label {label_a}) "
            f"and site_b={site_b.mode} (label {label_b}). Both-insertion is not supported."
        )

    # Keep smiles for logging
    a_smiles = g_a.meta.get("smiles")
    b_smiles = g_b.meta.get("smiles")

    warnings: List[str] = []

    # --- Prepare each graph: remove explicit H nodes (if present) on anchors, then delete dummy ---
    # We do H removals first so mapping is clean and deterministic.

    # Prepare A
    # Keep anchor/dummy indices in ORIGINAL-index space throughout. mapA always maps
    # original → current, so we use mapA[orig] to get the live index for each call.
    gA = g_a
    mapA = np.arange(int(gA.arrays.atomic_num.shape[0]), dtype=np.int64)
    anchors_a_orig = list(anchors_a)
    dummy_a_orig = site_a.dummy_idx
    for anc_orig in anchors_a_orig:
        anc_cur = int(mapA[anc_orig])
        if anc_cur < 0:
            raise ValueError("merge_by_labels: anchor in graph A was deleted unexpectedly during H removal")
        gA, m = _maybe_remove_one_explicit_h(gA, anc_cur)
        mapA = m[mapA]  # compose: mapA now maps original → post-this-removal

    # Apply final mapA to resolve all original indices → current indices
    anchors_a = [int(mapA[x]) for x in anchors_a_orig]
    if any(x < 0 for x in anchors_a):
        raise ValueError("merge_by_labels: anchor in graph A became invalid after H removal")
    site_a.dummy_idx = int(mapA[dummy_a_orig])
    if site_a.dummy_idx < 0:
        raise ValueError("merge_by_labels: dummy in graph A was deleted unexpectedly during H removal")

    # Now delete dummy in A
    gA, m = _delete_atoms_with_map(gA, [site_a.dummy_idx])
    mapA = m[mapA]
    anchors_a = [int(mapA[x]) for x in anchors_a_orig]
    if any(x < 0 for x in anchors_a):
        raise ValueError("merge_by_labels: anchor in graph A became invalid after deleting dummy")

    # Prepare B
    gB = g_b
    mapB = np.arange(int(gB.arrays.atomic_num.shape[0]), dtype=np.int64)
    anchors_b_orig = list(anchors_b)
    dummy_b_orig = site_b.dummy_idx
    for anc_orig in anchors_b_orig:
        anc_cur = int(mapB[anc_orig])
        if anc_cur < 0:
            raise ValueError("merge_by_labels: anchor in graph B was deleted unexpectedly during H removal")
        gB, m = _maybe_remove_one_explicit_h(gB, anc_cur)
        mapB = m[mapB]

    anchors_b = [int(mapB[x]) for x in anchors_b_orig]
    if any(x < 0 for x in anchors_b):
        raise ValueError("merge_by_labels: anchor in graph B became invalid after H removal")
    site_b.dummy_idx = int(mapB[dummy_b_orig])
    if site_b.dummy_idx < 0:
        raise ValueError("merge_by_labels: dummy in graph B was deleted unexpectedly during H removal")

    gB, m = _delete_atoms_with_map(gB, [site_b.dummy_idx])
    mapB = m[mapB]
    anchors_b = [int(mapB[x]) for x in anchors_b_orig]
    if any(x < 0 for x in anchors_b):
        raise ValueError("merge_by_labels: anchor in graph B became invalid after deleting dummy")

    # --- Concat graphs ---
    merged, offset = concat_graphs(gA, gB)

    # Remap B anchors into merged space
    anchors_b_merged = [int(x + offset) for x in anchors_b]
    anchors_a_merged = [int(x) for x in anchors_a]

    # --- Add bond(s) ---
    if mode == "substituent":
        merged = _append_bond(merged, anchors_a_merged[0], anchors_b_merged[0], bond_type_code=bond_type_code)
    else:
        # insertion: one side has 2 anchors and the other has 1 anchor
        if len(anchors_a_merged) == 2 and len(anchors_b_merged) == 1:
            center = anchors_b_merged[0]
            merged = _append_bond(merged, anchors_a_merged[0], center, bond_type_code=bond_type_code)
            merged = _append_bond(merged, anchors_a_merged[1], center, bond_type_code=bond_type_code)
        elif len(anchors_a_merged) == 1 and len(anchors_b_merged) == 2:
            center = anchors_a_merged[0]
            merged = _append_bond(merged, anchors_b_merged[0], center, bond_type_code=bond_type_code)
            merged = _append_bond(merged, anchors_b_merged[1], center, bond_type_code=bond_type_code)
        else:
            raise ValueError("merge_by_labels: internal error determining insertion anchors")

    # --- Meta: merge logs & smiles placeholder (non-authoritative unless RDKit validated) ---
    # Combine existing logs from both graphs
    logA = _get_merge_log(g_a.meta)
    logB = _get_merge_log(g_b.meta)
    merged.meta["merge_log"] = list(logA) + list(logB)

    # Default placeholder smiles unless we successfully validate
    merged.meta["smiles"] = f"MERGED({a_smiles})+({b_smiles})"

    # Append new merge entry
    _append_merge_log(
        merged,
        mode=mode,
        a_smiles=a_smiles,
        b_smiles=b_smiles,
        a_label=label_a,
        b_label=label_b,
        warnings=warnings if warnings else None,
    )

    # --- Optional RDKit validation & canonical smiles ---
    if validate_with_rdkit:
        try:
            from chem.build.molgraph_to_mol import molgraph_to_mol, molgraph_to_smiles
            # Build mol; molgraph_to_mol() already sanitizes internally.
            _ = molgraph_to_mol(merged)
            canon = molgraph_to_smiles(merged)
            merged.meta["smiles"] = canon
        except Exception as e:
            msg = f"RDKit validation failed: {type(e).__name__}: {e}"
            if on_rdkit_fail == "raise":
                raise
            # Warn via merge_log (append warning to last entry)
            ml = _get_merge_log(merged.meta)
            if ml:
                ml[-1].setdefault("warnings", [])
                ml[-1]["warnings"].append(msg)
                merged.meta["merge_log"] = ml
            # Keep placeholder smiles

    # --- Optional 3D coordinate generation for merged molecule ---
    if coords is not None:
        try:
            from chem.build.molgraph_to_mol import molgraph_to_mol
            from chem.build.create_molgraph import generate_coordinates
            
            # Convert merged graph to RDKit mol
            mol = molgraph_to_mol(merged)
            
            # Generate coordinates
            pos, coord_frame, coord_valid = generate_coordinates(
                mol,
                mode=coords,
                num_confs=num_confs,
                optimize=optimize_coords,
                random_seed=random_seed,
            )
            
            if pos is not None:
                # Successfully generated coordinates
                merged.arrays.pos = pos
                merged.arrays.coord_frame = coord_frame
                merged.arrays.coord_valid = coord_valid
                
                # Update metadata
                merged.meta["coords_mode"] = coords
                if coords == "ensemble":
                    merged.meta["num_confs"] = pos.shape[0]
                else:
                    merged.meta["num_confs"] = 1
            else:
                # Generation failed (returned None)
                msg = "Coordinate generation failed for merged molecule"
                ml = _get_merge_log(merged.meta)
                if ml:
                    ml[-1].setdefault("warnings", [])
                    ml[-1]["warnings"].append(msg)
                    merged.meta["merge_log"] = ml
                    
        except Exception as e:
            # Don't fail the merge, just log the error
            msg = f"Coordinate generation error: {type(e).__name__}: {e}"
            ml = _get_merge_log(merged.meta)
            if ml:
                ml[-1].setdefault("warnings", [])
                ml[-1]["warnings"].append(msg)
                merged.meta["merge_log"] = ml

    _invalidate_derived(merged)
    return merged
