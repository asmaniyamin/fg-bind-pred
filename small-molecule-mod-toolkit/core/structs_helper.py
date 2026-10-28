# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# NOTE: WRONG DIMENSIONS IE FOR (K,N,3) vs (N,3) FOR ENSEMBLE

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from core.structs import (
    MolArrays, MolGraph, DerivedCaches, DirtyFlags,
    AttachmentPointArray,
    BondType, BondDir, BondStereo, BondResonanceType
)

# -------------------------
# Validation
# -------------------------

def _require_1d(name: str, a: Optional[np.ndarray], n: int, dtype=None) -> None:
    if a is None:
        return
    if a.shape != (n,):
        raise ValueError(f"{name} must be shape ({n},), got {a.shape}")
    if dtype is not None and a.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {a.dtype}")

def _require_bond_1d(name: str, a: Optional[np.ndarray], m: int) -> None:
    if a is None:
        return
    if a.shape != (m,):
        raise ValueError(f"{name} must be shape ({m},), got {a.shape}")

def validate_molarrays(arr: MolArrays) -> None:
    N = int(arr.atomic_num.shape[0])
    if arr.formal_charge.shape != (N,):
        raise ValueError("formal_charge must be (N,)")

    if arr.bonds.ndim != 2 or arr.bonds.shape[1] != 2:
        raise ValueError("bonds must be (M,2)")
    M = int(arr.bonds.shape[0])

    if arr.bond_type.shape != (M,):
        raise ValueError("bond_type must be (M,)")

    # Optional arrays (check shape only; dtype checks can be added)
    _require_1d("isotope", arr.isotope, N)
    _require_1d("is_aromatic", arr.is_aromatic, N)
    _require_1d("hybridization", arr.hybridization, N)
    _require_1d("chiral_tag", arr.chiral_tag, N)
    _require_1d("cip_code", arr.cip_code, N)
    _require_1d("atom_map", arr.atom_map, N)
    _require_1d("attachment_label", arr.attachment_label, N)

    _require_1d("explicit_h", arr.explicit_h, N)
    _require_1d("implicit_h", arr.implicit_h, N)
    _require_1d("partial_charge", arr.partial_charge, N)

    if arr.pos is not None:
        if arr.pos.ndim == 2:
            if arr.pos.shape not in [(N, 2), (N, 3)]:
                raise ValueError(f"pos must be ({N},2) or ({N},3), got {arr.pos.shape}")
        elif arr.pos.ndim == 3:
            k, n, d = arr.pos.shape
            if n != N or d != 3:
                raise ValueError(f"pos ensemble must be (K,{N},3), got {arr.pos.shape}")
        else:
            raise ValueError(f"pos must have ndim 2 or 3, got {arr.pos.ndim}")

    _require_bond_1d("is_conjugated", arr.is_conjugated, M)
    _require_bond_1d("is_in_ring", arr.is_in_ring, M)
    _require_bond_1d("bond_dir", arr.bond_dir, M)
    _require_bond_1d("bond_stereo", arr.bond_stereo, M)
    _require_bond_1d("bond_resonance_type", arr.bond_resonance_type, M)

def validate_attachment_points(ap: AttachmentPointArray, N: int) -> None:
    P = int(ap.idx.shape[0])
    if ap.kind.shape != (P,) or ap.target.shape != (P,):
        raise ValueError("attachments idx/kind/target must all be (P,)")

    if np.any(ap.idx < 0) or np.any(ap.idx >= N):
        raise ValueError("attachments.idx contains out-of-range atom indices")

    # target allows -1
    bad = (ap.target < -1) | (ap.target >= N)
    if np.any(bad):
        raise ValueError("attachments.target contains out-of-range indices")

    if ap.label_id is not None and ap.label_id.shape != (P,):
        raise ValueError("attachments.label_id must be (P,)")

    if ap.is_insertion is not None and ap.is_insertion.shape != (P,):
        raise ValueError("attachments.is_insertion must be (P,)")

    if ap.insertion_anchors is not None and ap.insertion_anchors.shape != (P, 2):
        raise ValueError("attachments.insertion_anchors must be (P,2)")

def validate_molgraph(g: MolGraph) -> None:
    validate_molarrays(g.arrays)
    if g.attachments is not None:
        validate_attachment_points(g.attachments, g.arrays.atomic_num.shape[0])


# -------------------------
# Cache invalidation
# -------------------------

def invalidate(g: MolGraph, structure: bool, features: bool) -> None:
    # No methods in dataclasses: mutate flags/caches here.
    if structure:
        g.dirty.structure_dirty = True
        g.cache.edge_index = None
        g.cache.bond_pair_to_index = None
        g.cache.edge_attr = None
    if features:
        g.dirty.features_dirty = True
        g.cache.x = None
        g.cache.edge_attr = None