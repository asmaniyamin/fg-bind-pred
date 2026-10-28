# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/dummy/query.py
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from core.structs import MolGraph

def dummy_indices(g: MolGraph) -> np.ndarray:
    """Indices of all dummy atoms (atomic_num == 0)."""
    return np.where(g.arrays.atomic_num == 0)[0]

def dummy_labels(g: MolGraph) -> np.ndarray:
    """
    Returns a length-K object array of labels for each dummy index in dummy_indices(g).
    Unlabeled dummies are None.
    """
    d = dummy_indices(g)
    if g.arrays.attachment_label is None:
        return np.array([None] * len(d), dtype=object)
    return g.arrays.attachment_label[d].astype(object, copy=False)

def dummy_by_label(g: MolGraph, label: Any) -> Optional[int]:
    """Return atom index of dummy with given label, else None."""
    if g.arrays.attachment_label is None:
        return None
    target = str(label)
    d = dummy_indices(g)
    for idx in d:
        lab = g.arrays.attachment_label[int(idx)]
        if lab is not None and str(lab) == target:
            return int(idx)
    return None

def neighbors(g: MolGraph, atom_idx: int) -> np.ndarray:
    """Return neighbors of an atom using bonds list."""
    bonds = g.arrays.bonds
    if bonds.size == 0:
        return np.zeros((0,), dtype=np.int64)
    atom_idx = int(atom_idx)
    u = bonds[:, 0].astype(np.int64, copy=False)
    v = bonds[:, 1].astype(np.int64, copy=False)
    nbrs = np.concatenate([v[u == atom_idx], u[v == atom_idx]]).astype(np.int64, copy=False)
    return nbrs

def dummy_target(g: MolGraph, dummy_idx: int) -> Optional[int]:
    """
    If dummy has exactly one *non-dummy* neighbor, return it; else None.
    (Matches old behavior where target is meaningful only for substituent-type dummies.)
    """
    dummy_idx = int(dummy_idx)
    nbrs = neighbors(g, dummy_idx)
    if nbrs.size == 0:
        return None
    real = [int(n) for n in nbrs if int(g.arrays.atomic_num[int(n)]) != 0]
    return real[0] if len(real) == 1 else None

def is_insertion_dummy(g: MolGraph, dummy_idx: int) -> bool:
    """Insertion dummy heuristic: dummy has 2+ neighbors."""
    return neighbors(g, int(dummy_idx)).size >= 2
