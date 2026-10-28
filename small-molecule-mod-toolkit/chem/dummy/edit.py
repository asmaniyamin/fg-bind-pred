# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/dummy/edit.py
from __future__ import annotations
from typing import Any
import numpy as np
from core.structs import MolGraph
from chem.dummy.query import dummy_by_label

def _ensure_attachment_label_array(g: MolGraph) -> None:
    if g.arrays.attachment_label is None:
        n = int(g.arrays.atomic_num.shape[0])
        g.arrays.attachment_label = np.empty(n, dtype=object)
        g.arrays.attachment_label[:] = None

def set_dummy_label(g: MolGraph, idx: int, label: Any) -> MolGraph:
    idx = int(idx)
    if int(g.arrays.atomic_num[idx]) != 0:
        raise ValueError(f"Atom {idx} is not a dummy (atomic_num != 0)")
    _ensure_attachment_label_array(g)
    g.arrays.attachment_label[idx] = label
    return g

def clear_dummy_label(g: MolGraph, idx: int) -> MolGraph:
    idx = int(idx)
    if g.arrays.attachment_label is None:
        return g
    g.arrays.attachment_label[idx] = None
    return g

def relabel_dummy(g: MolGraph, old_label: Any, new_label: Any) -> MolGraph:
    idx = dummy_by_label(g, old_label)
    if idx is None:
        raise ValueError(f"No dummy found with label '{old_label}'")
    # optional: prevent duplicates among existing dummy labels
    if dummy_by_label(g, new_label) is not None:
        raise ValueError(f"Label '{new_label}' already exists")
    return set_dummy_label(g, idx, new_label)

def enforce_label_invariant(g: MolGraph) -> MolGraph:
    """
    Enforce: non-dummy atoms must have attachment_label = None.
    This is a canonical graph invariant.
    """
    if g.arrays.attachment_label is None:
        return g
    non_dummy = g.arrays.atomic_num != 0
    g.arrays.attachment_label[non_dummy] = None
    return g
