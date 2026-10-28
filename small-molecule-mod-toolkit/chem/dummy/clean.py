# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/dummy/clean.py
from __future__ import annotations
from typing import List
import numpy as np
from core.structs import MolGraph
from chem.dummy.query import dummy_indices

def remove_orphan_dummies_mark_only(g: MolGraph) -> MolGraph:
    """
    Mark orphan dummies (degree 0) in meta for later removal.
    Logs and stores indices without deleting atoms.
    """
    d = set(map(int, dummy_indices(g).tolist()))
    if not d:
        return g

    connected = set()
    for u, v in g.arrays.bonds:
        u = int(u); v = int(v)
        if u in d: connected.add(u)
        if v in d: connected.add(v)

    orphans = sorted(d - connected)
    if orphans:
        g.meta["orphan_dummies"] = orphans
    return g

def validate_dummy_invariants(g: MolGraph) -> List[str]:
    """
    Canonical invariants only:
      - all labels must live on dummy atoms only
      - dummy indices are atomic_num==0 (definition)
    """
    warnings: List[str] = []
    if g.arrays.attachment_label is None:
        return warnings

    # label leakage onto non-dummies
    leakage = np.where((g.arrays.atomic_num != 0) & (g.arrays.attachment_label != None))[0]  # noqa: E711
    if leakage.size > 0:
        warnings.append(f"Found attachment labels on non-dummy atoms: {leakage.tolist()}")

    return warnings
