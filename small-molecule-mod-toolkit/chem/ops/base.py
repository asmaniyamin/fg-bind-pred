# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/ops/base.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import time
import numpy as np

from core.structs import MolGraph

"""
Ask Claude to fix this idea
"""

# -----------------------------
# Change summaries + op records
# -----------------------------

@dataclass(slots=True)
class ChangeSummary:
    """
    Minimal, always-available summary of what changed.
    Extend later if you want finer-grained diffs.
    """
    n_atoms_before: int
    n_atoms_after: int
    n_bonds_before: int
    n_bonds_after: int

    # Optional index-level info (when op can provide it cheaply)
    atoms_removed_old: Optional[np.ndarray] = None   # indices in old graph
    atoms_added_new: Optional[np.ndarray] = None     # indices in new graph
    bonds_removed_old: Optional[np.ndarray] = None
    bonds_added_new: Optional[np.ndarray] = None

    # Freeform notes
    notes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class OpInfo:
    """
    Standard info payload returned by Transform.apply().
    This is what you log into mg.meta["op_log"].
    """
    op: str
    params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # Common mappings (present when relevant)
    old_to_new: Optional[np.ndarray] = None  # shape (N_old,), -1 for deleted
    new_to_old: Optional[np.ndarray] = None  # shape (N_new,)
    bond_perm: Optional[np.ndarray] = None   # optional (depends on op)

    changes: Optional[ChangeSummary] = None

    # Timestamp (helps reproducibility/debugging)
    t_unix: float = field(default_factory=lambda: time.time())


def _count_atoms_bonds(mg: MolGraph) -> tuple[int, int]:
    a = mg.arrays
    n = int(a.atomic_num.shape[0])
    m = int(a.bonds.shape[0])
    return n, m


def _make_change_summary(mg_before: MolGraph, mg_after: MolGraph) -> ChangeSummary:
    n0, m0 = _count_atoms_bonds(mg_before)
    n1, m1 = _count_atoms_bonds(mg_after)
    return ChangeSummary(
        n_atoms_before=n0,
        n_atoms_after=n1,
        n_bonds_before=m0,
        n_bonds_after=m1,
    )


def _np_to_jsonish(x: np.ndarray) -> Any:
    """
    Convert numpy arrays to a stable, JSON-ish representation.
    Operation logs retain numpy arrays for downstream consumers.
    """
    # Keep as numpy arrays so downstream code can use them directly.
    # If you later want strict JSON, switch to x.tolist().
    return np.asarray(x, dtype=np.int64)


def append_op_log(mg: MolGraph, info: OpInfo) -> None:
    """
    Append op info to mg.meta["op_log"] (creates list if missing).
    NOTE: mutates mg.meta in-place.
    """
    if mg.meta is None:
        mg.meta = {}

    log = mg.meta.get("op_log")
    if log is None:
        mg.meta["op_log"] = []
        log = mg.meta["op_log"]

    if not isinstance(log, list):
        mg.meta["op_log"] = [log]
        log = mg.meta["op_log"]

    entry: Dict[str, Any] = {
        "op": info.op,
        "params": dict(info.params),
        "warnings": list(info.warnings),
        "t_unix": float(info.t_unix),
    }

    if info.old_to_new is not None:
        entry["old_to_new"] = _np_to_jsonish(info.old_to_new)
    if info.new_to_old is not None:
        entry["new_to_old"] = _np_to_jsonish(info.new_to_old)
    if info.bond_perm is not None:
        entry["bond_perm"] = _np_to_jsonish(info.bond_perm)

    if info.changes is not None:
        cs = info.changes
        changes: Dict[str, Any] = {
            "n_atoms_before": int(cs.n_atoms_before),
            "n_atoms_after": int(cs.n_atoms_after),
            "n_bonds_before": int(cs.n_bonds_before),
            "n_bonds_after": int(cs.n_bonds_after),
            "notes": list(cs.notes),
        }
        if cs.atoms_removed_old is not None:
            changes["atoms_removed_old"] = _np_to_jsonish(cs.atoms_removed_old)
        if cs.atoms_added_new is not None:
            changes["atoms_added_new"] = _np_to_jsonish(cs.atoms_added_new)
        if cs.bonds_removed_old is not None:
            changes["bonds_removed_old"] = _np_to_jsonish(cs.bonds_removed_old)
        if cs.bonds_added_new is not None:
            changes["bonds_added_new"] = _np_to_jsonish(cs.bonds_added_new)

        entry["changes"] = changes

    log.append(entry)


# -----------------------------
# Transform base class
# -----------------------------

class Transform(ABC):
    """
    Base class for structural operations.

    Contract:
      - apply(mg) returns (mg2, info)
      - info.op must match self.name (we enforce in apply_transform)
      - mg2.meta['op_log'] gets appended by apply_transform()
    """

    name: str = "transform"

    @abstractmethod
    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        raise NotImplementedError # This is never meant to be run; Transform is not instantiable. Make an apply for each transformation itself.

    def inverse(self, info: OpInfo) -> Optional["Transform"]:
        """
        Optional: return a Transform that would reverse this op using info.
        Default is None.
        """
        return None


# -----------------------------
# Convenience runner
# -----------------------------

def apply_transform(mg: MolGraph, op: Transform) -> tuple[MolGraph, OpInfo]:
    """
    Apply a Transform, fill ChangeSummary if missing, and append to mg2.meta['op_log'].
    """
    mg2, info = op.apply(mg)

    # Ensure mg2 has a meta dict
    if mg2.meta is None:
        mg2.meta = {}

    # Enforce op name consistency
    if not info.op:
        info.op = op.name
    elif info.op != op.name:
        # Keep the record consistent even if the op forgot
        info.op = op.name

    # Fill changes if missing
    if info.changes is None:
        info.changes = _make_change_summary(mg, mg2)

    append_op_log(mg2, info)
    return mg2, info
