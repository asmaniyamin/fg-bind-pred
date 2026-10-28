# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# wrappers for merge/attach

# chem/ops/merge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Dict, List

import numpy as np

from core.structs import MolGraph, BondType
from chem.ops.base import Transform, OpInfo, ChangeSummary
from chem.merge.merge import merge_by_labels


@dataclass
class MergeByLabels(Transform):
    """
    Transform wrapper around chem.merge.merge.merge_by_labels().

    Usage:
        op = MergeByLabels(g_b, label_a=1, label_b=2, coords="canonical")
        merged, info = apply_transform(g_a, op)

    Notes:
      - This merge consumes TWO graphs (g_a is the apply() input; g_b is stored in the op).
      - Atom/bond index mappings across a merge are inherently two-input and not represented
        by a single old_to_new/new_to_old mapping; we leave those as None for now.
        If you later want them, add fields like a_old_to_new and b_old_to_new to OpInfo.params,
        or extend OpInfo to support multi-input mappings.
    """
    g_b: MolGraph
    label_a: Any
    label_b: Any

    bond_type_code: int = BondType.SINGLE
    validate_with_rdkit: bool = False
    on_rdkit_fail: Literal["warn", "raise"] = "warn"

    coords: Optional[Literal["canonical", "ensemble"]] = None
    num_confs: int = 10
    optimize_coords: bool = False
    random_seed: int = 42

    name: str = "merge_by_labels"

    def apply(self, g_a: MolGraph) -> Tuple[MolGraph, OpInfo]:
        # capture pre-counts BEFORE calling merge (robust even if underlying code mutates)
        nA = int(g_a.arrays.atomic_num.shape[0])
        mA = int(g_a.arrays.bonds.shape[0])
        nB = int(self.g_b.arrays.atomic_num.shape[0])
        mB = int(self.g_b.arrays.bonds.shape[0])

        warnings: List[str] = []

        try:
            merged = merge_by_labels(
                g_a,
                self.g_b,
                label_a=self.label_a,
                label_b=self.label_b,
                bond_type_code=int(self.bond_type_code),
                validate_with_rdkit=bool(self.validate_with_rdkit),
                on_rdkit_fail=self.on_rdkit_fail,
                coords=self.coords,
                num_confs=int(self.num_confs),
                optimize_coords=bool(self.optimize_coords),
                random_seed=int(self.random_seed),
            )
        except Exception as e:
            # surface failure as a warning and re-raise (caller can catch)
            # if you prefer "warn and return input", wrap this op in StopOnWarning.
            raise

        nM = int(merged.arrays.atomic_num.shape[0])
        mM = int(merged.arrays.bonds.shape[0])

        # Change summary: we can at least record totals and basic expected direction.
        cs = ChangeSummary(
            n_atoms_before=nA + nB,
            n_atoms_after=nM,
            n_bonds_before=mA + mB,
            n_bonds_after=mM,
            notes=[
                "merge is a two-input op; old_to_new/new_to_old mappings are not provided",
                "counts_before computed as nA+nB and mA+mB (pre-merge totals)",
            ],
        )

        # If merge_by_labels wrote warnings into merged.meta["merge_log"] last entry,
        # we can pull them into OpInfo.warnings too (optional but helpful).
        ml = merged.meta.get("merge_log") if merged.meta else None
        if isinstance(ml, list) and ml:
            last = ml[-1]
            if isinstance(last, dict) and last.get("warnings"):
                warnings.extend(list(last["warnings"]))

        info = OpInfo(
            op=self.name,
            params={
                "label_a": self.label_a,
                "label_b": self.label_b,
                "bond_type_code": int(self.bond_type_code),
                "validate_with_rdkit": bool(self.validate_with_rdkit),
                "on_rdkit_fail": self.on_rdkit_fail,
                "coords": self.coords,
                "num_confs": int(self.num_confs),
                "optimize_coords": bool(self.optimize_coords),
                "random_seed": int(self.random_seed),
                # useful for debugging
                "n_atoms_a": nA,
                "n_atoms_b": nB,
                "n_bonds_a": mA,
                "n_bonds_b": mB,
            },
            warnings=warnings,
            old_to_new=None,
            new_to_old=None,
            bond_perm=None,
            changes=cs,
        )
        return merged, info
