# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/ops/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from core.structs import MolGraph
from chem.ops.base import Transform, OpInfo, ChangeSummary, apply_transform


"""
Eventual use:
from chem.ops.pipeline import Compose
from chem.ops.atoms import DeleteAtom
from chem.ops.merge import MergeByLabels
from chem.ops.base import apply_transform

pipeline = Compose([
    DeleteAtom(3),
    MergeByLabels(1, 2),
])

mg2, _ = apply_transform(mg, pipeline)

for entry in mg2.meta["op_log"]:
    print(entry["op"], entry.get("params"))
"""


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class Compose(Transform):
    """
    Sequentially apply a list of Transforms.

    Behavior:
      - Each sub-op is applied via apply_transform(), so each sub-op logs itself.
      - Compose also returns its own OpInfo summary; apply_transform() will log that summary too.
      - Mappings are not propagated (ambiguous across multiple ops); use sub-op logs instead.
    """
    ops: List[Transform]
    name: str = "compose"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        warnings: List[str] = []
        for op in self.ops:
            mg, info = apply_transform(mg, op)
            warnings.extend(info.warnings)

        return mg, OpInfo(
            op=self.name,
            params={"n_ops": len(self.ops), "ops": [op.name for op in self.ops]},
            warnings=warnings,
        )


@dataclass
class Sequential(Compose):
    """
    Alias for Compose (common ML name).
    """
    name: str = "sequential"


@dataclass
class Repeat(Transform):
    """
    Repeat a Transform k times.
    """
    op: Transform
    k: int
    name: str = "repeat"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        if self.k < 0:
            raise ValueError("k must be >= 0")

        warnings: List[str] = []
        for _ in range(int(self.k)):
            mg, info = apply_transform(mg, self.op)
            warnings.extend(info.warnings)

        return mg, OpInfo(
            op=self.name,
            params={"k": int(self.k), "op": self.op.name},
            warnings=warnings,
        )


@dataclass
class Maybe(Transform):
    """
    Apply op with probability p (else no-op).
    """
    op: Transform
    p: float = 0.5
    seed: Optional[int] = None
    name: str = "maybe"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be in [0, 1]")

        rng = _rng_from_seed(self.seed)
        do = bool(rng.random() < self.p)

        warnings: List[str] = []
        if do:
            mg2, info = apply_transform(mg, self.op)
            warnings.extend(info.warnings)
            mg = mg2

        return mg, OpInfo(
            op=self.name,
            params={"p": float(self.p), "seed": self.seed, "op": self.op.name, "applied": do},
            warnings=warnings,
        )


@dataclass
class RandomChoice(Transform):
    """
    Choose exactly one op from ops (uniform or weighted) and apply it.
    """
    ops: Sequence[Transform]
    weights: Optional[Sequence[float]] = None
    seed: Optional[int] = None
    name: str = "random_choice"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        if len(self.ops) == 0:
            raise ValueError("ops must be non-empty")

        rng = _rng_from_seed(self.seed)

        if self.weights is None:
            idx = int(rng.integers(0, len(self.ops)))
        else:
            w = np.asarray(self.weights, dtype=float)
            if w.shape != (len(self.ops),):
                raise ValueError("weights must have same length as ops")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            if float(w.sum()) == 0.0:
                raise ValueError("weights must sum to > 0")
            p = w / w.sum()
            idx = int(rng.choice(len(self.ops), p=p))

        chosen = self.ops[idx]
        mg2, info = apply_transform(mg, chosen)

        return mg2, OpInfo(
            op=self.name,
            params={
                "seed": self.seed,
                "choice_idx": idx,
                "choice_op": chosen.name,
                "n_ops": len(self.ops),
                "weights": list(self.weights) if self.weights is not None else None,
            },
            warnings=list(info.warnings),
        )


@dataclass
class Conditional(Transform):
    """
    Apply one of two ops depending on predicate(mg).

    predicate should be pure (no mutation).
    """
    predicate: Callable[[MolGraph], bool]
    then_op: Transform
    else_op: Optional[Transform] = None
    name: str = "conditional"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        cond = bool(self.predicate(mg))
        warnings: List[str] = []

        if cond:
            mg2, info = apply_transform(mg, self.then_op)
            warnings.extend(info.warnings)
            mg = mg2
            chosen = self.then_op.name
        elif self.else_op is not None:
            mg2, info = apply_transform(mg, self.else_op)
            warnings.extend(info.warnings)
            mg = mg2
            chosen = self.else_op.name
        else:
            chosen = None

        return mg, OpInfo(
            op=self.name,
            params={
                "predicate": getattr(self.predicate, "__name__", "predicate"),
                "cond": cond,
                "then_op": self.then_op.name,
                "else_op": self.else_op.name if self.else_op is not None else None,
                "chosen": chosen,
            },
            warnings=warnings,
        )


@dataclass
class StopOnWarning(Transform):
    """
    Run an op, but if it emits any warnings, revert to input (no-op)
    and return an OpInfo indicating it was skipped.

    Useful for "try augmentation but keep only clean results".
    """
    op: Transform
    name: str = "stop_on_warning"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        mg2, info = apply_transform(mg, self.op)
        if info.warnings:
            # revert
            return mg, OpInfo(
                op=self.name,
                params={"op": self.op.name, "applied": False},
                warnings=list(info.warnings),
                changes=ChangeSummary(
                    n_atoms_before=int(mg.arrays.atomic_num.shape[0]),
                    n_atoms_after=int(mg.arrays.atomic_num.shape[0]),
                    n_bonds_before=int(mg.arrays.bonds.shape[0]),
                    n_bonds_after=int(mg.arrays.bonds.shape[0]),
                    notes=["reverted due to warnings"],
                ),
            )

        return mg2, OpInfo(
            op=self.name,
            params={"op": self.op.name, "applied": True},
            warnings=[],
        )
