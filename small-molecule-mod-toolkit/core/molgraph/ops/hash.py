# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

from core.structs import MolGraph
from chem.ops.base import Transform, OpInfo, ChangeSummary
from core.molgraph.hash import hash_molgraph


@dataclass
class HashMolGraph(Transform):
    mode: str = "stereo"          # "stereo" or "topology"
    write_to_meta: bool = True
    meta_key: Optional[str] = None
    name: str = "hash_molgraph"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        key = hash_molgraph(mg, mode=str(self.mode))

        if self.write_to_meta:
            mg.meta = dict(getattr(mg, "meta", {}) or {})
            k = self.meta_key or f"hash_{self.mode}"
            mg.meta[k] = key

        info = OpInfo(
            op=self.name,
            params={"mode": str(self.mode), "hash": key, "write_to_meta": bool(self.write_to_meta)},
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=n0,
                n_bonds_before=m0,
                n_bonds_after=m0,
                notes=["hash computed (no structural changes)"],
            ),
        )
        return mg, info
