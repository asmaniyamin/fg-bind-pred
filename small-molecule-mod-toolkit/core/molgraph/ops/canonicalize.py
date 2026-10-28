# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from core.structs import MolGraph
from chem.ops.base import Transform, OpInfo, ChangeSummary
from core.molgraph.canonicalize import canonicalize


@dataclass
class Canonicalize(Transform):
    iters: int = 4
    stereo: bool = True
    name: str = "canonicalize"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        mg2, old_to_new, new_to_old, bond_perm = canonicalize(
            mg, iters=int(self.iters), stereo=bool(self.stereo)
        )

        info = OpInfo(
            op=self.name,
            params={"iters": int(self.iters), "stereo": bool(self.stereo)},
            old_to_new=np.asarray(old_to_new, dtype=np.int64),
            new_to_old=np.asarray(new_to_old, dtype=np.int64),
            bond_perm=np.asarray(bond_perm, dtype=np.int64),
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=int(mg2.arrays.atomic_num.shape[0]),
                n_bonds_before=m0,
                n_bonds_after=int(mg2.arrays.bonds.shape[0]),
                notes=["canonical atom/bond order applied"],
            ),
        )
        return mg2, info
