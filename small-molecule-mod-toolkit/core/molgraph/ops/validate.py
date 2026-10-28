# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from core.structs import MolGraph
from chem.ops.base import Transform, OpInfo, ChangeSummary
from core.molgraph.validate import validate_molgraph, MolGraphValidationError


@dataclass
class Validate(Transform):
    strict: bool = True
    name: str = "validate"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        warnings = []
        if self.strict:
            # will raise MolGraphValidationError if invalid
            validate_molgraph(mg, strict=True)
        else:
            warnings = validate_molgraph(mg, strict=False)

        info = OpInfo(
            op=self.name,
            params={"strict": bool(self.strict)},
            warnings=list(warnings),
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=n0,
                n_bonds_before=m0,
                n_bonds_after=m0,
                notes=["validation only (no structural changes)"],
            ),
        )
        return mg, info
