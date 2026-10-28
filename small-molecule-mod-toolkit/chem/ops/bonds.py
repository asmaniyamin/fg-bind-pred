# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/ops/bonds.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Any, Dict, List
import numpy as np

from core.structs import MolGraph
from chem.ops.base import Transform, OpInfo, ChangeSummary

from chem.edit.bonds import add_bond, delete_bond, delete_bonds, edit_bond


def _identity_maps(n_atoms: int):
    old_to_new = np.arange(n_atoms, dtype=np.int64)
    new_to_old = old_to_new.copy()
    return old_to_new, new_to_old


@dataclass
class AddBond(Transform):
    """
    Wraps chem.edit.bonds.add_bond().
    """
    u: int
    v: int
    bond_type: int
    # optional per-bond fields (only used if the underlying arrays exist)
    is_conjugated: Optional[bool] = None
    is_in_ring: Optional[bool] = None
    bond_dir: Optional[int] = None
    bond_stereo: Optional[int] = None
    bond_resonance_type: Optional[int] = None

    name: str = "add_bond"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        fields: Dict[str, Any] = {}
        if self.is_conjugated is not None:
            fields["is_conjugated"] = bool(self.is_conjugated)
        if self.is_in_ring is not None:
            fields["is_in_ring"] = bool(self.is_in_ring)
        if self.bond_dir is not None:
            fields["bond_dir"] = int(self.bond_dir)
        if self.bond_stereo is not None:
            fields["bond_stereo"] = int(self.bond_stereo)
        if self.bond_resonance_type is not None:
            fields["bond_resonance_type"] = int(self.bond_resonance_type)

        mg2 = add_bond(mg, int(self.u), int(self.v), bond_type=int(self.bond_type), **fields)

        old_to_new, new_to_old = _identity_maps(n0)

        info = OpInfo(
            op=self.name,
            params={
                "u": int(self.u),
                "v": int(self.v),
                "bond_type": int(self.bond_type),
                **fields,
            },
            old_to_new=old_to_new,
            new_to_old=new_to_old,
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=n0,
                n_bonds_before=m0,
                n_bonds_after=int(mg2.arrays.bonds.shape[0]),
                notes=["bond added"],
            ),
        )
        return mg2, info


@dataclass
class DeleteBond(Transform):
    """
    Wraps chem.edit.bonds.delete_bond() by atom pair.
    """
    u: int
    v: int
    undirected: bool = True
    name: str = "delete_bond"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        mg2 = delete_bond(mg, int(self.u), int(self.v), undirected=bool(self.undirected))

        old_to_new, new_to_old = _identity_maps(n0)

        info = OpInfo(
            op=self.name,
            params={"u": int(self.u), "v": int(self.v), "undirected": bool(self.undirected)},
            old_to_new=old_to_new,
            new_to_old=new_to_old,
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=n0,
                n_bonds_before=m0,
                n_bonds_after=int(mg2.arrays.bonds.shape[0]),
                notes=["bond(s) deleted by pair"],
            ),
        )
        return mg2, info


@dataclass
class DeleteBonds(Transform):
    """
    Wraps chem.edit.bonds.delete_bonds() by bond indices.
    """
    bond_indices: Sequence[int]
    name: str = "delete_bonds"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        idxs = list(map(int, self.bond_indices))
        mg2 = delete_bonds(mg, idxs)

        old_to_new, new_to_old = _identity_maps(n0)

        info = OpInfo(
            op=self.name,
            params={"bond_indices": idxs},
            old_to_new=old_to_new,
            new_to_old=new_to_old,
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=n0,
                n_bonds_before=m0,
                n_bonds_after=int(mg2.arrays.bonds.shape[0]),
                bonds_removed_old=np.asarray(idxs, dtype=np.int64),
            ),
        )
        return mg2, info


@dataclass
class EditBond(Transform):
    """
    Wraps chem.edit.bonds.edit_bond() by atom pair (u,v).
    You can set bond_type and/or optional bond fields.
    """
    u: int
    v: int

    bond_type: Optional[int] = None
    is_conjugated: Optional[bool] = None
    is_in_ring: Optional[bool] = None
    bond_dir: Optional[int] = None
    bond_stereo: Optional[int] = None
    bond_resonance_type: Optional[int] = None

    name: str = "edit_bond"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        mg2 = edit_bond(
            mg,
            int(self.u),
            int(self.v),
            bond_type=int(self.bond_type) if self.bond_type is not None else None,
            is_conjugated=bool(self.is_conjugated) if self.is_conjugated is not None else None,
            is_in_ring=bool(self.is_in_ring) if self.is_in_ring is not None else None,
            bond_dir=int(self.bond_dir) if self.bond_dir is not None else None,
            bond_stereo=int(self.bond_stereo) if self.bond_stereo is not None else None,
            bond_resonance_type=int(self.bond_resonance_type) if self.bond_resonance_type is not None else None,
        )

        old_to_new, new_to_old = _identity_maps(n0)

        # We don't know the bond index without searching, so we just note "changed"
        params: Dict[str, Any] = {"u": int(self.u), "v": int(self.v)}
        if self.bond_type is not None: params["bond_type"] = int(self.bond_type)
        if self.is_conjugated is not None: params["is_conjugated"] = bool(self.is_conjugated)
        if self.is_in_ring is not None: params["is_in_ring"] = bool(self.is_in_ring)
        if self.bond_dir is not None: params["bond_dir"] = int(self.bond_dir)
        if self.bond_stereo is not None: params["bond_stereo"] = int(self.bond_stereo)
        if self.bond_resonance_type is not None: params["bond_resonance_type"] = int(self.bond_resonance_type)

        info = OpInfo(
            op=self.name,
            params=params,
            old_to_new=old_to_new,
            new_to_old=new_to_old,
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=n0,
                n_bonds_before=m0,
                n_bonds_after=m0,
                notes=["bond attributes edited (by pair)"],
            ),
        )
        return mg2, info
