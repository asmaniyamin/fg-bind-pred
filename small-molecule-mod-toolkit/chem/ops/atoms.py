# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# wrappers for atom-level edits

# chem/ops/atoms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Any, Dict
import numpy as np

from core.structs import MolGraph
from chem.ops.base import Transform, OpInfo, ChangeSummary
from chem.edit.atoms import (
    delete_atom,
    delete_atoms,
    add_atom,
    replace_atom,
    set_dummy_label,
    permute_atoms,
)


def _new_to_old_from_old_to_new(old_to_new: np.ndarray) -> np.ndarray:
    """
    Given old_to_new (len N_old, -1 for deleted), return new_to_old (len N_new).
    Assumes new indices are 0..N_new-1 and bijective among kept atoms.
    """
    kept_old = np.nonzero(old_to_new >= 0)[0]
    # sort kept_old by their new index
    order = np.argsort(old_to_new[kept_old], kind="mergesort")
    new_to_old = kept_old[order].astype(np.int64, copy=False)
    return new_to_old


@dataclass
class DeleteAtom(Transform):
    idx: int
    name: str = "delete_atom"

    def apply(self, mg: MolGraph):
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        mg2, old_to_new, new_to_old = delete_atoms(mg, [int(self.idx)], return_maps=True)

        info = OpInfo(
            op=self.name,
            params={"idx": int(self.idx)},
            old_to_new=old_to_new,
            new_to_old=new_to_old,
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=int(mg2.arrays.atomic_num.shape[0]),
                n_bonds_before=m0,
                n_bonds_after=int(mg2.arrays.bonds.shape[0]),
                atoms_removed_old=np.array([int(self.idx)], dtype=np.int64),
            ),
        )
        return mg2, info



@dataclass
class AddAtom(Transform):
    atomic_num: int
    formal_charge: int = 0
    isotope: int = 0
    aromatic: Optional[bool] = None
    name: str = "add_atom"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        mg2, idx_new = add_atom(
            mg,
            atomic_num=int(self.atomic_num),
            formal_charge=int(self.formal_charge),
            isotope=int(self.isotope),
            is_aromatic=self.aromatic,
        )

        info = OpInfo(
            op=self.name,
            params={
                "atomic_num": int(self.atomic_num),
                "formal_charge": int(self.formal_charge),
                "isotope": int(self.isotope),
                "is_aromatic": self.aromatic,
            },
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=int(mg2.arrays.atomic_num.shape[0]),
                n_bonds_before=m0,
                n_bonds_after=int(mg2.arrays.bonds.shape[0]),
                atoms_added_new=np.array([int(idx_new)], dtype=np.int64),
            ),
        )
        return mg2, info


@dataclass
class ReplaceAtom(Transform):
    idx: int
    atomic_num: int
    formal_charge: Optional[int] = None
    isotope: Optional[int] = None
    aromatic: Optional[bool] = None
    name: str = "replace_atom"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        # Only forward optional fields that are not None; passing None would cause
        # replace_atom's set_opt to call np.asarray(None, dtype=...) which raises TypeError.
        fields: Dict[str, Any] = {"atomic_num": int(self.atomic_num)}
        if self.formal_charge is not None:
            fields["formal_charge"] = int(self.formal_charge)
        if self.isotope is not None:
            fields["isotope"] = int(self.isotope)
        if self.aromatic is not None:
            fields["is_aromatic"] = bool(self.aromatic)

        mg2 = replace_atom(mg, idx=int(self.idx), **fields)

        info = OpInfo(
            op=self.name,
            params={
                "idx": int(self.idx),
                **fields,
            },
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=int(mg2.arrays.atomic_num.shape[0]),
                n_bonds_before=m0,
                n_bonds_after=int(mg2.arrays.bonds.shape[0]),
                notes=["atom attributes replaced"],
            ),
        )
        return mg2, info


@dataclass
class SetDummyLabel(Transform):
    idx: int
    label: int
    name: str = "set_dummy_label"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        mg2 = set_dummy_label(mg, idx=int(self.idx), label=int(self.label))

        info = OpInfo(
            op=self.name,
            params={"idx": int(self.idx), "label": int(self.label)},
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=int(mg2.arrays.atomic_num.shape[0]),
                n_bonds_before=m0,
                n_bonds_after=int(mg2.arrays.bonds.shape[0]),
                notes=["dummy label updated"],
            ),
        )
        return mg2, info


@dataclass
class PermuteAtoms(Transform):
    new_to_old: np.ndarray
    name: str = "permute_atoms"

    def apply(self, mg: MolGraph) -> Tuple[MolGraph, OpInfo]:
        n0 = int(mg.arrays.atomic_num.shape[0])
        m0 = int(mg.arrays.bonds.shape[0])

        new_to_old = np.asarray(self.new_to_old, dtype=np.int64)
        mg2, old_to_new, bond_perm = permute_atoms(mg, new_to_old)

        info = OpInfo(
            op=self.name,
            params={"n_atoms": int(n0)},
            old_to_new=old_to_new,
            new_to_old=new_to_old,
            bond_perm=bond_perm,
            changes=ChangeSummary(
                n_atoms_before=n0,
                n_atoms_after=int(mg2.arrays.atomic_num.shape[0]),
                n_bonds_before=m0,
                n_bonds_after=int(mg2.arrays.bonds.shape[0]),
                notes=["atom permutation applied"],
            ),
        )
        return mg2, info
