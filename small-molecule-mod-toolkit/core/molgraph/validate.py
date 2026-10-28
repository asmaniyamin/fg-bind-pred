# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# core/molgraph/validate.py

from __future__ import annotations
from typing import List
import numpy as np

from core.structs import MolGraph

"""
TO ADD:
validate_stereo(mg)

validate_valence(mg) (RDKit-backed)

validate_physics(mg) (bond lengths, clashes)

validate_ml_ready(mg) (no dummies, canonicalized, coords present)
"""

class MolGraphValidationError(ValueError):
    pass


def validate_molgraph(
    mg: MolGraph,
    *,
    strict: bool = True,
) -> List[str]:
    """
    Validate structural invariants of a MolGraph.

    Args:
        mg: MolGraph to validate
        strict: if True, raise on error; if False, collect warnings

    Returns:
        List of warning/error messages (empty if valid)
    """
    errors: List[str] = []
    arrays = mg.arrays

    # -------------------------
    # Atom-level invariants
    # -------------------------
    if arrays.atomic_num is None:
        errors.append("atomic_num is None")

    n_atoms = len(arrays.atomic_num)

    for name in [
        "formal_charge",
        "is_aromatic",
        "hybridization",
        "chiral_tag",
        "atom_map",
        "explicit_h",
        "implicit_h",
        "partial_charge",
        "pos",
    ]:
        arr = getattr(arrays, name, None)
        if arr is not None:
            if arr.shape[0] != n_atoms:
                errors.append(
                    f"{name} has length {arr.shape[0]} != n_atoms {n_atoms}"
                )

    # No NaNs in numeric atom arrays
    for name in ["formal_charge", "partial_charge"]:
        arr = getattr(arrays, name, None)
        if arr is not None and np.isnan(arr).any():
            errors.append(f"{name} contains NaNs")

    # -------------------------
    # Bond-level invariants
    # -------------------------
    bonds = arrays.bonds
    if bonds is None:
        errors.append("bonds is None")
        n_bonds = 0
    else:
        n_bonds = bonds.shape[0]

    if bonds is not None:
        if bonds.ndim != 2 or bonds.shape[1] != 2:
            errors.append("bonds must have shape (M, 2)")

        # indices valid
        if np.any(bonds < 0) or np.any(bonds >= n_atoms):
            errors.append("bond indices out of range")

        # no self-bonds
        if np.any(bonds[:, 0] == bonds[:, 1]):
            errors.append("self-bonds detected")

        # normalized order u < v
        if np.any(bonds[:, 0] > bonds[:, 1]):
            errors.append("bonds not normalized (u > v)")

    for name in [
        "bond_type",
        "bond_dir",
        "bond_stereo",
        "is_conjugated",
        "is_in_ring",
        "bond_resonance_type",
    ]:
        arr = getattr(arrays, name, None)
        if arr is not None:
            if arr.shape[0] != n_bonds:
                errors.append(
                    f"{name} has length {arr.shape[0]} != n_bonds {n_bonds}"
                )

    # -------------------------
    # Coordinate invariants
    # -------------------------
    pos = arrays.pos
    coord_valid = arrays.coord_valid

    if pos is not None:
        if pos.ndim == 2:
            if pos.shape != (n_atoms, 3):
                errors.append(
                    f"pos shape {pos.shape} invalid for canonical coords"
                )
            if coord_valid is not None and coord_valid.shape != (n_atoms,):
                errors.append("coord_valid shape mismatch for canonical coords")

        elif pos.ndim == 3:
            k, n, d = pos.shape
            if n != n_atoms or d != 3:
                errors.append(
                    f"pos shape {pos.shape} invalid for ensemble coords"
                )
            if coord_valid is not None and coord_valid.shape != (k, n_atoms):
                errors.append("coord_valid shape mismatch for ensemble coords")

        else:
            errors.append(f"pos has invalid ndim {pos.ndim}")

    # -------------------------
    # Attachment / fragments
    # -------------------------
    for fr in mg.fragments:
        if np.any(fr.atom_indices >= n_atoms):
            errors.append(f"fragment {fr.fragment_id} atom index out of range")

    # -------------------------
    # Finalize
    # -------------------------
    if errors and strict:
        raise MolGraphValidationError(
            "MolGraph validation failed:\n  - "
            + "\n  - ".join(errors)
        )

    return errors
