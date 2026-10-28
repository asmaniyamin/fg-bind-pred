# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
core/structs.py

NumPy-first, non-redundant dataclasses for mutation + PyTorch Geometric pipelines.
This file contains DATA ONLY (no methods). Put conversion, featurization, caching,
RDKit, and PyG-export logic in separate modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)
logger.debug("Loading structs.py...")


# =========================
# Enum codes (int-coded)
# =========================
logger.debug("Defining enum classes...")


class AtomHybridization:
    OTHER = 0
    SP = 1
    SP2 = 2
    SP3 = 3
    SP3D = 4
    SP3D2 = 5


class AtomChiralTag:
    UNSPECIFIED = 0
    CW = 1
    CCW = 2
    OTHER = 3


class AtomCIPCode:
    NONE = 0
    R = 1
    S = 2
    UNKNOWN = 3


class BondType:
    SINGLE = 0
    DOUBLE = 1
    TRIPLE = 2
    AROMATIC = 3


class BondDir:
    NONE = 0
    BEGINWEDGE = 1
    BEGINDASH = 2
    ENDDOWNRIGHT = 3
    ENDUPRIGHT = 4
    UNKNOWN = 5


class BondStereo:
    NONE = 0
    E = 1
    Z = 2
    CIS = 3
    TRANS = 4
    UP = 5
    DOWN = 6
    UNKNOWN = 7


class BondResonanceType:
    NONE = 0
    LOCALIZED = 1
    DELOCALIZED = 2
    AROMATIC = 3


class AttachmentKind:
    DUMMY = 0
    H_SUB = 1
    OPEN_VALENCE = 2


logger.debug("Enum classes defined")


# =========================
# Canonical arrays (truth)
# =========================
logger.debug("Defining MolArrays dataclass...")


@dataclass(slots=True)
class MolArrays:
    """
    Canonical chemistry state (single source of truth).
    N = number of atoms, M = number of undirected bonds.
    """
    # ---- REQUIRED fields (no defaults) ----
    atomic_num: np.ndarray            # int16 (N,)
    formal_charge: np.ndarray         # int8 (N,)
    bonds: np.ndarray                 # int32 (M, 2)
    bond_type: np.ndarray             # int8 (M,)

    # ---- OPTIONAL atom fields ----
    isotope: Optional[np.ndarray] = None
    is_aromatic: Optional[np.ndarray] = None
    hybridization: Optional[np.ndarray] = None
    chiral_tag: Optional[np.ndarray] = None
    cip_code: Optional[np.ndarray] = None
    atom_map: Optional[np.ndarray] = None
    attachment_label: Optional[np.ndarray] = None
    explicit_h: Optional[np.ndarray] = None
    implicit_h: Optional[np.ndarray] = None
    partial_charge: Optional[np.ndarray] = None

    # ---- OPTIONAL geometry fields ----
    pos: Optional[np.ndarray] = None   # float32, shape (N, 3) OR (K, N, 3)
    coord_frame: Optional[str] = None  # Options: rdkit, etkdg, openbabel, user ... [NOTE 2]
    coord_valid: Optional[np.ndarray] = None # pos/coord_frame become invalid after structural-altering operations --- bool, (N,) or (K,N)

    # ---- OPTIONAL bond fields ----
    is_conjugated: Optional[np.ndarray] = None
    is_in_ring: Optional[np.ndarray] = None
    bond_dir: Optional[np.ndarray] = None
    bond_stereo: Optional[np.ndarray] = None
    bond_resonance_type: Optional[np.ndarray] = None

    # ---- graph globals ----
    total_charge: int = 0
    multiplicity: Optional[int] = None


logger.debug("MolArrays defined")


# =========================
# Mutation/edit scaffolding
# =========================
logger.debug("Defining AttachmentPointArray...")


@dataclass(slots=True)
class AttachmentPointArray:
    """Vectorized attachment points."""
    idx: np.ndarray
    kind: np.ndarray
    target: np.ndarray
    label_id: Optional[np.ndarray] = None
    is_insertion: Optional[np.ndarray] = None
    insertion_anchors: Optional[np.ndarray] = None


logger.debug("Defining Editability...")


@dataclass(slots=True)
class Editability:
    """Mutation policy scores."""
    atom_scores: np.ndarray
    bond_scores: np.ndarray


logger.debug("Defining Fragment...")


@dataclass(slots=True)
class Fragment:
    """Ragged container for fragments."""
    fragment_id: str
    atom_indices: np.ndarray
    attachment_indices: np.ndarray
    role: Optional[str] = None
    origin: Optional[str] = None
    smiles: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


logger.debug("Defining ResonanceSystem...")


@dataclass(slots=True)
class ResonanceSystem:
    """Optional rich object view for resonance."""
    resonance_system_id: Optional[str] = None
    atom_indices: Optional[np.ndarray] = None
    bond_indices: Optional[np.ndarray] = None
    smarts: Optional[str] = None
    label: Optional[str] = None
    role: Optional[str] = None
    origin: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


logger.debug("Defining DerivedCaches...")


@dataclass(slots=True)
class DerivedCaches:
    """Derived tensors/caches (NOT canonical truth)."""
    edge_index: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    edge_attr: Optional[np.ndarray] = None
    bond_pair_to_index: Optional[Dict[tuple, int]] = None


logger.debug("Defining DirtyFlags...")


@dataclass(slots=True)
class DirtyFlags:
    """Dirty flags for derived caches only."""
    structure_dirty: bool = True
    features_dirty: bool = True


logger.debug("Defining MolGraph...")


@dataclass(slots=True)
class MolGraph:
    """Clean top-level molecular graph object."""
    arrays: MolArrays
    attachments: Optional[AttachmentPointArray] = None
    editability: Optional[Editability] = None
    fragments: List[Fragment] = field(default_factory=list)
    resonance_systems: List[ResonanceSystem] = field(default_factory=list)
    cache: DerivedCaches = field(default_factory=DerivedCaches)
    dirty: DirtyFlags = field(default_factory=DirtyFlags)
    meta: Dict[str, Any] = field(default_factory=dict)
    rdkit_mol: Any = field(default=None, repr=False)


logger.debug("All dataclasses defined")
logger.debug("structs.py loaded successfully")