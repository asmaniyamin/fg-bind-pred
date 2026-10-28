# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/build/build_utils.py

Internal helpers for RDKit → MolGraph conversion.

Contains:
- RDKit enum → integer code mappings
- Atom/bond array extraction functions
- Attachment point extraction
- Gasteiger charge computation
- SMILES parsing and cleaning (via MoleculeCleaner)
- PyG edge index utilities
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import math

from core.structs import (
    MolArrays,
    AttachmentPointArray,
    AtomHybridization,
    AtomChiralTag,
    BondType,
    BondDir,
    BondStereo,
    BondResonanceType,
    AttachmentKind,
)
from chem.clean.clean_molecule import MoleculeCleaner
from utils.logger import get_logger

logger = get_logger(__name__)
logger.debug("Loading build_utils.py...")


# =============================================================================
# RDKit Enum → Integer Code Mappings
# =============================================================================
logger.debug("Defining enum mappings...")

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: AtomHybridization.OTHER,
    Chem.rdchem.HybridizationType.S: AtomHybridization.OTHER,
    Chem.rdchem.HybridizationType.SP: AtomHybridization.SP,
    Chem.rdchem.HybridizationType.SP2: AtomHybridization.SP2,
    Chem.rdchem.HybridizationType.SP3: AtomHybridization.SP3,
    Chem.rdchem.HybridizationType.SP3D: AtomHybridization.SP3D,
    Chem.rdchem.HybridizationType.SP3D2: AtomHybridization.SP3D2,
    Chem.rdchem.HybridizationType.OTHER: AtomHybridization.OTHER,
}

CHIRAL_TAG_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: AtomChiralTag.UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: AtomChiralTag.CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: AtomChiralTag.CCW,
    Chem.rdchem.ChiralType.CHI_OTHER: AtomChiralTag.OTHER,
}

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE: BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE: BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC: BondType.AROMATIC,
}

BOND_DIR_MAP = {
    Chem.rdchem.BondDir.NONE: BondDir.NONE,
    Chem.rdchem.BondDir.BEGINWEDGE: BondDir.BEGINWEDGE,
    Chem.rdchem.BondDir.BEGINDASH: BondDir.BEGINDASH,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.ENDUPRIGHT: BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.UNKNOWN: BondDir.UNKNOWN,
}

BOND_STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREONONE: BondStereo.NONE,
    Chem.rdchem.BondStereo.STEREOE: BondStereo.E,
    Chem.rdchem.BondStereo.STEREOZ: BondStereo.Z,
    Chem.rdchem.BondStereo.STEREOCIS: BondStereo.CIS,
    Chem.rdchem.BondStereo.STEREOTRANS: BondStereo.TRANS,
    Chem.rdchem.BondStereo.STEREOANY: BondStereo.UNKNOWN,
}

logger.debug("Enum mappings defined")


# =============================================================================
# Atom Extraction
# =============================================================================

def extract_atom_arrays(mol: Chem.Mol, compute_charges: bool = True) -> dict:
    """Extract atom-level arrays from an RDKit molecule."""
    logger.debug("extract_atom_arrays() called")
    atoms = list(mol.GetAtoms())
    n = len(atoms)
    logger.debug(f"Found {n} atoms")

    logger.debug("Extracting atomic_num...")
    atomic_num = np.array([a.GetAtomicNum() for a in atoms], dtype=np.int16)
    
    logger.debug("Extracting formal_charge...")
    formal_charge = np.array([a.GetFormalCharge() for a in atoms], dtype=np.int8)

    logger.debug("Extracting isotope...")
    isotope = np.array([a.GetIsotope() for a in atoms], dtype=np.int16)
    
    logger.debug("Extracting is_aromatic...")
    is_aromatic = np.array([a.GetIsAromatic() for a in atoms], dtype=bool)
    
    logger.debug("Extracting hybridization...")
    hybridization = np.array(
        [HYBRIDIZATION_MAP.get(a.GetHybridization(), AtomHybridization.OTHER) for a in atoms],
        dtype=np.int8
    )
    
    logger.debug("Extracting chiral_tag...")
    chiral_tag = np.array(
        [CHIRAL_TAG_MAP.get(a.GetChiralTag(), AtomChiralTag.UNSPECIFIED) for a in atoms],
        dtype=np.int8
    )
    
    logger.debug("Extracting atom_map...")
    atom_map = np.array([a.GetAtomMapNum() for a in atoms], dtype=np.int32)

    logger.debug("Extracting attachment_label...")
    attachment_label = np.empty(n, dtype=object)
    for i, a in enumerate(atoms):
        if a.GetAtomicNum() == 0 and a.GetAtomMapNum() > 0:
            attachment_label[i] = a.GetAtomMapNum()
        else:
            attachment_label[i] = None

    logger.debug("Extracting explicit_h...")
    explicit_h = np.array([a.GetNumExplicitHs() for a in atoms], dtype=np.int8)
    
    logger.debug("Extracting implicit_h...")
    implicit_h = np.array([a.GetNumImplicitHs() for a in atoms], dtype=np.int8)

    logger.debug("Computing partial charges...")
    partial_charge = None
    if compute_charges:
        partial_charge = compute_gasteiger_charges(mol)

    logger.debug("Checking for conformer...")
    pos = None
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        pos = np.array([conf.GetAtomPosition(i) for i in range(n)], dtype=np.float32)

    logger.debug("extract_atom_arrays() complete")
    return {
        "atomic_num": atomic_num,
        "formal_charge": formal_charge,
        "isotope": isotope if isotope.any() else None,
        "is_aromatic": is_aromatic,
        "hybridization": hybridization,
        "chiral_tag": chiral_tag,
        "atom_map": atom_map if atom_map.any() else None,
        "attachment_label": attachment_label if any(x is not None for x in attachment_label) else None,
        "explicit_h": explicit_h,
        "implicit_h": implicit_h,
        "partial_charge": partial_charge,
        "pos": pos,
    }


# =============================================================================
# Bond Extraction
# =============================================================================

def extract_bond_arrays(mol: Chem.Mol) -> dict:
    """Extract bond-level arrays from an RDKit molecule."""
    logger.debug("extract_bond_arrays() called")
    bonds = list(mol.GetBonds())
    m = len(bonds)
    logger.debug(f"Found {m} bonds")

    if m == 0:
        logger.debug("No bonds, returning empty arrays")
        return {
            "bonds": np.zeros((0, 2), dtype=np.int32),
            "bond_type": np.zeros(0, dtype=np.int8),
            "is_conjugated": None,
            "is_in_ring": None,
            "bond_dir": None,
            "bond_stereo": None,
            "bond_resonance_type": None,
        }

    logger.debug("Extracting bond pairs...")
    bond_pairs = []
    for b in bonds:
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_pairs.append((min(u, v), max(u, v)))
    bonds_arr = np.array(bond_pairs, dtype=np.int32)

    logger.debug("Extracting bond_type...")
    bond_type = np.array(
        [BOND_TYPE_MAP.get(b.GetBondType(), BondType.SINGLE) for b in bonds],
        dtype=np.int8
    )

    logger.debug("Extracting is_conjugated...")
    is_conjugated = np.array([b.GetIsConjugated() for b in bonds], dtype=bool)
    
    logger.debug("Extracting is_in_ring...")
    is_in_ring = np.array([b.IsInRing() for b in bonds], dtype=bool)

    logger.debug("Extracting bond_dir...")
    bond_dir = np.array(
        [BOND_DIR_MAP.get(b.GetBondDir(), BondDir.NONE) for b in bonds],
        dtype=np.int8
    )
    
    logger.debug("Extracting bond_stereo...")
    bond_stereo = np.array(
        [BOND_STEREO_MAP.get(b.GetStereo(), BondStereo.NONE) for b in bonds],
        dtype=np.int8
    )

    logger.debug("Extracting bond_resonance_type...")
    bond_resonance_type = np.array(
        [infer_resonance_type(b) for b in bonds],
        dtype=np.int8
    )

    logger.debug("extract_bond_arrays() complete")
    return {
        "bonds": bonds_arr,
        "bond_type": bond_type,
        "is_conjugated": is_conjugated,
        "is_in_ring": is_in_ring,
        "bond_dir": bond_dir if bond_dir.any() else None,
        "bond_stereo": bond_stereo if bond_stereo.any() else None,
        "bond_resonance_type": bond_resonance_type,
    }


def infer_resonance_type(bond: Chem.Bond) -> int:
    """Infer resonance type for a bond."""
    if bond.GetIsAromatic():
        return BondResonanceType.AROMATIC
    elif bond.GetIsConjugated():
        return BondResonanceType.DELOCALIZED
    elif bond.GetBondType() in (
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
    ):
        return BondResonanceType.LOCALIZED
    else:
        return BondResonanceType.NONE


# =============================================================================
# Gasteiger Charges
# =============================================================================

def compute_gasteiger_charges(mol: Chem.Mol) -> np.ndarray:
    """
    Extract or compute Gasteiger charges, handling dummy atoms gracefully.
    
    If charges are already computed (stored as _GasteigerCharge properties),
    extracts them. Otherwise computes them first.
    """
    logger.debug("compute_gasteiger_charges() called")
    n = mol.GetNumAtoms()
    charges = np.zeros(n, dtype=np.float32)
    
    # Check if charges are already computed
    has_existing_charges = any(
        a.HasProp("_GasteigerCharge") for a in mol.GetAtoms() if a.GetAtomicNum() != 0
    )
    
    if has_existing_charges:
        logger.debug("Extracting existing Gasteiger charges...")
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.HasProp("_GasteigerCharge"):
                try:
                    val = float(atom.GetProp("_GasteigerCharge"))
                    charges[i] = 0.0 if math.isnan(val) else val
                except ValueError:
                    charges[i] = 0.0
        logger.debug("Charges extracted successfully")
    else:
        # Need to compute charges
        has_dummies = any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
        logger.debug(f"Computing charges (has_dummies: {has_dummies})")

        if has_dummies:
            logger.debug("Replacing dummies with H for charge computation...")
            mol_copy = Chem.RWMol(mol)
            dummy_indices = []
            for atom in mol_copy.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    dummy_indices.append(atom.GetIdx())
                    atom.SetAtomicNum(1)
                    atom.SetFormalCharge(0)
            try:
                logger.debug("Computing Gasteiger charges on modified mol...")
                AllChem.ComputeGasteigerCharges(mol_copy)
                for i, atom in enumerate(mol_copy.GetAtoms()):
                    if i not in dummy_indices and atom.HasProp("_GasteigerCharge"):
                        val = float(atom.GetProp("_GasteigerCharge"))
                        charges[i] = 0.0 if math.isnan(val) else val
                logger.debug("Charges computed successfully")
            except Exception as e:
                logger.warning(f"Charge computation failed: {e}")
        else:
            try:
                logger.debug("Computing Gasteiger charges...")
                AllChem.ComputeGasteigerCharges(mol)
                for i, atom in enumerate(mol.GetAtoms()):
                    if atom.HasProp("_GasteigerCharge"):
                        val = float(atom.GetProp("_GasteigerCharge"))
                        charges[i] = 0.0 if math.isnan(val) else val
                logger.debug("Charges computed successfully")
            except Exception as e:
                logger.warning(f"Charge computation failed: {e}")

    logger.debug("compute_gasteiger_charges() complete")
    return np.round(charges, 4)


# =============================================================================
# Attachment Points
# =============================================================================

def extract_attachment_points(mol: Chem.Mol) -> Optional[AttachmentPointArray]:
    """Extract attachment points (dummy atoms)."""
    logger.debug("extract_attachment_points() called")
    dummy_indices = []
    dummy_kinds = []
    dummy_targets = []
    dummy_labels = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            idx = atom.GetIdx()
            dummy_indices.append(idx)
            dummy_kinds.append(AttachmentKind.DUMMY)

            neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 0]
            target = neighbors[0].GetIdx() if neighbors else -1
            dummy_targets.append(target)

            label = atom.GetAtomMapNum() if atom.GetAtomMapNum() > 0 else None
            dummy_labels.append(label)

    logger.debug(f"Found {len(dummy_indices)} attachment points")

    if not dummy_indices:
        logger.debug("extract_attachment_points() complete (no attachments)")
        return None

    result = AttachmentPointArray(
        idx=np.array(dummy_indices, dtype=np.int32),
        kind=np.array(dummy_kinds, dtype=np.int8),
        target=np.array(dummy_targets, dtype=np.int32),
        label_id=np.array(dummy_labels, dtype=object) if any(dummy_labels) else None,
    )
    logger.debug("extract_attachment_points() complete")
    return result


# =============================================================================
# PyG Utilities
# =============================================================================

def build_edge_index(arrays: MolArrays) -> np.ndarray:
    """Build directed edge_index (2, 2M) from undirected bonds (M, 2)."""
    logger.debug("build_edge_index() called")
    bonds = arrays.bonds
    if bonds.shape[0] == 0:
        logger.debug("No bonds, returning empty edge_index")
        return np.zeros((2, 0), dtype=np.int64)

    forward = bonds.T
    backward = bonds[:, ::-1].T
    edge_index = np.concatenate([forward, backward], axis=1).astype(np.int64)
    logger.debug(f"edge_index shape: {edge_index.shape}")
    logger.debug("build_edge_index() complete")
    return edge_index


def build_bond_pair_to_index(arrays: MolArrays) -> dict:
    """Build lookup dict: (u, v) → bond index."""
    logger.debug("build_bond_pair_to_index() called")
    bonds = arrays.bonds
    lookup = {}
    for idx, (u, v) in enumerate(bonds):
        lookup[(u, v)] = idx
        lookup[(v, u)] = idx
    logger.debug(f"Built lookup with {len(lookup)} entries")
    logger.debug("build_bond_pair_to_index() complete")
    return lookup


# =============================================================================
# SMILES Parsing and Molecule Preparation
# =============================================================================

def parse_and_clean_smiles(
    smiles: str,
) -> Tuple[Chem.Mol, list]:
    """
    Parses a SMILES (without RDKit sanitization), runs MoleculeCleaner,
    then applies RDKit sanitization -> aromaticity -> kekulization.
    
    Args:
        smiles: SMILES string to parse
        
    Returns:
        Tuple of (mol, repair_log) where repair_log is a list of repair steps
        
    Raises:
        ValueError: If SMILES cannot be parsed or cleaned
    """
    logger.debug(f"parse_and_clean_smiles() called with: {smiles}")
    
    # 1) Parse without any RDKit sanitization
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        logger.error(f"Could not parse SMILES: {smiles}")
        raise ValueError(f"Unparsable SMILES: {smiles}")
    
    # 2) One-pass custom cleaning with log
    cleaner = MoleculeCleaner(verbose=False, return_log=True)
    mol, repair_log = cleaner.clean(mol)
    
    if mol is None:
        logger.warning(f"Cleaner failed for SMILES: {smiles}")
        logger.debug(f"Repair log: {repair_log}")
        raise ValueError(f"Could not clean SMILES: {smiles}\nRepair log: {repair_log}")
    
    logger.debug(f"parse_and_clean_smiles() complete, repair_log: {repair_log}")
    return mol, repair_log


def compute_gasteiger_with_capping(mol: Chem.Mol) -> Chem.Mol:
    """
    Computes Gasteiger partial charges for a molecule with dummy atoms.
    
    Dummy atoms (atomic number 0) are temporarily replaced with hydrogen atoms
    to allow charge computation. The computed charges are stored as atom properties
    (_GasteigerCharge) on the original molecule.
    
    Args:
        mol: RDKit Mol object, possibly containing dummy atoms
        
    Returns:
        The same molecule with Gasteiger charges added as atom properties
    """
    logger.debug("compute_gasteiger_with_capping() called")
    
    # Check for dummy atoms
    has_dummies = any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
    
    if has_dummies:
        logger.debug("Replacing dummies with H for charge computation...")
        # Create a copy for charge computation
        mol_copy = Chem.RWMol(mol)
        
        dummy_indices = set()
        for atom in mol_copy.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_indices.add(atom.GetIdx())
                atom.SetAtomicNum(1)
                atom.SetIsAromatic(False)
                atom.SetFormalCharge(0)
        
        try:
            AllChem.ComputeGasteigerCharges(mol_copy)
            
            # Transfer charges back to original atoms (excluding dummies)
            for i, (orig_atom, capped_atom) in enumerate(zip(mol.GetAtoms(), mol_copy.GetAtoms())):
                if i not in dummy_indices and capped_atom.HasProp("_GasteigerCharge"):
                    charge_val = capped_atom.GetProp("_GasteigerCharge")
                    orig_atom.SetProp("_GasteigerCharge", charge_val)
                elif i in dummy_indices:
                    orig_atom.SetProp("_GasteigerCharge", "0.0")
            
            logger.debug("Gasteiger charges computed with capping")
        except Exception as e:
            logger.warning(f"Gasteiger charge computation failed: {e}")
    else:
        try:
            AllChem.ComputeGasteigerCharges(mol)
            logger.debug("Gasteiger charges computed")
        except Exception as e:
            logger.warning(f"Gasteiger charge computation failed: {e}")
    
    logger.debug("compute_gasteiger_with_capping() complete")
    return mol


def prepare_molecule(
    smiles: str,
    add_hs: bool = True,
    compute_charges: bool = True,
) -> Tuple[Chem.Mol, list]:
    """
    Full molecule preparation pipeline: parse, clean, add Hs, compute charges.
    
    Args:
        smiles: SMILES string
        add_hs: Whether to add explicit hydrogens
        compute_charges: Whether to compute Gasteiger charges
        
    Returns:
        Tuple of (prepared mol, repair_log)
        
    Raises:
        ValueError: If SMILES cannot be parsed or cleaned
    """
    logger.debug(f"prepare_molecule() called with: {smiles}")
    
    mol, repair_log = parse_and_clean_smiles(smiles)
    
    if add_hs:
        logger.debug("Adding explicit hydrogens...")
        mol = Chem.AddHs(mol)
    
    if compute_charges:
        logger.debug("Computing Gasteiger charges...")
        mol = compute_gasteiger_with_capping(mol)
    
    logger.debug("prepare_molecule() complete")
    return mol, repair_log


logger.debug("build_utils.py loaded successfully")