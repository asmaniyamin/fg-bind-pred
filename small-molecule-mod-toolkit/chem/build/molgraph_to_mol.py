# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/build/molgraph_to_mol.py

Reconstructs an RDKit Mol object from a MolGraph representation.

This module provides the inverse of create_molgraph.py, converting
the NumPy-based MolGraph structure back to an RDKit Mol object.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem

from core.structs import (
    MolGraph,
    MolArrays,
    AtomHybridization,
    AtomChiralTag,
    BondType,
    BondDir,
    BondStereo,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Integer Code → RDKit Enum Mappings (reverse of build_utils.py)
# =============================================================================

CHIRAL_TAG_TO_RDKIT = {
    AtomChiralTag.UNSPECIFIED: rdchem.ChiralType.CHI_UNSPECIFIED,
    AtomChiralTag.CW: rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    AtomChiralTag.CCW: rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    AtomChiralTag.OTHER: rdchem.ChiralType.CHI_OTHER,
}

BOND_TYPE_TO_RDKIT = {
    BondType.SINGLE: rdchem.BondType.SINGLE,
    BondType.DOUBLE: rdchem.BondType.DOUBLE,
    BondType.TRIPLE: rdchem.BondType.TRIPLE,
    BondType.AROMATIC: rdchem.BondType.AROMATIC,
}

BOND_DIR_TO_RDKIT = {
    BondDir.NONE: rdchem.BondDir.NONE,
    BondDir.BEGINWEDGE: rdchem.BondDir.BEGINWEDGE,
    BondDir.BEGINDASH: rdchem.BondDir.BEGINDASH,
    BondDir.ENDDOWNRIGHT: rdchem.BondDir.ENDDOWNRIGHT,
    BondDir.ENDUPRIGHT: rdchem.BondDir.ENDUPRIGHT,
    BondDir.UNKNOWN: rdchem.BondDir.UNKNOWN,
}

BOND_STEREO_TO_RDKIT = {
    BondStereo.NONE: rdchem.BondStereo.STEREONONE,
    BondStereo.E: rdchem.BondStereo.STEREOE,
    BondStereo.Z: rdchem.BondStereo.STEREOZ,
    BondStereo.CIS: rdchem.BondStereo.STEREOCIS,
    BondStereo.TRANS: rdchem.BondStereo.STEREOTRANS,
    BondStereo.UNKNOWN: rdchem.BondStereo.STEREOANY,
}


# =============================================================================
# Atom Construction
# =============================================================================

def _create_rdkit_atom(arrays: MolArrays, idx: int) -> Chem.Atom:
    """Create an RDKit Atom from MolArrays at the given index."""
    logger.debug(f"Creating atom {idx}")
    
    atomic_num = int(arrays.atomic_num[idx])
    atom = Chem.Atom(atomic_num)
    
    # Atom map number
    if arrays.atom_map is not None and arrays.atom_map[idx] != 0:
        atom.SetAtomMapNum(int(arrays.atom_map[idx]))
    
    # Handle dummy atoms
    if atomic_num == 0:
        if arrays.attachment_label is not None and arrays.attachment_label[idx] is not None:
            label = arrays.attachment_label[idx]
            atom.SetProp("attachment_label", str(label))
            # Set atom map from label if it's an integer
            if isinstance(label, (int, np.integer)):
                atom.SetAtomMapNum(int(label))
    else:
        # Regular atoms
        atom.SetFormalCharge(int(arrays.formal_charge[idx]))
        
        if arrays.is_aromatic is not None:
            atom.SetIsAromatic(bool(arrays.is_aromatic[idx]))
        
        if arrays.isotope is not None and arrays.isotope[idx] != 0:
            atom.SetIsotope(int(arrays.isotope[idx]))
        
        if arrays.chiral_tag is not None:
            chiral_code = int(arrays.chiral_tag[idx])
            rdkit_chiral = CHIRAL_TAG_TO_RDKIT.get(chiral_code, rdchem.ChiralType.CHI_UNSPECIFIED)
            atom.SetChiralTag(rdkit_chiral)
    
    return atom


def _build_atoms(arrays: MolArrays) -> Tuple[Chem.RWMol, Dict[int, int]]:
    """Build all atoms and return the RWMol and index mapping."""
    logger.debug("Building atoms...")
    rw_mol = Chem.RWMol()
    atom_idx_map: Dict[int, int] = {}
    
    n_atoms = len(arrays.atomic_num)
    logger.debug(f"Creating {n_atoms} atoms")
    
    for i in range(n_atoms):
        atom = _create_rdkit_atom(arrays, i)
        new_idx = rw_mol.AddAtom(atom)
        atom_idx_map[i] = new_idx
    
    logger.debug("Atoms built successfully")
    return rw_mol, atom_idx_map


# =============================================================================
# Bond Construction
# =============================================================================

def _build_bonds(
    rw_mol: Chem.RWMol,
    arrays: MolArrays,
    atom_idx_map: Dict[int, int]
) -> None:
    """Add bonds to the RWMol from MolArrays."""
    logger.debug("Building bonds...")
    
    n_bonds = arrays.bonds.shape[0]
    logger.debug(f"Adding {n_bonds} bonds")
    
    for bond_idx in range(n_bonds):
        i, j = arrays.bonds[bond_idx]
        
        if i not in atom_idx_map or j not in atom_idx_map:
            logger.warning(f"Skipping bond {bond_idx}: atom index out of range ({i}, {j})")
            continue
        
        # Get bond type
        bond_type_code = int(arrays.bond_type[bond_idx])
        rdkit_bond_type = BOND_TYPE_TO_RDKIT.get(bond_type_code, rdchem.BondType.SINGLE)
        
        try:
            rw_mol.AddBond(atom_idx_map[i], atom_idx_map[j], rdkit_bond_type)
            bond = rw_mol.GetBondBetweenAtoms(atom_idx_map[i], atom_idx_map[j])
            
            # Set bond direction
            if arrays.bond_dir is not None:
                bond_dir_code = int(arrays.bond_dir[bond_idx])
                rdkit_bond_dir = BOND_DIR_TO_RDKIT.get(bond_dir_code, rdchem.BondDir.NONE)
                bond.SetBondDir(rdkit_bond_dir)
            
            # Set bond stereo
            if arrays.bond_stereo is not None:
                bond_stereo_code = int(arrays.bond_stereo[bond_idx])
                rdkit_bond_stereo = BOND_STEREO_TO_RDKIT.get(bond_stereo_code, rdchem.BondStereo.STEREONONE)
                bond.SetStereo(rdkit_bond_stereo)
            
            logger.debug(f"Added bond {i}-{j} (type={bond_type_code})")
            
        except Exception as e:
            logger.warning(f"Could not add bond {i}-{j}: {e}")
    
    logger.debug("Bonds built successfully")


# =============================================================================
# Aromaticity Handling
# =============================================================================

def _enforce_bond_aromaticity(mol: Chem.Mol) -> None:
    """Ensure bonds between aromatic atoms are marked aromatic."""
    logger.debug("Enforcing bond aromaticity...")
    
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        
        if a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)
            bond.SetIsConjugated(True)
            
            if bond.GetBondType() in (rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE):
                bond.SetBondType(rdchem.BondType.AROMATIC)
    
    logger.debug("Bond aromaticity enforced")


def _validate_aromatic_consistency(mol: Chem.Mol) -> None:
    """Check that aromatic atoms have aromatic bonds."""
    logger.debug("Validating aromatic consistency...")
    
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        
        if a1.GetIsAromatic() and a2.GetIsAromatic():
            if bond.GetBondType() not in (rdchem.BondType.AROMATIC, rdchem.BondType.UNSPECIFIED):
                raise ValueError(
                    f"Inconsistent aromatic bond: {a1.GetIdx()}-{a2.GetIdx()} "
                    f"(type={bond.GetBondType()})"
                )
    
    logger.debug("Aromatic consistency validated")


# =============================================================================
# Sanitization and Stereochemistry
# =============================================================================

def _sanitize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Sanitize the molecule.
    
    Args:
        mol: RDKit molecule to sanitize
        
    Returns:
        Sanitized molecule or None if sanitization fails
    """
    logger.debug("Sanitizing molecule...")
    
    try:
        # Do full sanitization - RDKit will properly perceive aromaticity
        # from the bond types we set (AROMATIC bonds will be recognized)
        Chem.SanitizeMol(mol)
        logger.debug("Sanitization successful")
        return mol
        
    except Exception as e:
        logger.error(f"Sanitization failed: {e}")
        # Try partial sanitization as fallback
        try:
            logger.debug("Attempting partial sanitization...")
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                    Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                    Chem.SanitizeFlags.SANITIZE_CLEANUP |
                    Chem.SanitizeFlags.SANITIZE_PROPERTIES
                )
            )
            logger.debug("Partial sanitization successful")
            return mol
        except Exception as e2:
            logger.error(f"Partial sanitization also failed: {e2}")
            return None


def _assign_stereochemistry(mol: Chem.Mol) -> None:
    """Assign stereochemistry from bond directions and chiral tags."""
    logger.debug("Assigning stereochemistry...")
    
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        logger.debug("Stereochemistry assigned")
    except Exception as e:
        logger.warning(f"Stereochemistry assignment failed: {e}")


# =============================================================================
# Main Conversion Function
# =============================================================================

def molgraph_to_mol(
    graph: MolGraph,
    sanitize: bool = True,
    remove_hs: bool = False,
) -> Optional[Chem.Mol]:
    """
    Convert a MolGraph back to an RDKit Mol object.
    
    Args:
        graph: MolGraph object to convert
        sanitize: Whether to sanitize the molecule
        remove_hs: Whether to remove explicit hydrogens from output
        
    Returns:
        RDKit Mol object or None if conversion fails
    """
    logger.debug("molgraph_to_mol() called")
    
    arrays = graph.arrays
    
    # Log input summary
    n_atoms = len(arrays.atomic_num)
    n_bonds = arrays.bonds.shape[0]
    logger.debug(f"Converting MolGraph: {n_atoms} atoms, {n_bonds} bonds")
    
    # Build atoms
    rw_mol, atom_idx_map = _build_atoms(arrays)
    
    # Build bonds
    _build_bonds(rw_mol, arrays, atom_idx_map)
    
    # Get molecule
    mol = rw_mol.GetMol()
    
    # Enforce aromatic bond types
    _enforce_bond_aromaticity(mol)
    
    # Validate aromaticity
    try:
        _validate_aromatic_consistency(mol)
    except ValueError as e:
        logger.error(f"Aromatic validation failed: {e}")
        return None
    
    # Sanitize
    if sanitize:
        mol = _sanitize_mol(mol)
        if mol is None:
            return None
    
    # Assign stereochemistry
    _assign_stereochemistry(mol)
    
    # Remove hydrogens if requested
    if remove_hs:
        logger.debug("Removing explicit hydrogens...")
        mol = Chem.RemoveHs(mol)
    
    # Generate SMILES for logging
    try:
        smiles = Chem.MolToSmiles(mol)
        logger.debug(f"Conversion complete: {smiles}")
    except Exception:
        logger.debug("Conversion complete (could not generate SMILES)")
    
    logger.debug("molgraph_to_mol() complete")
    return mol


def molgraph_to_smiles(
    graph: MolGraph,
    canonical: bool = True,
    isomeric: bool = True,
) -> Optional[str]:
    """
    Convenience function: MolGraph → SMILES string.
    
    Args:
        graph: MolGraph object to convert
        canonical: Whether to return canonical SMILES
        isomeric: Whether to include stereochemistry in SMILES
        
    Returns:
        SMILES string or None if conversion fails
    """
    logger.debug(f"molgraph_to_smiles() called (canonical={canonical}, isomeric={isomeric})")
    
    mol = molgraph_to_mol(graph)
    if mol is None:
        logger.error("Failed to convert MolGraph to Mol")
        return None
    
    try:
        smiles = Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
        logger.debug(f"Generated SMILES: {smiles}")
        return smiles
    except Exception as e:
        logger.error(f"Failed to generate SMILES: {e}")
        return None


logger.debug("molgraph_to_mol.py loaded successfully")