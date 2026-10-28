# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
detect_tools.py

Validation and diagnostic utilities for identifying structural issues in RDKit molecules.

This module provides a set of lightweight, stateless detection functions used to
check for common problems that may prevent successful sanitization, modeling,
or graph conversion.

Key detection capabilities include:
    - Dummy atoms in aromatic rings
    - Unconnected molecular fragments
    - Overvalent atoms (explicit valence > default valence)
    - Excessive formal charges
    - Missing stereochemistry or undefined chirality
    - Kekulization failures or ring perception issues

Functions:
    - detect_valence_errors
    - detect_dummies_in_aromatic_ring
    - detect_unconnected_fragments
    - detect_nonstandard_valence
    - detect_missing_stereochemistry
    - detect_formal_charge_imbalance
    - detect_kekulization_failure
    - detect_rings_broken
    - detect_overvalent_atoms
    - find_overvalent_atoms

These functions are used by the 'MoleculeCleaner' and fatal repair modules to
drive rule-based decision making and determine when a molecule needs repair.

All functions assume RDKit 'Mol' objects and are safe to call on raw or partially sanitized inputs.
"""

from rdkit import Chem
from rdkit.Chem import rdchem, GetPeriodicTable
from functools import lru_cache

@lru_cache(maxsize=None)
def _default_valence(atomic_num: int) -> int:
    return Chem.GetPeriodicTable().GetDefaultValence(atomic_num)

@lru_cache(maxsize=None)
def _valence_list(atomic_num: int) -> tuple:
    # tuple ensures lru_cache works
    return tuple(Chem.GetPeriodicTable().GetValenceList(atomic_num))

def detect_dummies_in_aromatic_ring(mol):
    """
    Checks if any dummy atoms are connected to aromatic atoms.

    This typically indicates an invalid or unstable aromatic system
    that may require correction during cleanup.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule to inspect.

    Returns:
        bool: True if a dummy atom is adjacent to an aromatic atom.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and any(nbr.GetIsAromatic() for nbr in atom.GetNeighbors()):
            return True
    return False

def detect_unconnected_fragments(mol):
    """
    Detects whether the molecule consists of multiple disconnected fragments.

    Useful for enforcing single-fragment constraints or isolating the largest component.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule to inspect.

    Returns:
        bool: True if more than one disconnected fragment is present.
    """
    return len(Chem.GetMolFrags(mol)) > 1

def detect_valence_errors(mol: Chem.Mol) -> bool:
    """
    Return True if any atom's bond order sum (bonds + implicit Hs)
    exceeds one of its allowed valences (from RDKit's PeriodicTable),
    adjusted by formal charge.
    """
    for atom in mol.GetAtoms():
        Z = atom.GetAtomicNum()
        if Z == 0:
            continue  # skip dummies

        total_val = atom.GetTotalValence()
        charge = atom.GetFormalCharge()

        valences = _valence_list(Z)
        if valences:
            allowed = max(v + charge for v in valences)
        else:
            allowed = _default_valence(Z) + charge

        allowed = max(allowed, 0)  # avoid negatives

        if total_val > allowed:
            return True

    return False

def detect_formal_charge_imbalance(mol, max_charge=2):
    """
    Detects atoms with formal charges exceeding a threshold.

    Args:
        mol (rdkit.Chem.Mol): Molecule to inspect.
        max_charge (int): Maximum allowed magnitude of formal charge.

    Returns:
        bool: True if any atom exceeds the allowed charge.
    """
    for atom in mol.GetAtoms():
        if abs(atom.GetFormalCharge()) > max_charge:
            return True
    return False

def detect_missing_stereochemistry(mol):
    """
    Checks for stereocenters that lack explicit stereochemical labels.

    Uses RDKit's internal stereochemistry assignment logic.

    Args:
        mol (rdkit.Chem.Mol): Molecule to check.

    Returns:
        bool: True if any chiral atoms are undefined.
    """
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    return any(atom.HasProp('_ChiralityPossible') and not atom.HasProp('_CIPCode') for atom in mol.GetAtoms())

def detect_nonstandard_valence_deprecated(mol):
    """
    Detects atoms with explicit valence values greater than their standard defaults.

    This often signals hypervalence, which can be chemically invalid for certain atoms.

    Args:
        mol (rdkit.Chem.Mol): Molecule to inspect.

    Returns:
        bool: True if any atom exceeds its standard valence.
    """
    for atom in mol.GetAtoms():
        if atom.GetExplicitValence() > Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum()):
            return True
    return False

def detect_nonstandard_valence(mol):
    """
    Detects atoms with explicit valence values greater than their standard defaults.

    This often signals hypervalence, which can be chemically invalid for certain atoms.

    Args:
        mol (rdkit.Chem.Mol): Molecule to inspect.

    Returns:
        bool: True if any atom exceeds its standard valence.
    """
    for atom in mol.GetAtoms():
        Z = atom.GetAtomicNum()

        # only skip checking valence for the dummy atom itself
        if Z == 0:
            continue

        # still include dummy bonds when computing valence of normal atoms
        if atom.GetExplicitValence() > _default_valence(Z):
            return True

    return False

def detect_kekulization_failure(mol):
    """
    Attempts Kekulization and returns True if it fails.

    Useful for identifying problematic aromatic systems before converting to Kekule form.

    Args:
        mol (rdkit.Chem.Mol): Molecule to test.

    Returns:
        bool: True if Kekulization fails.
    """
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return True
    return False

def detect_rings_broken(mol):
    """
    Checks whether RDKit detects any rings in the molecule.

    Returns True if ring info is missing or zero, which may indicate corruption.

    Args:
        mol (rdkit.Chem.Mol): Molecule to analyze.

    Returns:
        bool: True if the molecule has no detected rings.
    """
    ring_info = mol.GetRingInfo()
    return not ring_info.NumRings()  # No rings detected when expected

def find_overvalent_atoms(mol: Chem.Mol) -> list[Chem.Atom]:
    """
    Identifies and returns RDKit Atom objects with excessive valence.

    Args:
        mol (rdkit.Chem.Mol): Molecule to inspect.

    Returns:
        List[rdkit.Chem.Atom]: List of atoms with valence violations.
    """
    bad = []
    for atom in mol.GetAtoms():
        Z = atom.GetAtomicNum()
        if Z == 0:
            continue  # skip checking dummy atoms themselves

        default_val = _default_valence(Z)
        if atom.GetExplicitValence() > default_val:
            bad.append(atom)
    return bad