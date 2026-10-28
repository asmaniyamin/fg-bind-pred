# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/clean/fatal_repair_tools.py

Rule-based molecular repair utilities for fixing invalid chemical structures.

This module contains a series of conservative, chemistry-aware functions designed
to repair molecules that fail sanitization due to issues such as:
- Overvalent atoms
- Unconnected fragments
- Dummy atoms in aromatic rings
- Bad stereochemistry
- Protonation and tautomer inconsistencies

These "fatal" repairs are typically applied as a last resort when RDKit's
'SanitizeMol()' fails or when SMILES parsing produces unstable structures.

Key functions include:
    - fatal_repair_dummy_in_aromatic: Moves dummy atoms away from aromatic neighbors
    - fatal_repair_unconnected_fragments: Keeps the largest connected component
    - fatal_repair_remove_excess_hydrogens: Removes explicit H atoms to reduce valence
    - try_neutralise_common_hypervalence: Replaces known hypervalent SMARTS patterns
    - try_promote_hypervalent_atoms: Promotes S, P, Cl, etc., by assigning +1 formal charge
    - fatal_repair_valence / formal_charges / tautomer_normalization / stereochemistry: Misc. cleaning steps

These tools are orchestrated by 'MoleculeCleaner' and used to stabilize molecules
prior to graph conversion or property prediction.

Note:
    Repairs are destructive and prioritize RDKit validity over perfect chemical fidelity.
"""

from rdkit import Chem
import copy

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit.Chem.MolStandardize import rdMolStandardize

def fatal_repair_dummy_in_aromatic(mol):
    """
    Attempts to relocate dummy atoms from aromatic neighbors to nearby non-aromatic atoms.

    This repair avoids invalid or unstable aromatic systems caused by dummy atoms in rings.

    Args:
        mol (rdkit.Chem.Mol): Molecule to fix.

    Returns:
        rdkit.Chem.Mol: Repaired molecule if changes were made, otherwise original.
    """
    mol = copy.deepcopy(mol)
    rw_mol = Chem.RWMol(mol)
    changed = False

    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # dummy
            neighbors = atom.GetNeighbors()
            if any(nbr.GetIsAromatic() for nbr in neighbors):
                for nbr in neighbors:
                    if not nbr.GetIsAromatic():
                        dummy_idx = atom.GetIdx()
                        nbr_idx = nbr.GetIdx()
                        rw_mol.RemoveAtom(dummy_idx)
                        new_dummy = Chem.Atom(0)
                        rw_mol.AddAtom(new_dummy)
                        new_idx = rw_mol.GetNumAtoms() - 1
                        rw_mol.AddBond(nbr_idx, new_idx, Chem.rdchem.BondType.SINGLE)
                        changed = True
                        break
    return rw_mol.GetMol() if changed else mol

def fatal_repair_unconnected_fragments(mol):
    """
    If the molecule has disconnected fragments, keeps only the largest one.

    Args:
        mol (rdkit.Chem.Mol): Molecule with potentially multiple fragments.

    Returns:
        rdkit.Chem.Mol: Largest connected fragment.
    """
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        return max(frags, key=lambda m: m.GetNumAtoms())
    return mol

def fatal_repair_valence(mol):
    """
    Checks and repairs invalid valence. Returns None if the molecule is unfixable.

    Args:
        mol (rdkit.Chem.Mol): Molecule to sanitize.

    Returns:
        Optional[rdkit.Chem.Mol]: Sanitized molecule, or None if sanitization fails.
    """
    try:
        Chem.SanitizeMol(mol)
    except Chem.MolSanitizeException:
        return None  # signal to discard
    return mol

def fatal_repair_remove_excess_hydrogens(mol):
    """
    Removes all explicit hydrogens from the molecule to help resolve valence issues.

    Args:
        mol (rdkit.Chem.Mol): Molecule to clean.

    Returns:
        rdkit.Chem.Mol: Molecule with explicit hydrogens removed.
    """
    mol = Chem.RemoveHs(mol, sanitize=False)  # Remove only explicit Hs
    return mol


def fatal_repair_formal_charges(mol):
    """
    Updates RDKit property cache to correct or re-compute formal charges.

    Args:
        mol (rdkit.Chem.Mol): Molecule to update.

    Returns:
        rdkit.Chem.Mol: Molecule with refreshed charge state.
    """
    mol = copy.deepcopy(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol

def fatal_repair_stereochemistry(mol):
    """
    Assigns stereochemistry to atoms and bonds using RDKit's built-in routines.

    Args:
        mol (rdkit.Chem.Mol): Molecule to annotate.

    Returns:
        rdkit.Chem.Mol: Molecule with stereochemistry assigned.
    """
    mol = copy.deepcopy(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol

def fatal_repair_tautomer_normalization(mol):
    """
    Normalizes tautomeric and protonation states using RDKit's standardizer.

    Args:
        mol (rdkit.Chem.Mol): Molecule to normalize.

    Returns:
        rdkit.Chem.Mol: Standardized tautomeric form.
    """
    mol = copy.deepcopy(mol)
    normalizer = rdMolStandardize.Normalizer()
    return normalizer.normalize(mol)

def try_neutralise_overvalence(mol, bad_atoms):
    """
    Attempts to fix overvalent atoms by neutralizing formal charges.

    Applies only to atoms with atomic numbers 6 (C), 7 (N), or 8 (O) and non-zero charges.

    Args:
        mol (rdkit.Chem.Mol): Molecule to adjust.
        bad_atoms (List[Tuple[int, int, int]]): List of (atom_idx, valence, expected_valence).

    Returns:
        rdkit.Chem.Mol: Molecule with charges reset if fixable.
    """
    rw = Chem.RWMol(mol)
    changed = False
    for idx, vexp, vdef in bad_atoms:
        atom = rw.GetAtomWithIdx(idx)
        ch = atom.GetFormalCharge()
        if ch and atom.GetAtomicNum() in (6, 7, 8):   # C, N, O common [Note 1]
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)                 # let RDKit add Hs
            changed = True
    return rw.GetMol() if changed else mol

def try_promote_hypervalent_atoms(mol, bad_atoms):
    """
    Promotes overvalent atoms (e.g., P, S, halogens) to +1 formal charge.

    Applies to atoms like P5+, S6+, Cl+, etc., that commonly exceed default valence.

    Args:
        mol (rdkit.Chem.Mol): Molecule to repair.
        bad_atoms (List[Tuple[int, int, int]]): List of (atom_idx, valence, expected_valence).

    Returns:
        rdkit.Chem.Mol: Molecule with promoted formal charges if changed.
    """
    rw = Chem.RWMol(mol)
    changed = False
    for idx, vexp, vdef in bad_atoms:
        atom = rw.GetAtomWithIdx(idx) 
        if atom.GetAtomicNum() in (15, 16, 17, 35, 53):  # P S Cl Br I [Note 2]
            atom.SetFormalCharge(+1)
            atom.SetNoImplicit(False)
            changed = True
    return rw.GetMol() if changed else mol

_RESCUE_SMARTS = [
    # --- Nitro  N+(=O)O-  ->  N(=O)O  (already present) -------------------
    ("[N+](=O)[O-]", "N(=O)O"),

    # --- Sulfone / sulfonate  S+(=O)(=O)O-  ->  S(=O)(=O)O ----------------
    ("[S+](=O)(=O)[O-]", "S(=O)(=O)O"),

    # --- Quaternary amine  [N+](C)(C)C  ->  N(C)(C)  (already present) ----
    ("[N+](C)(C)C", "N(C)(C)"),

    # --- Anionic nitrile (both orientations) ------------------------------
    ("[C-]#N",  "C#N"),     #  *C- ≡ N
    ("N#[C-]",  "N#C"),     #  N ≡ C-*

    # --- Over-valent phosphoric acid (P-valence 6)-------------------------
    #      4 OH groups  ->  3 OH groups  (valence drops to 5 -> RDKit‐legal)
    ("P(=O)(O)(O)O",  "P(=O)(O)O"),
    # --- (possibly) keep the charge-balancing variant (after) this one ----
    ("[P+](=O)([O-])[O-]", "P(=O)(O)O"),
]

def try_neutralise_common_hypervalence(mol: Chem.Mol,
    max_iter: int = 15) -> Chem.Mol:
    """
    Attempts rule-based neutralization of known problematic substructures (e.g., nitro, sulfone).

    Replaces SMARTS-based overvalent patterns with chemically reasonable alternatives.
    Iterates until convergence or maximum attempts reached.

    Args:
        mol (rdkit.Chem.Mol): Molecule to fix.
        max_iter (int): Maximum number of transformation iterations.

    Returns:
        rdkit.Chem.Mol: Cleaned molecule, or original if no applicable changes.
    """
    rw       = Chem.RWMol(mol)
    previous = None          # SMILES snapshot to detect no-progress
    n_iter   = 0

    while n_iter < max_iter:
        n_iter  += 1
        changed  = False

        for smarts, repl_smiles in _RESCUE_SMARTS:
            patt = Chem.MolFromSmarts(smarts)
            repl = Chem.MolFromSmiles(repl_smiles, sanitize=False)

            if rw.HasSubstructMatch(patt):
                rw  = Chem.RWMol(
                        Chem.ReplaceSubstructs(rw.GetMol(),
                                               patt, repl,
                                               replaceAll=False)[0])
                changed = True
                break                       # restart pattern list

        # --- progress / convergence check ----------------------------
        smi_now = Chem.MolToSmiles(rw, canonical=True)
        if not changed or smi_now == previous:
            break
        previous = smi_now

    return rw.GetMol()

def fatal_repair_kekulization(mol):
    pass

def clear_nonring_aromaticity(mol):
    ri = mol.GetRingInfo()
    ring_atoms = set(i for ring in ri.AtomRings() for i in ring)
    ring_bonds = set(i for ring in ri.BondRings() for i in ring)

    for atom in mol.GetAtoms():
        if atom.GetIdx() not in ring_atoms:
            atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        if bond.GetIdx() not in ring_bonds:
            bond.SetIsAromatic(False)


def fatal_repair_all(mol):
    """
    Applies a full repair pipeline to the molecule in a single pass.
    This function has not been updated, nor is that planned.
    Retained for reference.

    Includes:
    - Dummy cleanup
    - Fragment isolation
    - Charge fix
    - Tautomer and stereochemistry normalization
    - Final valence check

    Args:
        mol (rdkit.Chem.Mol): Molecule to repair.

    Returns:
        rdkit.Chem.Mol: Fully repaired molecule.
    """
    mol = fatal_repair_dummy_in_aromatic(mol)
    mol = fatal_repair_unconnected_fragments(mol)
    mol = fatal_repair_formal_charges(mol)
    mol = fatal_repair_tautomer_normalization(mol)
    mol = fatal_repair_stereochemistry(mol)
    mol = fatal_repair_valence(mol)
    return mol