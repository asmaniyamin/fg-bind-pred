# -*- coding: utf-8 -*-
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/clean/clean_molecule.py

High-level molecule repair interface for sanitizing invalid or unstable RDKit Mol objects.

This module defines the 'MoleculeCleaner' class, which encapsulates a multi-stage
repair pipeline that attempts to resolve structural problems in SMILES-derived or
graph-generated molecules.

The cleaning pipeline includes:
    1. Dummy atom and fragment fixes
    2. Tautomer normalization and charge correction
    3. Stereochemistry assignment
    4. Valence and hydrogen fixes
    5. Hypervalence neutralization using SMARTS patterns

Key features:
    - Verbose logging of repair actions
    - Return of detailed repair logs alongside cleaned molecules
    - Fails gracefully with structured error messages if a molecule is unfixable

Designed for use in:
    - SMILES parsing workflows
    - Molecule graph restoration after mutation or merging
    - Preprocessing for model training and generation

Example usage:
    >>> cleaner = MoleculeCleaner(verbose=True, return_log=True)
    >>> mol, log = cleaner.clean(Chem.MolFromSmiles("invalid_smiles"))

Dependencies:
    - RDKit
    - fatal_repair_tools.py
    - detect_tools.py
    - kekulization_repair.py
"""

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

from rdkit import Chem
import chem.clean.detect_tools as detect
import chem.clean.fatal_repair_tools as repair
from chem.clean.kekulization_repair import fix_kekulization_modular

class MoleculeCleaner:
    """
    Rule-based molecule fixer that attempts to sanitize and repair invalid RDKit Mol objects.

    This class is typically used after SMILES parsing or graph reconstruction to
    detect and fix structural problems such as:
      - Dummy atoms in aromatic rings
      - Unconnected fragments
      - Invalid formal charges
      - Tautomer normalization
      - Overvalent atoms (hypervalence)
      - Stereochemistry inconsistencies

    It applies increasingly aggressive repair steps and can optionally return a
    detailed log of actions taken.

    Attributes:
        verbose (bool): Whether to print logs to stdout.
        return_log (bool): Whether to return a list of repair steps with the molecule.
        logs (List[str]): Stores cleanup steps taken during the last run.
    """
    def __init__(self, verbose: bool = False, return_log: bool = False, preserveAromaticity: bool = False):
        """
        Initializes the molecule cleaner.

        Args:
            verbose (bool): If True, prints the cleanup log after each run.
            return_log (bool): If True, returns a tuple (mol, log) instead of just mol.
            preserveAromaticity (bool): If True, 
        """
        self.verbose = verbose
        self.return_log = return_log
        self.logs = []
        self.preserveAromaticity = preserveAromaticity

    def clean_dep(self, mol: Chem.Mol):
        """
        Deprecated cleanup method that applied a simpler repair sequence.

        Fixed known failure modes such as dummy atoms in aromatic rings,
        disconnected fragments, charge normalization, and stereochemistry.

        Returns:
            Chem.Mol or Tuple[Chem.Mol, List[str]]: The repaired molecule, optionally with a log.
            Returns None (or (None, log)) if the molecule cannot be fixed.

        Note:
            This method is deprecated. Use 'clean()' instead for enhanced hypervalence handling.
        """
        self.logs.clear()

        self.logs.append(f"Attempting cleanup of: {Chem.MolToSmiles(mol)}")

        if detect.detect_dummies_in_aromatic_ring(mol):
            mol = repair.fatal_repair_dummy_in_aromatic(mol)
            self.logs.append('Moved dummy atom from aromatic ring')

        if detect.detect_unconnected_fragments(mol):
            mol = repair.fatal_repair_unconnected_fragments(mol)
            self.logs.append('Fixed unconnected fragments')

        mol = repair.fatal_repair_formal_charges(mol)
        mol = repair.fatal_repair_tautomer_normalization(mol)
        mol = repair.fatal_repair_stereochemistry(mol)

        if detect.detect_valence_errors(mol):
            mol = repair.fatal_repair_remove_excess_hydrogens(mol)
            mol = repair.fatal_repair_formal_charges(mol)
            if detect.detect_valence_errors(mol):
                self.logs.append('Unfixable valence error -- molecule discarded')
                return (None, self.logs) if self.return_log else None
            else:
                self.logs.append('Fixed valence error via H removal')

        if self.verbose and self.logs:
            print("Cleaner log:", self.logs)

        return (mol, self.logs) if self.return_log else mol

    def clean(self, mol: Chem.Mol):
        """
        Cleans and sanitizes a molecule by applying escalating repair strategies.

        Stages include:
            1. Fixing dummy atoms, fragments, charges, tautomers, stereochemistry
            2. Attempting RDKit sanitization
            3. Removing explicit hydrogens
            4. Neutralizing common hypervalent atoms
            5. Failing safely if unrecoverable

        Args:
            mol (Chem.Mol): The RDKit molecule to clean.

        Returns:
            Chem.Mol or Tuple[Chem.Mol, List[str]]: A repaired molecule,
            optionally with a list of log messages describing actions taken.
            Returns None (or (None, log)) if the molecule is unfixable.
        """
        self.logs.clear()
        smi0 = Chem.MolToSmiles(mol, canonical=True)
        self.logs.append(f"Attempting cleanup of: {smi0}")

        # ---------- 1. ultra-safe edits ------------------------------------
        #if detect.detect_dummies_in_aromatic_ring(mol): 
        #   [THIS SETTING HAS BEEN DISABLED: the project now has support for dummies in aromatic rings]
        #    mol = repair.fatal_repair_dummy_in_aromatic(mol)
        #    self.logs.append("Moved dummy atom from aromatic ring")

        if detect.detect_unconnected_fragments(mol):
            mol = repair.fatal_repair_unconnected_fragments(mol)
            self.logs.append("Kept largest fragment")

        # fast normalisations (never harm):
        mol = repair.fatal_repair_formal_charges(mol)
        mol = repair.fatal_repair_tautomer_normalization(mol)
        mol = repair.fatal_repair_stereochemistry(mol)

        # ---------- 2. RDKit sanitise (quick exit if already fine) ---------
        if not detect.detect_valence_errors(mol):
            return (mol, self.logs) if self.return_log else mol

        # ---------- 3. cheap generic fix: strip explicit H atoms -----------
        mol = repair.fatal_repair_remove_excess_hydrogens(mol)
        mol = repair.fatal_repair_formal_charges(mol)

        if not detect.detect_valence_errors(mol):
            self.logs.append("Fixed valence error via H removal")
            return (mol, self.logs) if self.return_log else mol

        # ---------- KEKULE -------------------------------------------------
        if detect.detect_kekulization_failure(mol):
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                try:
                    smi = Chem.MolToSmarts(mol)
                except Exception:
                    self.logs.append("Could not serialize molecule for Kekulization repair — skipping.")
                    return (None, self.logs) if self.return_log else None

            self.logs.append(f"Converted Kekule-error molecule to SMILES/SMARTS: {smi}")
            self.logs.append("Processing fallback strategy...")

            try:
                mol = fix_kekulization_modular(
                    smi,
                    verbose=self.verbose,
                    aromatizeIfPossible=self.preserveAromaticity
                )
                # restore ring aromaticity where possible
                Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_NONE)
            except Exception as e:
                self.logs.append(f"Kekulization fallback failed: {e}")
                return (None, self.logs) if self.return_log else None

            self.logs.append("Resolved Kekulization failure using fallback strategy.")

        # ---------- 4. very last resort: hyper-valence neutralizer ---------
        bad_atoms = detect.find_overvalent_atoms(mol)
        if bad_atoms:
            self.logs.append(
                f"Over-valent atoms detected: {[a.GetIdx() for a in bad_atoms]}"
            )

            mol = repair.try_neutralise_common_hypervalence(mol)
            mol = repair.fatal_repair_formal_charges(mol)   # recalc charges

            if not detect.detect_valence_errors(mol):
                self.logs.append("Neutralised hyper-valent centre(s)")
                return (mol, self.logs) if self.return_log else mol

        # ---------- 5. still broken -> give up -----------------------------
        self.logs.append("Unfixable valence error - molecule discarded")
        return (None, self.logs) if self.return_log else None

if __name__ == "__main__":
    """
    IGNORE: (was for manual testing)
    """
    smiles = "C1=CC=CC([1*])=C1"
    mol = Chem.MolFromSmiles(smiles)

    cleaner = MoleculeCleaner(verbose=True)
    cleaned_mol = cleaner.clean(mol)

    if cleaned_mol is None:
        print("Molecule discarded.")
    else:
        print("Cleaned SMILES:", Chem.MolToSmiles(cleaned_mol))

