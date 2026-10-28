# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import traceback
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import ResonanceMolSupplier
from chem.clean.fatal_repair_tools import clear_nonring_aromaticity

# --- Logging wrapper that passes kwargs ------------------------------------------------
def step_with_logging(step_fn, smiles, verbose, **kwargs):
    try:
        mol = step_fn(smiles, **kwargs)
        if mol:
            if verbose:
                print(f"[PASS] Success in {step_fn.__name__}")
            return mol
    except Exception as e:
        if verbose:
            print(f"[FAIL] Failed in {step_fn.__name__}:\n{traceback.format_exc()}")
    return None

# --- Step 1: Sanitize all except kekulize, then attempt Kekulize -----------------------
def try_sanitize_then_kekulize_preserving_aromatic(smiles, **kwargs):
    params = Chem.SmilesParserParams()
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles, params)
    flags = Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_KEKULIZE
    Chem.SanitizeMol(mol, flags)
    clear_nonring_aromaticity(mol)
    Chem.Kekulize(mol, clearAromaticFlags=False)
    return mol

# --- Step 2: Sanitize skipping aromaticity perception (aromatizeIfPossible=False) ------
def try_sanitize_without_aromatization(smiles, **kwargs):
    aromatize = kwargs.get("aromatizeIfPossible", False)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    flags = Chem.SanitizeFlags.SANITIZE_ALL
    Chem.rdmolops.SanitizeMol(mol, flags, aromatizeIfPossible=aromatize)
    return mol

# --- Step 3: Clear aromaticity flags and sanitize --------------------------------------
def try_clear_aromaticity_and_sanitize(smiles, **kwargs):
    params = Chem.SmilesParserParams()
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles, params)
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        bond.SetIsAromatic(False)
    Chem.SanitizeMol(mol)
    return mol

# --- Step 4: Add explicit hydrogens ----------------------------------------------------
def try_add_hs_and_sanitize(smiles, **kwargs):
    params = Chem.SmilesParserParams()
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles, params)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    return mol

# --- Step 5: Normalize charges / tautomers ---------------------------------------------
def try_normalize_and_sanitize(smiles, **kwargs):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    normalizer = rdMolStandardize.Normalizer()
    mol = normalizer.normalize(mol)
    Chem.SanitizeMol(mol)
    return mol

# --- Step 6: Try resonance forms and pick a good one -----------------------------------
def try_resonance_forms(smiles, **kwargs):
    mol = Chem.MolFromSmiles(smiles)
    resonance_forms = ResonanceMolSupplier(mol, Chem.KEKULE_ALL)
    for res_mol in resonance_forms:
        try:
            Chem.SanitizeMol(res_mol)
            return res_mol
        except:
            continue
    return None

# --- Main controller -------------------------------------------------------------------
def fix_kekulization_modular(smiles, verbose=True, extreme=False, aromatizeIfPossible=False):
    steps = [
        try_sanitize_then_kekulize_preserving_aromatic,
        try_sanitize_without_aromatization,
        try_add_hs_and_sanitize,
        try_normalize_and_sanitize,
        try_resonance_forms,
    ]
    
    if extreme:
        steps.append(try_clear_aromaticity_and_sanitize)

    for step in steps:
        mol = step_with_logging(step, smiles, verbose, aromatizeIfPossible=aromatizeIfPossible)
        if mol:
            return mol

    raise ValueError(f"[FAIL] Kekulization failed for SMILES: {smiles}")
