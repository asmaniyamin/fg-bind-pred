# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/functional_groups/__init__.py
"""
Functional group detection and matching for MolGraph structures.

This module provides integration with AccFG for accurate functional group
extraction, adapted to work seamlessly with the MolGraph/Transform system.

AccFG Library:
    - 321 common functional groups (amines, acids, esters, halides, etc.)
    - 211 heterocycles (pyridine, imidazole, thiophene, etc.)
    - Hierarchical relationships between groups (via networkx DiGraph)
    - SMARTS-based matching with atom indices
    - Visualization tools for highlighting functional groups

Quick Start:
    from chem.functional_groups import detect_functional_groups, list_all_functional_groups
    
    # See what's available (532 functional groups!)
    all_fgs = list_all_functional_groups()
    print(len(all_fgs))  # 532
    
    # Search for specific types
    amines = search_functional_groups("amine")
    
    # Detect in a molecule
    result = detect_functional_groups("CC(=O)O")
    print(result.names)  # ['carboxylic acid']
    
    # Get detailed match info with hierarchy
    result = detect_functional_groups("CC(=O)O", include_hierarchy=True)
    match = result.find_one("carboxylic acid")
    print(f"Atoms: {match.atom_indices}")
    print(f"SMARTS: {match.smarts}")
    print(f"Children: {match.children}")  # Sub-groups like 'carbonyl'
    
    # Visualize (requires rdkit, matplotlib, PIL)
    img_bytes = draw_mol_with_fgs("CC(=O)O")

Lite Mode:
    Use lite=True to exclude detailed sub-categories (faster, simpler):
    - Full mode: "amine", "primary aliphatic amine", "secondary aliphatic amine", etc.
    - Lite mode: just "amine"
    
    result = detect_functional_groups("CCN", lite=True)

Reference:
    Liu et al. "AccFG: Accurate Functional Group Extraction and Molecular
    Structure Comparison" J. Chem. Inf. Model. 2025, 65, 8593-8602.
    https://github.com/xuanliugit/AccFG
"""

from chem.functional_groups.match import (
    FunctionalGroupMatch,
    FunctionalGroupResult,
)

from chem.functional_groups.detect import (
    # Core detection
    detect_functional_groups,
    detect_from_molgraph,
    detect_from_mol,
    compare_molecules,
    # Library inspection
    get_accfg_instance,
    get_smarts_for_fg,
    list_all_functional_groups,
    search_functional_groups,
    # Hierarchy visualization
    print_fg_tree,
    # Molecule visualization
    draw_mol_with_fgs,
    draw_compare_mols,
    # Convenience functions
    find_carboxylic_acids,
    find_amines,
    find_halides,
    find_alcohols,
)

__all__ = [
    # Core classes
    "FunctionalGroupMatch",
    "FunctionalGroupResult",
    # Detection functions
    "detect_functional_groups",
    "detect_from_molgraph",
    "detect_from_mol",
    "compare_molecules",
    # Library inspection
    "get_accfg_instance",
    "get_smarts_for_fg",
    "list_all_functional_groups",
    "search_functional_groups",
    # Visualization
    "print_fg_tree",
    "draw_mol_with_fgs",
    "draw_compare_mols",
    # Convenience functions
    "find_carboxylic_acids",
    "find_amines",
    "find_halides",
    "find_alcohols",
]