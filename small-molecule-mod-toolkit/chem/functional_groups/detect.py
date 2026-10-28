# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/functional_groups/detect.py
"""
Functional group detection using AccFG.

This module wraps the AccFG library (532 functional groups) to provide
functional group detection that integrates with the MolGraph/Transform system.

AccFG provides:
  - 321 common functional groups (fgs_common.csv)  
  - 211 heterocycles (fgs_heterocycle.csv)
  - Hierarchical FG relationships via networkx DiGraph
  - Atom-level mapping for each match
  - Visualization tools (draw_mol_with_fgs, print_fg_tree)

The CSV files use special notation:
  - Lines starting with '#' are sub-categories (e.g., "#primary aliphatic amine")
  - Lines starting with '%' are comments
  - lite=True excludes '#'-prefixed sub-categories for faster/simpler matching

Reference:
    Liu et al. "AccFG: Accurate Functional Group Extraction and Molecular
    Structure Comparison" J. Chem. Inf. Model. 2025, 65, 8593-8602.
    https://github.com/xuanliugit/AccFG

Usage:
    from chem.functional_groups import detect_functional_groups, detect_from_molgraph
    
    # From SMILES
    result = detect_functional_groups("CC(=O)O")
    print(result.names)  # ['carboxylic acid']
    
    # With hierarchy graph
    result = detect_functional_groups("CC(=O)O", include_hierarchy=True)
    print_fg_tree(result.hierarchy, result.names)  # ASCII tree
    
    # Access the underlying AccFG instance for advanced usage
    afg = get_accfg_instance()
    print(len(afg.dict_fgs))  # 532 functional groups
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from chem.functional_groups.match import FunctionalGroupMatch, FunctionalGroupResult

if TYPE_CHECKING:
    from core.structs import MolGraph
    import networkx as nx

# Lazy-load AccFG to avoid import errors if not installed
_ACCFG_INSTANCE: Optional[Any] = None
_ACCFG_LITE_INSTANCE: Optional[Any] = None


def _get_accfg(lite: bool = False, user_defined_fgs: Optional[Dict[str, str]] = None) -> Any:
    """
    Get or create a cached AccFG instance.
    
    Args:
        lite: Use lite version - excludes '#'-prefixed sub-categories like
              "primary aliphatic amine" (keeps just "amine"). Faster but less specific.
        user_defined_fgs: Optional dict of {name: SMILES/SMARTS} for custom FGs.
              SMILES are auto-converted; SMARTS (containing ';') used directly.
    """
    global _ACCFG_INSTANCE, _ACCFG_LITE_INSTANCE
    
    try:
        from accfg import AccFG
    except ImportError as e:
        raise ImportError(
            "AccFG is not installed. Install it with: pip install accfg\n"
            "See https://github.com/xuanliugit/AccFG for documentation."
        ) from e
    
    # If user provides custom FGs, always create a new instance
    if user_defined_fgs:
        return AccFG(user_defined_fgs=user_defined_fgs, lite=lite, print_load_info=False)
    
    # Otherwise use cached instance
    if lite:
        if _ACCFG_LITE_INSTANCE is None:
            _ACCFG_LITE_INSTANCE = AccFG(lite=True, print_load_info=False)
        return _ACCFG_LITE_INSTANCE
    else:
        if _ACCFG_INSTANCE is None:
            _ACCFG_INSTANCE = AccFG(lite=False, print_load_info=False)
        return _ACCFG_INSTANCE


def get_accfg_instance(lite: bool = False) -> Any:
    """
    Get the AccFG instance for advanced usage.
    
    This exposes the full AccFG API including:
      - afg.dict_fgs: Dict of all {fg_name: SMARTS_pattern}
      - afg.dict_fgs_common: Just the common functional groups
      - afg.dict_fg_heterocycle: Just the heterocycles
      - afg.run(smiles, show_atoms=True, show_graph=True): Full detection
      - afg.run_mol(mol, ...): Detection from RDKit Mol
      - afg.run_freq(smiles): Get (fg_name, count) pairs
      
    Example:
        afg = get_accfg_instance()
        
        # See all available functional groups
        print(list(afg.dict_fgs.keys())[:20])
        
        # Get SMARTS for a specific group
        print(afg.dict_fgs.get("carboxylic acid"))
        
        # Run detection directly
        fgs, fg_graph = afg.run("CC(=O)O", show_atoms=True, show_graph=True)
        
        # fg_graph is a networkx DiGraph with hierarchy
        # Edges go from parent → child (e.g., "carboxylic acid" → "carbonyl")
        for node in fg_graph.nodes():
            print(f"{node}: {fg_graph.nodes[node].get('mapped_atoms')}")
    """
    return _get_accfg(lite=lite)


def get_smarts_for_fg(fg_name: str, lite: bool = False) -> Optional[str]:
    """
    Get the SMARTS pattern for a functional group by name.
    
    Useful for understanding exactly what atoms are matched and in what order.
    
    Example:
        >>> get_smarts_for_fg("carboxylic acid")
        '[CX3](=[OX1])[$([OX2H]),$([OX1-])]'
        
        >>> get_smarts_for_fg("primary aliphatic amine")
        '[NX3H2+0,NX4H3+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]'
    """
    afg = _get_accfg(lite=lite)
    return afg.dict_fgs.get(fg_name)


def print_fg_tree(fg_graph: "nx.DiGraph", root_names: List[str], show_atom_idx: bool = True) -> None:
    """
    Print the functional group hierarchy as an ASCII tree.
    
    This is a convenience wrapper around AccFG's print_fg_tree.
    
    Args:
        fg_graph: The hierarchy graph from detect_functional_groups(..., include_hierarchy=True)
        root_names: List of top-level FG names (usually result.names)
        show_atom_idx: Whether to show atom indices for each group
        
    Example:
        result = detect_functional_groups("CC(=O)O", include_hierarchy=True)
        print_fg_tree(result.hierarchy, result.names)
        # Output:
        # ├──carboxylic acid: ((1, 2, 3),)
        # │  ├──carbonyl: ((1, 2),)
        # │  └──hydroxy: ((3,),)
    """
    try:
        from accfg.draw import print_fg_tree as _print_fg_tree
        _print_fg_tree(fg_graph, root_names, show_atom_idx=show_atom_idx)
    except ImportError:
        # Fallback: simple print
        for name in root_names:
            atoms = fg_graph.nodes[name].get('mapped_atoms', []) if fg_graph else []
            print(f"├──{name}: {atoms}")


def detect_functional_groups(
    smiles: str,
    *,
    lite: bool = False,
    user_defined_fgs: Optional[Dict[str, str]] = None,
    include_hierarchy: bool = False,
    canonical: bool = True,
) -> FunctionalGroupResult:
    """
    Detect functional groups in a molecule from its SMILES string.
    
    Args:
        smiles: SMILES string of the molecule
        lite: Use AccFG lite mode - excludes detailed sub-categories like
              "primary aliphatic amine" (keeps just "amine"). Faster but less specific.
        user_defined_fgs: Optional dict of custom FGs {name: SMILES or SMARTS}.
              SMILES are auto-converted; SMARTS (containing ';') used directly.
        include_hierarchy: Include the full hierarchy graph (nx.DiGraph) in result.
              The graph has edges parent→child and nodes with 'mapped_atoms' attribute.
        canonical: Whether to canonicalize SMILES before processing (default True)
        
    Returns:
        FunctionalGroupResult containing all detected functional groups
        
    Example:
        >>> result = detect_functional_groups("CC(=O)O")
        >>> result.has("carboxylic acid")
        True
        >>> match = result.find_one("carboxylic acid")
        >>> match.atom_indices
        (1, 2, 3)
        
        # With hierarchy
        >>> result = detect_functional_groups("CC(=O)O", include_hierarchy=True)
        >>> print_fg_tree(result.hierarchy, result.names)
    """
    afg = _get_accfg(lite=lite, user_defined_fgs=user_defined_fgs)
    
    # Run AccFG - always request graph for hierarchy info
    fgs, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True, canonical=canonical)
    
    # Convert AccFG output to our match format
    # AccFG returns: {'FG Name': [(tuple of atom indices), ...], ...}
    # fg_graph is a networkx DiGraph with edges parent→child
    matches: List[FunctionalGroupMatch] = []
    
    for fg_name, occurrences in fgs.items():
        # Get hierarchy info from graph
        parent = None
        children: List[str] = []
        if fg_graph is not None:
            try:
                # In AccFG's graph, edges go parent → child
                # So predecessors are parents, successors are children
                preds = list(fg_graph.predecessors(fg_name))
                parent = preds[0] if preds else None
                children = list(fg_graph.successors(fg_name))
            except:
                pass
        
        # Get SMARTS pattern
        smarts = afg.dict_fgs.get(fg_name)
        
        for atom_tuple in occurrences:
            # Ensure it's a tuple of ints
            if not isinstance(atom_tuple, tuple):
                atom_tuple = tuple(atom_tuple)
            
            # Simple positional key_atoms - the SMARTS pattern determines order
            # Users can interpret based on the SMARTS pattern
            key_atoms = {f"atom_{i}": idx for i, idx in enumerate(atom_tuple)}
            
            match = FunctionalGroupMatch(
                name=fg_name,
                atom_indices=atom_tuple,
                key_atoms=key_atoms,
                parent=parent,
                children=children,
                smarts=smarts,
            )
            matches.append(match)
    
    return FunctionalGroupResult(
        matches=matches,
        hierarchy=fg_graph if include_hierarchy else None,
        smiles=smiles,
    )


def detect_from_molgraph(
    mg: "MolGraph",
    *,
    lite: bool = False,
    user_defined_fgs: Optional[Dict[str, str]] = None,
    include_hierarchy: bool = False,
) -> FunctionalGroupResult:
    """
    Detect functional groups in a MolGraph.
    
    This converts the MolGraph to SMILES first, then runs detection.
    Note: Atom indices in the result correspond to the MolGraph's atom ordering
    when using non-canonical SMILES.
    
    Args:
        mg: MolGraph to analyze
        lite: Use AccFG lite mode (excludes sub-categories)
        user_defined_fgs: Optional custom FG definitions
        include_hierarchy: Include hierarchy graph in result
        
    Returns:
        FunctionalGroupResult with matches indexed to MolGraph atoms
    """
    # Get or compute SMILES
    smiles = mg.meta.get("smiles") if mg.meta else None
    
    if smiles is None:
        # Need to convert MolGraph to SMILES
        try:
            from chem.build.molgraph_to_mol import molgraph_to_smiles
            # Use non-canonical to preserve atom ordering
            smiles = molgraph_to_smiles(mg, canonical=False, isomeric=True)
        except ImportError:
            raise ImportError(
                "Could not import molgraph_to_smiles. Ensure chem.build.molgraph_to_mol is available."
            )
        
        if smiles is None:
            raise ValueError("Could not convert MolGraph to SMILES for functional group detection")
    
    # Run detection on SMILES (don't canonicalize to preserve atom ordering)
    result = detect_functional_groups(
        smiles,
        lite=lite,
        user_defined_fgs=user_defined_fgs,
        include_hierarchy=include_hierarchy,
        canonical=False,  # Preserve atom ordering from MolGraph
    )
    
    return result


def detect_from_mol(
    mol: Any,  # rdkit.Chem.Mol
    *,
    lite: bool = False,
    user_defined_fgs: Optional[Dict[str, str]] = None,
    include_hierarchy: bool = False,
) -> FunctionalGroupResult:
    """
    Detect functional groups directly from an RDKit Mol object.
    
    Uses AccFG's run_mol() method for direct Mol processing.
    
    Args:
        mol: RDKit Mol object
        lite: Use AccFG lite mode
        user_defined_fgs: Optional custom FG definitions
        include_hierarchy: Include hierarchy graph in result
        
    Returns:
        FunctionalGroupResult with matches
    """
    afg = _get_accfg(lite=lite, user_defined_fgs=user_defined_fgs)
    
    # Use run_mol for direct Mol processing
    fgs, fg_graph = afg.run_mol(mol, show_atoms=True, show_graph=True)
    
    # Convert to our format
    matches: List[FunctionalGroupMatch] = []
    for fg_name, occurrences in fgs.items():
        # Get hierarchy info
        parent = None
        children: List[str] = []
        if fg_graph is not None:
            try:
                preds = list(fg_graph.predecessors(fg_name))
                parent = preds[0] if preds else None
                children = list(fg_graph.successors(fg_name))
            except:
                pass
        
        smarts = afg.dict_fgs.get(fg_name)
        
        for atom_tuple in occurrences:
            if not isinstance(atom_tuple, tuple):
                atom_tuple = tuple(atom_tuple)
            
            key_atoms = {f"atom_{i}": idx for i, idx in enumerate(atom_tuple)}
            
            match = FunctionalGroupMatch(
                name=fg_name,
                atom_indices=atom_tuple,
                key_atoms=key_atoms,
                parent=parent,
                children=children,
                smarts=smarts,
            )
            matches.append(match)
    
    return FunctionalGroupResult(
        matches=matches,
        hierarchy=fg_graph if include_hierarchy else None,
        smiles=None,
    )


def compare_molecules(
    smiles_a: str,
    smiles_b: str,
    similarity_threshold: float = 0.7,
) -> Tuple[Any, Any]:
    """
    Compare two molecules at the functional group level using AccFG.
    
    Returns the functional group differences between the two molecules,
    including both FG-level and alkane (backbone) differences.
    
    Args:
        smiles_a: SMILES of first molecule
        smiles_b: SMILES of second molecule
        similarity_threshold: Threshold for MCES similarity (default 0.7)
        
    Returns:
        Tuple of ((target_fgs, target_alkanes), (ref_fgs, ref_alkanes))
        where each *_fgs is a list of (fg_name, count, atom_indices_list) tuples
    """
    try:
        from accfg import compare_mols
    except ImportError as e:
        raise ImportError(
            "AccFG is not installed. Install it with: pip install accfg"
        ) from e
    
    return compare_mols(smiles_a, smiles_b, similarityThreshold=similarity_threshold)


# ============================================================================
# Visualization wrappers
# ============================================================================

def draw_mol_with_fgs(
    smiles: str,
    *,
    lite: bool = False,
    user_defined_fgs: Optional[Dict[str, str]] = None,
    with_legend: bool = True,
    with_atom_idx: bool = True,
    img_size: Tuple[int, int] = (500, 400),
) -> bytes:
    """
    Draw a molecule with functional groups highlighted.
    
    Returns PNG image bytes that can be displayed or saved.
    
    Args:
        smiles: SMILES string of the molecule
        lite: Use lite mode (fewer sub-categories)
        user_defined_fgs: Custom functional group definitions
        with_legend: Show legend with FG names
        with_atom_idx: Show atom indices on the molecule
        img_size: Image size as (width, height)
        
    Returns:
        PNG image as bytes
        
    Example:
        img_bytes = draw_mol_with_fgs("CC(=O)O")
        with open("acetic_acid_fgs.png", "wb") as f:
            f.write(img_bytes)
            
        # Or display in Jupyter
        from PIL import Image
        import io
        Image.open(io.BytesIO(img_bytes))
    """
    try:
        from accfg.draw import draw_mol_with_fgs as _draw
    except ImportError as e:
        raise ImportError(
            "AccFG visualization requires: pip install accfg\n"
            "Also requires: rdkit, matplotlib, PIL"
        ) from e
    
    afg = _get_accfg(lite=lite, user_defined_fgs=user_defined_fgs)
    return _draw(
        smiles,
        afg=afg,
        with_legend=with_legend,
        with_atom_idx=with_atom_idx,
        img_size=img_size,
    )


def draw_compare_mols(
    smiles_a: str,
    smiles_b: str,
    *,
    lite: bool = False,
    img_size: Tuple[int, int] = (500, 400),
) -> List[Any]:
    """
    Draw two molecules side-by-side with their functional group differences highlighted.
    
    Returns list of PIL Image objects.
    
    Args:
        smiles_a: SMILES of first molecule
        smiles_b: SMILES of second molecule
        lite: Use lite mode
        img_size: Size of each molecule image
        
    Returns:
        List of two PIL Image objects [mol_a_img, mol_b_img]
    """
    try:
        from accfg.draw import draw_compare_mols as _draw_compare
    except ImportError as e:
        raise ImportError(
            "AccFG visualization requires: pip install accfg"
        ) from e
    
    afg = _get_accfg(lite=lite)
    return _draw_compare(smiles_a, smiles_b, afg=afg, img_size=img_size)


# Convenience functions for common reaction queries

def find_carboxylic_acids(
    smiles_or_mg: Any,
    *,
    lite: bool = False,
) -> List[FunctionalGroupMatch]:
    """Find all carboxylic acid groups in a molecule."""
    if isinstance(smiles_or_mg, str):
        result = detect_functional_groups(smiles_or_mg, lite=lite)
    else:
        result = detect_from_molgraph(smiles_or_mg, lite=lite)
    return result.by_name("carboxylic acid")


def find_amines(
    smiles_or_mg: Any,
    *,
    primary_only: bool = False,
    lite: bool = False,
) -> List[FunctionalGroupMatch]:
    """
    Find all amine groups in a molecule.
    
    Args:
        smiles_or_mg: SMILES string or MolGraph
        primary_only: If True, only return primary amines
        lite: Use AccFG lite mode
    """
    if isinstance(smiles_or_mg, str):
        result = detect_functional_groups(smiles_or_mg, lite=lite)
    else:
        result = detect_from_molgraph(smiles_or_mg, lite=lite)
    
    amine_types = [
        "primary aliphatic amine",
        "primary aromatic amine",
    ]
    if not primary_only:
        amine_types.extend([
            "secondary aliphatic amine",
            "secondary mixed amine", 
            "secondary aromatic amine",
            "tertiary aliphatic amine",
            "tertiary mixed amine",
        ])
    
    matches = []
    for amine_type in amine_types:
        matches.extend(result.by_name(amine_type))
    return matches


def find_halides(
    smiles_or_mg: Any,
    *,
    alkyl_only: bool = True,
    lite: bool = False,
) -> List[FunctionalGroupMatch]:
    """
    Find all halide groups (for SN2 reactions).
    
    Args:
        smiles_or_mg: SMILES string or MolGraph
        alkyl_only: If True, only return alkyl halides (not aryl)
        lite: Use AccFG lite mode
    """
    if isinstance(smiles_or_mg, str):
        result = detect_functional_groups(smiles_or_mg, lite=lite)
    else:
        result = detect_from_molgraph(smiles_or_mg, lite=lite)
    
    halide_types = ["alkyl chloride", "alkyl bromide", "alkyl iodide", "alkyl fluoride"]
    if not alkyl_only:
        halide_types.extend(["aryl chloride", "aryl bromide", "aryl iodide", "aryl fluoride"])
    
    matches = []
    for halide_type in halide_types:
        matches.extend(result.by_name(halide_type))
    return matches


def find_alcohols(
    smiles_or_mg: Any,
    *,
    lite: bool = False,
) -> List[FunctionalGroupMatch]:
    """Find all alcohol (hydroxyl) groups in a molecule."""
    if isinstance(smiles_or_mg, str):
        result = detect_functional_groups(smiles_or_mg, lite=lite)
    else:
        result = detect_from_molgraph(smiles_or_mg, lite=lite)
    
    alcohol_types = ["primary hydroxyl", "secondary hydroxyl", "tertiary hydroxyl"]
    
    matches = []
    for alcohol_type in alcohol_types:
        matches.extend(result.by_name(alcohol_type))
    return matches


def list_all_functional_groups(lite: bool = False) -> List[str]:
    """
    List all available functional group names.
    
    AccFG provides 532 functional groups (321 common + 211 heterocycles).
    Use lite=True for a reduced set (~50% fewer, faster).
    
    Example:
        >>> fgs = list_all_functional_groups()
        >>> len(fgs)
        532
        >>> "carboxylic acid" in fgs
        True
        >>> "pyridine" in fgs
        True
    """
    afg = _get_accfg(lite=lite)
    return list(afg.dict_fgs.keys())


def search_functional_groups(query: str, lite: bool = False) -> List[str]:
    """
    Search for functional groups by name substring (case-insensitive).
    
    Example:
        >>> search_functional_groups("amine")
        ['amine', 'primary aliphatic amine', 'secondary aliphatic amine', ...]
        >>> search_functional_groups("pyri")
        ['pyridine', 'pyrimidine', 'pyrazine', ...]
    """
    all_fgs = list_all_functional_groups(lite=lite)
    query_lower = query.lower()
    return [fg for fg in all_fgs if query_lower in fg.lower()]