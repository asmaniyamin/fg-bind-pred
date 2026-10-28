# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/functional_groups/match.py

# IN PROGRESS - COMPLETE FOR CS273B PROJECT

"""
Dataclasses for functional group detection results.

These classes provide a bridge between AccFG's output and the MolGraph-based
Transform/Reaction system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(slots=True)
class FunctionalGroupMatch:
    """
    A single detected functional group with atom mappings.
    
    Attributes:
        name: The functional group name (e.g., "carboxylic_acid", "primary_amine")
        atom_indices: All atom indices involved in this functional group (tuple from AccFG)
        key_atoms: Named atoms within the group for reaction targeting.
                   e.g., {"carbonyl_c": 3, "oh_o": 4} for carboxylic acid
        parent: Parent functional group name in hierarchy (if any)
        children: Child functional group names in hierarchy
        smarts: The SMARTS pattern that matched (if available)
        
    Example:
        >>> match = FunctionalGroupMatch(
        ...     name="carboxylic_acid",
        ...     atom_indices=(5, 6, 7),
        ...     key_atoms={"carbonyl_c": 5, "carbonyl_o": 6, "oh_o": 7},
        ... )
        >>> match.carbonyl_c  # Access via attribute
        5
    """
    name: str
    atom_indices: Tuple[int, ...]
    key_atoms: Dict[str, int] = field(default_factory=dict)
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    smarts: Optional[str] = None
    
    def __getattr__(self, name: str) -> int:
        """Allow attribute-style access to key_atoms."""
        if name.startswith("_") or name in self.__slots__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self.key_atoms:
            return self.key_atoms[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (not in key_atoms)")
    
    @property
    def indices_array(self) -> np.ndarray:
        """Return atom_indices as numpy array."""
        return np.array(self.atom_indices, dtype=np.int64)
    
    def contains_atom(self, idx: int) -> bool:
        """Check if this functional group contains the given atom index."""
        return idx in self.atom_indices
    
    def overlaps_with(self, other: "FunctionalGroupMatch") -> bool:
        """Check if this functional group shares any atoms with another."""
        return bool(set(self.atom_indices) & set(other.atom_indices))


@dataclass(slots=True)
class FunctionalGroupResult:
    """
    Complete result of functional group detection on a molecule.
    
    Attributes:
        matches: List of all FunctionalGroupMatch objects
        hierarchy: The full hierarchy graph from AccFG (if requested)
        smiles: The SMILES string that was analyzed
        
    Methods:
        by_name: Get all matches of a specific functional group type
        find_one: Get exactly one match (raises if 0 or >1)
        atoms_to_groups: Map atom index → list of FG names containing it
    """
    matches: List[FunctionalGroupMatch]
    hierarchy: Optional[Any] = None  # AccFG's fg_graph
    smiles: Optional[str] = None
    
    def by_name(self, name: str) -> List[FunctionalGroupMatch]:
        """Get all matches with the given functional group name."""
        # Case-insensitive matching to handle AccFG's naming conventions
        name_lower = name.lower().replace("_", " ").replace("-", " ")
        return [
            m for m in self.matches 
            if m.name.lower().replace("_", " ").replace("-", " ") == name_lower
        ]
    
    def find_one(self, name: str, *, allow_multiple: bool = False) -> FunctionalGroupMatch:
        """
        Get exactly one match of the given type.
        
        Args:
            name: Functional group name to find
            allow_multiple: If True and multiple exist, return the first one
            
        Raises:
            ValueError: If no matches found, or if multiple found and allow_multiple=False
        """
        found = self.by_name(name)
        if not found:
            raise ValueError(f"No '{name}' functional group found in molecule")
        if len(found) > 1 and not allow_multiple:
            raise ValueError(
                f"Multiple '{name}' functional groups found ({len(found)}); "
                f"use allow_multiple=True or index with by_name()"
            )
        return found[0]
    
    def has(self, name: str) -> bool:
        """Check if the molecule contains the given functional group."""
        return len(self.by_name(name)) > 0
    
    def count(self, name: str) -> int:
        """Count occurrences of a functional group."""
        return len(self.by_name(name))
    
    @property
    def names(self) -> List[str]:
        """Get list of all unique functional group names found."""
        return list(dict.fromkeys(m.name for m in self.matches))
    
    def atoms_to_groups(self) -> Dict[int, List[str]]:
        """
        Build a mapping from atom index to functional group names.
        
        Useful for understanding which atoms are involved in which groups.
        """
        result: Dict[int, List[str]] = {}
        for match in self.matches:
            for idx in match.atom_indices:
                if idx not in result:
                    result[idx] = []
                result[idx].append(match.name)
        return result
    
    def non_fg_atoms(self, n_atoms: int) -> np.ndarray:
        """
        Return indices of atoms NOT involved in any functional group.
        
        Args:
            n_atoms: Total number of atoms in the molecule
            
        Returns:
            Array of atom indices not in any functional group (typically backbone carbons)
        """
        fg_atoms = set()
        for match in self.matches:
            fg_atoms.update(match.atom_indices)
        return np.array([i for i in range(n_atoms) if i not in fg_atoms], dtype=np.int64)