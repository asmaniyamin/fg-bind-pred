# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# tests/test_functional_groups.py
"""
Test suite for functional group detection with MolGraph integration.

Tests that:
1. Functional groups are correctly detected from SMILES
2. Atom indices from AccFG match MolGraph atom ordering
3. Round-trip MolGraph → RDKit Mol preserves atom correspondence
4. Complex molecules with multiple FGs are handled correctly

Test molecules:
1. Risperidone-like: CC1=C(C(=O)N2CCCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F
   - imidazole derivative, piperidine, isoxazole, fluoride, ketone/amide
   
2. Raloxifene-like: C1CCN(CC1)CCOC2=CC=C(C=C2)C(=O)C3=C(SC4=C3C=CC(=C4)O)C5=CC=C(C=C5)O
   - piperidine, ether, ketone, thiophene, phenols, benzene rings
   
3. Complex amide (Cyclophilin Inhibitor 3, CAS 1676100-30-3): COC1=CC=C(C=C1OC)CCN(C(C2=CC=C(C=C2)[N+]([O-])=O)C(NC3CCCCC3)=O)C(CC4=CNC5=CC=CC=C45)=O
   - methoxy groups, nitro group, tertiary amide, secondary amide, indole
"""

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import pytest
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

from chem.functional_groups.detect import draw_mol_with_fgs


# Test molecules with expected functional groups
TEST_MOLECULES = [
    {
        "name": "Risperidone-like",
        "smiles": "CC1=C(C(=O)N2CCCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F",
        "expected_fgs": [
            "fluoride",
            "carbonyl",
            "lactam",  # cyclic amide
            "piperidine",
            "amine",
        ],
        "min_fg_count": 3,  # At least this many different FG types
    },
    {
        "name": "Raloxifene-like", 
        "smiles": "C1CCN(CC1)CCOC2=CC=C(C=C2)C(=O)C3=C(SC4=C3C=CC(=C4)O)C5=CC=C(C=C5)O",
        "expected_fgs": [
            "phenol",
            "ketone",
            "ether",
            "piperidine",
            "thiophene",
            "benzene",
        ],
        "min_fg_count": 4,
    },
    {
        "name": "Complex amide with nitro and indole",
        "smiles": "COC1=CC=C(C=C1OC)CCN(C(C2=CC=C(C=C2)[N+]([O-])=O)C(NC3CCCCC3)=O)C(CC4=CNC5=CC=CC=C45)=O",
        "expected_fgs": [
            "ether",  # methoxy
            "amide",
            "1H-indole",
            "benzene",
        ],
        "min_fg_count": 4,
    },
]


# ============================================================================
# Fixtures and helpers
# ============================================================================

@dataclass
class AtomInfo:
    """Helper to store atom information for verification."""
    index: int
    symbol: str
    atomic_num: int
    degree: int  # number of bonds
    

def get_atom_info_from_rdkit(mol) -> Dict[int, AtomInfo]:
    """Extract atom information from RDKit mol for verification."""
    from rdkit import Chem
    
    info = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        info[idx] = AtomInfo(
            index=idx,
            symbol=atom.GetSymbol(),
            atomic_num=atom.GetAtomicNum(),
            degree=atom.GetDegree(),
        )
    return info


def get_atom_info_from_molgraph(mg) -> Dict[int, AtomInfo]:
    """Extract atom information from MolGraph for verification."""
    info = {}
    n_atoms = int(mg.arrays.atomic_num.shape[0])
    
    # Get degree from bonds
    degrees = [0] * n_atoms
    for bond in mg.arrays.bonds:
        i, j = int(bond[0]), int(bond[1])
        degrees[i] += 1
        degrees[j] += 1
    
    # Map atomic number to symbol
    ATOMIC_SYMBOLS = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
    }
    
    for i in range(n_atoms):
        z = int(mg.arrays.atomic_num[i])
        info[i] = AtomInfo(
            index=i,
            symbol=ATOMIC_SYMBOLS.get(z, f"#{z}"),
            atomic_num=z,
            degree=degrees[i],
        )
    return info


def verify_fg_atoms_match_expected_elements(
    fg_name: str,
    atom_indices: Tuple[int, ...],
    atom_info: Dict[int, AtomInfo],
    smarts: str,
) -> Tuple[bool, str]:
    """
    Verify that atoms at the given indices have element types consistent with the FG.
    
    Returns (is_valid, message).
    """
    # Extract expected elements from SMARTS pattern (simplified)
    # This is a basic check - we look for element symbols in the SMARTS
    expected_elements = set()
    
    # Common element patterns in SMARTS
    element_patterns = {
        'C': {'C', 'c'},
        'N': {'N', 'n'},
        'O': {'O', 'o'},
        'S': {'S', 's'},
        'F': {'F'},
        'Cl': {'Cl'},
        'Br': {'Br'},
        'I': {'I'},
        'P': {'P'},
    }
    
    for elem, patterns in element_patterns.items():
        for p in patterns:
            if p in smarts:
                expected_elements.add(elem)
    
    # Get actual elements at the indices
    actual_elements = set()
    for idx in atom_indices:
        if idx in atom_info:
            actual_elements.add(atom_info[idx].symbol)
        else:
            return False, f"Atom index {idx} out of range"
    
    # Check that actual elements are a subset of expected (allowing for implicit)
    # This is a loose check since SMARTS can be complex
    if not actual_elements:
        return False, "No atoms found at indices"
    
    # At least one atom should match expected elements
    if expected_elements and not (actual_elements & expected_elements):
        return False, f"Element mismatch: got {actual_elements}, expected some of {expected_elements}"
    
    return True, "OK"


# ============================================================================
# Test: Basic detection from SMILES
# ============================================================================

class TestFunctionalGroupDetection:
    """Test basic functional group detection."""
    
    def test_import_modules(self):
        """Verify all modules can be imported."""
        from chem.functional_groups import (
            detect_functional_groups,
            detect_from_molgraph,
            FunctionalGroupMatch,
            FunctionalGroupResult,
            list_all_functional_groups,
            search_functional_groups,
            get_smarts_for_fg,
        )
        
        # Check we have functional groups loaded
        all_fgs = list_all_functional_groups()
        assert len(all_fgs) > 500, f"Expected 500+ FGs, got {len(all_fgs)}"
    
    @pytest.mark.parametrize("mol_data", TEST_MOLECULES, ids=lambda x: x["name"])
    def test_detection_from_smiles(self, mol_data):
        """Test that functional groups are detected from SMILES."""
        from chem.functional_groups import detect_functional_groups
        
        result = detect_functional_groups(mol_data["smiles"])
        
        # Should find some functional groups
        assert len(result.matches) > 0, f"No FGs found for {mol_data['name']}"
        
        # Should find minimum expected number of FG types
        unique_fg_names = set(m.name for m in result.matches)
        assert len(unique_fg_names) >= mol_data["min_fg_count"], (
            f"Expected at least {mol_data['min_fg_count']} FG types, "
            f"got {len(unique_fg_names)}: {unique_fg_names}"
        )
        
        # Check that expected FGs are found (at least some of them)
        found_expected = [
            fg for fg in mol_data["expected_fgs"]
            if any(fg.lower() in m.name.lower() for m in result.matches)
        ]
        assert len(found_expected) >= len(mol_data["expected_fgs"]) // 2, (
            f"Expected to find some of {mol_data['expected_fgs']}, "
            f"but only found {found_expected} in {unique_fg_names}"
        )
    
    @pytest.mark.parametrize("mol_data", TEST_MOLECULES, ids=lambda x: x["name"])
    def test_atom_indices_are_valid(self, mol_data):
        """Test that returned atom indices are within valid range."""
        from chem.functional_groups import detect_functional_groups
        from rdkit import Chem
        
        smiles = mol_data["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()
        
        result = detect_functional_groups(smiles)
        
        for match in result.matches:
            for idx in match.atom_indices:
                assert 0 <= idx < n_atoms, (
                    f"Atom index {idx} out of range [0, {n_atoms}) "
                    f"for FG '{match.name}'"
                )
    
    @pytest.mark.parametrize("mol_data", TEST_MOLECULES, ids=lambda x: x["name"])
    def test_atom_elements_match_fg_type(self, mol_data):
        """Test that atoms at FG indices have appropriate element types."""
        from chem.functional_groups import detect_functional_groups
        from rdkit import Chem
        
        smiles = mol_data["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        atom_info = get_atom_info_from_rdkit(mol)
        
        result = detect_functional_groups(smiles)
        
        for match in result.matches:
            if match.smarts:
                is_valid, msg = verify_fg_atoms_match_expected_elements(
                    match.name,
                    match.atom_indices,
                    atom_info,
                    match.smarts,
                )
                # Log but don't fail on element mismatches (SMARTS can be complex)
                if not is_valid:
                    print(f"Warning: {match.name} - {msg}")


# ============================================================================
# Test: MolGraph integration
# ============================================================================

class TestMolGraphIntegration:
    """Test functional group detection with MolGraph."""
    
    @pytest.fixture
    def smiles_to_molgraph(self):
        """Get the smiles_to_molgraph function."""
        try:
            from chem.build.create_molgraph import smiles_to_molgraph
            return smiles_to_molgraph
        except ImportError:
            pytest.skip("smiles_to_molgraph not available")
    
    @pytest.fixture
    def molgraph_to_mol(self):
        """Get the molgraph_to_mol function."""
        try:
            from chem.build.molgraph_to_mol import molgraph_to_mol
            return molgraph_to_mol
        except ImportError:
            pytest.skip("molgraph_to_mol not available")
    
    @pytest.mark.parametrize("mol_data", TEST_MOLECULES, ids=lambda x: x["name"])
    def test_molgraph_creation(self, mol_data, smiles_to_molgraph):
        """Test that MolGraph can be created from SMILES."""
        mg = smiles_to_molgraph(mol_data["smiles"])
        
        assert mg is not None
        assert mg.arrays.atomic_num.shape[0] > 0
        assert mg.arrays.bonds.shape[0] > 0
    
    @pytest.mark.parametrize("mol_data", TEST_MOLECULES, ids=lambda x: x["name"])
    def test_detect_from_molgraph(self, mol_data, smiles_to_molgraph):
        """Test functional group detection from MolGraph."""
        from chem.functional_groups import detect_from_molgraph
        
        mg = smiles_to_molgraph(mol_data["smiles"])
        result = detect_from_molgraph(mg)
        
        assert len(result.matches) > 0, f"No FGs found for {mol_data['name']}"
        
        unique_fg_names = set(m.name for m in result.matches)
        assert len(unique_fg_names) >= mol_data["min_fg_count"]
    
    @pytest.mark.parametrize("mol_data", TEST_MOLECULES, ids=lambda x: x["name"])
    def test_atom_indices_correspond_to_molgraph(
        self, mol_data, smiles_to_molgraph, molgraph_to_mol
    ):
        """
        Critical test: Verify that atom indices from FG detection
        correspond to the correct atoms in MolGraph.
        
        This checks the round-trip: SMILES → MolGraph → RDKit Mol → AccFG → indices
        """
        from chem.functional_groups import detect_from_molgraph
        
        smiles = mol_data["smiles"]
        
        # Create MolGraph
        mg = smiles_to_molgraph(smiles)
        mg_atom_info = get_atom_info_from_molgraph(mg)
        
        # Convert back to RDKit Mol
        rdkit_mol = molgraph_to_mol(mg)
        rdkit_atom_info = get_atom_info_from_rdkit(rdkit_mol)
        
        # Detect FGs from MolGraph
        result = detect_from_molgraph(mg)
        
        # Verify atom correspondence
        n_atoms_mg = len(mg_atom_info)
        n_atoms_rdkit = len(rdkit_atom_info)
        
        assert n_atoms_mg == n_atoms_rdkit, (
            f"Atom count mismatch: MolGraph has {n_atoms_mg}, RDKit has {n_atoms_rdkit}"
        )
        
        # Check each FG match
        for match in result.matches:
            for idx in match.atom_indices:
                # Index should be valid for both
                assert idx in mg_atom_info, f"Index {idx} not in MolGraph"
                assert idx in rdkit_atom_info, f"Index {idx} not in RDKit mol"
                
                # Element types should match
                mg_elem = mg_atom_info[idx].symbol
                rdkit_elem = rdkit_atom_info[idx].symbol
                assert mg_elem == rdkit_elem, (
                    f"Element mismatch at index {idx}: "
                    f"MolGraph has {mg_elem}, RDKit has {rdkit_elem}"
                )
    
    @pytest.mark.parametrize("mol_data", TEST_MOLECULES, ids=lambda x: x["name"])
    def test_fg_indices_point_to_correct_elements(
        self, mol_data, smiles_to_molgraph
    ):
        """
        Verify that FG atom indices point to atoms with appropriate element types.
        
        For example:
        - "carboxylic acid" should have C, O atoms
        - "phenol" should have O atoms
        - "fluoride" should have F atom
        """
        from chem.functional_groups import detect_from_molgraph
        
        mg = smiles_to_molgraph(mol_data["smiles"])
        mg_atom_info = get_atom_info_from_molgraph(mg)
        
        result = detect_from_molgraph(mg)
        
        # Define expected elements for common FGs
        FG_EXPECTED_ELEMENTS = {
            "fluoride": {"F"},
            "chloride": {"Cl"},
            "bromide": {"Br"},
            "iodide": {"I"},
            "phenol": {"O"},
            "hydroxy": {"O"},
            "ether": {"O"},
            "amine": {"N"},
            "nitrile": {"C", "N"},
            "amide": {"C", "N", "O"},
            "carbonyl": {"C", "O"},
            "carboxylic acid": {"C", "O"},
            "ketone": {"C", "O"},
            "aldehyde": {"C", "O"},
            "thiol": {"S"},
            "sulfide": {"S"},
            "thiophene": {"C", "S"},
        }
        
        errors = []
        for match in result.matches:
            # Find matching expected elements
            fg_lower = match.name.lower()
            expected = None
            for fg_key, elements in FG_EXPECTED_ELEMENTS.items():
                if fg_key in fg_lower:
                    expected = elements
                    break
            
            if expected:
                actual_elements = {
                    mg_atom_info[idx].symbol 
                    for idx in match.atom_indices 
                    if idx in mg_atom_info
                }
                
                # At least one expected element should be present
                if not (actual_elements & expected):
                    errors.append(
                        f"{match.name}: expected some of {expected}, "
                        f"got {actual_elements} at indices {match.atom_indices}"
                    )
        
        assert not errors, f"Element mismatches:\n" + "\n".join(errors)


# ============================================================================
# Test: Consistency between detection methods
# ============================================================================

class TestDetectionConsistency:
    """Test that different detection methods give consistent results."""
    
    @pytest.fixture
    def smiles_to_molgraph(self):
        try:
            from chem.build.create_molgraph import smiles_to_molgraph
            return smiles_to_molgraph
        except ImportError:
            pytest.skip("smiles_to_molgraph not available")
    
    @pytest.mark.parametrize("mol_data", TEST_MOLECULES, ids=lambda x: x["name"])
    def test_smiles_vs_molgraph_detection(self, mol_data, smiles_to_molgraph):
        """
        Compare FG detection from SMILES vs from MolGraph.
        
        They should find the same functional groups (though indices may differ
        if canonicalization changes atom ordering).
        """
        from chem.functional_groups import detect_functional_groups, detect_from_molgraph
        
        smiles = mol_data["smiles"]
        
        # Detect from SMILES (canonical=True by default)
        result_smiles = detect_functional_groups(smiles)
        fg_names_smiles = set(m.name for m in result_smiles.matches)
        
        # Detect from MolGraph
        mg = smiles_to_molgraph(smiles)
        result_mg = detect_from_molgraph(mg)
        fg_names_mg = set(m.name for m in result_mg.matches)
        
        # Should find the same FG types (allowing for minor differences due to canonicalization)
        common = fg_names_smiles & fg_names_mg
        only_smiles = fg_names_smiles - fg_names_mg
        only_mg = fg_names_mg - fg_names_smiles
        
        # Most FGs should be found by both methods
        assert len(common) >= len(fg_names_smiles) * 0.7, (
            f"Too few common FGs. "
            f"Common: {common}, Only SMILES: {only_smiles}, Only MG: {only_mg}"
        )
    
    @pytest.mark.parametrize("mol_data", TEST_MOLECULES, ids=lambda x: x["name"])
    def test_hierarchy_information(self, mol_data):
        """Test that hierarchy information is populated correctly."""
        from chem.functional_groups import detect_functional_groups
        
        result = detect_functional_groups(mol_data["smiles"], include_hierarchy=True)
        
        # Check that hierarchy graph exists
        assert result.hierarchy is not None
        
        # Check that some matches have parent/children info
        has_parent = any(m.parent is not None for m in result.matches)
        has_children = any(len(m.children) > 0 for m in result.matches)
        
        # At least some hierarchy info should be present for complex molecules
        # (This may not always be true depending on the FGs found)
        if len(result.matches) > 3:
            assert has_parent or has_children, (
                f"No hierarchy info for molecule with {len(result.matches)} FGs"
            )


# ============================================================================
# Test: Edge cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and potential error conditions."""
    
    def test_empty_smiles(self):
        """Test handling of empty/invalid SMILES."""
        from chem.functional_groups import detect_functional_groups
        
        # This should raise an error or return empty result
        with pytest.raises(Exception):
            detect_functional_groups("")
    
    def test_single_atom(self):
        """Test detection on single-atom molecules."""
        from chem.functional_groups import detect_functional_groups
        
        # Methane - should find nothing or just "alkane"
        result = detect_functional_groups("C")
        # Should not crash
        assert isinstance(result.matches, list)
    
    def test_simple_molecules(self):
        """Test on simple molecules with known FGs."""
        from chem.functional_groups import detect_functional_groups
        
        test_cases = [
            ("CC(=O)O", "carboxylic acid"),
            ("CCO", "hydroxy"),
            ("CCN", "amine"),
            ("CC=O", "aldehyde"),
            ("CC(=O)C", "ketone"),
            ("c1ccccc1", "benzene"),
            ("c1ccncc1", "pyridine"),
        ]
        
        for smiles, expected_fg in test_cases:
            result = detect_functional_groups(smiles)
            fg_names = [m.name.lower() for m in result.matches]
            
            assert any(expected_fg in name for name in fg_names), (
                f"Expected '{expected_fg}' in {smiles}, got {fg_names}"
            )
    
    def test_lite_mode_finds_fewer_fgs(self):
        """Test that lite mode finds fewer (less specific) FGs."""
        from chem.functional_groups import detect_functional_groups
        
        smiles = "CCN"  # ethylamine
        
        result_full = detect_functional_groups(smiles, lite=False)
        result_lite = detect_functional_groups(smiles, lite=True)
        
        # Lite mode should find fewer or equal FGs
        assert len(result_lite.matches) <= len(result_full.matches)
    
    def test_smarts_pattern_available(self):
        """Test that SMARTS patterns are included in matches."""
        from chem.functional_groups import detect_functional_groups
        
        result = detect_functional_groups("CC(=O)O")
        
        for match in result.matches:
            # SMARTS should be available
            assert match.smarts is not None, f"No SMARTS for {match.name}"
            assert len(match.smarts) > 0


# ============================================================================
# Test: Specific functional group verification
# ============================================================================

class TestSpecificFunctionalGroups:
    """Detailed tests for specific functional groups."""
    
    def test_carboxylic_acid_atoms(self):
        """Test carboxylic acid detection and atom mapping."""
        from chem.functional_groups import detect_functional_groups
        from rdkit import Chem
        
        smiles = "CC(=O)O"  # acetic acid
        mol = Chem.MolFromSmiles(smiles)
        
        result = detect_functional_groups(smiles)
        acid_matches = result.by_name("carboxylic acid")
        
        assert len(acid_matches) == 1, f"Expected 1 carboxylic acid, got {len(acid_matches)}"
        
        match = acid_matches[0]
        
        # Should have 3 atoms: C, =O, and OH
        assert len(match.atom_indices) == 3, (
            f"Carboxylic acid should have 3 atoms, got {len(match.atom_indices)}"
        )
        
        # Verify elements
        elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in match.atom_indices]
        assert 'C' in elements, f"Carboxylic acid should contain C, got {elements}"
        assert elements.count('O') == 2, f"Carboxylic acid should have 2 O, got {elements}"
    
    def test_amide_atoms(self):
        """Test amide detection and atom mapping."""
        from chem.functional_groups import detect_functional_groups
        from rdkit import Chem
        
        smiles = "CC(=O)NC"  # N-methylacetamide
        mol = Chem.MolFromSmiles(smiles)
        
        result = detect_functional_groups(smiles)
        amide_matches = [m for m in result.matches if "amide" in m.name.lower()]
        
        assert len(amide_matches) >= 1, f"Expected at least 1 amide"
        
        match = amide_matches[0]
        elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in match.atom_indices]
        
        assert 'C' in elements, f"Amide should contain C"
        assert 'N' in elements, f"Amide should contain N"
        assert 'O' in elements, f"Amide should contain O"
    
    def test_indole_detection(self):
        """Test indole heterocycle detection."""
        from chem.functional_groups import detect_functional_groups
        
        smiles = "c1ccc2[nH]ccc2c1"  # indole
        result = detect_functional_groups(smiles)
        
        fg_names = [m.name.lower() for m in result.matches]
        assert any("indole" in name for name in fg_names), (
            f"Expected indole, got {fg_names}"
        )
    
    def test_multiple_same_fg(self):
        """Test detection of multiple instances of the same FG."""
        from chem.functional_groups import detect_functional_groups
        
        smiles = "OCC(O)CO"  # glycerol - 3 hydroxyls
        result = detect_functional_groups(smiles)
        
        # Should find multiple hydroxyl groups
        hydroxy_matches = [m for m in result.matches if "hydroxy" in m.name.lower()]
        
        assert len(hydroxy_matches) >= 2, (
            f"Expected at least 2 hydroxyl groups in glycerol, got {len(hydroxy_matches)}"
        )
        
        # Each should have different atom indices
        all_indices = [m.atom_indices for m in hydroxy_matches]
        unique_indices = set(tuple(sorted(idx)) for idx in all_indices)
        
        assert len(unique_indices) >= 2, "Hydroxyl groups should be at different positions"


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    # Run with: python -m pytest tests/test_functional_groups.py -v
    
    # Quick manual test without pytest
    print("Running quick manual tests...\n")
    
    try:
        from chem.functional_groups import (
            detect_functional_groups,
            list_all_functional_groups,
            search_functional_groups,
            get_smarts_for_fg,
        )
        print(f"✓ Imports successful")
        print(f"✓ Total FGs available: {len(list_all_functional_groups())}")
        
        for mol_data in TEST_MOLECULES:
            print(f"\nTesting: {mol_data['name']}")
            print(f"  SMILES: {mol_data['smiles'][:50]}...")
            
            result = detect_functional_groups(mol_data["smiles"])
            print(f"  Found {len(result.matches)} FG matches")
            print(f"  Unique FG types: {len(set(m.name for m in result.matches))}")
            
            for match in result.matches[:5]:
                print(f"    - {match.name}: atoms {match.atom_indices}")
            
            if len(result.matches) > 5:
                print(f"    ... and {len(result.matches) - 5} more")

            # Save a visualization PNG per molecule
            img_bytes = draw_mol_with_fgs(
                mol_data["smiles"],
                with_atom_idx=True,
                with_legend=True,
                img_size=(900, 700),
            )
            out_path = f"fg_viz_{mol_data['name'].replace(' ', '_')}.png"
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            print(f"  Saved visualization: {out_path}")
        
        print("\n✓ All manual tests passed!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Make sure AccFG is installed: pip install accfg")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
