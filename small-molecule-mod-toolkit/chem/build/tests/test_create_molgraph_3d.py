# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/build/tests/test_create_molgraph_3d.py

Comprehensive tests for 3D coordinate generation in create_molgraph.py

Tests cover:
- All three coordinate modes (None, canonical, ensemble)
- Shape validation for pos, coord_valid
- Metadata tracking (coord_frame)
- Optimization options
- Edge cases and failure handling
- Integration with other options (add_hs, compute_charges)
- Round-trip consistency
"""

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import sys
import logging
import numpy as np
from rdkit import Chem

from utils.logger import get_logger
from chem.build.create_molgraph import (
    smiles_to_molgraph,
    mol_to_molgraph,
    generate_coordinates,
)

# Get logger
logger = get_logger(__name__, level=logging.INFO)


class TestResult:
    """Simple test result tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record_pass(self, test_name: str):
        self.passed += 1
        print(f"✓ PASS: {test_name}")
    
    def record_fail(self, test_name: str, reason: str):
        self.failed += 1
        error_msg = f"✗ FAIL: {test_name} - {reason}"
        self.errors.append(error_msg)
        print(error_msg)
    
    def summary(self):
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        if self.errors:
            print("\nFailed tests:")
            for error in self.errors:
                print(f"  {error}")
        print("=" * 70)
        return self.failed == 0


results = TestResult()


def assert_equal(actual, expected, test_name: str):
    """Assert equality and record result."""
    if actual == expected:
        results.record_pass(test_name)
        return True
    else:
        results.record_fail(test_name, f"Expected {expected}, got {actual}")
        return False


def assert_true(condition, test_name: str, reason: str = ""):
    """Assert condition is True and record result."""
    if condition:
        results.record_pass(test_name)
        return True
    else:
        results.record_fail(test_name, reason or "Condition was False")
        return False


def assert_shape(array, expected_shape, test_name: str):
    """Assert array shape and record result."""
    if array is None:
        results.record_fail(test_name, f"Array is None, expected shape {expected_shape}")
        return False
    if array.shape == expected_shape:
        results.record_pass(test_name)
        return True
    else:
        results.record_fail(test_name, f"Expected shape {expected_shape}, got {array.shape}")
        return False


def assert_not_none(obj, test_name: str):
    """Assert object is not None and record result."""
    if obj is not None:
        results.record_pass(test_name)
        return True
    else:
        results.record_fail(test_name, "Object is None")
        return False


def assert_is_none(obj, test_name: str):
    """Assert object is None and record result."""
    if obj is None:
        results.record_pass(test_name)
        return True
    else:
        results.record_fail(test_name, f"Object is not None: {obj}")
        return False


# =============================================================================
# Test Suite 1: Basic Coordinate Mode Tests
# =============================================================================

def test_no_coords_default():
    """Test default behavior - no coordinate generation."""
    print("\n" + "=" * 70)
    print("TEST SUITE 1: Basic Coordinate Mode Tests")
    print("=" * 70)
    
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles)
    
    assert_not_none(mg, "test_no_coords_default: MolGraph created")
    assert_is_none(mg.arrays.pos, "test_no_coords_default: pos is None")
    assert_is_none(mg.arrays.coord_frame, "test_no_coords_default: coord_frame is None")
    assert_is_none(mg.arrays.coord_valid, "test_no_coords_default: coord_valid is None")


def test_canonical_coords():
    """Test canonical mode - single conformer."""
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg, "test_canonical_coords: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_canonical_coords: pos is not None")
    
    n_atoms = len(mg.arrays.atomic_num)
    assert_shape(mg.arrays.pos, (n_atoms, 3), "test_canonical_coords: pos shape is (N, 3)")
    assert_equal(mg.arrays.coord_frame, "etkdg", "test_canonical_coords: coord_frame is 'etkdg'")
    assert_shape(mg.arrays.coord_valid, (n_atoms,), "test_canonical_coords: coord_valid shape is (N,)")
    assert_true(
        mg.arrays.coord_valid.all(),
        "test_canonical_coords: all coordinates initially valid"
    )


def test_ensemble_coords():
    """Test ensemble mode - multiple conformers."""
    smiles = "CCO"
    num_confs = 5
    mg = smiles_to_molgraph(smiles, coords="ensemble", num_confs=num_confs)
    
    assert_not_none(mg, "test_ensemble_coords: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_ensemble_coords: pos is not None")
    
    n_atoms = len(mg.arrays.atomic_num)
    actual_confs = mg.arrays.pos.shape[0]
    
    # Note: RDKit may generate fewer conformers than requested
    assert_true(
        actual_confs <= num_confs,
        "test_ensemble_coords: number of conformers <= requested",
        f"Got {actual_confs} conformers, requested {num_confs}"
    )
    assert_true(
        actual_confs > 0,
        "test_ensemble_coords: at least one conformer generated"
    )
    assert_shape(
        mg.arrays.pos,
        (actual_confs, n_atoms, 3),
        "test_ensemble_coords: pos shape is (K, N, 3)"
    )
    assert_equal(
        mg.arrays.coord_frame,
        "etkdg",
        "test_ensemble_coords: coord_frame is 'etkdg'"
    )
    assert_shape(
        mg.arrays.coord_valid,
        (actual_confs, n_atoms),
        "test_ensemble_coords: coord_valid shape is (K, N)"
    )
    assert_true(
        mg.arrays.coord_valid.all(),
        "test_ensemble_coords: all coordinates initially valid"
    )


# =============================================================================
# Test Suite 2: Optimization Tests
# =============================================================================

def test_canonical_with_optimization():
    """Test canonical mode with MMFF optimization."""
    print("\n" + "=" * 70)
    print("TEST SUITE 2: Optimization Tests")
    print("=" * 70)
    
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, coords="canonical", optimize_coords=True)
    
    assert_not_none(mg, "test_canonical_with_optimization: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_canonical_with_optimization: pos is not None")
    assert_equal(
        mg.arrays.coord_frame,
        "etkdg-mmff",
        "test_canonical_with_optimization: coord_frame is 'etkdg-mmff'"
    )


def test_ensemble_with_optimization():
    """Test ensemble mode with MMFF optimization."""
    smiles = "CCO"
    mg = smiles_to_molgraph(
        smiles,
        coords="ensemble",
        num_confs=3,
        optimize_coords=True
    )
    
    assert_not_none(mg, "test_ensemble_with_optimization: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_ensemble_with_optimization: pos is not None")
    assert_equal(
        mg.arrays.coord_frame,
        "etkdg-mmff",
        "test_ensemble_with_optimization: coord_frame is 'etkdg-mmff'"
    )


# =============================================================================
# Test Suite 3: Shape and Data Type Tests
# =============================================================================

def test_coordinate_dtype():
    """Test that coordinates are float32."""
    print("\n" + "=" * 70)
    print("TEST SUITE 3: Shape and Data Type Tests")
    print("=" * 70)
    
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg.arrays.pos, "test_coordinate_dtype: pos is not None")
    assert_equal(
        mg.arrays.pos.dtype,
        np.float32,
        "test_coordinate_dtype: pos dtype is float32"
    )


def test_coord_valid_dtype():
    """Test that coord_valid is boolean."""
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg.arrays.coord_valid, "test_coord_valid_dtype: coord_valid is not None")
    assert_equal(
        mg.arrays.coord_valid.dtype,
        bool,
        "test_coord_valid_dtype: coord_valid dtype is bool"
    )


def test_coordinate_values_reasonable():
    """Test that coordinate values are reasonable (not NaN, not inf)."""
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg.arrays.pos, "test_coordinate_values_reasonable: pos is not None")
    assert_true(
        np.isfinite(mg.arrays.pos).all(),
        "test_coordinate_values_reasonable: all coordinates are finite"
    )
    assert_true(
        not np.isnan(mg.arrays.pos).any(),
        "test_coordinate_values_reasonable: no NaN coordinates"
    )


def test_coordinate_variance():
    """Test that coordinates have reasonable variance (not all zeros)."""
    smiles = "CCCC"  # Longer chain should have more variance
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg.arrays.pos, "test_coordinate_variance: pos is not None")
    variance = np.var(mg.arrays.pos)
    assert_true(
        variance > 0.01,
        "test_coordinate_variance: coordinates have non-trivial variance",
        f"Variance was {variance}"
    )


# =============================================================================
# Test Suite 4: Integration with Other Options
# =============================================================================

def test_coords_with_explicit_hydrogens():
    """Test coordinate generation with explicit hydrogens."""
    print("\n" + "=" * 70)
    print("TEST SUITE 4: Integration with Other Options")
    print("=" * 70)
    
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, add_hs=True, coords="canonical")
    
    assert_not_none(mg, "test_coords_with_explicit_hydrogens: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_coords_with_explicit_hydrogens: pos is not None")
    
    n_atoms = len(mg.arrays.atomic_num)
    # CCO with explicit H should have 9 atoms (C-C-O + 6 H)
    assert_true(
        n_atoms == 9,
        "test_coords_with_explicit_hydrogens: correct number of atoms with H",
        f"Expected 9 atoms, got {n_atoms}"
    )
    assert_shape(
        mg.arrays.pos,
        (n_atoms, 3),
        "test_coords_with_explicit_hydrogens: pos shape matches atom count"
    )


def test_coords_with_charge_computation():
    """Test coordinate generation with charge computation."""
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, compute_charges=True, coords="canonical")
    
    assert_not_none(mg, "test_coords_with_charge_computation: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_coords_with_charge_computation: pos is not None")
    assert_not_none(
        mg.arrays.partial_charge,
        "test_coords_with_charge_computation: partial charges computed"
    )


def test_coords_with_keep_rdkit_mol():
    """Test coordinate generation with keep_rdkit_mol=True."""
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, keep_rdkit_mol=True, coords="canonical")
    
    assert_not_none(mg, "test_coords_with_keep_rdkit_mol: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_coords_with_keep_rdkit_mol: pos is not None")
    assert_not_none(mg.rdkit_mol, "test_coords_with_keep_rdkit_mol: rdkit_mol is stored")
    
    # Check that the RDKit mol has conformers
    assert_true(
        mg.rdkit_mol.GetNumConformers() > 0,
        "test_coords_with_keep_rdkit_mol: RDKit mol has conformers"
    )


# =============================================================================
# Test Suite 5: Reproducibility Tests
# =============================================================================

def test_random_seed_reproducibility():
    """Test that random_seed makes generation reproducible."""
    print("\n" + "=" * 70)
    print("TEST SUITE 5: Reproducibility Tests")
    print("=" * 70)
    
    smiles = "CCCC"
    seed = 42
    
    mg1 = smiles_to_molgraph(smiles, coords="canonical", random_seed=seed)
    mg2 = smiles_to_molgraph(smiles, coords="canonical", random_seed=seed)
    
    assert_not_none(mg1.arrays.pos, "test_random_seed_reproducibility: pos1 is not None")
    assert_not_none(mg2.arrays.pos, "test_random_seed_reproducibility: pos2 is not None")
    
    assert_true(
        np.allclose(mg1.arrays.pos, mg2.arrays.pos),
        "test_random_seed_reproducibility: same seed gives same coordinates"
    )


def test_different_seeds_give_different_coords():
    """Test that different seeds give different coordinates."""
    smiles = "CCCC"
    
    mg1 = smiles_to_molgraph(smiles, coords="canonical", random_seed=42)
    mg2 = smiles_to_molgraph(smiles, coords="canonical", random_seed=123)
    
    assert_not_none(mg1.arrays.pos, "test_different_seeds: pos1 is not None")
    assert_not_none(mg2.arrays.pos, "test_different_seeds: pos2 is not None")
    
    assert_true(
        not np.allclose(mg1.arrays.pos, mg2.arrays.pos),
        "test_different_seeds: different seeds give different coordinates"
    )


# =============================================================================
# Test Suite 6: Edge Cases and Failure Handling
# =============================================================================

def test_single_atom_molecule():
    """Test coordinate generation for single atom."""
    print("\n" + "=" * 70)
    print("TEST SUITE 6: Edge Cases and Failure Handling")
    print("=" * 70)
    
    smiles = "C"
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg, "test_single_atom_molecule: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_single_atom_molecule: pos is not None")
    assert_shape(
        mg.arrays.pos,
        (1, 3),
        "test_single_atom_molecule: single atom has (1, 3) shape"
    )


def test_cyclic_molecule():
    """Test coordinate generation for cyclic molecules."""
    smiles = "C1CCCCC1"  # Cyclohexane
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg, "test_cyclic_molecule: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_cyclic_molecule: pos is not None")
    assert_shape(
        mg.arrays.pos,
        (6, 3),
        "test_cyclic_molecule: cyclohexane has (6, 3) shape"
    )


def test_aromatic_molecule():
    """Test coordinate generation for aromatic molecules."""
    smiles = "c1ccccc1"  # Benzene
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg, "test_aromatic_molecule: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_aromatic_molecule: pos is not None")
    assert_shape(
        mg.arrays.pos,
        (6, 3),
        "test_aromatic_molecule: benzene has (6, 3) shape"
    )


def test_complex_molecule():
    """Test coordinate generation for complex molecules."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg, "test_complex_molecule: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_complex_molecule: pos is not None")
    
    n_atoms = len(mg.arrays.atomic_num)
    assert_shape(
        mg.arrays.pos,
        (n_atoms, 3),
        "test_complex_molecule: aspirin has correct shape"
    )


def test_strained_molecule():
    """Test coordinate generation for strained molecules (may fail)."""
    smiles = "C1CC1C1CC1"  # Two fused cyclopropanes
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    # This may or may not generate coordinates depending on RDKit version
    # Just check that it doesn't crash
    assert_not_none(mg, "test_strained_molecule: MolGraph created (doesn't crash)")
    
    if mg.arrays.pos is not None:
        print("  Note: Strained molecule successfully generated coordinates")
    else:
        print("  Note: Strained molecule failed to generate coordinates (expected)")


def test_invalid_smiles():
    """Test that invalid SMILES returns None gracefully."""
    smiles = "INVALID_SMILES_123"
    mg = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_is_none(mg, "test_invalid_smiles: invalid SMILES returns None")


# =============================================================================
# Test Suite 7: mol_to_molgraph Direct Tests
# =============================================================================

def test_mol_to_molgraph_canonical():
    """Test mol_to_molgraph with canonical coordinates."""
    print("\n" + "=" * 70)
    print("TEST SUITE 7: mol_to_molgraph Direct Tests")
    print("=" * 70)
    
    mol = Chem.MolFromSmiles("CCO")
    mg = mol_to_molgraph(mol, coords="canonical")
    
    assert_not_none(mg, "test_mol_to_molgraph_canonical: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_mol_to_molgraph_canonical: pos is not None")
    assert_shape(
        mg.arrays.pos,
        (3, 3),
        "test_mol_to_molgraph_canonical: pos has correct shape"
    )


def test_mol_to_molgraph_ensemble():
    """Test mol_to_molgraph with ensemble coordinates."""
    mol = Chem.MolFromSmiles("CCO")
    mg = mol_to_molgraph(mol, coords="ensemble", num_confs=3)
    
    assert_not_none(mg, "test_mol_to_molgraph_ensemble: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_mol_to_molgraph_ensemble: pos is not None")
    
    # Check 3D shape
    assert_true(
        mg.arrays.pos.ndim == 3,
        "test_mol_to_molgraph_ensemble: pos is 3D array"
    )


def test_mol_to_molgraph_preserves_input():
    """Test that mol_to_molgraph doesn't modify the input molecule."""
    mol = Chem.MolFromSmiles("CCO")
    original_num_confs = mol.GetNumConformers()
    
    mg = mol_to_molgraph(mol, coords="canonical")
    
    # Input molecule should not have conformers added (we make a copy)
    assert_equal(
        mol.GetNumConformers(),
        original_num_confs,
        "test_mol_to_molgraph_preserves_input: input mol unchanged"
    )


# =============================================================================
# Test Suite 8: Metadata Tests
# =============================================================================

def test_metadata_storage():
    """Test that metadata is properly stored."""
    print("\n" + "=" * 70)
    print("TEST SUITE 8: Metadata Tests")
    print("=" * 70)
    
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, coords="ensemble", num_confs=5)
    
    assert_true(
        "coords_mode" in mg.meta,
        "test_metadata_storage: coords_mode in metadata"
    )
    assert_equal(
        mg.meta.get("coords_mode"),
        "ensemble",
        "test_metadata_storage: coords_mode is 'ensemble'"
    )
    assert_true(
        "num_confs" in mg.meta,
        "test_metadata_storage: num_confs in metadata"
    )


def test_metadata_not_stored_when_no_coords():
    """Test that coordinate metadata is not stored when coords=None."""
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles)
    
    assert_true(
        "coords_mode" not in mg.meta,
        "test_metadata_not_stored_when_no_coords: no coords_mode when coords=None"
    )


# =============================================================================
# Test Suite 9: Ensemble Conformer Count Tests
# =============================================================================

def test_ensemble_requested_vs_actual():
    """Test that ensemble generates reasonable number of conformers."""
    print("\n" + "=" * 70)
    print("TEST SUITE 9: Ensemble Conformer Count Tests")
    print("=" * 70)
    
    smiles = "CCCCCCCC"  # Longer chain allows more conformers
    num_confs = 20
    mg = smiles_to_molgraph(smiles, coords="ensemble", num_confs=num_confs)
    
    assert_not_none(mg.arrays.pos, "test_ensemble_requested_vs_actual: pos is not None")
    
    actual_confs = mg.arrays.pos.shape[0]
    print(f"  Requested {num_confs} conformers, got {actual_confs}")
    
    # RDKit may generate fewer conformers than requested
    assert_true(
        actual_confs > 0,
        "test_ensemble_requested_vs_actual: at least one conformer generated"
    )


def test_ensemble_minimum_conformers():
    """Test ensemble with num_confs=1."""
    smiles = "CCO"
    mg = smiles_to_molgraph(smiles, coords="ensemble", num_confs=1)
    
    assert_not_none(mg.arrays.pos, "test_ensemble_minimum_conformers: pos is not None")
    
    # Even with num_confs=1, shape should be (K, N, 3)
    assert_true(
        mg.arrays.pos.ndim == 3,
        "test_ensemble_minimum_conformers: pos is 3D even with 1 conformer"
    )


# =============================================================================
# Test Suite 10: generate_coordinates Function Tests
# =============================================================================

def test_generate_coordinates_canonical():
    """Test generate_coordinates function directly - canonical."""
    print("\n" + "=" * 70)
    print("TEST SUITE 10: generate_coordinates Function Tests")
    print("=" * 70)
    
    mol = Chem.MolFromSmiles("CCO")
    pos, coord_frame, coord_valid = generate_coordinates(mol, mode="canonical")
    
    assert_not_none(pos, "test_generate_coordinates_canonical: pos is not None")
    assert_equal(
        coord_frame,
        "etkdg",
        "test_generate_coordinates_canonical: coord_frame is 'etkdg'"
    )
    assert_not_none(
        coord_valid,
        "test_generate_coordinates_canonical: coord_valid is not None"
    )


def test_generate_coordinates_ensemble():
    """Test generate_coordinates function directly - ensemble."""
    mol = Chem.MolFromSmiles("CCO")
    pos, coord_frame, coord_valid = generate_coordinates(
        mol,
        mode="ensemble",
        num_confs=3
    )
    
    assert_not_none(pos, "test_generate_coordinates_ensemble: pos is not None")
    assert_equal(
        coord_frame,
        "etkdg",
        "test_generate_coordinates_ensemble: coord_frame is 'etkdg'"
    )
    assert_true(
        pos.ndim == 3,
        "test_generate_coordinates_ensemble: pos is 3D"
    )


def test_generate_coordinates_with_optimization():
    """Test generate_coordinates with optimization."""
    mol = Chem.MolFromSmiles("CCO")
    pos, coord_frame, coord_valid = generate_coordinates(
        mol,
        mode="canonical",
        optimize=True
    )
    
    assert_not_none(pos, "test_generate_coordinates_with_optimization: pos is not None")
    assert_equal(
        coord_frame,
        "etkdg-mmff",
        "test_generate_coordinates_with_optimization: coord_frame is 'etkdg-mmff'"
    )


def test_generate_coordinates_empty_mol():
    """Test generate_coordinates with empty molecule."""
    mol = Chem.MolFromSmiles("")  # Empty molecule
    if mol is not None and mol.GetNumAtoms() == 0:
        pos, coord_frame, coord_valid = generate_coordinates(mol, mode="canonical")
        assert_is_none(pos, "test_generate_coordinates_empty_mol: pos is None for empty mol")
    else:
        print("  Note: Could not create empty molecule, skipping test")


# =============================================================================
# Test Suite 11: Coordinate Consistency Tests
# =============================================================================

def test_ensemble_conformers_different():
    """Test that ensemble conformers are different from each other."""
    print("\n" + "=" * 70)
    print("TEST SUITE 11: Coordinate Consistency Tests")
    print("=" * 70)
    
    smiles = "CCCCCC"  # Longer chain for more conformational freedom
    mg = smiles_to_molgraph(smiles, coords="ensemble", num_confs=3)
    
    assert_not_none(mg.arrays.pos, "test_ensemble_conformers_different: pos is not None")
    
    if mg.arrays.pos.shape[0] >= 2:
        # Check that first two conformers are different
        conf1 = mg.arrays.pos[0]
        conf2 = mg.arrays.pos[1]
        
        assert_true(
            not np.allclose(conf1, conf2),
            "test_ensemble_conformers_different: conformers are different"
        )
    else:
        print("  Note: Only one conformer generated, skipping difference test")


def test_canonical_single_call_consistency():
    """Test that calling canonical twice gives different results (no seed)."""
    smiles = "CCCC"
    
    # Don't specify random_seed, should be different
    mg1 = smiles_to_molgraph(smiles, coords="canonical")
    mg2 = smiles_to_molgraph(smiles, coords="canonical")
    
    assert_not_none(mg1.arrays.pos, "test_canonical_consistency: pos1 is not None")
    assert_not_none(mg2.arrays.pos, "test_canonical_consistency: pos2 is not None")
    
    # Without specifying seed, results may vary
    # (This is more of a documentation test than assertion)
    print(f"  Note: Without seed, coords vary: {not np.allclose(mg1.arrays.pos, mg2.arrays.pos)}")


# =============================================================================
# Test Suite 12: Integration Tests
# =============================================================================

def test_full_pipeline_canonical():
    """Test full pipeline with canonical coordinates."""
    print("\n" + "=" * 70)
    print("TEST SUITE 12: Integration Tests")
    print("=" * 70)
    
    smiles = "CC(=O)O"  # Acetic acid
    mg = smiles_to_molgraph(
        smiles,
        add_hs=False,
        compute_charges=True,
        keep_rdkit_mol=False,
        coords="canonical",
        optimize_coords=False,
        random_seed=42,
    )
    
    assert_not_none(mg, "test_full_pipeline_canonical: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_full_pipeline_canonical: has coordinates")
    assert_not_none(mg.arrays.partial_charge, "test_full_pipeline_canonical: has charges")
    assert_is_none(mg.rdkit_mol, "test_full_pipeline_canonical: no rdkit_mol stored")


def test_full_pipeline_ensemble():
    """Test full pipeline with ensemble coordinates."""
    smiles = "CC(=O)O"
    mg = smiles_to_molgraph(
        smiles,
        add_hs=True,
        compute_charges=True,
        keep_rdkit_mol=True,
        coords="ensemble",
        num_confs=5,
        optimize_coords=True,
        random_seed=123,
    )
    
    assert_not_none(mg, "test_full_pipeline_ensemble: MolGraph created")
    assert_not_none(mg.arrays.pos, "test_full_pipeline_ensemble: has coordinates")
    assert_true(
        mg.arrays.pos.ndim == 3,
        "test_full_pipeline_ensemble: has ensemble coordinates"
    )
    assert_not_none(mg.arrays.partial_charge, "test_full_pipeline_ensemble: has charges")
    assert_not_none(mg.rdkit_mol, "test_full_pipeline_ensemble: has rdkit_mol")


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests."""
    logger.info("Starting comprehensive 3D coordinate generation tests...")
    
    # Suite 1: Basic coordinate modes
    test_no_coords_default()
    test_canonical_coords()
    test_ensemble_coords()
    
    # Suite 2: Optimization
    test_canonical_with_optimization()
    test_ensemble_with_optimization()
    
    # Suite 3: Shape and data types
    test_coordinate_dtype()
    test_coord_valid_dtype()
    test_coordinate_values_reasonable()
    test_coordinate_variance()
    
    # Suite 4: Integration with other options
    test_coords_with_explicit_hydrogens()
    test_coords_with_charge_computation()
    test_coords_with_keep_rdkit_mol()
    
    # Suite 5: Reproducibility
    test_random_seed_reproducibility()
    test_different_seeds_give_different_coords()
    
    # Suite 6: Edge cases
    test_single_atom_molecule()
    test_cyclic_molecule()
    test_aromatic_molecule()
    test_complex_molecule()
    test_strained_molecule()
    test_invalid_smiles()
    
    # Suite 7: mol_to_molgraph
    test_mol_to_molgraph_canonical()
    test_mol_to_molgraph_ensemble()
    test_mol_to_molgraph_preserves_input()
    
    # Suite 8: Metadata
    test_metadata_storage()
    test_metadata_not_stored_when_no_coords()
    
    # Suite 9: Ensemble conformer counts
    test_ensemble_requested_vs_actual()
    test_ensemble_minimum_conformers()
    
    # Suite 10: generate_coordinates function
    test_generate_coordinates_canonical()
    test_generate_coordinates_ensemble()
    test_generate_coordinates_with_optimization()
    test_generate_coordinates_empty_mol()
    
    # Suite 11: Coordinate consistency
    test_ensemble_conformers_different()
    test_canonical_single_call_consistency()
    
    # Suite 12: Integration tests
    test_full_pipeline_canonical()
    test_full_pipeline_ensemble()
    
    # Print summary
    success = results.summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.exception(f"Fatal error in test suite: {e}")
        sys.exit(1)
