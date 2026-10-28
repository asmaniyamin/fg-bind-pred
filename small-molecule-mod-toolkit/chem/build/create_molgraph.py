# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/build/create_molgraph.py

Minimal RDKit → MolGraph (structs.py) conversion.

Provides two entry points:
- mol_to_molgraph(mol) - convert RDKit Mol to MolGraph
- smiles_to_molgraph(smiles) - convenience wrapper for SMILES strings
"""

from __future__ import annotations

from typing import Optional, Literal
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from core.structs import MolGraph, MolArrays
from chem.build.build_utils import (
    extract_atom_arrays,
    extract_bond_arrays,
    extract_attachment_points,
    prepare_molecule,
)
from utils.logger import get_logger

logger = get_logger(__name__)
logger.debug("Loading create_molgraph.py...")


def generate_coordinates(
    mol: Chem.Mol,
    mode: Literal["canonical", "ensemble"] = "canonical",
    num_confs: int = 10,
    random_seed: int = 42,
    optimize: bool = False,
    max_iterations: int = 5,
) -> tuple[Optional[np.ndarray], Optional[str], Optional[np.ndarray]]:
    """
    Generate 3D coordinates for a molecule using RDKit ETKDG.
    
    Args:
        mol: RDKit molecule object (will be modified with conformers)
        mode: "canonical" for single conformer (N,3) or "ensemble" for multiple (K,N,3)
        num_confs: Number of conformers to generate (only used for ensemble mode)
        random_seed: Random seed for reproducibility
        optimize: Whether to run MMFF optimization after embedding
        max_iterations: Maximum embedding attempts per conformer
        
    Returns:
        Tuple of (positions, coord_frame, coord_valid):
            - positions: numpy array of shape (N,3) or (K,N,3), or None if generation fails
            - coord_frame: string identifier (e.g., "etkdg", "etkdg-mmff")
            - coord_valid: boolean mask of shape (N,) or (K,N), all True initially
    """
    logger.debug(f"Generating coordinates in {mode} mode...")
    
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        logger.warning("Cannot generate coordinates for molecule with 0 atoms")
        return None, None, None
    
    # Set up ETKDG parameters
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.maxIterations = max_iterations
    params.useRandomCoords = True

    try:
        if mode == "canonical":
            # Generate single conformer
            logger.debug("Embedding single conformer...")
            conf_id = AllChem.EmbedMolecule(mol, params)
            
            if conf_id == -1:
                logger.error("Failed to embed conformer")
                return None, None, None
            
            # Optional optimization
            coord_frame = "etkdg"
            if optimize:
                logger.debug("Optimizing conformer with MMFF...")
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                    if ff is not None:
                        ff.Minimize()
                        coord_frame = "etkdg-mmff"
                    else:
                        logger.warning("Failed to create force field, skipping optimization")
                else:
                    logger.warning("Failed to get MMFF properties, skipping optimization")
            
            # Extract positions
            conf = mol.GetConformer(conf_id)
            positions = np.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(num_atoms)
            ], dtype=np.float32)
            
            # All coordinates valid initially
            coord_valid = np.ones(num_atoms, dtype=bool)
            
            logger.debug(f"Generated canonical conformer with shape {positions.shape}")
            return positions, coord_frame, coord_valid
            
        elif mode == "ensemble":
            # Generate multiple conformers
            logger.debug(f"Embedding {num_confs} conformers...")
            conf_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=num_confs,
                params=params
            )
            
            if len(conf_ids) == 0:
                logger.error("Failed to embed any conformers")
                return None, None, None
            
            logger.debug(f"Successfully embedded {len(conf_ids)} conformers")
            
            # Optional optimization
            coord_frame = "etkdg"
            if optimize:
                logger.debug("Optimizing conformers with MMFF...")
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is not None:
                    for conf_id in conf_ids:
                        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                        if ff is not None:
                            ff.Minimize()
                    coord_frame = "etkdg-mmff"
                else:
                    logger.warning("Failed to get MMFF properties, skipping optimization")
            
            # Extract positions from all conformers
            positions_list = []
            for conf_id in conf_ids:
                conf = mol.GetConformer(conf_id)
                conf_positions = np.array([
                    [conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(num_atoms)
                ], dtype=np.float32)
                positions_list.append(conf_positions)
            
            # Stack into (K, N, 3) array
            positions = np.stack(positions_list, axis=0)
            
            # All coordinates valid initially, shape (K, N)
            coord_valid = np.ones((len(conf_ids), num_atoms), dtype=bool)
            
            logger.debug(f"Generated ensemble with shape {positions.shape}")
            return positions, coord_frame, coord_valid
            
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'canonical' or 'ensemble'")
            
    except Exception as e:
        logger.error(f"Error during coordinate generation: {e}")
        return None, None, None


def mol_to_molgraph(
    mol: Chem.Mol,
    compute_charges: bool = True,
    keep_rdkit_mol: bool = False,
    coords: Optional[Literal["canonical", "ensemble"]] = None,
    num_confs: int = 10,
    optimize_coords: bool = False,
    random_seed: int = 42,
) -> MolGraph:
    """
    Convert an RDKit Mol object to a MolGraph.
    
    Args:
        mol: RDKit molecule object
        compute_charges: Whether to compute Gasteiger partial charges
        keep_rdkit_mol: Whether to store the RDKit mol in the output
        coords: Coordinate generation mode:
            - None: Don't generate coordinates (default)
            - "canonical": Generate single conformer, pos.shape = (N, 3)
            - "ensemble": Generate multiple conformers, pos.shape = (K, N, 3)
        num_confs: Number of conformers for ensemble mode (default: 10)
        optimize_coords: Whether to optimize coordinates with MMFF (default: False)
        random_seed: Random seed for coordinate generation (default: 42)
        
    Returns:
        MolGraph object with optional 3D coordinates
    """
    logger.debug("mol_to_molgraph() called")
    
    # Generate coordinates if requested
    pos = None
    coord_frame = None
    coord_valid = None
    
    if coords is not None:
        logger.debug(f"Generating coordinates in {coords} mode...")
        # Make a copy to avoid modifying the input molecule
        mol_copy = Chem.Mol(mol)
        pos, coord_frame, coord_valid = generate_coordinates(
            mol_copy,
            mode=coords,
            num_confs=num_confs,
            optimize=optimize_coords,
            random_seed=random_seed,
        )
        # Use the molecule with conformers if generation succeeded
        if pos is not None:
            mol = mol_copy
            logger.debug(f"Coordinates generated successfully: shape={pos.shape}, frame={coord_frame}")
        else:
            logger.warning("Coordinate generation failed, proceeding without coordinates")
    
    logger.debug("Extracting atom arrays...")
    atom_data = extract_atom_arrays(mol, compute_charges=compute_charges)
    logger.debug("Atom arrays extracted")

    logger.debug("Extracting bond arrays...")
    bond_data = extract_bond_arrays(mol)
    logger.debug("Bond arrays extracted")

    # Override pos from atom_data if we generated coordinates
    if pos is not None:
        atom_data["pos"] = pos

    logger.debug("Building MolArrays...")
    arrays = MolArrays(
        # Required fields first (positional order mattering)
        atomic_num=atom_data["atomic_num"],
        formal_charge=atom_data["formal_charge"],
        bonds=bond_data["bonds"],
        bond_type=bond_data["bond_type"],
        # Optional atom fields
        isotope=atom_data["isotope"],
        is_aromatic=atom_data["is_aromatic"],
        hybridization=atom_data["hybridization"],
        chiral_tag=atom_data["chiral_tag"],
        atom_map=atom_data["atom_map"],
        attachment_label=atom_data["attachment_label"],
        explicit_h=atom_data["explicit_h"],
        implicit_h=atom_data["implicit_h"],
        partial_charge=atom_data["partial_charge"],
        pos=atom_data["pos"],
        coord_frame=coord_frame,
        coord_valid=coord_valid,
        # Optional bond fields
        is_conjugated=bond_data["is_conjugated"],
        is_in_ring=bond_data["is_in_ring"],
        bond_dir=bond_data["bond_dir"],
        bond_stereo=bond_data["bond_stereo"],
        bond_resonance_type=bond_data["bond_resonance_type"],
        # Globals
        total_charge=Chem.GetFormalCharge(mol),
    )
    logger.debug("MolArrays built")

    logger.debug("Extracting attachment points...")
    attachments = extract_attachment_points(mol)
    logger.debug("Attachment points extracted")

    logger.debug("Building MolGraph...")
    smiles = Chem.MolToSmiles(mol)
    logger.debug(f"SMILES: {smiles}")
    
    result = MolGraph(
        arrays=arrays,
        attachments=attachments,
        rdkit_mol=mol if keep_rdkit_mol else None,
        meta={"smiles": smiles},
    )
    logger.debug("mol_to_molgraph() complete")
    return result


def smiles_to_molgraph(
    smiles: str,
    add_hs: bool = False,
    compute_charges: bool = True,
    keep_rdkit_mol: bool = False,
    coords: Optional[Literal["canonical", "ensemble"]] = None,
    num_confs: int = 10,
    optimize_coords: bool = False,
    random_seed: int = 42,
) -> Optional[MolGraph]:
    """
    Convenience function: SMILES string → MolGraph.
    
    Uses MoleculeCleaner to parse and sanitize the SMILES string, then converts
    to a MolGraph with atom/bond arrays and optional 3D coordinates.
    
    Args:
        smiles: SMILES string
        add_hs: Whether to add explicit hydrogens
        compute_charges: Whether to compute Gasteiger partial charges
        keep_rdkit_mol: Whether to store the RDKit mol in the output
        coords: Coordinate generation mode:
            - None: Don't generate coordinates (default)
            - "canonical": Generate single conformer, pos.shape = (N, 3)
            - "ensemble": Generate multiple conformers, pos.shape = (K, N, 3)
        num_confs: Number of conformers for ensemble mode (default: 10)
        optimize_coords: Whether to optimize coordinates with MMFF (default: False)
        random_seed: Random seed for coordinate generation (default: 42)
        
    Returns:
        MolGraph or None if conversion fails
    """
    logger.debug(f"smiles_to_molgraph() called with: {smiles}")
    
    try:
        mol, repair_log = prepare_molecule(
            smiles,
            add_hs=add_hs,
            compute_charges=compute_charges,
        )
    except ValueError as e:
        logger.error(f"Failed to prepare molecule: {e}")
        return None
    
    logger.debug("Molecule prepared successfully")
    
    # Convert to MolGraph - charges are already on mol as properties,
    # extract_atom_arrays will find them
    result = mol_to_molgraph(
        mol,
        compute_charges=compute_charges,  # Will extract existing charges or compute if needed
        keep_rdkit_mol=keep_rdkit_mol,
        coords=coords,
        num_confs=num_confs,
        optimize_coords=optimize_coords,
        random_seed=random_seed,
    )
    
    # Store repair log and original SMILES in meta
    if result is not None:
        result.meta["input_smiles"] = smiles
        result.meta["repair_log"] = repair_log
        if coords is not None:
            result.meta["coords_mode"] = coords
            result.meta["num_confs"] = num_confs if coords == "ensemble" else 1
    
    logger.debug("smiles_to_molgraph() complete")
    return result


logger.debug("create_molgraph.py loaded successfully")