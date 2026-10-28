# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
chem/build/tests/molgraph_structure.py

Demonstration of SMILES → MolGraph conversion.
"""

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import sys
import logging

from utils.logger import get_logger
from chem.build.create_molgraph import smiles_to_molgraph
from chem.build.molgraph_to_mol import molgraph_to_mol, molgraph_to_smiles
from chem.build.build_utils import build_edge_index, build_bond_pair_to_index

# Get logger - set to DEBUG to see all debug messages, INFO to hide them
logger = get_logger(__name__, level=logging.INFO)


def print_molgraph(mg, label: str):
    """Pretty-print a MolGraph object."""
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"SMILES: {mg.meta.get('smiles')}")
    print()

    arr = mg.arrays
    n_atoms = len(arr.atomic_num)
    n_bonds = arr.bonds.shape[0]

    print(f"[MolArrays] {n_atoms} atoms, {n_bonds} bonds")
    print()

    # --- Atoms ---
    print("ATOMS:")
    print("-" * 50)
    print(f"  {'idx':<4} {'Z':<4} {'chg':<4} {'arom':<5} {'hyb':<4} {'chir':<5}")
    print("-" * 50)
    for i in range(n_atoms):
        z = arr.atomic_num[i]
        chg = arr.formal_charge[i]
        arom = arr.is_aromatic[i] if arr.is_aromatic is not None else False
        hyb = arr.hybridization[i] if arr.hybridization is not None else 0
        chir = arr.chiral_tag[i] if arr.chiral_tag is not None else 0
        print(f"  {i:<4} {z:<4} {chg:<4} {str(arom):<5} {hyb:<4} {chir:<5}")
    print()

    # --- Bonds ---
    print("BONDS:")
    print("-" * 60)
    print(f"  {'idx':<4} {'u-v':<8} {'type':<5} {'conj':<5} {'ring':<5} {'stereo':<6} {'res':<4}")
    print("-" * 60)
    for i in range(n_bonds):
        u, v = arr.bonds[i]
        bt = arr.bond_type[i]
        conj = arr.is_conjugated[i] if arr.is_conjugated is not None else False
        ring = arr.is_in_ring[i] if arr.is_in_ring is not None else False
        stereo = arr.bond_stereo[i] if arr.bond_stereo is not None else 0
        res = arr.bond_resonance_type[i] if arr.bond_resonance_type is not None else 0
        print(f"  {i:<4} {u}-{v:<5} {bt:<5} {str(conj):<5} {str(ring):<5} {stereo:<6} {res:<4}")
    print()

    # --- Attachment Points ---
    if mg.attachments is not None:
        print("ATTACHMENTS:")
        print("-" * 40)
        att = mg.attachments
        for i in range(len(att.idx)):
            idx = att.idx[i]
            kind = att.kind[i]
            target = att.target[i]
            label = att.label_id[i] if att.label_id is not None else None
            print(f"  idx={idx}, kind={kind}, target={target}, label={label}")
        print()

    # --- Partial Charges ---
    if arr.partial_charge is not None:
        print("PARTIAL CHARGES:")
        print(f"  {arr.partial_charge}")
        print()

    # --- Globals ---
    print(f"TOTAL CHARGE: {arr.total_charge}")
    print()

    # --- Derived (edge_index for PyG) ---
    logger.debug("Building edge_index...")
    edge_index = build_edge_index(arr)
    print(f"EDGE INDEX (directed, shape {edge_index.shape}):")
    print(f"  {edge_index}")
    print()


def main():
    logger.debug("main() called")
    
    # Test molecule 1: E/Z stereochemistry with dummy atom
    smiles1 = r"[*:1]C/C=C\C"
    logger.debug(f"Processing molecule 1: {smiles1}")
    mg1 = smiles_to_molgraph(smiles1, keep_rdkit_mol=True)

    if mg1:
        logger.debug("Molecule 1 converted successfully")
        print_molgraph(mg1, "Molecule 1: E/Z isomer with dummy")
    else:
        print(f"Failed to parse: {smiles1}")

    # Test molecule 2: Chiral center with dummy atom
    smiles2 = r"[*:1][C@H](Br)C(=O)O"
    logger.debug(f"Processing molecule 2: {smiles2}")
    mg2 = smiles_to_molgraph(smiles2, keep_rdkit_mol=True)

    if mg2:
        logger.debug("Molecule 2 converted successfully")
        print_molgraph(mg2, "Molecule 2: Chiral center with dummy")
    else:
        print(f"Failed to parse: {smiles2}")

    # Print legend for integer codes
    print("=" * 70)
    print("  LEGEND (Integer Codes)")
    print("=" * 70)
    print("""
HYBRIDIZATION:  0=OTHER, 1=SP, 2=SP2, 3=SP3, 4=SP3D, 5=SP3D2
CHIRAL_TAG:     0=UNSPECIFIED, 1=CW, 2=CCW, 3=OTHER
BOND_TYPE:      0=SINGLE, 1=DOUBLE, 2=TRIPLE, 3=AROMATIC
BOND_STEREO:    0=NONE, 1=E, 2=Z, 3=CIS, 4=TRANS, 5=UP, 6=DOWN, 7=UNKNOWN
RESONANCE:      0=NONE, 1=LOCALIZED, 2=DELOCALIZED, 3=AROMATIC
ATTACH_KIND:    0=DUMMY, 1=H_SUB, 2=OPEN_VALENCE
""")

    # Test round-trip conversion
    print("=" * 70)
    print("  ROUND-TRIP TEST (SMILES → MolGraph → SMILES)")
    print("=" * 70)
    print()
    
    test_smiles = [
        r"[*:1]C/C=C\C",           # E/Z with dummy
        r"[*:1][C@H](Br)C(=O)O",   # Chiral with dummy
        "c1ccccc1",                 # Benzene (aromatic)
        "CC(=O)O",                  # Acetic acid
        "C1CCCCC1",                 # Cyclohexane
    ]
    
    for smiles in test_smiles:
        logger.debug(f"Testing round-trip for: {smiles}")
        mg = smiles_to_molgraph(smiles)
        if mg is None:
            print(f"  {smiles:<30} → FAILED (parse error)")
            continue
            
        recovered = molgraph_to_smiles(mg)
        if recovered is None:
            print(f"  {smiles:<30} → FAILED (conversion error)")
            continue
            
        print(f"  {smiles:<30} → {recovered}")
    
    print()
    logger.debug("main() complete")


if __name__ == "__main__":
    logger.debug("__name__ == '__main__'")
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main(): {e}")
        sys.exit(1)
    
    logger.debug("Script finished")
