# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/extract/find_pocket_unprocessed.py

Extract ligand-centered binding pockets from a holo mmCIF structure using
open-source tooling only.

Bond source priority for the unprocessed path:
    1. PDBx/mmCIF ``_chem_comp_bond`` templates when available
    2. OpenBabel bond perception from the ligand's 3D coordinates as fallback

For each detected ligand, the script writes:
    * an SDF file for the ligand with 3D coordinates
    * a JSON file containing ligand atoms, ligand bonds, and nearby pocket
      residues with atom coordinates and residue identities
    * a manifest JSON summarizing all extracted ligands

Example:
    python -m protein.extract.find_pocket_unprocessed input_holo.cif --output-dir pockets
"""

from __future__ import annotations

import argparse
from typing import Iterable

from . import _cli, _core, _ligands, _pocket
from ._core import (
    AtomRecord,
    COFACTOR_LOOKUP,
    COMMON_EXCLUDED_LIGANDS,
    LigandGroup,
    PocketExtractionError,
    ResidueKey,
    SimpleAtom,
    SimpleChain,
    SimpleResidue,
    StructConnLink,
    WATER_RESNAMES,
    build_component_metadata,
    build_struct_conn_links,
    build_template_bonds,
    classify_cofactor,
    classify_ligand_group,
    detect_ligand_groups,
    index_model_residues,
    load_structure_and_cif,
    residue_key_from_biopython,
)
from ._ligands import build_ligand_molecule


def _build_unprocessed_ligand_factory(cif_dict: dict[str, object]) -> _pocket.LigandBuilder:
    """Return a :data:`_pocket.LigandBuilder` closure for the unprocessed path.

    Called once per run by ``_pocket.run_pocket_extraction`` with the
    post-expansion ``cif_dict``; the ``_chem_comp_bond`` templates built here
    then match the structure actually being extracted (important when
    ``--biological-assembly`` rewrites the CIF upstream).
    """
    template_bonds = _core.build_template_bonds(cif_dict)

    def build(ligand_group, ligand_residues, struct_conn_links):
        mol, atom_records = _ligands.build_ligand_molecule(
            ligand_id=ligand_group.ligand_id,
            ligand_residues=ligand_residues,
            template_bonds=template_bonds,
            struct_conn_links=struct_conn_links,
        )
        return mol, atom_records, None

    return build


def extract_pockets(args: argparse.Namespace) -> dict[str, object]:
    """Extract ligands and their surrounding pockets from an mmCIF file."""
    return _pocket.run_pocket_extraction(
        args,
        build_ligand_factory=_build_unprocessed_ligand_factory,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract ligand-centered pockets from a holo mmCIF using "
            "Biopython, RDKit, and OpenBabel fallback bond perception."
        )
    )
    parser.add_argument("input_path", help="Input holo preprocessed mmCIF/CIF file.")
    _cli.add_common_pocket_arguments(parser)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    manifest = extract_pockets(args)
    print(f"Extracted {manifest['ligand_count']} ligand pocket(s) into {manifest['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
