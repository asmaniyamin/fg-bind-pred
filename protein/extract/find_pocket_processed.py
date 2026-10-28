# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/extract/find_pocket_processed.py

Extract ligand-centered binding pockets from a preprocessed holo mmCIF.

Pipeline:
    fetch.py -> preprocess.py -> find_pocket_processed.py

Unlike the unprocessed extractor, this module treats actual per-instance
connection records in ``_struct_conn`` as the primary ligand bond source.
That is the expected path for preprocessed mmCIF files that preserve explicit
connection records.

If a processed file still lacks usable intra-ligand connection records, the
script can optionally fall back to OpenBabel bond perception with
``--allow-openbabel-fallback``.

"""

from __future__ import annotations

import argparse
from typing import Iterable

from . import _cli, _core, _ligands, _pocket
from ._core import PocketExtractionError


PROCESSED_OUTPUT_SUFFIX = "_processed_pockets"


def _build_processed_ligand_factory(args: argparse.Namespace) -> "_pocket.LigandBuilderFactory":
    """Return a factory that yields a :data:`_pocket.LigandBuilder` for the processed path.

    The processed builder prefers explicit ``_struct_conn`` bond records and
    falls back to OpenBabel bond perception only when
    ``--allow-openbabel-fallback`` is set. The bond-source tag distinguishes
    the two outcomes in the pocket JSON and manifest.
    """
    allow_openbabel_fallback = bool(args.allow_openbabel_fallback)

    def factory(_cif_dict: dict[str, object]) -> _pocket.LigandBuilder:
        def build(ligand_group, ligand_residues, struct_conn_links):
            try:
                mol, atom_records = _ligands.build_ligand_molecule_from_struct_conn_links(
                    ligand_id=ligand_group.ligand_id,
                    ligand_residues=ligand_residues,
                    struct_conn_links=struct_conn_links,
                )
                return mol, atom_records, "struct_conn"
            except PocketExtractionError:
                if not allow_openbabel_fallback:
                    raise
                mol, atom_records = _ligands.build_ligand_molecule_with_openbabel(
                    ligand_id=ligand_group.ligand_id,
                    ligand_residues=ligand_residues,
                )
                return mol, atom_records, "openbabel_fallback"

        return build

    return factory


def extract_pockets(args: argparse.Namespace) -> dict[str, object]:
    """Extract ligand pockets from a processed holo mmCIF file."""
    return _pocket.run_pocket_extraction(
        args,
        build_ligand_factory=_build_processed_ligand_factory(args),
        is_grouping_bond=_core.is_bond_like_connection,
        output_suffix=PROCESSED_OUTPUT_SUFFIX,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract ligand-centered pockets from a processed holo mmCIF, "
            "using explicit _struct_conn records as the primary ligand bond source."
        )
    )
    parser.add_argument("input_path", help="Input processed holo mmCIF/CIF file.")
    _cli.add_common_pocket_arguments(parser)
    parser.add_argument(
        "--allow-openbabel-fallback",
        action="store_true",
        help=(
            "Fallback to OpenBabel bond perception if the processed mmCIF "
            "_struct_conn records are missing or incomplete for a ligand."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    manifest = extract_pockets(args)
    print(f"Extracted {manifest['ligand_count']} ligand pocket(s) into {manifest['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
