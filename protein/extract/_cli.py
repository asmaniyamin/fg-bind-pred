# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/extract/_cli.py

Shared argparse helpers for the unprocessed and processed extractors.

Both extractors expose the same eight common flags. Each extractor wires
its own additional flags on top:

    * ``find_pocket_unprocessed``: only the common surface.
    * ``find_pocket_processed``:   adds ``--allow-openbabel-fallback``.
"""

from __future__ import annotations

import argparse

from ._core import DEFAULT_MIN_HEAVY_ATOMS, DEFAULT_POCKET_RADIUS


def add_common_pocket_arguments(parser: argparse.ArgumentParser) -> None:
    """Wire the eight flags both extractors share onto ``parser``.

    The caller has already added ``input_path`` (and any extractor-specific
    flags).
    """
    parser.add_argument(
        "--output-dir",
        help="Directory for extracted ligand SDFs and pocket JSON files.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_POCKET_RADIUS,
        help=f"Pocket radius in angstrom (default: {DEFAULT_POCKET_RADIUS}).",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=0,
        help="Zero-based model index to process (default: 0).",
    )
    parser.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=DEFAULT_MIN_HEAVY_ATOMS,
        help=(
            "Minimum heavy atoms required for a ligand candidate "
            f"(default: {DEFAULT_MIN_HEAVY_ATOMS})."
        ),
    )
    parser.add_argument(
        "--include-resname",
        action="append",
        default=[],
        help="Force-include one residue name as a ligand candidate. May be repeated.",
    )
    parser.add_argument(
        "--exclude-resname",
        action="append",
        default=[],
        help="Force-exclude one residue name from ligand detection. May be repeated.",
    )
    parser.add_argument(
        "--include-hetero-pocket",
        action="store_true",
        help="Include nearby non-water hetero residues in the pocket output.",
    )
    parser.add_argument(
        "--include-water",
        action="store_true",
        help="Include nearby waters in the pocket output.",
    )
    parser.add_argument(
        "--exclude-cofactors",
        action="store_true",
        help=(
            "Skip known cofactors (HEM, FAD, NAD, FMN, CoA, PLP, etc.). "
            "--include-resname overrides this for named residues."
        ),
    )
    parser.add_argument(
        "--biological-assembly",
        action="store_true",
        help=(
            "Expand the asymmetric unit to the biological assembly using gemmi "
            "before extracting pockets. Symmetry copies receive distinct chain "
            "IDs (A, A2, A3, ...)."
        ),
    )
    parser.add_argument(
        "--assembly-id",
        default="1",
        help=(
            "Biological assembly identifier to apply when "
            "--biological-assembly is set (default: '1')."
        ),
    )
    parser.add_argument(
        "--strict-validity",
        action="store_true",
        help=(
            "Raise PocketExtractionError when the post-cleanup ligand fails "
            "RDKit sanitization or atom-completeness audit. Default is "
            "observe-only: the failure is recorded under pocket_payload"
            "['validation'] and the manifest entry, but extraction continues."
        ),
    )
    parser.add_argument(
        "--strict-batch",
        action="store_true",
        help=(
            "Abort the whole run on the first per-ligand failure. Default "
            "is fail-soft: each failure is captured (with traceback) into "
            "failed_ligands.json next to the manifest, and the loop "
            "continues to the next ligand."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip ligands whose pocket JSON and SDF already exist in the "
            "output directory. The skipped manifest entry is reconstructed "
            "from the prior pocket JSON and tagged status='skipped'. "
            "Default re-extracts everything."
        ),
    )
    parser.add_argument(
        "--force-resume",
        action="store_true",
        help=(
            "Allow --resume even when the input mmCIF's sha256 differs "
            "from the prior run's manifest. Default refuses (silently "
            "mixing two extractions is rarely what you want); pass this "
            "to deliberately reuse cached pocket JSONs from a different "
            "input version."
        ),
    )
