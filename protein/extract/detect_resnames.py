# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/extract/detect_resnames.py

Detect candidate ligand residue names from a *raw* mmCIF (one that still
carries the ``_atom_site.group_PDB`` column).

The processed-pocket extractor (``find_pocket_processed``) requires explicit
``--include-resname`` flags because Schrödinger's ``structconvert -PDBx``
strips the ``group_PDB`` column from its output. Without ``group_PDB``, the
extractor can't tell HETATM from ATOM rows and therefore can't run its
HETATM-based ligand heuristic.

Scan the raw CIF for HETATM ligands before running PrepWizard, then pass
the detected residue names to the processed-pocket extractor.

Programmatic usage::

    from protein.extract.detect_resnames import detect_ligand_resnames_from_raw_cif
    resnames = detect_ligand_resnames_from_raw_cif(Path("1stp.cif"))
    # -> ['BTN']

CLI usage::

    python -m protein.extract.detect_resnames RAW_CIF
    # prints space-separated resnames on stdout

Parsing requires neither Gemmi nor Biopython.
``EXCLUDED_RESNAMES`` defines excluded residue names.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# Conservative exclude list: waters, common ions, common buffers /
# cryoprotectants, and embedded post-translational modifications that
# appear as HETATM rows but are part of the protein chain rather than
# distinct ligands. Keep chemically meaningful small molecules (PO4,
# SO4, GDP, ATP, HEM, etc.) - those are real binders.
EXCLUDED_RESNAMES: frozenset[str] = frozenset({
    # Waters
    "HOH", "WAT", "DOD", "H2O",
    # Monoatomic ions
    "NA", "K", "CL", "BR", "I", "F", "MG", "CA", "ZN", "MN", "FE",
    "CU", "CO", "NI", "CD", "HG", "CS", "RB", "SR", "BA", "LI",
    # Crystallization additives / cryoprotectants / buffers
    "GOL", "EDO", "PEG", "MPD", "DMS", "TRS", "BME", "IOD", "FMT",
    "MES", "EPE", "CIT", "BCT", "ACT", "ACY", "IMD", "NH4", "TPP",
    "PG4", "PGE", "PE4", "BOG", "BTB", "DTT", "ICI",
    # Small halides / oxo-anions almost always crystallization buffer
    "NO3", "AZI",
    # Non-standard / post-translationally modified residues that appear
    # as HETATM but are part of the protein chain (NOT ligands). Common
    # PTMs seen in the PDB; extend as needed.
    "MSE",                                # Se-methionine
    "SEP", "TPO", "PTR",                  # phospho-Ser/Thr/Tyr
    "CSO", "CSD", "OCS", "CME", "CSS",    # oxidized / modified cysteines
    "HYP",                                # hydroxyproline
    "HIC", "MLY", "MLZ", "M3L",           # methylated Lys / His
    "KCX",                                # carbamylated Lys
    "ALY",                                # acetylated Lys
    "PCA",                                # pyroglutamate
    "FME", "NLE",                         # formyl-Met, norleucine
    "ACE", "NH2",                         # N/C-term caps
})

DEFAULT_MIN_HEAVY_ATOMS = 6


def detect_ligand_resnames_from_raw_cif(
    cif_path: Path,
    *,
    min_heavy_atoms: int = DEFAULT_MIN_HEAVY_ATOMS,
    excluded_resnames: Iterable[str] = EXCLUDED_RESNAMES,
) -> list[str]:
    """Scan a raw mmCIF for HETATM residues that look like real ligands.

    A resname is "real ligand" if at least one instance of it (i.e. one
    ``(chain, seq)`` group) has ``>= min_heavy_atoms`` non-hydrogen atoms
    in the ``_atom_site`` loop, AND the resname isn't in
    ``excluded_resnames``. The ``>=`` over the *max* per-instance count
    (rather than the sum) is intentional: if any single instance is large
    enough, the resname survives - useful for bio-units where the same
    ligand appears in multiple symmetry copies.

    Returns the surviving resnames sorted lexicographically.

    The CIF reader is a minimal hand-rolled parser tailored to the
    ``_atom_site`` loop in well-formed RCSB mmCIF files. It does not
    handle multi-line rows, escaped strings, or non-standard whitespace
    - preprocessed Schrödinger output that lacks ``group_PDB`` is
    explicitly out of scope (no HETATM rows survive the column
    presence check). For exotic CIFs use a real parser (``gemmi``).
    """
    excluded_set = frozenset(excluded_resnames)
    cif_lines = Path(cif_path).read_text().splitlines()

    # Locate the _atom_site loop_ block: a "loop_" line, then a run of
    # "_atom_site.<col>" header lines, terminated by the first non-header
    # line. We require the Cartn_x column as a sanity check that we found
    # the right loop (some CIFs have multiple loop_ blocks).
    cols: list[str] = []
    data_start: int | None = None
    i = 0
    while i < len(cif_lines):
        if cif_lines[i].strip() == "loop_":
            j = i + 1
            local_cols: list[str] = []
            while j < len(cif_lines) and cif_lines[j].startswith("_atom_site."):
                local_cols.append(cif_lines[j].strip().split(".", 1)[1])
                j += 1
            if any("Cartn_x" in c for c in local_cols):
                cols = local_cols
                data_start = j
                break
        i += 1

    if data_start is None:
        return []

    col_index = {c: k for k, c in enumerate(cols)}
    grp_i = col_index.get("group_PDB")
    resn_i = col_index.get("auth_comp_id") or col_index.get("label_comp_id")
    chain_i = col_index.get("auth_asym_id") or col_index.get("label_asym_id")
    seq_i = col_index.get("auth_seq_id") or col_index.get("label_seq_id")
    elem_i = col_index.get("type_symbol")

    # Without group_PDB / resname / chain / seq we can't classify HETATM
    # rows. Bail early - caller will get an empty list, which the shell
    # script interprets as "no candidate ligands".
    if None in (grp_i, resn_i, chain_i, seq_i):
        return []

    # (resname) -> {(chain, seq): heavy_atom_count}
    instances: dict[str, dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    required_min_cols = max(grp_i, resn_i, chain_i, seq_i)
    for line in cif_lines[data_start:]:
        # End of loop: blank line, comment, next header, or another loop_.
        stripped = line.strip()
        if not line or line.startswith("#") or line.startswith("_") or stripped == "loop_":
            break
        parts = line.split()
        if len(parts) <= required_min_cols:
            continue
        if parts[grp_i] != "HETATM":
            continue
        resname = parts[resn_i]
        if resname in excluded_set:
            continue
        # Skip hydrogens for heavy-atom count.
        if elem_i is not None and len(parts) > elem_i and parts[elem_i] in ("H", "D"):
            continue
        instances[resname][(parts[chain_i], parts[seq_i])] += 1

    # Keep resnames where at least one instance hits the heavy-atom floor.
    keepers = sorted(
        resname
        for resname, inst_counts in instances.items()
        if inst_counts and max(inst_counts.values()) >= min_heavy_atoms
    )
    return keepers


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``python -m protein.extract.detect_resnames``."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect candidate ligand residue names from a raw mmCIF "
            "(must still have the _atom_site.group_PDB column). Prints "
            "the surviving resnames space-separated on stdout."
        )
    )
    parser.add_argument("cif_path", help="Path to the raw mmCIF file.")
    parser.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=DEFAULT_MIN_HEAVY_ATOMS,
        help=(
            "Minimum heavy-atom count required for at least one instance "
            f"of a resname to survive (default: {DEFAULT_MIN_HEAVY_ATOMS})."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point. Prints space-separated resnames on stdout."""
    args = parse_args(argv)
    resnames = detect_ligand_resnames_from_raw_cif(
        Path(args.cif_path),
        min_heavy_atoms=args.min_heavy_atoms,
    )
    print(" ".join(resnames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
