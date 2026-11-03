# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Helpers for receptor structure repair prior to AmberTools processing."""

from __future__ import annotations

from pathlib import Path

PROTEIN_BACKBONE_ATOMS = {"N", "CA", "C", "O"}
NTERM_BACKBONE_H_NAMES = {"H", "H1", "H2", "H3"}
MAX_PEPTIDE_BOND_GAP_ANGSTROM = 2.5


def _pdb_atom_line(
    serial,
    rec,
    name,
    alt,
    resname,
    chain,
    resseq,
    icode,
    x,
    y,
    z,
    occ,
    bfac,
    element,
):
    """Format a single PDB ATOM/HETATM record (80 cols)."""
    name = name.strip()
    element = element.strip().upper()
    if len(name) >= 4 or len(element) == 2:
        name_field = f"{name:<4s}"
    else:
        name_field = f" {name:<3s}"
    return (
        f"{rec:<6s}"
        f"{serial % 100000:5d} "
        f"{name_field}"
        f"{(alt or ' '):1s}"
        f"{resname:>3s} "
        f"{(chain or 'A'):1s}"
        f"{resseq:4d}"
        f"{(icode or ' '):1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occ:6.2f}{bfac:6.2f}"
        f"          "
        f"{element:>2s}"
    )


def _residue_backbone_missing(atoms):
    """Return the missing protein-backbone atoms for one residue."""
    present = {atom["name"] for atom in atoms}
    return sorted(PROTEIN_BACKBONE_ATOMS - present)


def _residue_atom(atoms, atom_name):
    """Return one atom record by name from a residue table."""
    for atom in atoms:
        if atom["name"] == atom_name:
            return atom
    return None


def _distance(atom_a, atom_b):
    dx = atom_a["x"] - atom_b["x"]
    dy = atom_a["y"] - atom_b["y"]
    dz = atom_a["z"] - atom_b["z"]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def prune_incomplete_terminal_residues(rec_keys_prot, residues):
    """Drop only terminal protein residues missing backbone heavy atoms."""
    kept = []
    pruned = []
    by_chain = {}
    for key in rec_keys_prot:
        by_chain.setdefault(key[0], []).append(key)

    for chain, keys in by_chain.items():
        left = 0
        right = len(keys) - 1

        while left <= right:
            missing = _residue_backbone_missing(residues[keys[left]])
            if not missing:
                break
            pruned.append((keys[left], missing))
            left += 1

        while right >= left:
            missing = _residue_backbone_missing(residues[keys[right]])
            if not missing:
                break
            pruned.append((keys[right], missing))
            right -= 1

        for key in keys[left:right + 1]:
            missing = _residue_backbone_missing(residues[key])
            if missing:
                chain_id, seq, _, comp = key
                raise RuntimeError(
                    "Internal protein residue is missing backbone atoms and "
                    f"cannot be safely auto-pruned: {comp} {chain_id}{seq} "
                    f"(missing {', '.join(missing)})"
                )
            kept.append(key)

    return kept, pruned


def should_strip_protein_hydrogen(comp, atom_name, *, is_chain_start):
    """Strip protein hydrogens prior to Amber template assignment.

    Schrodinger-prepared receptors frequently carry hydrogen names like
    ``GLU HB1`` or ``LYS HG1`` that do not match ff14SB's expected
    side-chain template names.  `tleap` then aborts with "does not have a
    type" errors even though the heavy-atom scaffold is valid.

    The most robust fix is to drop all protein hydrogens here and let
    AmberTools rebuild them implicitly from the residue template during
    `loadpdb`/`saveamberparm`.
    """
    atom_name = atom_name.strip()
    if atom_name and atom_name[0] in {"H", "D", "T"}:
        return True
    if atom_name == "HXT":
        return True
    if is_chain_start and atom_name in NTERM_BACKBONE_H_NAMES:
        return True
    if comp == "ASN" and (
        atom_name in {"HD1", "HD2", "HD11", "HD12", "HD21", "HD22"}
        or atom_name.startswith("HD1")
        or atom_name.startswith("HD2")
    ):
        return True
    if comp == "GLN" and (
        atom_name in {"HE1", "HE2", "HE11", "HE12", "HE21", "HE22"}
        or atom_name.startswith("HE1")
        or atom_name.startswith("HE2")
    ):
        return True
    # Bare "HD"/"HE" on LYS and bare "HD" on PRO lack the trailing digit(s)
    # that ff14SB requires; tleap cannot assign a type and will abort.
    if comp == "LYS" and atom_name in {"HD", "HE"}:
        return True
    if comp == "PRO" and atom_name == "HD":
        return True
    return False


def _parse_pdb_atom_records(path):
    """Parse ATOM/HETATM records from one PDB into grouped residue tables."""
    protein_residues = {}
    protein_order = []
    ion_residues = {}
    ion_order = []

    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue

        rec = line[:6]
        name = line[12:16].strip()
        alt = line[16].strip() or " "
        resname = line[17:20].strip()
        chain = line[21].strip() or "A"
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            resseq = 0
        icode = line[26].strip() or " "
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError:
            continue
        try:
            occ = float(line[54:60].strip() or "1.0")
        except ValueError:
            occ = 1.0
        try:
            bfac = float(line[60:66].strip() or "0.0")
        except ValueError:
            bfac = 0.0
        element = line[76:78].strip() or name[:1]

        key = (chain, resseq, icode, resname)
        atom = {
            "name": name,
            "alt": alt,
            "element": element,
            "x": x,
            "y": y,
            "z": z,
            "occ": occ,
            "bfac": bfac,
        }

        if rec.startswith("ATOM"):
            if key not in protein_residues:
                protein_residues[key] = []
                protein_order.append(key)
            protein_residues[key].append(atom)
        else:
            if key not in ion_residues:
                ion_residues[key] = []
                ion_order.append(key)
            ion_residues[key].append(atom)

    return protein_residues, protein_order, ion_residues, ion_order


def _deduplicate_residue_atoms(residues, order):
    """Drop duplicate atom names within each residue, keeping first occurrence."""
    removed = []
    for key in order:
        seen = set()
        deduped = []
        removed_count = 0
        for atom in residues[key]:
            atom_name = atom["name"].strip()
            if atom_name in seen:
                removed_count += 1
                continue
            seen.add(atom_name)
            deduped.append(atom)
        if removed_count:
            residues[key] = deduped
            chain, seq, _, comp = key
            removed.append(f"{comp} {chain}{seq} ({removed_count})")
    return removed


def _find_chain_breaks(residues, order):
    """Return segment starts where adjacent residues are too far apart."""
    segment_starts = set()
    breaks = []
    by_chain = {}
    for key in order:
        by_chain.setdefault(key[0], []).append(key)

    for keys in by_chain.values():
        if keys:
            segment_starts.add(keys[0])
        for prev_key, next_key in zip(keys, keys[1:]):
            prev_c = _residue_atom(residues[prev_key], "C")
            next_n = _residue_atom(residues[next_key], "N")
            if prev_c is None or next_n is None:
                continue
            distance = _distance(prev_c, next_n)
            if distance > MAX_PEPTIDE_BOND_GAP_ANGSTROM:
                segment_starts.add(next_key)
                breaks.append((prev_key, next_key, distance))
    return segment_starts, breaks


def repair_receptor_pdb(path, *, printer=print):
    """Apply WaterMap-style residue cleanup to the split receptor PDB."""
    path = Path(path)
    protein_residues, protein_order, ion_residues, ion_order = _parse_pdb_atom_records(path)
    if not protein_order:
        return

    removed_duplicates = _deduplicate_residue_atoms(protein_residues, protein_order)
    protein_order, pruned_terminal = prune_incomplete_terminal_residues(
        protein_order, protein_residues
    )
    segment_start_keys, chain_breaks = _find_chain_breaks(protein_residues, protein_order)

    rec_lines = []
    serial = 0
    prev_chain = None
    new_seq = 0

    for key in protein_order:
        chain, _, _, comp = key
        is_segment_start = key in segment_start_keys
        if chain != prev_chain:
            if prev_chain is not None:
                rec_lines.append("TER")
            prev_chain = chain
            new_seq = 0
        elif is_segment_start:
            rec_lines.append("TER")
        new_seq += 1
        for atom in protein_residues[key]:
            if should_strip_protein_hydrogen(
                comp, atom["name"], is_chain_start=is_segment_start
            ):
                continue
            serial += 1
            rec_lines.append(
                _pdb_atom_line(
                    serial,
                    "ATOM  ",
                    atom["name"],
                    atom["alt"],
                    comp,
                    chain,
                    new_seq,
                    " ",
                    atom["x"],
                    atom["y"],
                    atom["z"],
                    atom["occ"],
                    atom["bfac"],
                    atom["element"],
                )
            )
    if protein_order:
        rec_lines.append("TER")

    for key in ion_order:
        chain, seq, icode, comp = key
        for atom in ion_residues[key]:
            serial += 1
            rec_lines.append(
                _pdb_atom_line(
                    serial,
                    "HETATM",
                    atom["name"],
                    atom["alt"],
                    comp,
                    chain,
                    seq,
                    icode,
                    atom["x"],
                    atom["y"],
                    atom["z"],
                    atom["occ"],
                    atom["bfac"],
                    atom["element"],
                )
            )
        rec_lines.append("TER")
    rec_lines.append("END")
    path.write_text("\n".join(rec_lines) + "\n")

    if removed_duplicates:
        printer(
            "  removed duplicate protein atoms: "
            + ", ".join(removed_duplicates)
        )
    if pruned_terminal:
        summary = ", ".join(
            f"{comp} {chain}{seq} (missing {', '.join(missing)})"
            for (chain, seq, _, comp), missing in pruned_terminal
        )
        printer(f"  repaired terminal residues after PDB write: {summary}")
    if chain_breaks:
        summary = ", ".join(
            f"{prev_comp} {prev_chain}{prev_seq} -> {next_comp} {next_chain}{next_seq} "
            f"({distance:.3f} A)"
            for (prev_chain, prev_seq, _, prev_comp), (next_chain, next_seq, _, next_comp), distance
            in chain_breaks
        )
        printer(f"  inserted TER across backbone gaps: {summary}")
