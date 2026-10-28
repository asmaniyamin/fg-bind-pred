#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Single-snapshot MM-GBSA on CIF complexes via AmberTools.

Pipeline per CIF:
  1. Parse _atom_site loop -> split into receptor (protein + monatomic ions)
     and ligand (non-standard HET) PDBs.
  2. Clean receptor with pdb4amber; parameterize ligand with antechamber
     (GAFF2 + AM1-BCC) and parmchk2 (frcmod).
  3. Build complex / receptor / ligand prmtop+inpcrd with tleap (subprocess).
  4. Drive MMPBSA.py in-process via MMPBSA_mods.main.MMPBSA_App using
     complex.inpcrd as the 1-frame trajectory. Each system runs in its
     own multiprocessing child so MMPBSA global state stays clean.
"""

from __future__ import annotations

import builtins
import concurrent.futures as cf
import csv
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

from structure_repair import (
    prune_incomplete_terminal_residues,
    repair_receptor_pdb,
    should_strip_protein_hydrogen,
)

STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HIE", "HID", "HIP", "CYX", "ASH", "GLH", "LYN",
}
MONATOMIC_IONS = {
    "CA", "MG", "ZN", "FE", "MN", "CU", "NA", "K", "CL", "BR", "CS", "LI",
    "RB", "SR", "BA", "NI", "CO", "CD", "HG", "F",
}

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
PDBIND_DIR = REPO_ROOT / "PDBind"
LP_PDBBIND_ALL_CSV = PDBIND_DIR / "LP_PDBBind_all_complexes.csv"
LP_PDBBIND_SOURCE_CSV = PDBIND_DIR / "LP-PDBBind" / "dataset" / "LP_PDBBind.csv"
DEFAULT_THREADS_PER_COMPLEX = 2
DEFAULT_MAX_TOTAL_THREADS = 16
DEFAULT_LIGAND_SELECTION_MODE = "lp_pdbbind"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
)
SMILES_AROMATIC_ELEMENTS = {
    "b": "B",
    "c": "C",
    "n": "N",
    "o": "O",
    "p": "P",
    "s": "S",
}
SMILES_TWO_LETTER_ELEMENTS = {
    "Ag", "Al", "Ar", "As", "At", "Au", "Ba", "Be", "Bi", "Br", "Ca", "Cd",
    "Ce", "Cl", "Co", "Cr", "Cs", "Cu", "Dy", "Er", "Eu", "Fe", "Ga", "Gd",
    "Ge", "He", "Hg", "Ho", "In", "Ir", "Kr", "La", "Li", "Lu", "Mg", "Mn",
    "Mo", "Na", "Nb", "Nd", "Ne", "Ni", "Os", "Pb", "Pd", "Po", "Pr", "Pt",
    "Rb", "Re", "Rh", "Rn", "Ru", "Sb", "Sc", "Se", "Si", "Sm", "Sn", "Sr",
    "Ta", "Tb", "Tc", "Te", "Ti", "Tl", "Tm", "Xe", "Yb", "Zn", "Zr",
}
LIGAND_SELECTION_MODES = ("heuristic", "lp_pdbbind", "lp_pdbbind_strict")
_PRINT_LOCK = threading.Lock()


def print(*args, **kwargs):
    """Serialize console output so concurrent complexes stay readable."""
    kwargs.setdefault("flush", True)
    with _PRINT_LOCK:
        builtins.print(*args, **kwargs)


def configure_thread_environment(thread_count):
    """Apply a per-complex thread cap inherited by child processes."""
    thread_count = str(thread_count)
    for var in THREAD_ENV_VARS:
        os.environ[var] = thread_count
    if "AMBERHOME" not in os.environ:
        for candidate in (os.environ.get("CONDA_PREFIX"), sys.prefix):
            if not candidate:
                continue
            if (Path(candidate) / "dat" / "mmpbsa").exists():
                os.environ["AMBERHOME"] = candidate
                break


def compute_max_complex_workers(max_total_threads, threads_per_complex, num_complexes):
    """Return the number of complexes that may run concurrently."""
    return min(num_complexes, max(1, max_total_threads // threads_per_complex))


@lru_cache(maxsize=1)
def load_lp_pdbbind_rows():
    """Return LP-PDBBind rows keyed by lowercase PDB ID."""
    for path in (LP_PDBBIND_ALL_CSV, LP_PDBBIND_SOURCE_CSV):
        if path.exists():
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                rows = {}
                for row in reader:
                    pdb_id = (row.get("pdb_id") or row.get("") or "").strip().lower()
                    if pdb_id:
                        rows[pdb_id] = row
                return rows
    return {}


def _parse_smiles_bracket_element(token):
    """Return the element symbol encoded at the start of one bracket atom."""
    i = 0
    while i < len(token) and token[i].isdigit():
        i += 1
    if i >= len(token):
        return None
    char = token[i]
    if char.isupper():
        if (
            i + 1 < len(token)
            and token[i + 1].islower()
            and char + token[i + 1] in SMILES_TWO_LETTER_ELEMENTS
        ):
            return char + token[i + 1]
        return char
    return SMILES_AROMATIC_ELEMENTS.get(char)


def smiles_heavy_atom_counts(smiles):
    """Count heavy atoms from a SMILES string without requiring RDKit/OpenBabel."""
    counts = {}
    i = 0
    while i < len(smiles):
        char = smiles[i]
        if char == "[":
            end = smiles.find("]", i + 1)
            if end == -1:
                break
            elem = _parse_smiles_bracket_element(smiles[i + 1:end])
            if elem and elem != "H":
                elem = elem.upper()
                counts[elem] = counts.get(elem, 0) + 1
            i = end + 1
            continue
        if char.isupper():
            if (
                i + 1 < len(smiles)
                and smiles[i + 1].islower()
                and char + smiles[i + 1] in SMILES_TWO_LETTER_ELEMENTS
            ):
                elem = char + smiles[i + 1]
                i += 2
            else:
                elem = char
                i += 1
            if elem != "H":
                elem = elem.upper()
                counts[elem] = counts.get(elem, 0) + 1
            continue
        if char in SMILES_AROMATIC_ELEMENTS:
            elem = SMILES_AROMATIC_ELEMENTS[char].upper()
            counts[elem] = counts.get(elem, 0) + 1
        i += 1
    return counts


def residue_heavy_atom_counts(atoms):
    """Return heavy-atom element counts for one residue."""
    counts = {}
    for atom in atoms:
        elem = atom["element"].strip().upper()
        if elem in {"", "H", "D", "T"}:
            continue
        counts[elem] = counts.get(elem, 0) + 1
    return counts


def average_occupancy(key, residues):
    atoms = residues[key]
    return sum(atom["occ"] for atom in atoms) / max(1, len(atoms))


def choose_ligand_copy(candidate_keys, residues, cif_path, selection_label):
    """Pick one residue copy by occupancy, with deterministic tie-breaking."""
    max_occ = max(average_occupancy(key, residues) for key in candidate_keys)
    keys = [
        key for key in candidate_keys
        if abs(average_occupancy(key, residues) - max_occ) < 1e-6
    ]
    if len(keys) > 1:
        rng = random.Random(f"{cif_path.stem}:{selection_label}")
        keys = [rng.choice(keys)]
        chain, seq, _, comp = keys[0]
        print(
            "  multiple ligand copies matched occupancy; "
            f"selected {comp} {chain}{seq}"
        )
    return keys[0]


def select_ligand_key(cif_path, residues, ligand_candidate_keys, ligand_selection_mode):
    """Choose one ligand residue from the CIF and return its residue key."""
    if ligand_selection_mode in {"lp_pdbbind", "lp_pdbbind_strict"}:
        strict_lp = ligand_selection_mode == "lp_pdbbind_strict"
        lp_rows = load_lp_pdbbind_rows()
        pdb_id = cif_path.stem.split("_")[0].lower()
        lp_row = lp_rows.get(pdb_id)
        smiles = (lp_row or {}).get("smiles", "").strip()
        if smiles:
            target_counts = smiles_heavy_atom_counts(smiles)
            matching_keys = [
                key for key in ligand_candidate_keys
                if residue_heavy_atom_counts(residues[key]) == target_counts
            ]
            if matching_keys:
                selected = choose_ligand_copy(
                    matching_keys, residues, cif_path, f"lp_pdbbind:{smiles}"
                )
                chain, seq, _, comp = selected
                print(
                    "  LP-PDBBind matched ligand by heavy-atom composition: "
                    f"{comp} {chain}{seq}"
                )
                return selected
            if strict_lp:
                raise RuntimeError(
                    "LP-PDBBind strict ligand selection could not find an exact "
                    f"heavy-atom composition match for {pdb_id}"
                )
            print(
                "  LP-PDBBind SMILES did not exactly match any ligand candidate; "
                "falling back to heuristic ligand selection"
            )
        else:
            if strict_lp:
                raise RuntimeError(
                    f"LP-PDBBind strict ligand selection has no SMILES for {pdb_id}"
                )
            print(
                "  LP-PDBBind has no SMILES for this PDB; "
                "falling back to heuristic ligand selection"
            )

    ligand_counts = {}
    for key in ligand_candidate_keys:
        ligand_counts[key[3]] = ligand_counts.get(key[3], 0) + len(residues[key])
    ligand_resname = max(ligand_counts, key=ligand_counts.get)
    resname_keys = [key for key in ligand_candidate_keys if key[3] == ligand_resname]
    return choose_ligand_copy(resname_keys, residues, cif_path, ligand_resname)


# -------------------------- CIF -> atoms --------------------------

def parse_cif_atom_site(path):
    """Return (column_names, rows) for the _atom_site loop_."""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                cols.append(lines[j].strip())
                j += 1
            if cols:
                data = []
                while j < len(lines):
                    s = lines[j].rstrip()
                    if not s.strip() or s.startswith("#") or s.startswith("loop_"):
                        break
                    data.append(s.split())
                    j += 1
                return cols, data
        i += 1
    raise RuntimeError(f"No _atom_site loop found in {path}")


def pdb_atom_line(serial, rec, name, alt, resname, chain, resseq, icode,
                  x, y, z, occ, bfac, element):
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


def split_cif(cif_path, rec_pdb, lig_pdb, ligand_selection_mode=DEFAULT_LIGAND_SELECTION_MODE):
    """Write receptor.pdb (protein + monatomic ions) and ligand.pdb.

    Groups atoms per residue so Schrodinger-prepared CIFs (heavy atoms first,
    hydrogens appended at the end) produce contiguous residues in the PDB.
    Inserts TER across numbering gaps within a chain.

    Returns (ligand_resname, ligand_net_charge_from_cif).
    """
    cols, rows = parse_cif_atom_site(cif_path)

    def col(name, default=None):
        return cols.index(name) if name in cols else default

    ci_elem  = col("_atom_site.type_symbol")
    ci_name  = col("_atom_site.label_atom_id")
    ci_alt   = col("_atom_site.label_alt_id")
    ci_comp  = col("_atom_site.label_comp_id")
    ci_seq   = col("_atom_site.label_seq_id")
    ci_chain = col("_atom_site.auth_asym_id", col("_atom_site.label_asym_id"))
    ci_ins   = col("_atom_site.pdbx_PDB_ins_code")
    ci_x     = col("_atom_site.Cartn_x")
    ci_y     = col("_atom_site.Cartn_y")
    ci_z     = col("_atom_site.Cartn_z")
    ci_occ   = col("_atom_site.occupancy")
    ci_bfac  = col("_atom_site.B_iso_or_equiv")
    ci_chg   = col("_atom_site.pdbx_formal_charge")

    def val(r, idx, default=""):
        if idx is None:
            return default
        v = r[idx]
        return default if v in (".", "?") else v

    def alt_ok(a):
        return a in ("", "A")

    def resseq_int(s):
        try:
            return int(s)
        except ValueError:
            return 0

    # Group atoms by residue; preserve first-seen order.
    residues = {}
    order = []
    for r in rows:
        comp = r[ci_comp]
        if comp == "HOH":
            continue
        alt = val(r, ci_alt)
        if not alt_ok(alt):
            continue
        element = val(r, ci_elem, "C")
        name    = val(r, ci_name, element)
        chain   = val(r, ci_chain, "A") or "A"
        seq     = resseq_int(val(r, ci_seq, "0"))
        icode   = val(r, ci_ins) or " "
        try:
            x = float(r[ci_x]); y = float(r[ci_y]); z = float(r[ci_z])
        except ValueError:
            continue
        try:
            occ = float(val(r, ci_occ, "1.0"))
        except ValueError:
            occ = 1.0
        try:
            bfac = float(val(r, ci_bfac, "0.0"))
        except ValueError:
            bfac = 0.0
        chg = val(r, ci_chg, "0") or "0"
        key = (chain, seq, icode, comp)
        if key not in residues:
            residues[key] = []
            order.append(key)
        residues[key].append({
            "name": name, "alt": alt, "element": element,
            "x": x, "y": y, "z": z, "occ": occ, "bfac": bfac, "charge": chg,
        })

    # Partition residues.
    rec_keys_prot = [k for k in order if k[3] in STD_AA]
    rec_keys_ion  = [k for k in order if k[3] in MONATOMIC_IONS]
    lig_candidate_keys = [
        k for k in order
        if k[3] not in STD_AA and k[3] != "HOH" and k[3] not in MONATOMIC_IONS
    ]
    if not lig_candidate_keys:
        raise RuntimeError(f"No ligand candidate found in {cif_path}")

    rec_keys_prot, pruned_terminal = prune_incomplete_terminal_residues(
        rec_keys_prot, residues
    )
    if pruned_terminal:
        summary = ", ".join(
            f"{comp} {chain}{seq} (missing {', '.join(missing)})"
            for (chain, seq, _, comp), missing in pruned_terminal
        )
        print(f"  pruned incomplete terminal residues: {summary}")

    lig_key = select_ligand_key(
        cif_path, residues, lig_candidate_keys, ligand_selection_mode
    )
    ligand_resname = lig_key[3]
    lig_keys = [lig_key]

    # Write receptor.pdb: protein residues (TER on chain change or num gap),
    # then each ion as its own HETATM with its own TER.
    # Strip only specific hydrogens known to confuse Amber's templates, while
    # preserving heavy atoms and the rest of the deposited protonation state.
    # Missing residues in the middle of a chain (numbering gaps) are NOT
    # TER-separated here; we let tleap form one long peptide bond across
    # the gap. The spurious bond energy cancels in ΔG_bind because it
    # appears identically in the receptor and the complex.
    rec_lines = []
    ser = 0
    prev_chain = None
    new_seq = 0
    chain_offsets = {}
    for index, key in enumerate(rec_keys_prot):
        chain_offsets.setdefault(key[0], []).append(index)
    chain_start_keys = {
        rec_keys_prot[indexes[0]] for indexes in chain_offsets.values() if indexes
    }

    for k in rec_keys_prot:
        chain, _, icode, comp = k
        if chain != prev_chain:
            if prev_chain is not None:
                rec_lines.append("TER")
            new_seq = 0
            prev_chain = chain
        new_seq += 1
        is_chain_start = k in chain_start_keys
        for a in residues[k]:
            if should_strip_protein_hydrogen(
                comp, a["name"], is_chain_start=is_chain_start
            ):
                continue
            ser += 1
            rec_lines.append(pdb_atom_line(
                ser, "ATOM  ", a["name"], a["alt"], comp,
                chain, new_seq, " ", a["x"], a["y"], a["z"],
                a["occ"], a["bfac"], a["element"]))
    if rec_keys_prot:
        rec_lines.append("TER")
    for k in rec_keys_ion:
        chain, seq, icode, comp = k
        for a in residues[k]:
            ser += 1
            rec_lines.append(pdb_atom_line(
                ser, "HETATM", a["name"], a["alt"], comp,
                chain, seq, icode, a["x"], a["y"], a["z"],
                a["occ"], a["bfac"], a["element"]))
        rec_lines.append("TER")
    rec_lines.append("END")
    rec_pdb.write_text("\n".join(rec_lines) + "\n")
    repair_receptor_pdb(rec_pdb, printer=print)

    # Write ligand.pdb
    lig_lines = []
    lig_net_charge = 0
    ser = 0
    for k in lig_keys:
        chain, seq, icode, comp = k
        for a in residues[k]:
            ser += 1
            try:
                lig_net_charge += int(a["charge"])
            except ValueError:
                pass
            lig_lines.append(pdb_atom_line(
                ser, "HETATM", a["name"], a["alt"], comp,
                chain, seq, icode, a["x"], a["y"], a["z"],
                a["occ"], a["bfac"], a["element"]))
    lig_lines.append("TER")
    lig_lines.append("END")
    lig_pdb.write_text("\n".join(lig_lines) + "\n")
    return ligand_resname, lig_net_charge


# -------------------------- shell helpers --------------------------

def run(cmd, cwd, log=None):
    """Run a command; raise on non-zero exit. Tee stdout+stderr to log."""
    print(f"  $ {' '.join(cmd)}  (cwd={cwd.name})")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if log is not None:
        log.write_text((proc.stdout or "") + "\n----STDERR----\n" + (proc.stderr or ""))
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-30:])
        tail_err = "\n".join((proc.stderr or "").splitlines()[-30:])
        raise RuntimeError(
            f"{cmd[0]} failed (exit {proc.returncode}).\n"
            f"--- stdout tail ---\n{tail}\n--- stderr tail ---\n{tail_err}"
        )


def clean_receptor(rec_pdb, cleaned, wdir):
    """Run pdb4amber; return list of (res_i, res_j) 1-based SS-bond pairs."""
    # --no-conect: we feed SS bonds to tleap via explicit `bond` commands;
    # pdb4amber's CONECT serials can disagree with atom ordering and cause
    # tleap to bond the wrong atom pair when loadpdb consumes them.
    run(["pdb4amber", "-i", rec_pdb.name, "-o", cleaned.name,
         "--no-conect",
         "-l", "pdb4amber.log"],
        cwd=wdir, log=wdir / "pdb4amber.stdio")

    ss_file = cleaned.with_name(cleaned.stem + "_sslink")
    pairs = []
    if ss_file.exists():
        for line in ss_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pairs.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    return pairs


def parameterize_ligand(lig_pdb, wdir, resname, net_charge):
    """antechamber (GAFF2 + AM1-BCC) + parmchk2. Returns (mol2, frcmod)."""
    mol2   = wdir / "ligand.mol2"
    frcmod = wdir / "ligand.frcmod"
    run([
        "antechamber",
        "-i", lig_pdb.name, "-fi", "pdb",
        "-o", mol2.name,   "-fo", "mol2",
        "-c", "bcc", "-nc", str(net_charge),
        "-at", "gaff2",
        "-rn", resname[:3],
        "-pf", "y",
        "-s", "2",
    ], cwd=wdir, log=wdir / "antechamber.log")
    run([
        "parmchk2", "-i", mol2.name, "-f", "mol2",
        "-o", frcmod.name, "-s", "gaff2",
    ], cwd=wdir, log=wdir / "parmchk2.log")
    return mol2, frcmod


# GB Born radii calibrated to reproduce experimental ion hydration ΔG.
# None of Amber's standard radius sets (mbondi/mbondi2/mbondi3) define these;
# they all fall back to the 1.5 Å default, which overestimates Ca2+ solvation
# by ~62 kcal/mol vs. the experimental ΔG_hyd = -380.8 kcal/mol.
# R = -166 * q^2 / ΔG_hyd_exp → Ca2+: 1.74 Å.  Screen kept at OBC default 0.80.
_GB_RADIUS_OVERRIDE = {
    "Ca2+": (1.74, 0.80),   # Ca2+: R from ΔG_hyd_exp -380.8 kcal/mol
    "Co2+": (1.31, 0.80),   # Co2+: ΔG_hyd_exp ~ -507 kcal/mol
    "Mg2+": (1.39, 0.80),   # Mg2+: ΔG_hyd_exp -455.5 kcal/mol
    "Mn2+": (1.28, 0.80),   # Mn2+: ΔG_hyd_exp -516.3 kcal/mol
    "Zn2+": (1.31, 0.80),   # Zn2+: ΔG_hyd_exp -484.6 kcal/mol
}
_GBSA_MODEL_OVERRIDE = {
    # Amber 24 `sander` rejects these atom types in the `gbsa=2` LCPO surface-area model.
    "Co2+": 1,
    "Mn2+": 1,
    "Zn2+": 1,
    "i":    1,  # GAFF2 iodine
}

_MASK_MODE_DESCRIPTIONS = {
    "ghost": "zero charge + LJ; removes all masked-atom nonbonded interactions",
    "charge_only": "zero charges only; keeps steric packing/LJ",
    "lj_only": "zero LJ only; keeps electrostatics",
    "charge_redistribute": (
        "zero masked charges and spread the removed charge over the "
        "remaining ligand atoms"
    ),
}


def _masked_prmtop_paths(wdir, mode):
    """Return (complex_prmtop, ligand_prmtop) paths for one mask mode."""
    return (
        wdir / f"complex_{mode}.prmtop",
        wdir / f"ligand_{mode}.prmtop",
    )


def _apply_mask_mode(parm, mask_names, mode, target_resname=None):
    """Modify one ParmEd topology in place for the requested mask mode."""
    from parmed.tools.addljtype import AddLJType

    ligand_atoms = [
        atom for atom in parm.atoms
        if target_resname is None or atom.residue.name == target_resname
    ]
    masked_atoms = [atom for atom in ligand_atoms if atom.name in mask_names]
    if not masked_atoms:
        return {
            "atom_names": frozenset(),
            "removed_charge": 0.0,
            "redistributed_charge": 0.0,
        }

    removed_charge = sum(atom.charge for atom in masked_atoms)
    redistributed_charge = 0.0

    def _zero_masked_lj():
        selected = {id(atom) for atom in masked_atoms}
        sel_atms = [1 if id(atom) in selected else 0 for atom in parm.atoms]
        radius14 = 0.0 if parm.chamber else None
        epsilon14 = 0.0 if parm.chamber else None
        AddLJType(parm, sel_atms, 0.0, 0.0, radius14, epsilon14)
        parm.load_atom_info()

    if mode == "ghost":
        _zero_masked_lj()
        for atom in masked_atoms:
            atom.charge = 0.0
    elif mode == "charge_only":
        for atom in masked_atoms:
            atom.charge = 0.0
    elif mode == "lj_only":
        _zero_masked_lj()
    elif mode == "charge_redistribute":
        unmasked_atoms = [atom for atom in ligand_atoms if atom.name not in mask_names]
        if not unmasked_atoms:
            raise ValueError(
                "charge_redistribute requires at least one unmasked ligand atom"
            )
        redistributed_charge = removed_charge / len(unmasked_atoms)
        for atom in masked_atoms:
            atom.charge = 0.0
        for atom in unmasked_atoms:
            atom.charge += redistributed_charge
    else:
        raise ValueError(
            f"Unknown mask mode {mode!r}; choose from {sorted(_MASK_MODE_DESCRIPTIONS)}"
        )

    return {
        "atom_names": frozenset(atom.name for atom in masked_atoms),
        "removed_charge": removed_charge,
        "redistributed_charge": redistributed_charge,
    }


def apply_ligand_mask(wdir, mask_atom_names, mode="ghost"):
    """Apply one ligand masking mode and write mode-specific prmtops."""
    import parmed as pmd

    mask_names = set(mask_atom_names)

    lig_parm = pmd.load_file(str(wdir / "ligand.prmtop"))
    lig_resname = lig_parm.residues[0].name
    info = _apply_mask_mode(lig_parm, mask_names, mode)
    if not info["atom_names"]:
        available = sorted({a.name for a in lig_parm.atoms})
        raise ValueError(
            f"None of {sorted(mask_names)} found in ligand atoms {available}"
        )

    complex_prmtop, ligand_prmtop = _masked_prmtop_paths(wdir, mode)
    lig_parm.save(str(ligand_prmtop), overwrite=True)

    com_parm = pmd.load_file(str(wdir / "complex.prmtop"))
    _apply_mask_mode(com_parm, mask_names, mode, target_resname=lig_resname)
    com_parm.save(str(complex_prmtop), overwrite=True)

    print(
        f"  {mode} mask on {len(info['atom_names'])} ligand atoms "
        f"{sorted(info['atom_names'])}"
    )
    if mode in ("ghost", "charge_only", "charge_redistribute"):
        print(f"    removed charge: {info['removed_charge']:+.4f} e")
    if mode == "charge_redistribute":
        print(
            "    redistributed per unmasked ligand atom: "
            f"{info['redistributed_charge']:+.6f} e"
        )

    return info


def apply_ligand_ghost_mask(wdir, mask_atom_names):
    """Zero charge + LJ params for specified ligand atoms."""
    return apply_ligand_mask(wdir, mask_atom_names, mode="ghost")


def apply_ligand_charge_only_mask(wdir, mask_atom_names):
    """Zero charges for specified ligand atoms while keeping LJ intact."""
    return apply_ligand_mask(wdir, mask_atom_names, mode="charge_only")


def apply_ligand_lj_only_mask(wdir, mask_atom_names):
    """Zero LJ params for specified ligand atoms while keeping charges intact."""
    return apply_ligand_mask(wdir, mask_atom_names, mode="lj_only")


def apply_ligand_charge_redistribute_mask(wdir, mask_atom_names):
    """Zero masked charges and redistribute them over the rest of the ligand."""
    return apply_ligand_mask(wdir, mask_atom_names, mode="charge_redistribute")


def patch_metal_gb_radii(wdir):
    """Patch Born radii for divalent metal ions in all three prmtops.

    Uses parmed to overwrite the solvent_radius stored in the prmtop after
    tleap assigns the 1.5 Å default fallback.  Must be called before MMPBSA.
    """
    import parmed as pmd
    for prmtop_name in ("complex.prmtop", "receptor.prmtop", "ligand.prmtop"):
        p = wdir / prmtop_name
        parm = pmd.load_file(str(p))
        changed = {}
        for atom in parm.atoms:
            if atom.type in _GB_RADIUS_OVERRIDE:
                old_r = atom.solvent_radius
                new_r, new_s = _GB_RADIUS_OVERRIDE[atom.type]
                atom.solvent_radius = new_r
                atom.screen = new_s
                changed[atom.type] = (old_r, new_r)
        if changed:
            parm.save(str(p), overwrite=True)
            for atype, (old_r, new_r) in changed.items():
                print(f"  patched {atype} GB radius {old_r:.3f} → {new_r:.3f} Å "
                      f"in {prmtop_name}")


def build_topologies(wdir, rec_pdb, lig_mol2, lig_frcmod, ss_pairs):
    """Run tleap to save complex/receptor/ligand prmtop+inpcrd."""
    tleap_in = wdir / "tleap.in"
    bond_lines = "\n".join(
        f"bond REC.{i}.SG REC.{j}.SG" for i, j in ss_pairs
    )
    script = f"""\
source leaprc.protein.ff14SB
source leaprc.gaff2
loadoff atomic_ions.lib
loadamberparams frcmod.ionsjc_tip3p
loadamberparams frcmod.ions234lm_126_tip3p
loadamberparams {lig_frcmod.name}
LIG = loadmol2 {lig_mol2.name}
REC = loadpdb {rec_pdb.name}
{bond_lines}
check REC
check LIG
COM = combine {{ REC LIG }}
saveamberparm LIG ligand.prmtop   ligand.inpcrd
saveamberparm REC receptor.prmtop receptor.inpcrd
saveamberparm COM complex.prmtop  complex.inpcrd
savepdb      COM complex.pdb
quit
"""
    tleap_in.write_text(script)
    run(["tleap", "-f", tleap_in.name], cwd=wdir, log=wdir / "tleap.log")
    for out in ("complex.prmtop", "receptor.prmtop", "ligand.prmtop",
                "complex.inpcrd"):
        if not (wdir / out).exists():
            raise RuntimeError(f"tleap did not produce {out} — see tleap.log")
    patch_metal_gb_radii(wdir)


# -------------------------- MMPBSA.py (GB mode) --------------------------

def write_mmpbsa_input(path):
    path.write_text("""\
Single-snapshot MM-GBSA with per-residue decomposition
&general
  startframe=1, endframe=1, interval=1,
  verbose=2, keep_files=0,
/
&gb
  igb=5, saltcon=0.150,
/
&decomp
  idecomp=1, dec_verbose=3,
/
""")


def choose_gbsa_mode(prmtop_path):
    """Return the gbsa mode that should be used for one topology."""
    import parmed as pmd

    parm = pmd.load_file(str(prmtop_path))
    triggered_types = sorted({
        atom.type
        for atom in parm.atoms
        if atom.type in _GBSA_MODEL_OVERRIDE
    })
    # lj_only masking zeros rmin to 0.0; LCPO (gbsa=2) crashes on zero-radius atoms.
    if not triggered_types and any(atom.rmin == 0.0 for atom in parm.atoms):
        return 1, ()

    if not triggered_types:
        return 2, ()

    gbsa_mode = min(_GBSA_MODEL_OVERRIDE[atype] for atype in triggered_types)
    return gbsa_mode, tuple(triggered_types)


def _mmpbsa_output_paths(wdir, output_tag=None):
    """Return result/decomp/csv paths for baseline or one mask mode."""
    if output_tag:
        tag = output_tag.upper()
        return {
            "results": wdir / f"FINAL_RESULTS_MMGBSA_{tag}.dat",
            "csv": wdir / f"FINAL_RESULTS_MMGBSA_{tag}.csv",
            "decomp": wdir / f"FINAL_DECOMP_MMGBSA_{tag}.dat",
        }
    return {
        "results": wdir / "FINAL_RESULTS_MMGBSA.dat",
        "csv": wdir / "FINAL_RESULTS_MMGBSA.csv",
        "decomp": wdir / "FINAL_DECOMP_MMGBSA.dat",
    }


def _mmpbsa_child_variant(
    wdir_str,
    complex_prmtop_name,
    receptor_prmtop_name,
    ligand_prmtop_name,
    output_tag,
    gbsa_mode,
):
    """Runs inside a spawned child; imports MMPBSA_mods and executes."""
    wdir = Path(wdir_str)
    os.chdir(wdir)
    from MMPBSA_mods import main
    from MMPBSA_mods.fake_mpi import MPI as FakeMPI

    outputs = _mmpbsa_output_paths(wdir, output_tag=output_tag or None)
    main.setup_run()
    app = main.MMPBSA_App(FakeMPI)
    app.get_cl_args([
        "-O",
        "-i", "mmpbsa.in",
        "-o", outputs["results"].name,
        "-eo", outputs["csv"].name,
        "-do", outputs["decomp"].name,
        "-cp", complex_prmtop_name,
        "-rp", receptor_prmtop_name,
        "-lp", ligand_prmtop_name,
        "-y",  "complex.inpcrd",
    ])
    app.read_input_file()
    app.process_input()
    app.INPUT["gbsa"] = gbsa_mode
    app.check_for_bad_input()
    app.loadcheck_prmtops()
    app.file_setup()
    app.setup_mem()
    app.run_mmpbsa()
    app.parse_output_files()
    app.write_final_outputs()
    app.finalize()


def run_mmpbsa(wdir):
    """Run MMPBSA.py via its Python API in a clean child process."""
    return run_mmpbsa_variant(wdir)


def run_mmpbsa_variant(
    wdir,
    complex_prmtop_name="complex.prmtop",
    receptor_prmtop_name="receptor.prmtop",
    ligand_prmtop_name="ligand.prmtop",
    output_tag=None,
):
    """Run MMPBSA.py with caller-selected topologies and output file suffix."""
    gbsa_mode, triggered_types = choose_gbsa_mode(wdir / complex_prmtop_name)
    if triggered_types:
        label = output_tag or "baseline"
        print(
            f"  using gbsa={gbsa_mode} for {label} MMPBSA because "
            f"`gbsa=2` does not support {', '.join(triggered_types)}"
        )
    write_mmpbsa_input(wdir / "mmpbsa.in")
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_mmpbsa_child_variant,
        args=(
            str(wdir),
            complex_prmtop_name,
            receptor_prmtop_name,
            ligand_prmtop_name,
            output_tag or "",
            gbsa_mode,
        ),
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        label = output_tag or "baseline"
        raise RuntimeError(f"MMPBSA {label} child exited with code {p.exitcode}")
    return _mmpbsa_output_paths(wdir, output_tag=output_tag)


def run_mmpbsa_masked(wdir, mode="ghost"):
    """Run MMPBSA with one mode-specific masked ligand topology."""
    complex_prmtop, ligand_prmtop = _masked_prmtop_paths(wdir, mode)
    return run_mmpbsa_variant(
        wdir,
        complex_prmtop_name=complex_prmtop.name,
        receptor_prmtop_name="receptor.prmtop",
        ligand_prmtop_name=ligand_prmtop.name,
        output_tag=mode,
    )


# -------------------------- per-residue decomp parser --------------------------

def parse_decomp(decomp_path):
    """Parse FINAL_DECOMP_MMGBSA.dat and return per-residue GB DELTA contributions.

    The file layout per GB section:
      DELTAS:
      Total Energy Decomposition:
      Residue,Location,Internal,,,van der Waals,...,TOTAL,,   <- col header
      ,,Avg.,Std. Dev.,Std. Err. of Mean,...                  <- sub-header
      RES  N,R RES  N,avg,std,err, avg,std,err, ...           <- data (18 numeric cols)

    Column positions (0-based after splitting on comma):
      0: residue label   1: location
      2: Internal avg    5: vdW avg    8: Elec avg
      11: Polar avg      14: NonPolar avg   17: TOTAL avg

    Returns list of (label, internal, vdw, elec, polar, nonpolar, total)
    sorted by |total|, descending. Uses only the first DELTAS block (GB).
    """
    lines = decomp_path.read_text().splitlines()

    # Find first "DELTAS:" line.
    start = next((i for i, l in enumerate(lines) if l.strip() == "DELTAS:"), None)
    if start is None:
        return []

    # Skip 3 header lines (DELTAS:, "Total Energy Decomposition:", col header, sub-header).
    data_start = start + 4
    rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("DELTAS:") or line.startswith("Backbone"):
            break
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 18:
            continue
        label = parts[0]
        try:
            internal  = float(parts[2])
            vdw       = float(parts[5])
            elec      = float(parts[8])
            polar     = float(parts[11])
            nonpolar  = float(parts[14])
            total     = float(parts[17])
        except ValueError:
            continue
        rows.append((label, internal, vdw, elec, polar, nonpolar, total))

    rows.sort(key=lambda r: abs(r[6]), reverse=True)
    return rows


def print_decomp_table(rows, top_n=15, label=""):
    if not rows:
        print("  (no decomp data found)")
        return
    header = f"{'Residue':<14} {'Internal':>10} {'vdW':>10} {'Elec':>10} {'Polar':>10} {'NonPolar':>10} {'TOTAL':>10}"
    sep = "-" * len(header)
    print(f"\n  --- GB Per-Residue ΔG Contributions {label}(top {min(top_n, len(rows))} by |TOTAL|) ---")
    print(f"  {header}")
    print(f"  {sep}")
    for row in rows[:top_n]:
        label_s, internal, vdw, elec, polar, nonpolar, total = row
        print(f"  {label_s:<14} {internal:>10.3f} {vdw:>10.3f} {elec:>10.3f} {polar:>10.3f} {nonpolar:>10.3f} {total:>10.3f}")
    print(f"  {sep}")


def parse_result_totals(results_path):
    """Return {"GB": value} DELTA TOTAL entries from a results file."""
    totals = {}
    section = None
    for line in results_path.read_text().splitlines():
        s = line.strip()
        if s == "GENERALIZED BORN:":
            section = "GB"
            continue
        if section and s.startswith("DELTA TOTAL"):
            parts = s.split()
            if len(parts) >= 3:
                try:
                    totals[section] = float(parts[2])
                except ValueError:
                    pass
                section = None
    return totals


def print_mask_comparison(baseline_totals, masked_totals_by_mode):
    """Print baseline vs. mask-mode GB/PB DELTA TOTAL values."""
    header = (
        f"{'Mode':<20} {'GB ΔG':>10} {'ΔGB':>10} "
        f"{'PB ΔG':>10} {'ΔPB':>10}"
    )
    sep = "-" * len(header)
    print("\n  --- Mask Mode Comparison ---")
    print(f"  {header}")
    print(f"  {sep}")

    base_gb = baseline_totals.get("GB")
    base_pb = baseline_totals.get("PB")
    print(
        f"  {'baseline':<20} "
        f"{base_gb if base_gb is not None else float('nan'):>10.3f} "
        f"{0.0:>10.3f} "
        f"{base_pb if base_pb is not None else float('nan'):>10.3f} "
        f"{0.0:>10.3f}"
    )

    for mode, totals in masked_totals_by_mode.items():
        gb = totals.get("GB")
        pb = totals.get("PB")
        dgb = gb - base_gb if gb is not None and base_gb is not None else float("nan")
        dpb = pb - base_pb if pb is not None and base_pb is not None else float("nan")
        print(
            f"  {mode:<20} "
            f"{gb if gb is not None else float('nan'):>10.3f} "
            f"{dgb:>10.3f} "
            f"{pb if pb is not None else float('nan'):>10.3f} "
            f"{dpb:>10.3f}"
        )
    print(f"  {sep}")


# -------------------------- driver --------------------------

def process_cif(
    cif_path,
    workroot,
    mask_atom_names=None,
    mask_modes=None,
    ligand_selection_mode=DEFAULT_LIGAND_SELECTION_MODE,
):
    """Process one CIF through the full MM-GBSA pipeline.

    If mask_atom_names is given, one or more extra MMPBSA runs are performed
    with mode-specific masked topologies so the baseline and alternate masking
    assumptions can be compared side by side.
    """
    if mask_atom_names:
        mask_modes = tuple(mask_modes or ("ghost",))

    name = cif_path.stem.split("_")[0]
    wdir = workroot / name
    if wdir.exists():
        shutil.rmtree(wdir)
    wdir.mkdir(parents=True)
    print(f"\n=== {name} ({cif_path.name}) ===")

    rec_pdb = wdir / "receptor_raw.pdb"
    lig_pdb = wdir / "ligand_raw.pdb"
    lig_resname, lig_nc = split_cif(
        cif_path,
        rec_pdb,
        lig_pdb,
        ligand_selection_mode=ligand_selection_mode,
    )
    print(f"  ligand: {lig_resname}  net_charge: {lig_nc:+d}")

    rec_clean = wdir / "receptor_amber.pdb"
    ss = clean_receptor(rec_pdb, rec_clean, wdir)
    if ss:
        print(f"  disulfide pairs (1-based residues): {ss}")

    mol2, frcmod = parameterize_ligand(lig_pdb, wdir, lig_resname, lig_nc)
    build_topologies(wdir, rec_clean, mol2, frcmod, ss)
    baseline_outputs = run_mmpbsa(wdir)

    results = baseline_outputs["results"]
    decomp  = baseline_outputs["decomp"]
    baseline_totals = parse_result_totals(results)
    print(f"  results -> {results}")
    if decomp.exists():
        rows = parse_decomp(decomp)
        print_decomp_table(rows, top_n=15, label=f"({name}) ")
        print(f"  decomp  -> {decomp}")

    if mask_atom_names:
        masked_totals_by_mode = {}
        for mode in mask_modes:
            print(
                f"\n  --- {mode} mask on {sorted(mask_atom_names)} "
                f"({_MASK_MODE_DESCRIPTIONS[mode]}) ---"
            )
            apply_ligand_mask(wdir, mask_atom_names, mode=mode)
            masked_outputs = run_mmpbsa_masked(wdir, mode=mode)
            masked_results = masked_outputs["results"]
            masked_decomp = masked_outputs["decomp"]
            masked_totals_by_mode[mode] = parse_result_totals(masked_results)
            print(f"  {mode} results -> {masked_results}")
            if masked_decomp.exists():
                rows = parse_decomp(masked_decomp)
                print_decomp_table(rows, top_n=15, label=f"({name} {mode}) ")
                print(f"  {mode} decomp  -> {masked_decomp}")
        print_mask_comparison(baseline_totals, masked_totals_by_mode)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MM-GBSA on CIF complexes via AmberTools."
    )
    parser.add_argument(
        "--mask", nargs="+", metavar="ATOM_NAME",
        help=(
            "Ligand atom names to mask in one or more extra MMPBSA runs "
            "(e.g. --mask C1 O2 N3)."
        ),
    )
    parser.add_argument(
        "--mask-mode", nargs="+", choices=sorted(_MASK_MODE_DESCRIPTIONS),
        metavar="MODE",
        help=(
            "Masking mode(s) to run for the selected atoms. Default: ghost. "
            "Pass multiple modes to compare them in one run."
        ),
    )
    parser.add_argument(
        "--threads-per-complex",
        type=int,
        default=DEFAULT_THREADS_PER_COMPLEX,
        help=(
            "Thread cap to apply inside each complex workflow "
            f"(default: {DEFAULT_THREADS_PER_COMPLEX})."
        ),
    )
    parser.add_argument(
        "--max-total-threads",
        type=int,
        default=DEFAULT_MAX_TOTAL_THREADS,
        help=(
            "Global thread budget across all concurrently running complexes "
            f"(default: {DEFAULT_MAX_TOTAL_THREADS})."
        ),
    )
    parser.add_argument(
        "--ligand-selection-mode",
        choices=LIGAND_SELECTION_MODES,
        default=DEFAULT_LIGAND_SELECTION_MODE,
        help=(
            "How to choose one ligand residue per CIF. 'heuristic' uses the "
            "largest non-standard residue name then occupancy tie-breaks; "
            "'lp_pdbbind' prefers an exact LP-PDBBind SMILES heavy-atom "
            "composition match when available and otherwise falls back to the "
            "heuristic; 'lp_pdbbind_strict' requires that exact LP-PDBBind match "
            "and fails otherwise. Default: lp_pdbbind."
        ),
    )
    args = parser.parse_args()
    if args.mask_mode and not args.mask:
        parser.error("--mask-mode requires --mask")
    if args.threads_per_complex < 1:
        parser.error("--threads-per-complex must be >= 1")
    if args.max_total_threads < 1:
        parser.error("--max-total-threads must be >= 1")
    if args.max_total_threads < args.threads_per_complex:
        parser.error(
            "--max-total-threads must be >= --threads-per-complex"
        )
    mask_atom_names = set(args.mask) if args.mask else None
    mask_modes = tuple(args.mask_mode) if args.mask_mode else None

    cifs = sorted(ROOT.glob("*.cif"))
    if not cifs:
        print("No .cif files found in", ROOT)
        return 1
    workroot = ROOT / "mmpbsa_work"
    workroot.mkdir(exist_ok=True)
    max_complex_workers = compute_max_complex_workers(
        args.max_total_threads,
        args.threads_per_complex,
        len(cifs),
    )
    configure_thread_environment(args.threads_per_complex)
    print(
        f"Configured {args.threads_per_complex} thread(s) per complex; "
        f"running up to {max_complex_workers} complex(es) concurrently "
        f"within a {args.max_total_threads}-thread budget."
    )
    print(f"Ligand selection mode: {args.ligand_selection_mode}")

    failures = {}
    results = {}
    with cf.ThreadPoolExecutor(max_workers=max_complex_workers) as executor:
        future_to_cif = {
            executor.submit(
                process_cif,
                cif,
                workroot,
                mask_atom_names,
                mask_modes,
                args.ligand_selection_mode,
            ): cif
            for cif in cifs
        }
        for future in cf.as_completed(future_to_cif):
            cif = future_to_cif[future]
            try:
                results[cif.name] = future.result()
            except Exception as e:
                failures[cif.name] = str(e)
                print(f"\n=== {cif.stem.split('_')[0]} ({cif.name}) ===")
                print(f"  !! failed: {e}")

    print("\n========== summary ==========")
    for cif in cifs:
        result = results.get(cif.name)
        if result is not None:
            print(f"OK  {result.relative_to(ROOT)}")
        elif cif.name in failures:
            err = failures[cif.name]
            print(f"FAIL {cif.name}: {err.splitlines()[0][:200]}")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
