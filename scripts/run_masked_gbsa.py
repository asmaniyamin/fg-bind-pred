# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
scripts/run_masked_gbsa.py

Connect outputs/fg_masks/ to the run_mmpbsa.py pipeline.

For a given PDB ID the script:
  1. Locates the CIF file (PDBind dataset or a prepared CIF already present
     in the mmpbsa_dir).
  2. Loads the functional-group mask JSON from outputs/fg_masks/{pdb_id}/.
     If no JSON exists there, it runs the extraction pipeline inline and
     writes the result to outputs/fg_masks/ before proceeding.
  3. Collects atom names where functional_group_mask == 1 (atoms inside any
     detected functional group).
  4. Ensures the CIF is present in mmpbsa_dir (copies it if necessary).
  5. Invokes run_mmpbsa.py --mask <atom_names> as a subprocess from mmpbsa_dir
     and streams the output.

Usage examples:
  # Standard PDBind structure (CIF taken from PDBind/PDB_100_4292026/):
  python scripts/run_masked_gbsa.py \\
      --pdb-id 4pox \\
      --mmpbsa-dir /path/to/mmpbsa_dir/

  # Prepared CIF already sitting in mmpbsa_dir (no copy needed):
  python scripts/run_masked_gbsa.py \\
      --pdb-id 3p6h \\
      --mmpbsa-dir /Users/tiavasudeva/Downloads/test-masking-273b-04-21-2026/

  # Explicit CIF override (copies to mmpbsa_dir):
  python scripts/run_masked_gbsa.py \\
      --pdb-id 3p6h \\
      --cif /some/path/3p6h_prepared.cif \\
      --mmpbsa-dir /path/to/mmpbsa_dir/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Bootstrap repo root so sibling packages resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# FG mask helpers
# ---------------------------------------------------------------------------

def load_mask_json(pdb_id: str, fg_masks_dir: Path) -> dict | None:
    """Return the first ligand JSON for pdb_id, or None if not found."""
    structure_dir = fg_masks_dir / pdb_id.lower()
    if not structure_dir.is_dir():
        return None
    jsons = sorted(structure_dir.glob("*.json"))
    if not jsons:
        return None
    return json.loads(jsons[0].read_text(encoding="utf-8"))


def extract_mask_inline(cif_path: Path, pdb_id: str, fg_masks_dir: Path) -> dict:
    """Run FG extraction on cif_path and write JSON to fg_masks_dir/{pdb_id}/."""
    print(f"  No cached FG mask found — running extraction from {cif_path.name} ...")

    from protein.extract.find_pocket_unprocessed import (
        PocketExtractionError,
        build_component_metadata,
        build_struct_conn_links,
        build_template_bonds,
        build_ligand_molecule,
        detect_ligand_groups,
        index_model_residues,
        load_structure_and_cif,
    )
    from chem.functional_groups.detect import detect_from_mol
    from rdkit import Chem

    structure, cif_dict = load_structure_and_cif(cif_path)
    model = structure[0]
    comp_meta = build_component_metadata(cif_dict)
    tmpl = build_template_bonds(cif_dict)
    sc = build_struct_conn_links(cif_dict)
    rl = index_model_residues(model)

    groups = detect_ligand_groups(
        residue_lookup=rl,
        component_metadata=comp_meta,
        struct_conn_links=sc,
        include_resnames=set(),
        exclude_resnames=set(),
        min_heavy_atoms=6,
    )
    if not groups:
        raise RuntimeError(f"No ligand groups detected in {cif_path}")

    group = groups[0]
    residues = [rl[k] for k in group.residue_keys]
    mol, atom_records = build_ligand_molecule(
        ligand_id=group.ligand_id,
        ligand_residues=residues,
        residue_lookup=rl,
        template_bonds=tmpl,
        struct_conn_links=sc,
    )

    n_atoms = mol.GetNumAtoms()
    try:
        smiles: str | None = Chem.MolToSmiles(mol)
    except Exception:
        smiles = None

    try:
        fg_result = detect_from_mol(mol)
    except Exception as exc:
        print(f"  [WARN] FG detection failed: {exc}")
        fg_result = None

    fg_matches_data = []
    atoms_to_groups: dict[str, list[str]] = {}
    mask = [0] * n_atoms
    non_fg_atoms = list(range(n_atoms))

    if fg_result and fg_result.matches:
        fg_matches_data = [
            {
                "name": m.name,
                "atom_indices": list(m.atom_indices),
                "smarts": m.smarts,
                "parent": m.parent,
                "children": list(m.children),
            }
            for m in fg_result.matches
        ]
        a2g = fg_result.atoms_to_groups()
        atoms_to_groups = {str(k): v for k, v in a2g.items()}
        fg_atom_set: set[int] = set()
        for m in fg_result.matches:
            fg_atom_set.update(m.atom_indices)
        mask = [1 if i in fg_atom_set else 0 for i in range(n_atoms)]
        non_fg_atoms = fg_result.non_fg_atoms(n_atoms).tolist()
    else:
        print(f"  [WARN] 0 functional groups detected — mask will be all zeros")

    atom_records_data = [
        {
            "idx": r.atom_index,
            "atom_name": r.atom_name,
            "element": r.element,
            "resname": r.residue_key.resname,
            "chain_id": r.residue_key.chain_id,
            "resseq": r.residue_key.resseq,
            "x": r.x,
            "y": r.y,
            "z": r.z,
        }
        for r in atom_records
    ]

    payload = {
        "structure_id": pdb_id.lower(),
        "ligand_id": group.ligand_id,
        "n_atoms": n_atoms,
        "smiles": smiles,
        "functional_groups": fg_matches_data,
        "atoms_to_groups": atoms_to_groups,
        "functional_group_mask": mask,
        "non_fg_atoms": non_fg_atoms,
        "atom_records": atom_records_data,
    }

    out_dir = fg_masks_dir / pdb_id.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group.ligand_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Wrote FG mask to {out_path.relative_to(_REPO_ROOT)}")
    return payload


# ---------------------------------------------------------------------------
# CIF resolution
# ---------------------------------------------------------------------------

def resolve_cif(
    pdb_id: str,
    mmpbsa_dir: Path,
    cif_override: Path | None,
    pdbbind_root: Path,
) -> Path:
    """Return the CIF path that run_mmpbsa.py should use, copying if needed.

    Priority:
      1. --cif explicit override → copy to mmpbsa_dir if not already there
      2. PDBind/{pdb_id}/{pdb_id}.cif → copy to mmpbsa_dir
      3. Any *.cif matching *{pdb_id}* already in mmpbsa_dir (case-insensitive)
    """
    pdb_lower = pdb_id.lower()
    pdb_upper = pdb_id.upper()

    def copy_if_needed(src: Path) -> Path:
        dest = mmpbsa_dir / src.name
        if dest.resolve() == src.resolve():
            return dest
        if not dest.exists():
            shutil.copy2(src, dest)
            print(f"  Copied {src.name} → {mmpbsa_dir}")
        else:
            print(f"  {src.name} already present in mmpbsa_dir")
        return dest

    if cif_override is not None:
        if not cif_override.is_file():
            raise FileNotFoundError(f"--cif path not found: {cif_override}")
        return copy_if_needed(cif_override)

    pdbbind_cif = pdbbind_root / pdb_lower / f"{pdb_lower}.cif"
    if pdbbind_cif.is_file():
        return copy_if_needed(pdbbind_cif)

    # Look for any matching CIF already in mmpbsa_dir (e.g. prepared CIFs).
    candidates = [
        p for p in mmpbsa_dir.glob("*.cif")
        if pdb_lower in p.name.lower() or pdb_upper in p.name
    ]
    if candidates:
        chosen = sorted(candidates)[0]
        print(f"  Using existing CIF in mmpbsa_dir: {chosen.name}")
        return chosen

    raise FileNotFoundError(
        f"No CIF found for '{pdb_id}' in PDBind ({pdbbind_cif}) "
        f"or mmpbsa_dir ({mmpbsa_dir}). "
        f"Pass --cif to specify the file explicitly."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MM-GBSA with functional-group-derived ligand masks."
    )
    p.add_argument("--pdb-id", required=True, help="PDB ID to process (e.g. 3p6h).")
    p.add_argument(
        "--mmpbsa-dir",
        required=True,
        help="Directory containing run_mmpbsa.py (and optionally prepared CIF files).",
    )
    p.add_argument(
        "--cif",
        default=None,
        help="Override path to a specific CIF file (skips PDBind lookup).",
    )
    p.add_argument(
        "--fg-masks-dir",
        default=str(_REPO_ROOT / "outputs" / "fg_masks"),
        help="Root of fg_masks output directory (default: outputs/fg_masks/).",
    )
    p.add_argument(
        "--pdbbind-root",
        default=str(_REPO_ROOT / "PDBind" / "PDB_100_4292026"),
        help="Root of PDBind CIF dataset.",
    )
    p.add_argument(
        "--mask-mode",
        nargs="+",
        default=["ghost"],
        choices=["ghost", "charge_only", "lj_only", "charge_redistribute"],
        help="Masking mode(s) to pass to run_mmpbsa.py (default: ghost).",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use when calling run_mmpbsa.py.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    pdb_id = args.pdb_id.lower()
    mmpbsa_dir = Path(args.mmpbsa_dir).expanduser().resolve()
    fg_masks_dir = Path(args.fg_masks_dir).expanduser().resolve()
    pdbbind_root = Path(args.pdbbind_root).expanduser().resolve()
    cif_override = Path(args.cif).expanduser().resolve() if args.cif else None

    mmpbsa_script = mmpbsa_dir / "run_mmpbsa.py"
    if not mmpbsa_script.is_file():
        print(f"[ERROR] run_mmpbsa.py not found in {mmpbsa_dir}")
        return 1

    print(f"\n{'='*60}")
    print(f"  PDB ID        : {pdb_id.upper()}")
    print(f"  mmpbsa_dir    : {mmpbsa_dir}")
    print(f"  fg_masks_dir  : {fg_masks_dir}")
    print(f"{'='*60}\n")

    # --- Step 1: Load or extract FG mask ---
    print("[1/3] Resolving functional-group mask ...")
    mask_data = load_mask_json(pdb_id, fg_masks_dir)
    if mask_data is not None:
        print(f"  Loaded cached mask: {pdb_id}/{mask_data['ligand_id']}.json")
    else:
        # Need the CIF to run extraction; resolve it first without copying.
        # We resolve here so extract_mask_inline can use it, then copy after.
        try:
            cif_path = resolve_cif(pdb_id, mmpbsa_dir, cif_override, pdbbind_root)
        except FileNotFoundError as exc:
            print(f"[ERROR] Cannot resolve CIF for inline extraction: {exc}")
            return 1
        mask_data = extract_mask_inline(cif_path, pdb_id, fg_masks_dir)

    # --- Step 2: Extract masked atom names ---
    atom_records = mask_data.get("atom_records", [])
    fg_mask = mask_data.get("functional_group_mask", [])
    ligand_id = mask_data.get("ligand_id", "?")
    n_atoms = mask_data.get("n_atoms", 0)

    masked_atom_names = [
        rec["atom_name"]
        for rec, bit in zip(atom_records, fg_mask)
        if bit == 1
    ]

    n_masked = len(masked_atom_names)
    n_unmasked = n_atoms - n_masked
    density = n_masked / n_atoms if n_atoms > 0 else 0.0

    print(f"\n  Ligand        : {ligand_id}")
    print(f"  Total atoms   : {n_atoms}")
    print(f"  Masked (FG=1) : {n_masked} ({density:.0%})")
    print(f"  Unmasked      : {n_unmasked}")
    print(f"  FG types      : {[fg['name'] for fg in mask_data.get('functional_groups', [])]}")
    print(f"  Atom names    : {masked_atom_names}")

    if not masked_atom_names:
        print("\n[WARN] No atoms are masked (mask is all zeros). "
              "run_mmpbsa.py --mask will receive no atom names.")

    # --- Step 3: Ensure CIF in mmpbsa_dir ---
    print("\n[2/3] Resolving CIF ...")
    try:
        resolve_cif(pdb_id, mmpbsa_dir, cif_override, pdbbind_root)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    # --- Step 4: Run run_mmpbsa.py ---
    print("\n[3/3] Invoking run_mmpbsa.py ...")
    cmd = [
        args.python,
        str(mmpbsa_script),
        "--mask", *masked_atom_names,
        "--mask-mode", *args.mask_mode,
    ]
    print(f"  $ {' '.join(cmd)}\n")

    proc = subprocess.run(cmd, cwd=mmpbsa_dir)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
