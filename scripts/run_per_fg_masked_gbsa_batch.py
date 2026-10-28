# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Run masked MM/GBSA one functional group at a time for successful baseline runs.

This script bridges:
  - the successful baseline MM/GBSA work directories under `mmpbsa_work/`
  - the precomputed functional-group JSON files under `outputs/fg_masks/`
  - the masking machinery implemented in `temp_dirs/mmgbsa-temp/run_mmpbsa.py`

For each selected successful structure, the script:
  1. Finds the completed baseline work directory.
  2. Chooses the matching ligand mask JSON for the ligand that was actually
     used in `ligand_raw.pdb`.
  3. Expands that JSON into one masking job per detected functional-group
     match.
  4. Stages a dedicated output directory per functional group so masked GBSA
     outputs do not overwrite one another.
  5. Reuses `run_mmpbsa.py` to apply the mask and run MM/GBSA with
     per-residue decomposition enabled.

By default it auto-selects up to 3 completed baseline runs with non-empty
functional-group annotations.

Recommended usage:
    conda run -n openmm_env python scripts/run_per_fg_masked_gbsa_batch.py

Dry run:
    python scripts/run_per_fg_masked_gbsa_batch.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MMPBSA_DIR = REPO_ROOT / "temp_dirs" / "mmgbsa-temp"
DEFAULT_WORKROOT = DEFAULT_MMPBSA_DIR / "mmpbsa_work"
DEFAULT_FG_MASKS_DIR = REPO_ROOT / "outputs" / "fg_masks"
DEFAULT_OUTPUT_ROOT = DEFAULT_MMPBSA_DIR / "per_fg_masked_gbsa"
DEFAULT_PDBBIND_ROOT = REPO_ROOT / "PDBind" / "PDB_100_4292026"
DEFAULT_THREADS_PER_COMPLEX = 2
MASK_MODE_CHOICES = ("ghost", "charge_only", "lj_only", "charge_redistribute")
STAGED_INPUT_FILES = (
    "complex.prmtop",
    "receptor.prmtop",
    "ligand.prmtop",
    "complex.inpcrd",
    "FINAL_RESULTS_MMGBSA.dat",
    "FINAL_RESULTS_MMGBSA.csv",
    "FINAL_DECOMP_MMGBSA.dat",
    "ligand_raw.pdb",
    "receptor_raw.pdb",
    "complex.pdb",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run masked MM/GBSA one functional group at a time on a small "
            "batch of successful baseline work directories."
        )
    )
    p.add_argument(
        "--mmpbsa-dir",
        default=str(DEFAULT_MMPBSA_DIR),
        help="Directory containing run_mmpbsa.py (default: temp_dirs/mmgbsa-temp).",
    )
    p.add_argument(
        "--workroot",
        default=str(DEFAULT_WORKROOT),
        help="Directory containing successful baseline work dirs.",
    )
    p.add_argument(
        "--fg-masks-dir",
        default=str(DEFAULT_FG_MASKS_DIR),
        help="Directory containing extract_fg_masks JSON outputs.",
    )
    p.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for per-functional-group masked runs.",
    )
    p.add_argument(
        "--pdbbind-root",
        default=str(DEFAULT_PDBBIND_ROOT),
        help="Root of the PDBbind CIF dataset, used only if FG JSONs must be regenerated.",
    )
    p.add_argument(
        "--pdb-ids",
        nargs="+",
        default=None,
        help=(
            "Specific successful baseline PDB IDs to process. "
            "If omitted, auto-select up to --limit-structures successful runs "
            "with non-empty functional-group annotations."
        ),
    )
    p.add_argument(
        "--limit-structures",
        type=int,
        default=3,
        help="Maximum number of structures to auto-select when --pdb-ids is omitted.",
    )
    p.add_argument(
        "--max-groups-per-structure",
        type=int,
        default=None,
        help="Optional cap on the number of FG matches run for each structure.",
    )
    p.add_argument(
        "--mask-mode",
        nargs="+",
        default=["ghost", "charge_only", "lj_only"],
        choices=MASK_MODE_CHOICES,
        help=(
            "Masking mode(s) to run for each functional group "
            "(default: ghost charge_only lj_only)."
        ),
    )
    p.add_argument(
        "--threads-per-complex",
        type=int,
        default=DEFAULT_THREADS_PER_COMPLEX,
        help="Thread cap to export for each masked MM/GBSA run (default: 2).",
    )
    p.add_argument(
        "--amber-prefix",
        default=None,
        help=(
            "Optional AmberTools/conda prefix whose bin/ directory should be "
            "prepended to PATH and whose dat/mmpbsa directory should define "
            "AMBERHOME."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate existing per-group output directories.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the jobs and write the summary manifest without running MM/GBSA.",
    )
    p.add_argument(
        "--summary-path",
        default=None,
        help=(
            "Optional explicit summary CSV path. "
            "Defaults to <output-root>/summary.csv."
        ),
    )
    p.add_argument(
        "--skip-summary",
        action="store_true",
        help="Do not write summary.csv. Useful when launching one structure per process in parallel.",
    )
    return p.parse_args(argv)


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_run_mmpbsa_module(mmpbsa_dir: Path):
    module_name = "run_mmpbsa"
    if str(mmpbsa_dir) not in sys.path:
        sys.path.insert(0, str(mmpbsa_dir))
    cached = sys.modules.get(module_name)
    if cached is not None:
        cached_file = getattr(cached, "__file__", None)
        if cached_file and Path(cached_file).resolve() == (mmpbsa_dir / "run_mmpbsa.py").resolve():
            return cached
        del sys.modules[module_name]
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "group"


def format_path_for_logs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_ligand_raw_residues(ligand_raw_path: Path) -> dict[str, Any]:
    residues: list[tuple[str, str, str]] = []
    atom_names: list[str] = []
    for line in ligand_raw_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        chain_id = line[21].strip()
        resseq = line[22:26].strip()
        atom_name = line[12:16].strip()
        key = (resname, chain_id, resseq)
        if key not in residues:
            residues.append(key)
        atom_names.append(atom_name)
    return {
        "residues": residues,
        "atom_names": atom_names,
        "unique_atom_names": sorted(set(atom_names)),
        "has_duplicate_atom_names": len(atom_names) != len(set(atom_names)),
    }


def residue_key_from_atom_records(atom_records: list[dict[str, Any]]) -> tuple[str, str, str] | None:
    if not atom_records:
        return None
    first = atom_records[0]
    return (
        str(first.get("resname", "")).strip(),
        str(first.get("chain_id", "")).strip(),
        str(first.get("resseq", "")).strip(),
    )


def load_fg_jsons_for_pdb(pdb_id: str, fg_masks_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    structure_dir = fg_masks_dir / pdb_id.lower()
    if not structure_dir.is_dir():
        return []
    results = []
    for json_path in sorted(structure_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        results.append((json_path, data))
    return results


def resolve_cif(pdb_id: str, mmpbsa_dir: Path, pdbbind_root: Path) -> Path:
    pdb_lower = pdb_id.lower()
    direct = pdbbind_root / pdb_lower / f"{pdb_lower}.cif"
    if direct.is_file():
        return direct
    candidates = sorted(
        p for p in mmpbsa_dir.glob("*.cif") if pdb_lower in p.name.lower()
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No CIF found for {pdb_id} in {pdbbind_root} or {mmpbsa_dir}"
    )


def ensure_fg_jsons_exist(
    pdb_id: str,
    fg_masks_dir: Path,
    mmpbsa_dir: Path,
    pdbbind_root: Path,
    *,
    allow_extract: bool = True,
) -> list[tuple[Path, dict[str, Any]]]:
    jsons = load_fg_jsons_for_pdb(pdb_id, fg_masks_dir)
    if jsons:
        return jsons
    if not allow_extract:
        return []

    cif_path = resolve_cif(pdb_id, mmpbsa_dir, pdbbind_root)
    extract_mod = load_module_from_path(
        "extract_fg_masks_module",
        REPO_ROOT / "scripts" / "extract_fg_masks.py",
    )
    extract_mod.process_cif(cif_path, fg_masks_dir, min_heavy_atoms=6)
    return load_fg_jsons_for_pdb(pdb_id, fg_masks_dir)


def rescue_mask_json_for_workdir(
    pdb_id: str,
    workdir: Path,
    fg_masks_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    extract_mod = load_module_from_path(
        "extract_fg_masks_module",
        REPO_ROOT / "scripts" / "extract_fg_masks.py",
    )
    return extract_mod.rescue_mask_json_from_workdir(
        pdb_id=pdb_id,
        workdir=workdir,
        output_dir=fg_masks_dir,
    )


def select_mask_payload_for_workdir(
    pdb_id: str,
    workdir: Path,
    fg_masks_dir: Path,
    mmpbsa_dir: Path,
    pdbbind_root: Path,
    *,
    allow_extract: bool = True,
    allow_rescue: bool = True,
) -> tuple[Path, dict[str, Any], list[str]]:
    notes: list[str] = []
    ligand_info = parse_ligand_raw_residues(workdir / "ligand_raw.pdb")
    ligand_residues = set(ligand_info["residues"])

    jsons = ensure_fg_jsons_exist(
        pdb_id,
        fg_masks_dir,
        mmpbsa_dir,
        pdbbind_root,
        allow_extract=allow_extract,
    )
    if not jsons:
        if not allow_rescue:
            raise RuntimeError(
                f"No FG JSON available for {pdb_id} and rescue is disabled"
            )
        rescue_path, rescue_data = rescue_mask_json_for_workdir(pdb_id, workdir, fg_masks_dir)
        notes.append(
            f"{pdb_id}: built FG JSON from mmpbsa_work ligand files because CIF extraction produced none"
        )
        return rescue_path, rescue_data, notes

    matches: list[tuple[Path, dict[str, Any], tuple[str, str, str]]] = []
    for json_path, data in jsons:
        residue_key = residue_key_from_atom_records(data.get("atom_records", []))
        if residue_key is not None and residue_key in ligand_residues:
            matches.append((json_path, data, residue_key))

    if not matches:
        if not allow_rescue:
            raise RuntimeError(
                f"No FG JSON residue match available for {pdb_id} and rescue is disabled"
            )
        rescue_path, rescue_data = rescue_mask_json_for_workdir(pdb_id, workdir, fg_masks_dir)
        notes.append(
            f"{pdb_id}: regenerated FG JSON from mmpbsa_work ligand files plus LP-PDBBind SMILES"
        )
        return rescue_path, rescue_data, notes

    if len(matches) == 1:
        json_path, data, _ = matches[0]
        if build_group_specs(data):
            return json_path, data, notes
        if not allow_rescue:
            raise RuntimeError(
                f"Matched FG JSON for {pdb_id} has zero groups and rescue is disabled"
            )
        rescue_path, rescue_data = rescue_mask_json_for_workdir(pdb_id, workdir, fg_masks_dir)
        notes.append(
            f"{pdb_id}: replaced zero-group CIF FG JSON with rescue JSON from mmpbsa_work ligand files"
        )
        return rescue_path, rescue_data, notes

    atom_signatures = {
        tuple(rec.get("atom_name", "") for rec in data.get("atom_records", []))
        for _, data, _ in matches
    }
    group_signatures = {
        tuple(
            (spec["group_name"], tuple(spec["atom_names"]))
            for spec in build_group_specs(data)
        )
        for _, data, _ in matches
    }
    if len(atom_signatures) == 1 or len(group_signatures) == 1:
        notes.append(
            f"{pdb_id}: {len(matches)} ligand copies matched the successful workdir; "
            "using the first JSON as a representative atom-name mask."
        )
        json_path, data, _ = matches[0]
        if build_group_specs(data):
            return json_path, data, notes
        if not allow_rescue:
            raise RuntimeError(
                f"Matched FG JSONs for {pdb_id} are empty and rescue is disabled"
            )
        rescue_path, rescue_data = rescue_mask_json_for_workdir(pdb_id, workdir, fg_masks_dir)
        notes.append(
            f"{pdb_id}: multiple matched JSONs were empty; using rescue JSON from mmpbsa_work ligand files"
        )
        return rescue_path, rescue_data, notes

    matched_paths = [format_path_for_logs(path) for path, _, _ in matches]
    raise RuntimeError(
        f"Multiple non-equivalent FG JSONs matched {pdb_id}: {matched_paths}"
    )


def build_group_specs(mask_data: dict[str, Any]) -> list[dict[str, Any]]:
    atom_records = mask_data.get("atom_records", [])
    groups = mask_data.get("functional_groups", [])
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for idx, fg in enumerate(groups, 1):
        atom_indices = [int(i) for i in fg.get("atom_indices", [])]
        atom_names: list[str] = []
        for atom_idx in atom_indices:
            if atom_idx < 0 or atom_idx >= len(atom_records):
                continue
            atom_name = str(atom_records[atom_idx].get("atom_name", "")).strip()
            if atom_name and atom_name not in atom_names:
                atom_names.append(atom_name)
        if not atom_names:
            continue

        dedupe_key = (str(fg.get("name", f"group_{idx}")), tuple(atom_names))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        specs.append(
            {
                "match_index": idx,
                "group_name": str(fg.get("name", f"group_{idx}")),
                "group_slug": f"{idx:02d}_{slugify(str(fg.get('name', f'group_{idx}')))}",
                "atom_indices": atom_indices,
                "atom_names": atom_names,
                "smarts": fg.get("smarts"),
                "parent": fg.get("parent"),
                "children": list(fg.get("children", [])),
            }
        )

    return specs


def discover_successful_workdirs(workroot: Path) -> list[Path]:
    results = []
    for workdir in sorted(p for p in workroot.iterdir() if p.is_dir()):
        if (workdir / "FINAL_RESULTS_MMGBSA.dat").is_file() and (workdir / "ligand_raw.pdb").is_file():
            results.append(workdir)
    return results


def stage_group_workdir(source_workdir: Path, dest_workdir: Path, force: bool) -> None:
    if dest_workdir.exists():
        if not force:
            return
        shutil.rmtree(dest_workdir)
    dest_workdir.mkdir(parents=True, exist_ok=True)
    for filename in STAGED_INPUT_FILES:
        src = source_workdir / filename
        if src.exists():
            shutil.copy2(src, dest_workdir / filename)


def masked_output_paths(group_workdir: Path, mode: str) -> dict[str, Path]:
    tag = mode.upper()
    return {
        "results": group_workdir / f"FINAL_RESULTS_MMGBSA_{tag}.dat",
        "csv": group_workdir / f"FINAL_RESULTS_MMGBSA_{tag}.csv",
        "decomp": group_workdir / f"FINAL_DECOMP_MMGBSA_{tag}.dat",
    }


def auto_detect_amber_prefix(explicit_prefix: Path | None) -> Path | None:
    candidates = []
    if explicit_prefix is not None:
        candidates.append(explicit_prefix)
    for raw in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if raw:
            candidates.append(Path(raw).expanduser().resolve())
    for candidate in candidates:
        if (candidate / "bin" / "cpptraj").exists() and (candidate / "dat" / "mmpbsa").exists():
            return candidate
    return None


def configure_amber_environment(amber_prefix: Path | None, threads_per_complex: int) -> Path | None:
    detected = auto_detect_amber_prefix(amber_prefix)
    if detected is not None:
        os.environ["AMBERHOME"] = str(detected)
        os.environ["PATH"] = str(detected / "bin") + os.pathsep + os.environ.get("PATH", "")
    for var in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GOTO_NUM_THREADS",
    ):
        os.environ[var] = str(threads_per_complex)
    return detected


def summarise_top_residues(rows: list[tuple[Any, ...]], top_n: int = 5) -> str:
    top = rows[:top_n]
    parts = [f"{label}:{total:.3f}" for label, _, _, _, _, _, total in top]
    return "; ".join(parts)


def write_group_metadata(
    dest_workdir: Path,
    pdb_id: str,
    source_workdir: Path,
    mask_json_path: Path,
    ligand_id: str,
    group_spec: dict[str, Any],
    notes: list[str],
) -> None:
    payload = {
        "pdb_id": pdb_id,
        "source_workdir": str(source_workdir),
        "mask_json": str(mask_json_path),
        "ligand_id": ligand_id,
        "group": group_spec,
        "notes": notes,
    }
    (dest_workdir / "mask_group.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def collect_group_row(
    pdb_id: str,
    ligand_id: str,
    mask_json_path: Path,
    source_workdir: Path,
    dest_workdir: Path,
    group_spec: dict[str, Any],
    mode: str,
    baseline_gb: float | None,
    masked_results_path: Path,
    masked_decomp_path: Path,
    run_mmpbsa_mod,
    notes: list[str],
) -> dict[str, Any]:
    masked_totals = run_mmpbsa_mod.parse_result_totals(masked_results_path)
    masked_gb = masked_totals.get("GB")
    delta_gb = None
    if baseline_gb is not None and masked_gb is not None:
        delta_gb = masked_gb - baseline_gb
    rows = run_mmpbsa_mod.parse_decomp(masked_decomp_path)
    return {
        "pdb_id": pdb_id,
        "ligand_id": ligand_id,
        "mask_json": format_path_for_logs(mask_json_path),
        "source_workdir": format_path_for_logs(source_workdir),
        "group_dir": format_path_for_logs(dest_workdir),
        "group_index": group_spec["match_index"],
        "group_name": group_spec["group_name"],
        "group_slug": group_spec["group_slug"],
        "atom_names": " ".join(group_spec["atom_names"]),
        "atom_indices": " ".join(str(i) for i in group_spec["atom_indices"]),
        "mask_mode": mode,
        "baseline_gb": "" if baseline_gb is None else f"{baseline_gb:.6f}",
        "masked_gb": "" if masked_gb is None else f"{masked_gb:.6f}",
        "delta_gb": "" if delta_gb is None else f"{delta_gb:.6f}",
        "results_path": format_path_for_logs(masked_results_path),
        "decomp_path": format_path_for_logs(masked_decomp_path),
        "top_residues": summarise_top_residues(rows),
        "notes": " | ".join(notes),
    }


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id",
        "ligand_id",
        "mask_json",
        "source_workdir",
        "group_dir",
        "group_index",
        "group_name",
        "group_slug",
        "atom_names",
        "atom_indices",
        "mask_mode",
        "baseline_gb",
        "masked_gb",
        "delta_gb",
        "results_path",
        "decomp_path",
        "top_residues",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    mmpbsa_dir = Path(args.mmpbsa_dir).expanduser().resolve()
    workroot = Path(args.workroot).expanduser().resolve()
    fg_masks_dir = Path(args.fg_masks_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    pdbbind_root = Path(args.pdbbind_root).expanduser().resolve()
    amber_prefix = Path(args.amber_prefix).expanduser().resolve() if args.amber_prefix else None

    mmpbsa_script = mmpbsa_dir / "run_mmpbsa.py"
    if not mmpbsa_script.is_file():
        print(f"[ERROR] Missing run_mmpbsa.py at {mmpbsa_script}")
        return 1
    if not workroot.is_dir():
        print(f"[ERROR] Missing baseline workroot: {workroot}")
        return 1

    run_mmpbsa_mod = import_run_mmpbsa_module(mmpbsa_dir)
    amber_home = configure_amber_environment(amber_prefix, args.threads_per_complex)

    successful_workdirs = discover_successful_workdirs(workroot)
    if not successful_workdirs:
        print(f"[ERROR] No successful baseline workdirs found under {workroot}")
        return 1

    requested_pdb_ids = [p.lower() for p in args.pdb_ids] if args.pdb_ids else None
    selection: list[dict[str, Any]] = []

    for workdir in successful_workdirs:
        pdb_id = workdir.name.lower()
        if requested_pdb_ids is not None and pdb_id not in requested_pdb_ids:
            continue

        try:
            mask_json_path, mask_data, notes = select_mask_payload_for_workdir(
                pdb_id,
                workdir,
                fg_masks_dir,
                mmpbsa_dir,
                pdbbind_root,
            )
        except Exception as exc:
            print(f"[SKIP] {pdb_id}: {exc}")
            continue

        group_specs = build_group_specs(mask_data)
        if not group_specs:
            print(f"[SKIP] {pdb_id}: no functional-group matches in {mask_json_path.name}")
            continue

        if args.max_groups_per_structure is not None:
            group_specs = group_specs[: args.max_groups_per_structure]

        baseline_totals = run_mmpbsa_mod.parse_result_totals(workdir / "FINAL_RESULTS_MMGBSA.dat")
        selection.append(
            {
                "pdb_id": pdb_id,
                "workdir": workdir,
                "mask_json_path": mask_json_path,
                "mask_data": mask_data,
                "group_specs": group_specs,
                "notes": notes,
                "baseline_gb": baseline_totals.get("GB"),
            }
        )

    if requested_pdb_ids is None:
        selection = selection[: args.limit_structures]

    if not selection:
        print("[ERROR] No eligible successful runs were selected.")
        return 1

    print(f"\n{'=' * 72}")
    print("Per-FG Masked GBSA Batch")
    print(f"{'=' * 72}")
    print(f"mmpbsa_dir         : {mmpbsa_dir}")
    print(f"workroot           : {workroot}")
    print(f"fg_masks_dir       : {fg_masks_dir}")
    print(f"output_root        : {output_root}")
    print(f"mask_mode(s)       : {args.mask_mode}")
    print(f"threads_per_complex: {args.threads_per_complex}")
    print(f"amber_prefix       : {amber_home if amber_home else '(not detected)'}")
    print(f"dry_run            : {args.dry_run}")
    print(f"{'=' * 72}\n")

    for item in selection:
        print(
            f"  {item['pdb_id']}: {len(item['group_specs'])} FG match(es) from "
            f"{format_path_for_logs(item['mask_json_path'])}"
        )
        for note in item["notes"]:
            print(f"    note: {note}")
    print()

    if not args.dry_run:
        try:
            import parmed  # noqa: F401
        except ImportError:
            print(
                "[ERROR] parmed is not importable in this Python environment. "
                "Run this script with the AmberTools/OpenMM environment, e.g. "
                "`conda run -n openmm_env python scripts/run_per_fg_masked_gbsa_batch.py`."
            )
            return 1

    summary_rows: list[dict[str, Any]] = []

    for item in selection:
        pdb_id = item["pdb_id"]
        workdir = item["workdir"]
        mask_json_path = item["mask_json_path"]
        mask_data = item["mask_data"]
        baseline_gb = item["baseline_gb"]
        ligand_id = str(mask_data.get("ligand_id", "?"))

        print(f"\n[{pdb_id}] baseline GB = {baseline_gb if baseline_gb is not None else 'n/a'}")

        for group_spec in item["group_specs"]:
            group_dir = output_root / pdb_id / group_spec["group_slug"]
            group_notes = list(item["notes"])
            print(
                f"  - {group_spec['group_slug']}: {group_spec['group_name']} "
                f"on {group_spec['atom_names']}"
            )

            stage_group_workdir(workdir, group_dir, force=args.force)
            write_group_metadata(
                group_dir,
                pdb_id,
                workdir,
                mask_json_path,
                ligand_id,
                group_spec,
                group_notes,
            )

            for mode in args.mask_mode:
                outputs = masked_output_paths(group_dir, mode)
                if args.dry_run:
                    summary_rows.append(
                        {
                            "pdb_id": pdb_id,
                            "ligand_id": ligand_id,
                            "mask_json": format_path_for_logs(mask_json_path),
                            "source_workdir": format_path_for_logs(workdir),
                            "group_dir": format_path_for_logs(group_dir),
                            "group_index": group_spec["match_index"],
                            "group_name": group_spec["group_name"],
                            "group_slug": group_spec["group_slug"],
                            "atom_names": " ".join(group_spec["atom_names"]),
                            "atom_indices": " ".join(str(i) for i in group_spec["atom_indices"]),
                            "mask_mode": mode,
                            "baseline_gb": "" if baseline_gb is None else f"{baseline_gb:.6f}",
                            "masked_gb": "",
                            "delta_gb": "",
                            "results_path": format_path_for_logs(outputs["results"]),
                            "decomp_path": format_path_for_logs(outputs["decomp"]),
                            "top_residues": "",
                            "notes": "dry-run" + (f" | {' | '.join(group_notes)}" if group_notes else ""),
                        }
                    )
                    continue

                if outputs["results"].exists() and outputs["decomp"].exists() and not args.force:
                    print(f"      {mode}: existing outputs found, reusing")
                    summary_rows.append(
                        collect_group_row(
                            pdb_id,
                            ligand_id,
                            mask_json_path,
                            workdir,
                            group_dir,
                            group_spec,
                            mode,
                            baseline_gb,
                            outputs["results"],
                            outputs["decomp"],
                            run_mmpbsa_mod,
                            group_notes + ["reused existing outputs"],
                        )
                    )
                    continue

                run_mmpbsa_mod.configure_thread_environment(args.threads_per_complex)
                run_mmpbsa_mod.apply_ligand_mask(
                    group_dir,
                    group_spec["atom_names"],
                    mode=mode,
                )
                run_mmpbsa_mod.run_mmpbsa_masked(group_dir, mode=mode)

                summary_rows.append(
                    collect_group_row(
                        pdb_id,
                        ligand_id,
                        mask_json_path,
                        workdir,
                        group_dir,
                        group_spec,
                        mode,
                        baseline_gb,
                        outputs["results"],
                        outputs["decomp"],
                        run_mmpbsa_mod,
                        group_notes,
                    )
                )

    if not args.skip_summary:
        summary_path = (
            Path(args.summary_path).expanduser().resolve()
            if args.summary_path
            else output_root / "summary.csv"
        )
        write_summary_csv(summary_rows, summary_path)
        print(f"\nWrote summary: {format_path_for_logs(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
