#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Generic batch runner for ligand-centered pocket extraction over a prepared root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
import traceback
from pathlib import Path
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protein.extract import find_pocket_processed as fpp  # noqa: E402
from protein.extract._core import DEFAULT_POCKET_RADIUS, DEFAULT_MIN_HEAVY_ATOMS  # noqa: E402


class _Args:
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        radius: float,
        min_heavy_atoms: int,
        include_hetero_pocket: bool,
        include_water: bool,
        allow_openbabel_fallback: bool,
    ) -> None:
        self.input_path = str(input_path)
        self.output_dir = str(output_dir)
        self.radius = radius
        self.model_index = 0
        self.min_heavy_atoms = min_heavy_atoms
        self.include_resname = []
        self.exclude_resname = []
        self.include_hetero_pocket = include_hetero_pocket
        self.include_water = include_water
        self.allow_openbabel_fallback = allow_openbabel_fallback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pockets for every <id>/<id>.cif under a prepared dataset root."
    )
    parser.add_argument("--dataset-root", required=True, help="Prepared dataset root.")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional explicit summary JSON path. Defaults to <dataset-root>/_batch_pocket_summary.json",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Optional explicit text log path. Defaults to <dataset-root>/_batch_pocket_log.txt",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_POCKET_RADIUS,
        help="Pocket radius in angstroms.",
    )
    parser.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=DEFAULT_MIN_HEAVY_ATOMS,
        help="Minimum heavy atoms for ligand candidacy.",
    )
    parser.add_argument("--include-water", action="store_true", help="Include water in pocket outputs.")
    parser.add_argument(
        "--include-hetero-pocket",
        action="store_true",
        help="Include nearby hetero residues in pocket outputs.",
    )
    parser.add_argument(
        "--disable-openbabel-fallback",
        action="store_true",
        help="Disable OpenBabel fallback for Schrödinger-exported CIF bond recovery.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of structures to process in parallel.",
    )
    return parser.parse_args(argv)


def iter_pdb_dirs(root: Path):
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        cif = child / f"{child.name}.cif"
        if cif.is_file():
            yield child, cif


def run_one_structure(
    pdb_dir: Path,
    cif_path: Path,
    *,
    radius: float,
    min_heavy_atoms: int,
    include_hetero_pocket: bool,
    include_water: bool,
    allow_openbabel_fallback: bool,
) -> dict[str, object]:
    out_dir = pdb_dir / f"{pdb_dir.name}_pockets"
    pocket_args = _Args(
        cif_path,
        out_dir,
        radius=radius,
        min_heavy_atoms=min_heavy_atoms,
        include_hetero_pocket=include_hetero_pocket,
        include_water=include_water,
        allow_openbabel_fallback=allow_openbabel_fallback,
    )
    record: dict[str, object] = {
        "pdb_id": pdb_dir.name,
        "input_cif": str(cif_path),
        "output_dir": str(out_dir),
    }
    start = time.time()
    try:
        manifest = fpp.extract_pockets(pocket_args)
        record["status"] = "ok"
        record["ligand_count"] = manifest["ligand_count"]
        record["ligands"] = [
            {
                "ligand_id": entry["ligand_id"],
                "pocket_residue_count": entry["pocket_residue_count"],
                "ligand_atom_count": entry["ligand_atom_count"],
                "ligand_bond_count": entry["ligand_bond_count"],
                "bond_source": entry["bond_source"],
            }
            for entry in manifest["ligands"]
        ]
        record["elapsed_seconds"] = round(time.time() - start, 2)
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=4)
        record["elapsed_seconds"] = round(time.time() - start, 2)
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else dataset_root / "_batch_pocket_summary.json"
    )
    log_path = (
        Path(args.log_path).expanduser().resolve()
        if args.log_path
        else dataset_root / "_batch_pocket_log.txt"
    )

    pdb_dirs = list(iter_pdb_dirs(dataset_root))
    print(f"Found {len(pdb_dirs)} prepared structures under {dataset_root}", flush=True)

    results = []
    log_lines = []
    t0 = time.time()
    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(pdb_dirs), desc="Extracting pockets", unit="pdb")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                run_one_structure,
                pdb_dir,
                cif_path,
                radius=args.radius,
                min_heavy_atoms=args.min_heavy_atoms,
                include_hetero_pocket=args.include_hetero_pocket,
                include_water=args.include_water,
                allow_openbabel_fallback=not args.disable_openbabel_fallback,
            ): (index, pdb_dir.name)
            for index, (pdb_dir, cif_path) in enumerate(pdb_dirs, start=1)
        }
        for future in as_completed(future_map):
            index, pdb_id = future_map[future]
            record = future.result()
            if record["status"] == "ok":
                line = (
                    f"[{index:4d}/{len(pdb_dirs)}] {pdb_id}: "
                    f"OK, {record['ligand_count']} ligand(s), {record['elapsed_seconds']:.1f}s"
                )
            else:
                line = f"[{index:4d}/{len(pdb_dirs)}] {pdb_id}: ERROR {record['error']}"
            print(line, flush=True)
            log_lines.append(line)
            results.append(record)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    ok=sum(1 for item in results if item["status"] == "ok"),
                    errors=sum(1 for item in results if item["status"] != "ok"),
                )
    if progress is not None:
        progress.close()

    summary = {
        "dataset_root": str(dataset_root),
        "total": len(results),
        "ok": sum(1 for record in results if record["status"] == "ok"),
        "errors": sum(1 for record in results if record["status"] != "ok"),
        "ligand_total": sum(record.get("ligand_count", 0) for record in results),
        "elapsed_seconds": round(time.time() - t0, 2),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(
        f"\nDone: {summary['ok']}/{summary['total']} ok, "
        f"{summary['errors']} errors, {summary['ligand_total']} ligands total, "
        f"{summary['elapsed_seconds']:.1f}s"
    )
    print(f"Summary: {summary_path}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
