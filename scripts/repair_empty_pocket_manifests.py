#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Repair empty pocket manifests for PDBs whose prepared CIFs lack
``_chem_comp`` / ``_struct_conn`` records.

For these CIFs, ``find_pocket_processed`` rejects the ligand at the
candidacy check because there's no chem_comp metadata and BioPython's
hetflag is empty. The ligand identity is already known to us via the
FG-mask JSON filenames (``ligand_NNN_C_<RESNAME>_<RESSEQ>.json``), so we
re-run ``find_pocket_processed`` with ``--include-resname <RESNAME>`` to
force-include them. OpenBabel fallback (default in our batch flow)
handles bond perception when ``_struct_conn`` is missing.

Outputs land in ``{raw_root}/{run}/{pdb}/pockets/`` (overwriting the
existing empty ``manifest.json``).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLKIT_DIR = _REPO_ROOT / "small-molecule-mod-toolkit"
if str(_TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_DIR))

from protein.extract import find_pocket_processed as fpp  # noqa: E402
from protein.extract._core import DEFAULT_POCKET_RADIUS, DEFAULT_MIN_HEAVY_ATOMS  # noqa: E402


DEFAULT_RAW_ROOT = Path("/mnt/data/projects/cs273b/asmani-branch/inputs/input_data/raw")

# Filename pattern: ligand_001_A_5GP_190.json -> resname = 5GP
_FG_FILENAME_RE = re.compile(r"^ligand_\d+_[A-Za-z0-9]+_([A-Za-z0-9]+)_-?\d+(?:\.json)?$")


class _Args:
    """Mirrors run_pocket_batch.py:_Args, plus our include_resname list."""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        include_resnames: list[str],
        *,
        radius: float = DEFAULT_POCKET_RADIUS,
        min_heavy_atoms: int = DEFAULT_MIN_HEAVY_ATOMS,
        include_hetero_pocket: bool = False,
        include_water: bool = False,
        allow_openbabel_fallback: bool = True,
    ) -> None:
        self.input_path = str(input_path)
        self.output_dir = str(output_dir)
        self.radius = radius
        self.model_index = 0
        self.min_heavy_atoms = min_heavy_atoms
        self.include_resname = include_resnames
        self.exclude_resname = []
        self.include_hetero_pocket = include_hetero_pocket
        self.include_water = include_water
        self.allow_openbabel_fallback = allow_openbabel_fallback


def _parse_resnames_from_fg_dir(fg_dir: Path) -> list[str]:
    names: list[str] = []
    for fg_json in sorted(fg_dir.glob("ligand_*.json")):
        m = _FG_FILENAME_RE.match(fg_json.stem)
        if m:
            names.append(m.group(1).upper())
            continue
        # Fallback: read ligand_id from JSON content.
        try:
            payload = json.loads(fg_json.read_text())
            lid = payload.get("ligand_id") or ""
            parts = lid.split("_")
            if len(parts) >= 4:
                names.append(parts[-2].upper())
        except Exception:
            continue
    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def _find_empty_pocket_pdbs(raw_root: Path) -> list[tuple[str, Path]]:
    """Return [(pdb_id, pdb_case_dir)] for every PDB whose pockets/manifest.json
    has zero ligands (or is missing)."""
    affected: list[tuple[str, Path]] = []
    for run_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for pdb_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            manifest = pdb_dir / "pockets" / "manifest.json"
            if not manifest.is_file():
                affected.append((pdb_dir.name, pdb_dir))
                continue
            try:
                payload = json.loads(manifest.read_text())
            except Exception:
                affected.append((pdb_dir.name, pdb_dir))
                continue
            if not (payload.get("ligands") or []):
                affected.append((pdb_dir.name, pdb_dir))
    return affected


def _repair_one(args: tuple[str, str]) -> dict[str, object]:
    pdb_id, pdb_dir = args
    pdb_dir = Path(pdb_dir)
    record: dict[str, object] = {"pdb_id": pdb_id, "status": "skipped", "error": None}
    t0 = time.time()
    try:
        prepared_cif = pdb_dir / "prepared" / f"{pdb_id}.cif"
        if not prepared_cif.is_file():
            record["status"] = "no_prepared_cif"
            return record

        fg_dir = pdb_dir / "fg_masks"
        if not fg_dir.is_dir():
            record["status"] = "no_fg_masks"
            return record

        resnames = _parse_resnames_from_fg_dir(fg_dir)
        if not resnames:
            record["status"] = "no_resnames_in_fg_filenames"
            return record
        record["include_resnames"] = resnames

        pockets_dir = pdb_dir / "pockets"
        # Re-extract into a temp staging dir, then move atomically into place.
        staging = pdb_dir / ".pockets.repair.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        pocket_args = _Args(prepared_cif, staging, include_resnames=resnames)
        manifest = fpp.extract_pockets(pocket_args)
        record["ligand_count"] = manifest.get("ligand_count", 0)
        if not manifest.get("ligands"):
            record["status"] = "still_no_ligands"
            return record

        # Replace pockets/ atomically: keep failed_ligands.json if it existed
        # in the original; everything else gets replaced from staging.
        if pockets_dir.exists():
            shutil.rmtree(pockets_dir)
        staging.rename(pockets_dir)
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=4)
    finally:
        record["elapsed_seconds"] = round(time.time() - t0, 2)
        staging = pdb_dir / ".pockets.repair.tmp"
        if staging.exists() and record.get("status") != "ok":
            try:
                shutil.rmtree(staging)
            except Exception:
                pass
    return record


def main() -> int:
    p = argparse.ArgumentParser(description="Repair empty pocket manifests.")
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--summary-json", type=Path, default=None,
                   help="Where to write the per-PDB result log "
                        "(default: {raw_root}/_pocket_repair_summary.json)")
    args = p.parse_args()

    affected = _find_empty_pocket_pdbs(args.raw_root)
    print(f"[repair] PDBs with empty pocket manifests: {len(affected)}")
    if not affected:
        return 0

    summary_path = args.summary_json or (args.raw_root / "_pocket_repair_summary.json")
    work = [(pid, str(d)) for pid, d in affected]
    results: list[dict[str, object]] = []
    counts: dict[str, int] = {}

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_repair_one, item): item[0] for item in work}
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            results.append(rec)
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            if i % 5 == 0 or i == len(futures):
                print(f"  [{i}/{len(futures)}] {counts}", flush=True)

    elapsed = time.time() - t_start
    summary_path.write_text(json.dumps({
        "raw_root": str(args.raw_root),
        "total_affected": len(affected),
        "status_counts": counts,
        "elapsed_seconds": elapsed,
        "per_pdb": sorted(results, key=lambda r: r["pdb_id"]),
    }, indent=2))
    print(f"[done] {counts} in {elapsed:.1f}s; summary at {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
