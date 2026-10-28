#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST_ROOT = Path("/mnt/data/projects/cs273b/asmani-branch/inputs/input_data/raw")


@dataclass(frozen=True)
class BatchConfig:
    label: str
    prepared_root: Path
    fg_root: Path
    baseline_root: Path
    masked_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle completed baseline/MMGBSA and masked FG decomposition inputs."
    )
    parser.add_argument(
        "--dest-root",
        default=str(DEFAULT_DEST_ROOT),
        help="Destination root for the bundled raw inputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned copies without writing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of structures to copy in parallel.",
    )
    return parser.parse_args()


def iter_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(child for child in path.iterdir() if child.is_dir())


def baseline_complete(workdir: Path) -> bool:
    return (workdir / "FINAL_RESULTS_MMGBSA.dat").is_file()


def masked_group_complete(group_dir: Path) -> bool:
    required = [
        group_dir / "FINAL_RESULTS_MMGBSA_GHOST.dat",
        group_dir / "FINAL_RESULTS_MMGBSA_CHARGE_ONLY.dat",
        group_dir / "FINAL_RESULTS_MMGBSA_LJ_ONLY.dat",
        group_dir / "FINAL_DECOMP_MMGBSA_GHOST.dat",
        group_dir / "FINAL_DECOMP_MMGBSA_CHARGE_ONLY.dat",
        group_dir / "FINAL_DECOMP_MMGBSA_LJ_ONLY.dat",
        group_dir / "mask_group.json",
    ]
    return all(path.exists() for path in required)


def ensure_parent(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, *, overwrite: bool, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return False
    ensure_parent(dst, dry_run=dry_run)
    if not dry_run:
        shutil.copy2(src, dst)
    return True


def copy_tree_selected(
    src_dir: Path,
    dst_dir: Path,
    *,
    names: list[str] | None,
    overwrite: bool,
    dry_run: bool,
) -> int:
    if not src_dir.exists():
        return 0
    copied = 0
    if names is None:
        candidates = sorted(src_dir.iterdir())
    else:
        candidates = [src_dir / name for name in names]
    for src in candidates:
        if not src.exists():
            continue
        dst = dst_dir / src.name
        if src.is_dir():
            if dry_run:
                copied += 1
                continue
            if dst.exists() and overwrite:
                shutil.rmtree(dst)
            if not dst.exists():
                shutil.copytree(src, dst)
                copied += 1
        else:
            copied += int(copy_file(src, dst, overwrite=overwrite, dry_run=dry_run))
    return copied


def pocket_has_success(pocket_dir: Path) -> bool:
    manifest = pocket_dir / "manifest.json"
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text())
    except Exception:
        return False
    return bool(payload.get("ligands"))


def bundle_structure(
    *,
    cfg: BatchConfig,
    workdir: Path,
    dest_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, int]:
    pdb_id = workdir.name
    structure_dest = dest_root / cfg.label / pdb_id
    stats = {
        "baseline_structures": 1,
        "masked_structures": 0,
        "masked_groups": 0,
        "files_copied": 0,
    }

    baseline_files = [
        "FINAL_RESULTS_MMGBSA.dat",
        "FINAL_RESULTS_MMGBSA.csv",
        "FINAL_DECOMP_MMGBSA.dat",
        "ligand_raw.pdb",
        "ligand.mol2",
        "mmpbsa.in",
    ]
    prepared_files = [f"{pdb_id}.cif", f"{pdb_id}.pdb", f"{pdb_id}.maegz"]
    masked_files = [
        "mask_group.json",
        "FINAL_RESULTS_MMGBSA.dat",
        "FINAL_RESULTS_MMGBSA.csv",
        "FINAL_DECOMP_MMGBSA.dat",
        "FINAL_RESULTS_MMGBSA_GHOST.dat",
        "FINAL_RESULTS_MMGBSA_GHOST.csv",
        "FINAL_DECOMP_MMGBSA_GHOST.dat",
        "FINAL_RESULTS_MMGBSA_CHARGE_ONLY.dat",
        "FINAL_RESULTS_MMGBSA_CHARGE_ONLY.csv",
        "FINAL_DECOMP_MMGBSA_CHARGE_ONLY.dat",
        "FINAL_RESULTS_MMGBSA_LJ_ONLY.dat",
        "FINAL_RESULTS_MMGBSA_LJ_ONLY.csv",
        "FINAL_DECOMP_MMGBSA_LJ_ONLY.dat",
    ]

    stats["files_copied"] += copy_tree_selected(
        workdir,
        structure_dest / "baseline",
        names=baseline_files,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    prepared_dir = cfg.prepared_root / pdb_id
    stats["files_copied"] += copy_tree_selected(
        prepared_dir,
        structure_dest / "prepared",
        names=prepared_files,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    pocket_dir = prepared_dir / f"{pdb_id}_pockets"
    if pocket_dir.exists():
        pocket_names = ["manifest.json", "failed_ligands.json"]
        if pocket_has_success(pocket_dir):
            pocket_names = None
        stats["files_copied"] += copy_tree_selected(
            pocket_dir,
            structure_dest / "pockets",
            names=pocket_names,
            overwrite=overwrite,
            dry_run=dry_run,
        )

    fg_dir = cfg.fg_root / pdb_id
    if fg_dir.exists():
        stats["files_copied"] += copy_tree_selected(
            fg_dir,
            structure_dest / "fg_masks",
            names=None,
            overwrite=overwrite,
            dry_run=dry_run,
        )

    masked_dir = cfg.masked_root / pdb_id
    if masked_dir.exists():
        complete_group_dirs = [group for group in iter_dirs(masked_dir) if masked_group_complete(group)]
        if complete_group_dirs:
            stats["masked_structures"] = 1
        for group_dir in complete_group_dirs:
            stats["masked_groups"] += 1
            stats["files_copied"] += copy_tree_selected(
                group_dir,
                structure_dest / "masked" / group_dir.name,
                names=masked_files,
                overwrite=overwrite,
                dry_run=dry_run,
            )

    return stats


def bundle_batch(
    cfg: BatchConfig,
    *,
    dest_root: Path,
    overwrite: bool,
    dry_run: bool,
    workers: int,
) -> dict[str, int]:
    stats = {
        "baseline_structures": 0,
        "masked_structures": 0,
        "masked_groups": 0,
        "files_copied": 0,
    }
    complete_workdirs = [workdir for workdir in iter_dirs(cfg.baseline_root) if baseline_complete(workdir)]
    if workers <= 1:
        for workdir in complete_workdirs:
            structure_stats = bundle_structure(
                cfg=cfg,
                workdir=workdir,
                dest_root=dest_root,
                overwrite=overwrite,
                dry_run=dry_run,
            )
            for key, value in structure_stats.items():
                stats[key] += value
        return stats

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                bundle_structure,
                cfg=cfg,
                workdir=workdir,
                dest_root=dest_root,
                overwrite=overwrite,
                dry_run=dry_run,
            )
            for workdir in complete_workdirs
        ]
        for future in cf.as_completed(futures):
            structure_stats = future.result()
            for key, value in structure_stats.items():
                stats[key] += value

    return stats


def main() -> int:
    args = parse_args()
    dest_root = Path(args.dest_root).expanduser().resolve()

    batches = [
        BatchConfig(
            label="100_run",
            prepared_root=Path("/mnt/data/projects/cs273b/asmani-branch/PDBind/PDB_100_4292026"),
            fg_root=REPO_ROOT / "outputs" / "fg_masks",
            baseline_root=REPO_ROOT / "temp_dirs" / "mmgbsa-temp" / "mmpbsa_work",
            masked_root=Path("/mnt/data/projects/cs273b/asmani-branch/temp_dirs/mmgbsa-temp/per_fg_masked_gbsa"),
        ),
        BatchConfig(
            label="1000_run",
            prepared_root=Path("/mnt/data/projects/cs273b/asmani-branch/PDBind/PDB_1000_20260520"),
            fg_root=Path("/mnt/data/projects/cs273b/asmani-branch/outputs/fg_masks_1000_20260520"),
            baseline_root=Path("/mnt/data/projects/cs273b/asmani-branch/temp_dirs/mmgbsa_1000_20260520/mmpbsa_work"),
            masked_root=Path("/mnt/data/projects/cs273b/asmani-branch/temp_dirs/mmgbsa_1000_20260520/per_fg_masked_gbsa"),
        ),
        BatchConfig(
            label="2000_run",
            prepared_root=Path("/mnt/data/projects/cs273b/asmani-branch/PDBind/PDB_2000_20260523"),
            fg_root=Path("/mnt/data/projects/cs273b/asmani-branch/outputs/fg_masks_2000_20260523"),
            baseline_root=Path("/mnt/data/projects/cs273b/asmani-branch/temp_dirs/mmgbsa_2000_20260523/mmpbsa_work"),
            masked_root=Path("/mnt/data/projects/cs273b/asmani-branch/temp_dirs/mmgbsa_2000_20260523/per_fg_masked_gbsa"),
        ),
    ]

    totals = {
        "baseline_structures": 0,
        "masked_structures": 0,
        "masked_groups": 0,
        "files_copied": 0,
    }
    batch_reports: dict[str, dict[str, int]] = {}
    for cfg in batches:
        stats = bundle_batch(
            cfg,
            dest_root=dest_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            workers=max(1, args.workers),
        )
        batch_reports[cfg.label] = stats
        for key, value in stats.items():
            totals[key] += value

    report = {
        "dest_root": str(dest_root),
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "batches": batch_reports,
        "totals": totals,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
