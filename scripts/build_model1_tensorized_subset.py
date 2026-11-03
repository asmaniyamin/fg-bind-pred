#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Build an exact-count tensorized subset for model1 experiments by symlinking
selected cached structures and root-level schema files into a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLKIT_DIR = _REPO_ROOT / "small-molecule-mod-toolkit"
for _directory in (_REPO_ROOT, _TOOLKIT_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/data/projects/cs273b/asmani-branch/inputs/May20-2026-input_data/tensorized"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/model1_subset")


def _stable_score(token: str, seed: int) -> float:
    payload = f"{seed}:{token}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def pocket_hash_split(
    pdb_ids: list[str],
    pocket_sequences: dict[str, str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for pdb_id in sorted(pdb_ids):
        token = pocket_sequences.get(pdb_id) or pdb_id
        score = _stable_score(token, seed)
        if score < test_frac:
            test_ids.append(pdb_id)
        elif score < test_frac + val_frac:
            val_ids.append(pdb_id)
        else:
            train_ids.append(pdb_id)
    return train_ids, val_ids, test_ids


def ligand_split(
    pdb_ids: list[str],
    ligand_smiles: dict[str, str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for pdb_id in sorted(pdb_ids):
        token = ligand_smiles.get(pdb_id) or pdb_id
        score = _stable_score(token, seed)
        if score < test_frac:
            test_ids.append(pdb_id)
        elif score < test_frac + val_frac:
            val_ids.append(pdb_id)
        else:
            train_ids.append(pdb_id)
    return train_ids, val_ids, test_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tensorized subset with exact train/val/test counts by selecting "
            "from stable split buckets and symlinking the chosen cached structures."
        )
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split-type", choices=["pocket_hash", "ligand"], default="pocket_hash")
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--floor-to-available",
        action="store_true",
        help=(
            "Clamp each requested split count to the available size of its stable "
            "split bucket instead of failing."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before creating the subset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected counts and sample IDs without writing anything.",
    )
    parser.add_argument(
        "--manifest-name",
        default="subset_manifest.json",
        help="Filename for the subset manifest written inside output-dir.",
    )
    return parser.parse_args(argv)


def discover_ids(source_root: Path) -> list[str]:
    ids: set[str] = set()
    for entry in sorted(source_root.iterdir()):
        if entry.is_dir() and (entry / f"{entry.name}.npz").is_file():
            ids.add(entry.name)
        elif entry.is_file() and entry.suffix == ".npz":
            ids.add(entry.stem)
    return sorted(ids)


def metadata_path_for(source_root: Path, pdb_id: str) -> Path:
    nested = source_root / pdb_id / f"{pdb_id}.metadata.json"
    flat = source_root / f"{pdb_id}.metadata.json"
    if nested.is_file():
        return nested
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"Metadata JSON not found for {pdb_id}")


def load_metadata(source_root: Path, pdb_id: str) -> dict:
    return json.loads(metadata_path_for(source_root, pdb_id).read_text())


def compute_split_buckets(
    source_root: Path,
    pdb_ids: list[str],
    split_type: str,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if split_type == "ligand":
        ligand_smiles = {
            pdb_id: (load_metadata(source_root, pdb_id).get("smiles") or pdb_id)
            for pdb_id in pdb_ids
        }
        return ligand_split(pdb_ids, ligand_smiles, val_frac, test_frac, seed)

    pocket_sequences = {}
    for pdb_id in pdb_ids:
        residue_keys = load_metadata(source_root, pdb_id).get("residue_keys", [])
        pocket_sequences[pdb_id] = "-".join(str(key[3]) for key in residue_keys)
    return pocket_hash_split(pdb_ids, pocket_sequences, val_frac, test_frac, seed)


def ensure_output_dir(output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if not output_dir.exists():
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Rerun with --overwrite to replace it."
        )
    if dry_run:
        return
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_subset(
    source_root: Path,
    output_dir: Path,
    selected_ids: list[str],
    overwrite: bool,
    dry_run: bool,
) -> None:
    ensure_output_dir(output_dir, overwrite=overwrite, dry_run=dry_run)
    if dry_run:
        return
    for top in sorted(source_root.iterdir()):
        if top.is_file() and top.name.startswith("_"):
            os.symlink(top, output_dir / top.name)
    for pdb_id in selected_ids:
        os.symlink(source_root / pdb_id, output_dir / pdb_id)


def write_manifest(
    output_dir: Path,
    manifest_name: str,
    payload: dict[str, object],
    dry_run: bool,
) -> None:
    if dry_run:
        return
    (output_dir / manifest_name).write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = Path(args.source_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    pdb_ids = discover_ids(source_root)
    if not pdb_ids:
        raise RuntimeError(f"No cached tensorized structures found under {source_root}")

    train_bucket, val_bucket, test_bucket = compute_split_buckets(
        source_root=source_root,
        pdb_ids=pdb_ids,
        split_type=args.split_type,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    requested_counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    available_counts = {
        "train": len(train_bucket),
        "val": len(val_bucket),
        "test": len(test_bucket),
    }
    adjusted_counts = dict(requested_counts)
    if args.floor_to_available:
        adjusted_counts = {
            split: min(requested_counts[split], available_counts[split])
            for split in requested_counts
        }

    if (
        available_counts["train"] < adjusted_counts["train"]
        or available_counts["val"] < adjusted_counts["val"]
        or available_counts["test"] < adjusted_counts["test"]
    ):
        raise RuntimeError(
            "Requested counts exceed available IDs in split buckets: "
            f"train={available_counts['train']} val={available_counts['val']} test={available_counts['test']}"
        )

    train_ids = train_bucket[: adjusted_counts["train"]]
    val_ids = val_bucket[: adjusted_counts["val"]]
    test_ids = test_bucket[: adjusted_counts["test"]]
    selected_ids = train_ids + val_ids + test_ids

    manifest = {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "split_type": args.split_type,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "floor_to_available": args.floor_to_available,
        "requested_counts": requested_counts,
        "available_bucket_counts": available_counts,
        "adjusted_counts": adjusted_counts,
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": len(selected_ids),
        },
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }

    build_subset(
        source_root=source_root,
        output_dir=output_dir,
        selected_ids=selected_ids,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    write_manifest(
        output_dir=output_dir,
        manifest_name=args.manifest_name,
        payload=manifest,
        dry_run=args.dry_run,
    )

    print(f"source_root {source_root}")
    print(f"output_dir {output_dir}")
    print(
        "requested_counts "
        f"train={requested_counts['train']} val={requested_counts['val']} test={requested_counts['test']}"
    )
    print(
        "available_bucket_counts "
        f"train={available_counts['train']} val={available_counts['val']} test={available_counts['test']}"
    )
    print(
        "selected_counts "
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} total={len(selected_ids)}"
    )
    print("train_ids_sample", ",".join(train_ids[: min(10, len(train_ids))]))
    print("val_ids_sample", ",".join(val_ids[: min(10, len(val_ids))]))
    print("test_ids_sample", ",".join(test_ids[: min(10, len(test_ids))]))
    if args.dry_run:
        print("dry_run True")
    else:
        print(f"manifest {output_dir / args.manifest_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
