#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Tensorize every PDB under inputs/input_data/raw/{100_run,1000_run}/ in
parallel, writing one ``.npz`` + ``.metadata.json`` per PDB into a flat
output directory.

Combines both runs into a single output dir (100/1000 not delineated).
Uses a process pool because tensorize_sample is CPU-bound (RDKit + numpy).
Errors are logged per-PDB and don't abort the batch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Make repository packages available when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLKIT_DIR = _REPO_ROOT / "small-molecule-mod-toolkit"
if str(_TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_DIR))

import numpy as np  # noqa: E402

from dataloader import (  # noqa: E402
    DatasetRoots,
    DEFAULT_TENSOR_VOCAB,
    build_feature_schema,
    build_sample,
    tensorize_sample,
)
from dataloader.collate import _flatten_array_tree, _jsonify  # noqa: E402


DEFAULT_RAW_ROOT = Path("/mnt/data/projects/cs273b/asmani-branch/inputs/input_data/raw")
DEFAULT_OUTPUT = Path("/mnt/data/projects/cs273b/asmani-branch/inputs/input_data/tensorized")


def _worker_init(case_roots: list[str]) -> None:
    # Each worker gets its own DatasetRoots (the lazy pdb index is per-instance).
    global _ROOTS
    _ROOTS = DatasetRoots(case_roots=[Path(p) for p in case_roots])


def _tensorize_one(args: tuple[str, str, bool]) -> tuple[str, str, str | None, float]:
    pdb_id, output_root, overwrite = args
    output_root = Path(output_root)
    pdb_dir = output_root / pdb_id
    arrays_path = pdb_dir / f"{pdb_id}.npz"
    metadata_path = pdb_dir / f"{pdb_id}.metadata.json"
    if not overwrite and arrays_path.exists() and metadata_path.exists():
        return pdb_id, "skipped", None, 0.0
    t0 = time.time()
    try:
        sample = build_sample(pdb_id, roots=_ROOTS)
        tensorized = tensorize_sample(sample, vocab=DEFAULT_TENSOR_VOCAB)
        arrays: dict[str, np.ndarray] = {}
        for block_key in ("ligand", "functional_groups", "pocket", "mmgbsa"):
            _flatten_array_tree(block_key, tensorized[block_key], arrays)
        pdb_dir.mkdir(parents=True, exist_ok=True)
        # np.savez_compressed appends '.npz' if the filename doesn't end in
        # it, so the tmp name must already end in .npz to avoid double-suffix.
        tmp_arrays = pdb_dir / f".{pdb_id}.tmp.npz"
        tmp_meta = pdb_dir / f".{pdb_id}.metadata.tmp.json"
        np.savez_compressed(tmp_arrays, **arrays)
        tmp_meta.write_text(json.dumps(_jsonify(tensorized["metadata"]), indent=2))
        tmp_arrays.replace(arrays_path)
        tmp_meta.replace(metadata_path)
        return pdb_id, "ok", None, time.time() - t0
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return pdb_id, "error", tb, time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser(description="Tensorize all raw PDBs into one flat dir")
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
                   help=f"directory containing 100_run/ and 1000_run/ (default: {DEFAULT_RAW_ROOT})")
    p.add_argument("--case-dir", action="append", default=None,
                   help="override: explicit case dir (repeat to add more); overrides --raw-root scanning")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"flat output directory (default: {DEFAULT_OUTPUT})")
    p.add_argument("--workers", type=int, default=20, help="worker processes (default: 20)")
    p.add_argument("--overwrite", action="store_true", help="re-tensorize PDBs whose outputs already exist")
    p.add_argument("--limit", type=int, default=None, help="cap total PDBs (for smoke runs)")
    p.add_argument("--pdb-ids", nargs="*", default=None, help="explicit subset of PDB ids to tensorize")
    args = p.parse_args()

    if args.case_dir:
        case_roots = [Path(c) for c in args.case_dir]
    else:
        case_roots = sorted(p for p in args.raw_root.iterdir() if p.is_dir())
    if not case_roots:
        print(f"error: no case directories under {args.raw_root}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    discovery_roots = DatasetRoots(case_roots=case_roots)
    if args.pdb_ids:
        pdb_ids = list(args.pdb_ids)
    else:
        pdb_ids = discovery_roots.discover_pdb_ids(require_baseline=True)
    if args.limit:
        pdb_ids = pdb_ids[: args.limit]

    print(f"[tensorize] case_roots: {[str(p) for p in case_roots]}")
    print(f"[tensorize] output: {args.output}")
    print(f"[tensorize] workers: {args.workers}")
    print(f"[tensorize] pdb count: {len(pdb_ids)}")

    # Persist the feature schema once so all shards share it.
    schema_path = args.output / "_feature_schema.json"
    schema_path.write_text(json.dumps(build_feature_schema(), indent=2))

    case_roots_str = [str(p) for p in case_roots]
    work_items = [(pid, str(args.output), args.overwrite) for pid in pdb_ids]

    summary = {"ok": 0, "skipped": 0, "error": 0}
    errors: list[dict[str, str]] = []
    durations: list[float] = []
    t_start = time.time()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(case_roots_str,),
    ) as pool:
        futures = {pool.submit(_tensorize_one, item): item[0] for item in work_items}
        for i, fut in enumerate(as_completed(futures), start=1):
            pdb_id, status, err, dt = fut.result()
            summary[status] += 1
            durations.append(dt)
            if status == "error":
                errors.append({"pdb_id": pdb_id, "error": err or ""})
            if i % 50 == 0 or i == len(futures):
                elapsed = time.time() - t_start
                ok = summary["ok"]; sk = summary["skipped"]; er = summary["error"]
                rate = i / max(elapsed, 1e-9)
                eta = (len(futures) - i) / max(rate, 1e-9)
                print(f"  [{i}/{len(futures)}] ok={ok} skipped={sk} error={er} "
                      f"rate={rate:.1f}/s eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t_start
    summary_path = args.output / "_tensorize_summary.json"
    summary_path.write_text(json.dumps({
        "total": len(pdb_ids),
        "ok": summary["ok"],
        "skipped": summary["skipped"],
        "error": summary["error"],
        "elapsed_seconds": elapsed,
        "errors": errors,
        "case_roots": case_roots_str,
        "workers": args.workers,
    }, indent=2))

    print(f"[done] {summary} in {elapsed:.1f}s -> {args.output}")
    if errors:
        print(f"[errors] {len(errors)} (see {summary_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
