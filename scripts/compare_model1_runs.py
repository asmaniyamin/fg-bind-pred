#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed model1 runs.")
    parser.add_argument(
        "--output-root",
        default="/mnt/data/projects/cs273b/asmani-branch/models/model1",
        help="Root directory containing models/model1/runs.",
    )
    parser.add_argument("--prefix", default=None, help="Only include run directories whose name starts with this prefix.")
    parser.add_argument("--contains", default=None, help="Only include run directories whose name contains this substring.")
    parser.add_argument(
        "--sort",
        default="test_r",
        choices=[
            "test_r",
            "val_r",
            "std_ratio",
            "mae_gain",
            "delta_r",
            "collapsed",
        ],
        help="Column used to sort rows.",
    )
    parser.add_argument("--limit", type=int, default=30, help="Maximum number of rows to print.")
    return parser.parse_args()


def _metric(payload: dict, path: str):
    value = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _to_float(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _fmt(value, digits: int = 3) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "nan"
    return f"{numeric:+.{digits}f}"


def _load_single(run_dir: Path) -> dict | None:
    summary_path = run_dir / "metrics" / "training_summary.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text())
    return {
        "run_name": run_dir.name,
        "kind": "single",
        "n_splits": 1,
        "val_r": _metric(payload, "best.best_val_contact_total_pearson_r"),
        "test_r": _metric(payload, "test_metrics.contact_total.pearson_r"),
        "test_r_std": None,
        "comp_sum_r": _metric(payload, "test_metrics.contact_components_sum.pearson_r"),
        "std_ratio": _metric(payload, "test_metrics.diagnostics.contact_total.pred_target_std_ratio"),
        "mae_gain": _metric(payload, "test_metrics.diagnostics.contact_total_mae_improvement_vs_baseline"),
        "collapsed": _metric(payload, "test_metrics.diagnostics.collapsed_component_channels_below_0_1_ratio"),
        "delta_r": _metric(payload, "test_metrics.delta_total.pearson_r"),
    }


def _load_repeated(run_dir: Path) -> dict | None:
    summary_path = run_dir / "metrics" / "repeated_split_summary.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text())
    aggregate = payload.get("aggregate", {})
    return {
        "run_name": run_dir.name,
        "kind": "repeat",
        "n_splits": int(payload.get("num_splits", 0)),
        "val_r": _metric(aggregate, "best_val_contact_total_pearson_r.mean"),
        "test_r": _metric(aggregate, "test_contact_total_pearson_r.mean"),
        "test_r_std": _metric(aggregate, "test_contact_total_pearson_r.std"),
        "comp_sum_r": _metric(aggregate, "test_contact_components_sum_pearson_r.mean"),
        "std_ratio": _metric(aggregate, "test_contact_total_std_ratio.mean"),
        "mae_gain": _metric(aggregate, "test_contact_total_mae_improvement_vs_baseline.mean"),
        "collapsed": _metric(aggregate, "test_collapsed_component_channels.mean"),
        "delta_r": _metric(aggregate, "test_delta_total_pearson_r.mean"),
    }


def _iter_rows(runs_dir: Path, prefix: str | None, contains: str | None) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if prefix and not run_dir.name.startswith(prefix):
            continue
        if contains and contains not in run_dir.name:
            continue
        row = _load_repeated(run_dir) or _load_single(run_dir)
        if row is not None:
            rows.append(row)
    return rows


def _sort_key(row: dict, field: str) -> tuple[int, float]:
    numeric = _to_float(row.get(field))
    if numeric is None:
        return (1, float("-inf"))
    return (0, numeric)


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.output_root) / "runs"
    rows = _iter_rows(runs_dir, args.prefix, args.contains)
    reverse = args.sort != "collapsed"
    rows.sort(key=lambda row: _sort_key(row, args.sort), reverse=reverse)

    print("run_name\tkind\tn_splits\tval_r\ttest_r\ttest_r_std\tcomp_sum_r\tstd_ratio\tmae_gain\tcollapsed\tdelta_r")
    for row in rows[: args.limit]:
        print(
            "\t".join(
                [
                    str(row["run_name"]),
                    str(row["kind"]),
                    str(row["n_splits"]),
                    _fmt(row["val_r"]),
                    _fmt(row["test_r"]),
                    _fmt(row["test_r_std"]),
                    _fmt(row["comp_sum_r"]),
                    _fmt(row["std_ratio"]),
                    _fmt(row["mae_gain"]),
                    _fmt(row["collapsed"]),
                    _fmt(row["delta_r"]),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
