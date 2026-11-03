#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from plot import apply_plot_styles


EXCLUDE_TOKENS = (
    "smoke",
    "probe",
    "subset10",
    "subset50",
    "subset889",
    "gpu_test",
    "gpu_long",
    "patch_smoke",
    "pair_smoke",
    "verify",
)

FAMILY_ORDER = [
    "m1small",
    "m1grid15",
    "m2reg500",
    "m2arch500",
    "m3scale889",
    "m4repeat5h",
]

FAMILY_LABELS = {
    "m1small": "m1 small",
    "m1grid15": "m1 grid15",
    "m2reg500": "m2 reg500",
    "m2arch500": "m2 arch500",
    "m3scale889": "m3 scale889",
    "m4repeat5h": "m4 repeat5h",
}

FAMILY_COLORS = {
    "m1small": "#4e79a7",
    "m1grid15": "#59a14f",
    "m2reg500": "#f28e2b",
    "m2arch500": "#e15759",
    "m3scale889": "#76b7b2",
    "m4repeat5h": "#b07aa1",
}


@dataclass
class RunRow:
    run_name: str
    family: str
    kind: str
    n_splits: int
    val_r: float | None
    test_r: float | None
    test_r_std: float | None
    contact_mae: float | None
    contact_mae_std: float | None
    std_ratio: float | None
    std_ratio_std: float | None
    collapsed: float | None
    collapsed_std: float | None
    delta_r: float | None
    delta_r_std: float | None
    delta_mae: float | None
    delta_mae_std: float | None
    train_loss: float | None
    val_loss: float | None
    best_epoch: float | None
    num_epochs_completed: float | None
    train_count: float | None
    val_count: float | None
    test_count: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create science+ieee figures summarizing completed model1 runs.")
    parser.add_argument(
        "--output-root",
        default="/mnt/data/projects/cs273b/asmani-branch/models/model1",
        help="Root directory containing the external model1 runs directory.",
    )
    parser.add_argument(
        "--report-dir",
        default="models/model1/reports/model1_run_report_20260521",
        help="Directory inside the repo where figures and CSVs will be written.",
    )
    parser.add_argument(
        "--plot-style",
        default="science+ieee",
        help="Plot style spec passed through the local ./plot scienceplots loader.",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Include smoke/probe/debug subset runs that are excluded by default.",
    )
    return parser.parse_args()


def _metric(payload: dict, path: str):
    value = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _to_float(value) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _safe_mean(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _safe_std(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(xs) < 2:
        return 0.0 if xs else None
    return float(np.std(xs, ddof=0))


def _family(run_name: str) -> str:
    return run_name.split("_")[0]


def _is_debug_run(run_name: str) -> bool:
    lowered = run_name.lower()
    return any(token in lowered for token in EXCLUDE_TOKENS)


def _load_single(run_dir: Path) -> RunRow | None:
    summary_path = run_dir / "metrics" / "training_summary.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text())
    best_epoch_metrics = _metric(payload, "best.best_epoch_metrics") or {}
    train_metrics = best_epoch_metrics.get("train", {})
    val_metrics = best_epoch_metrics.get("val", {})
    return RunRow(
        run_name=run_dir.name,
        family=_family(run_dir.name),
        kind="single",
        n_splits=1,
        val_r=_to_float(_metric(payload, "best.best_val_contact_total_pearson_r")),
        test_r=_to_float(_metric(payload, "test_metrics.contact_total.pearson_r")),
        test_r_std=None,
        contact_mae=_to_float(_metric(payload, "test_metrics.contact_total.mae")),
        contact_mae_std=None,
        std_ratio=_to_float(_metric(payload, "test_metrics.diagnostics.contact_total.pred_target_std_ratio")),
        std_ratio_std=None,
        collapsed=_to_float(_metric(payload, "test_metrics.diagnostics.collapsed_component_channels_below_0_1_ratio")),
        collapsed_std=None,
        delta_r=_to_float(_metric(payload, "test_metrics.delta_total.pearson_r")),
        delta_r_std=None,
        delta_mae=_to_float(_metric(payload, "test_metrics.delta_total.mae")),
        delta_mae_std=None,
        train_loss=_to_float(train_metrics.get("loss")),
        val_loss=_to_float(val_metrics.get("loss")),
        best_epoch=_to_float(_metric(payload, "best.best_epoch")),
        num_epochs_completed=_to_float(payload.get("num_epochs_completed")),
        train_count=_to_float(_metric(payload, "dataset_counts.train")),
        val_count=_to_float(_metric(payload, "dataset_counts.val")),
        test_count=_to_float(_metric(payload, "dataset_counts.test")),
    )


def _load_repeated(run_dir: Path) -> RunRow | None:
    summary_path = run_dir / "metrics" / "repeated_split_summary.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text())
    aggregate = payload.get("aggregate", {})
    split_summaries = []
    for split_dir in sorted((run_dir / "splits").glob("split_*")):
        training_summary_path = split_dir / "metrics" / "training_summary.json"
        if training_summary_path.is_file():
            split_summaries.append(json.loads(training_summary_path.read_text()))
    train_losses = []
    val_losses = []
    best_epochs = []
    num_epochs_completed = []
    for split_payload in split_summaries:
        best_epoch_metrics = _metric(split_payload, "best.best_epoch_metrics") or {}
        train_metrics = best_epoch_metrics.get("train", {})
        val_metrics = best_epoch_metrics.get("val", {})
        train_losses.append(_to_float(train_metrics.get("loss")))
        val_losses.append(_to_float(val_metrics.get("loss")))
        best_epochs.append(_to_float(_metric(split_payload, "best.best_epoch")))
        num_epochs_completed.append(_to_float(split_payload.get("num_epochs_completed")))
    return RunRow(
        run_name=run_dir.name,
        family=_family(run_dir.name),
        kind="repeat",
        n_splits=int(payload.get("num_splits", 0)),
        val_r=_to_float(_metric(aggregate, "best_val_contact_total_pearson_r.mean")),
        test_r=_to_float(_metric(aggregate, "test_contact_total_pearson_r.mean")),
        test_r_std=_to_float(_metric(aggregate, "test_contact_total_pearson_r.std")),
        contact_mae=_to_float(_metric(aggregate, "test_contact_total_mae.mean")),
        contact_mae_std=_to_float(_metric(aggregate, "test_contact_total_mae.std")),
        std_ratio=_to_float(_metric(aggregate, "test_contact_total_std_ratio.mean")),
        std_ratio_std=_to_float(_metric(aggregate, "test_contact_total_std_ratio.std")),
        collapsed=_to_float(_metric(aggregate, "test_collapsed_component_channels.mean")),
        collapsed_std=_to_float(_metric(aggregate, "test_collapsed_component_channels.std")),
        delta_r=_to_float(_metric(aggregate, "test_delta_total_pearson_r.mean")),
        delta_r_std=_to_float(_metric(aggregate, "test_delta_total_pearson_r.std")),
        delta_mae=_to_float(_metric(aggregate, "test_delta_total_mae.mean")),
        delta_mae_std=_to_float(_metric(aggregate, "test_delta_total_mae.std")),
        train_loss=_safe_mean(train_losses),
        val_loss=_safe_mean(val_losses),
        best_epoch=_safe_mean(best_epochs),
        num_epochs_completed=_safe_mean(num_epochs_completed),
        train_count=_to_float(_metric(aggregate, "train_count.mean")),
        val_count=_to_float(_metric(aggregate, "val_count.mean")),
        test_count=_to_float(_metric(aggregate, "test_count.mean")),
    )


def collect_rows(runs_root: Path, include_debug: bool) -> list[RunRow]:
    rows: list[RunRow] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if not include_debug and _is_debug_run(run_dir.name):
            continue
        row = _load_repeated(run_dir) or _load_single(run_dir)
        if row is None:
            continue
        if row.family not in FAMILY_LABELS:
            if not include_debug:
                continue
        rows.append(row)
    return rows


def _short_repeat_label(run_name: str) -> str:
    match = re.match(r"m4repeat5h_([a-z0-9]+)_", run_name)
    base = match.group(1) if match else run_name
    if "_rw0.1_" in run_name:
        return f"{base}+rw"
    return base


def _finite_xy(rows: Iterable[RunRow], x_attr: str, y_attr: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for row in rows:
        x = getattr(row, x_attr)
        y = getattr(row, y_attr)
        if x is None or y is None:
            continue
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            continue
        pairs.append((float(x), float(y)))
    if not pairs:
        return np.array([]), np.array([])
    xs, ys = zip(*pairs)
    return np.asarray(xs), np.asarray(ys)


def _write_csv(rows: list[RunRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RunRow.__dataclass_fields__.keys())
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _scatter_family(ax, rows: list[RunRow], x_attr: str, y_attr: str, *, y_log: bool = False) -> None:
    for family in [fam for fam in FAMILY_ORDER if any(row.family == fam for row in rows)]:
        family_rows = [row for row in rows if row.family == family]
        single_rows = [row for row in family_rows if row.kind == "single"]
        repeat_rows = [row for row in family_rows if row.kind == "repeat"]
        color = FAMILY_COLORS.get(family, "#666666")
        xs, ys = _finite_xy(single_rows, x_attr, y_attr)
        if xs.size:
            ax.scatter(xs, ys, s=16, alpha=0.32, color=color, linewidths=0.0)
        xs, ys = _finite_xy(repeat_rows, x_attr, y_attr)
        if xs.size:
            ax.scatter(xs, ys, s=120, alpha=0.95, color=color, marker="*", edgecolors="black", linewidths=0.6)
    if y_log:
        ax.set_yscale("log")


def _family_boxplot(ax, rows: list[RunRow], attr: str, *, y_log: bool = False, hline: float | None = None) -> None:
    families = [fam for fam in FAMILY_ORDER if any(row.family == fam and getattr(row, attr) is not None for row in rows)]
    data = []
    colors = []
    labels = []
    for family in families:
        xs = [float(getattr(row, attr)) for row in rows if row.family == family and getattr(row, attr) is not None]
        if not xs:
            continue
        data.append(xs)
        colors.append(FAMILY_COLORS.get(family, "#666666"))
        labels.append(f"{FAMILY_LABELS.get(family, family)}\n(n={len(xs)})")
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.0)
    for whisker in bp["whiskers"]:
        whisker.set_color("#555555")
        whisker.set_linewidth(0.8)
    for cap in bp["caps"]:
        cap.set_color("#555555")
        cap.set_linewidth(0.8)
    rng = np.random.default_rng(42)
    for idx, (xs, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.normal(loc=idx, scale=0.06, size=len(xs))
        ax.scatter(jitter, xs, s=12, color=color, alpha=0.45, linewidths=0.0)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if y_log:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(hline, color="#444444", linestyle="--", linewidth=0.9)


def make_overview_figure(rows: list[RunRow], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    _scatter_family(ax, rows, "test_r", "contact_mae")
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Test contact_total Pearson r")
    ax.set_ylabel("Test contact_total MAE (kcal/mol)")
    ax.set_title("All completed runs: contact accuracy")

    best_repeat = max((row for row in rows if row.kind == "repeat" and row.test_r is not None), key=lambda row: row.test_r, default=None)
    if best_repeat is not None and best_repeat.contact_mae is not None:
        ax.annotate(
            f"best repeat mean\n{best_repeat.run_name.split('_s42')[0]}",
            xy=(best_repeat.test_r, best_repeat.contact_mae),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=7,
        )

    ax = axes[0, 1]
    _scatter_family(ax, rows, "val_r", "test_r")
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.9)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
    xs, ys = _finite_xy(rows, "val_r", "test_r")
    if xs.size:
        lo = min(xs.min(), ys.min(), -0.15)
        hi = max(xs.max(), ys.max(), 0.35)
        ax.plot([lo, hi], [lo, hi], color="#999999", linestyle=":", linewidth=0.9)
        corr = float(np.corrcoef(xs, ys)[0, 1]) if xs.size > 1 else float("nan")
        ax.set_title(f"Validation-test alignment is weak (r={corr:.2f})")
    else:
        ax.set_title("Validation-test alignment")
    ax.set_xlabel("Validation contact_total Pearson r")
    ax.set_ylabel("Test contact_total Pearson r")

    ax = axes[1, 0]
    _family_boxplot(ax, rows, "test_r")
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
    ax.set_ylabel("Test contact_total Pearson r")
    ax.set_title("By sweep family: contact correlation")

    ax = axes[1, 1]
    _family_boxplot(ax, rows, "delta_mae")
    ax.set_ylabel("Test delta_total MAE (kcal/mol)")
    ax.set_title("By sweep family: whole-complex error")

    handles = []
    labels = []
    for family in [fam for fam in FAMILY_ORDER if any(row.family == fam for row in rows)]:
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=FAMILY_COLORS[family], markersize=6))
        labels.append(FAMILY_LABELS[family])
    handles.append(plt.Line2D([0], [0], marker="*", linestyle="", color="black", markersize=9))
    labels.append("repeated-split aggregate")
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_repeat_breakdown(rows: list[RunRow], out_path: Path) -> None:
    repeat_rows = [row for row in rows if row.family == "m4repeat5h" and row.kind == "repeat"]
    repeat_rows.sort(key=lambda row: (row.test_r is None, -(row.test_r or -999.0)))
    labels = [_short_repeat_label(row.run_name) for row in repeat_rows]
    x = np.arange(len(repeat_rows))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    ys = [row.test_r for row in repeat_rows]
    yerr = [row.test_r_std or 0.0 for row in repeat_rows]
    ax.errorbar(x, ys, yerr=yerr, fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=3, linewidth=1.2)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
    ax.set_ylabel("Test contact_total Pearson r")
    ax.set_title("Repeated sweep: contact correlation")

    ax = axes[0, 1]
    ys = [row.contact_mae for row in repeat_rows]
    yerr = [row.contact_mae_std or 0.0 for row in repeat_rows]
    ax.errorbar(x, ys, yerr=yerr, fmt="o", color="#e15759", ecolor="#e15759", capsize=3, linewidth=1.2)
    ax.set_ylabel("Test contact_total MAE (kcal/mol)")
    ax.set_title("Repeated sweep: contact MAE")

    ax = axes[1, 0]
    ys = [row.delta_mae for row in repeat_rows]
    yerr = [row.delta_mae_std or 0.0 for row in repeat_rows]
    ax.errorbar(x, ys, yerr=yerr, fmt="o", color="#59a14f", ecolor="#59a14f", capsize=3, linewidth=1.2)
    ax.set_ylabel("Test delta_total MAE (kcal/mol)")
    ax.set_title("Repeated sweep: whole-complex error")

    ax = axes[1, 1]
    ys = [row.std_ratio for row in repeat_rows]
    yerr = [row.std_ratio_std or 0.0 for row in repeat_rows]
    ax.errorbar(x, ys, yerr=yerr, fmt="o", color="#b07aa1", ecolor="#b07aa1", capsize=3, linewidth=1.2)
    ax.axhline(0.1, color="#666666", linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("Test contact_total std ratio")
    ax.set_title("Repeated sweep: collapse diagnostic")
    for xi, row in zip(x, repeat_rows):
        if row.collapsed is not None:
            ax.annotate(f"{row.collapsed:.2f} ch", (xi, row.std_ratio), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_family_summary(rows: list[RunRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "n_runs",
        "mean_test_contact_r",
        "mean_test_contact_mae",
        "mean_test_std_ratio",
        "mean_test_delta_r",
        "mean_test_delta_mae",
        "mean_train_loss",
        "mean_val_loss",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for family in [fam for fam in FAMILY_ORDER if any(row.family == fam for row in rows)]:
            family_rows = [row for row in rows if row.family == family]
            writer.writerow(
                {
                    "family": family,
                    "n_runs": len(family_rows),
                    "mean_test_contact_r": _safe_mean(row.test_r for row in family_rows),
                    "mean_test_contact_mae": _safe_mean(row.contact_mae for row in family_rows),
                    "mean_test_std_ratio": _safe_mean(row.std_ratio for row in family_rows),
                    "mean_test_delta_r": _safe_mean(row.delta_r for row in family_rows),
                    "mean_test_delta_mae": _safe_mean(row.delta_mae for row in family_rows),
                    "mean_train_loss": _safe_mean(row.train_loss for row in family_rows),
                    "mean_val_loss": _safe_mean(row.val_loss for row in family_rows),
                }
            )


def main() -> int:
    args = parse_args()
    apply_plot_styles(args.plot_style)

    runs_root = Path(args.output_root) / "runs"
    report_dir = Path(args.report_dir)
    rows = collect_rows(runs_root, include_debug=args.include_debug)

    if not rows:
        raise SystemExit("No completed runs found.")

    rows.sort(key=lambda row: (FAMILY_ORDER.index(row.family) if row.family in FAMILY_ORDER else 999, row.run_name))
    _write_csv(rows, report_dir / "completed_runs_summary.csv")
    _write_family_summary(rows, report_dir / "family_summary.csv")
    make_overview_figure(rows, report_dir / "completed_runs_overview.png")
    make_repeat_breakdown(rows, report_dir / "m4repeat5h_breakdown.png")
    print(f"Wrote report artifacts to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
