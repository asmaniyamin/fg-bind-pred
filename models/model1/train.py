# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile

import torch

from dataloader.torch_adapter import make_dataloader
from plot import apply_plot_styles, available_plot_styles, register_scienceplots

from .config import DataConfig, ModelConfig, TrainingConfig
from .data.dataset import PerFGDataset
from .data.label_builder import build_targets_from_batch
from .model.model import PerFGModel
from .training.loss import MultiTaskUncertaintyLoss
from .training.trainer import Trainer

DEFAULT_TASK_NAMES = ["contact_total", "vdw", "elec", "polar", "nonpolar", "preorg", "delta_total"]


def parse_args() -> argparse.Namespace:
    register_scienceplots()
    model_defaults = ModelConfig()
    training_defaults = TrainingConfig()
    data_defaults = DataConfig(data_root="dataloader/_export_check2")
    parser = argparse.ArgumentParser(description="Train the model1 per-FG model")
    parser.add_argument("--data-root", default=data_defaults.data_root)
    parser.add_argument("--cache-dir", default=data_defaults.cache_dir)
    parser.add_argument("--case-root", action="append", default=list(data_defaults.case_roots))
    parser.add_argument(
        "--output-root",
        default="outputs/model1",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=training_defaults.batch_size)
    parser.add_argument("--epochs", "--num-epochs", dest="epochs", type=int, default=training_defaults.num_epochs)
    parser.add_argument("--lr", type=float, default=training_defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=training_defaults.weight_decay)
    parser.add_argument("--warmup-steps", type=int, default=training_defaults.warmup_steps)
    parser.add_argument("--grad-clip", type=float, default=training_defaults.grad_clip)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=training_defaults.gradient_accumulation_steps)
    parser.add_argument(
        "--scheduler-type",
        choices=["cosine_warmup", "constant_warmup", "plateau", "none"],
        default=training_defaults.scheduler_type,
    )
    parser.add_argument("--min-lr", type=float, default=training_defaults.min_lr)
    parser.add_argument("--plateau-factor", type=float, default=training_defaults.plateau_factor)
    parser.add_argument("--plateau-patience", type=int, default=training_defaults.plateau_patience)
    parser.add_argument("--plateau-threshold", type=float, default=training_defaults.plateau_threshold)
    parser.add_argument("--early-stopping-metric", default=training_defaults.early_stopping_metric)
    parser.add_argument(
        "--early-stopping-mode",
        choices=["max", "min"],
        default=training_defaults.early_stopping_mode,
    )
    parser.add_argument(
        "--freeze-module",
        action="append",
        default=list(training_defaults.freeze_modules),
        choices=["encoders", "ligand_gnn", "fg_pool", "fg_context", "cross_attention", "contact_head", "preorg_head", "delta_head"],
    )
    parser.add_argument("--freeze-epochs", type=int, default=training_defaults.freeze_epochs)
    parser.add_argument("--print-batch-every", type=int, default=training_defaults.print_batch_every)
    parser.add_argument(
        "--loss-type",
        choices=["huber", "mse", "mae", "log_cosh", "charbonnier", "tukey", "quantile", "ccc"],
        default="huber",
    )
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--tukey-c", type=float, default=4.685)
    parser.add_argument("--quantile-q", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-loss-margin", type=float, default=0.0)
    parser.add_argument(
        "--variance-penalty-weight",
        type=float,
        default=0.0,
        help="Anti-collapse penalty. If >0, adds -w*log(pred.std+eps) per contact channel "
             "to push the model away from constant predictions.",
    )
    parser.add_argument(
        "--train-log-vars",
        action="store_true",
        default=False,
        help="If set, include MultiTaskUncertaintyLoss.log_vars in the AdamW optimizer "
             "(Kendall σ-learning). OFF by default — empirically destabilizes training in "
             "this setup because the optimizer harvests negative log_var rewards faster "
             "than the model learns the underlying task.",
    )
    parser.add_argument("--delta-task-weight", type=float, default=0.25)
    parser.add_argument("--contact-target-clip", type=float, default=0.0)
    parser.add_argument("--delta-target-clip", type=float, default=0.0)
    parser.add_argument(
        "--task-mode",
        choices=["auto", "full", "contact_only", "contact_delta"],
        default="auto",
        help="Which loss tasks to optimize. 'auto' drops degenerate train-set tasks.",
    )
    parser.add_argument(
        "--degenerate-std-threshold",
        type=float,
        default=1e-4,
        help="Drop non-contact tasks in auto mode when train-set target std is below this threshold.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=training_defaults.early_stopping_patience,
    )
    parser.add_argument("--num-workers", type=int, default=training_defaults.num_workers)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=training_defaults.pin_memory,
    )
    parser.add_argument("--hidden-dim", type=int, default=model_defaults.hidden_dim)
    parser.add_argument("--ligand-edge-dim", type=int, default=model_defaults.ligand_edge_dim)
    parser.add_argument("--fg-edge-dim", type=int, default=model_defaults.fg_edge_dim)
    parser.add_argument("--num-ligand-gnn-layers", type=int, default=model_defaults.num_ligand_gnn_layers)
    parser.add_argument("--num-fg-attn-layers", type=int, default=model_defaults.num_fg_attn_layers)
    parser.add_argument("--num-fg-global-attn-layers", type=int, default=model_defaults.num_fg_global_attn_layers)
    parser.add_argument("--num-cross-attn-layers", type=int, default=model_defaults.num_cross_attn_layers)
    parser.add_argument("--num-heads", type=int, default=model_defaults.num_heads)
    parser.add_argument("--num-fg-graph-heads", type=int, default=model_defaults.num_fg_graph_heads)
    parser.add_argument("--num-cross-attn-heads", type=int, default=model_defaults.num_cross_attn_heads)
    parser.add_argument("--dropout", type=float, default=model_defaults.dropout)
    parser.add_argument("--aev-rc", type=float, default=model_defaults.aev_rc)
    parser.add_argument("--aev-eta", type=float, default=model_defaults.aev_eta)
    parser.add_argument("--aev-num-rs", type=int, default=model_defaults.aev_num_rs)
    parser.add_argument("--aev-num-ecif-types", type=int, default=model_defaults.aev_num_ecif_types)
    parser.add_argument("--aev-dim", type=int, default=model_defaults.aev_dim)
    parser.add_argument("--use-aev", action=argparse.BooleanOptionalAction, default=model_defaults.use_aev)
    parser.add_argument("--fg-edge-cutoff", type=float, default=model_defaults.fg_edge_cutoff)
    parser.add_argument("--cross-attn-cutoff", type=float, default=model_defaults.cross_attn_cutoff)
    parser.add_argument(
        "--use-cross-attention",
        action=argparse.BooleanOptionalAction,
        default=model_defaults.use_cross_attention,
    )
    parser.add_argument("--num-rbf-kernels", type=int, default=model_defaults.num_rbf_kernels)
    parser.add_argument("--rbf-d-min", type=float, default=model_defaults.rbf_d_min)
    parser.add_argument("--rbf-d-max", type=float, default=model_defaults.rbf_d_max)
    parser.add_argument("--split-type", choices=["pocket_hash", "ligand"], default=data_defaults.split_type)
    parser.add_argument("--val-frac", type=float, default=data_defaults.val_frac)
    parser.add_argument("--test-frac", type=float, default=data_defaults.test_frac)
    parser.add_argument("--seed", type=int, default=data_defaults.seed)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed used for grouped train/val/test splitting. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--repeat-splits",
        type=int,
        default=1,
        help="If >1, train the same configuration across repeated grouped holdout splits and aggregate the results.",
    )
    parser.add_argument(
        "--split-seed-stride",
        type=int,
        default=1,
        help="Increment added to the split seed between repeated grouped holdout runs.",
    )
    parser.add_argument("--min-fg-count", type=int, default=data_defaults.min_fg_count)
    parser.add_argument(
        "--exclude-zero-fg",
        action=argparse.BooleanOptionalAction,
        default=data_defaults.exclude_zero_fg,
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=data_defaults.max_cases,
        help="If >0, cap discovered PDB IDs to this many before splitting.",
    )
    parser.add_argument(
        "--max-abs-delta-total",
        type=float,
        default=training_defaults.max_abs_delta_total,
        help="If >0, ignore delta_total supervision/metrics for complexes whose absolute DELTA TOTAL exceeds this threshold.",
    )
    parser.add_argument(
        "--require-baseline",
        action=argparse.BooleanOptionalAction,
        default=data_defaults.require_baseline,
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--plot-style",
        default="science+ieee",
        help=(
            "Professional plot style spec, e.g. 'science+ieee', 'science', "
            "'science+nature', 'science+scatter', or comma-separated style names."
        ),
    )
    parser.add_argument(
        "--list-plot-styles",
        action="store_true",
        help="List vendored professional plotting styles and exit.",
    )
    return parser.parse_args()


def build_dataloaders(
    data_config: DataConfig,
    training_config: TrainingConfig,
):
    train_dataset = PerFGDataset(data_config.data_root, data_config, split="train")
    val_dataset = PerFGDataset(data_config.data_root, data_config, split="val")
    test_dataset = PerFGDataset(data_config.data_root, data_config, split="test")
    loader_kwargs = {
        "batch_size": training_config.batch_size,
        "num_workers": training_config.num_workers,
        "pin_memory": training_config.pin_memory,
        "collate": "dict",
    }
    return (
        make_dataloader(train_dataset, shuffle=True, **loader_kwargs),
        make_dataloader(val_dataset, shuffle=False, **loader_kwargs),
        make_dataloader(test_dataset, shuffle=False, **loader_kwargs),
    )


def build_model(config: ModelConfig) -> PerFGModel:
    return PerFGModel(config)


def build_optimizer(model: PerFGModel, config: TrainingConfig, loss_fn: torch.nn.Module | None = None):
    params = list(model.parameters())
    if loss_fn is not None:
        params += [p for p in loss_fn.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(optimizer, config: TrainingConfig, num_training_steps: int):
    scheduler_type = getattr(config, "scheduler_type", "cosine_warmup")
    if scheduler_type == "none":
        config.scheduler_step_on = "batch"
        return None
    if scheduler_type == "plateau":
        config.scheduler_step_on = "epoch"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            threshold=config.plateau_threshold,
            min_lr=config.min_lr,
        )
    warmup = max(int(config.warmup_steps), 1)
    config.scheduler_step_on = "batch"

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        if scheduler_type == "constant_warmup":
            return 1.0
        progress = (step - warmup) / float(max(num_training_steps - warmup, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return max(cosine, config.min_lr / max(config.lr, 1e-12))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_split_seed(args: argparse.Namespace) -> int:
    return int(args.split_seed if args.split_seed is not None else args.seed)


def _split_seed_sequence(args: argparse.Namespace) -> list[int]:
    base = _resolve_split_seed(args)
    repeats = max(int(args.repeat_splits), 1)
    stride = int(args.split_seed_stride)
    return [base + idx * stride for idx in range(repeats)]


def _task_target_and_mask(task_name: str, targets: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]):
    if task_name == "contact_total":
        return targets["contact_total"], masks["contact_total"]
    if task_name == "vdw":
        return targets["contact"][:, 0], masks["contact"]
    if task_name == "elec":
        return targets["contact"][:, 1], masks["contact"]
    if task_name == "polar":
        return targets["contact"][:, 2], masks["contact"]
    if task_name == "nonpolar":
        return targets["contact"][:, 3], masks["contact"]
    if task_name == "preorg":
        return targets["preorg"].squeeze(-1), masks["preorg"]
    if task_name == "aux_elec":
        return targets["aux_elec"], masks["aux_elec"]
    if task_name == "aux_vdw":
        return targets["aux_vdw"], masks["aux_vdw"]
    if task_name == "delta_total":
        return targets["delta_total"], masks["delta_total"]
    raise KeyError(task_name)


def estimate_task_statistics(
    loader,
    cutoff: float,
    task_names: list[str],
    *,
    max_abs_delta_total: float = 0.0,
) -> dict[str, dict[str, float]]:
    values: dict[str, list[torch.Tensor]] = {name: [] for name in task_names}
    counts = {name: 0 for name in task_names}
    for batch in loader:
        targets, masks = build_targets_from_batch(batch, cutoff, max_abs_delta_total=max_abs_delta_total)
        for task_name in task_names:
            target, mask = _task_target_and_mask(task_name, targets, masks)
            mask = mask.bool()
            if not mask.any():
                continue
            selected = target[mask]
            if selected.numel() == 0:
                continue
            values[task_name].append(selected.detach().cpu().reshape(-1))
            counts[task_name] += int(selected.numel())
    stats: dict[str, dict[str, float]] = {}
    for task_name in task_names:
        if not values[task_name]:
            stats[task_name] = {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "absmean": 0.0,
                "median": 0.0,
                "robust_std": 0.0,
            }
            continue
        x = torch.cat(values[task_name])
        median = float(x.median().item())
        mad = float((x - median).abs().median().item())
        stats[task_name] = {
            "count": int(x.numel()),
            "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=False).item()),
            "absmean": float(x.abs().mean().item()),
            "median": median,
            "robust_std": 1.4826 * mad,
        }
    return stats


def select_task_setup(
    task_mode: str,
    task_stats: dict[str, dict[str, float]],
    degenerate_std_threshold: float,
    delta_task_weight: float,
) -> tuple[list[str], dict[str, float], dict[str, float], dict[str, float]]:
    component_tasks = ["vdw", "elec", "polar", "nonpolar"]
    contact_tasks = ["contact_total", *component_tasks]
    if task_mode == "contact_only":
        active = list(contact_tasks)
    elif task_mode == "contact_delta":
        active = contact_tasks + ["delta_total"]
    elif task_mode == "full":
        active = contact_tasks + ["preorg", "delta_total"]
    elif task_mode == "auto":
        active = list(contact_tasks)
        for task_name in ["preorg", "delta_total"]:
            stats = task_stats.get(task_name, {})
            count = int(stats.get("count", 0))
            std = float(stats.get("std", 0.0))
            if count > 0 and std > degenerate_std_threshold:
                active.append(task_name)
    else:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    task_scales = {}
    task_static_weights = {}
    task_target_clips: dict[str, float] = {}
    for task_name in active:
        stats = task_stats.get(task_name, {})
        std = float(stats.get("std", 0.0))
        robust_std = float(stats.get("robust_std", 0.0))
        if task_name == "delta_total":
            scale = robust_std if robust_std > 0.0 else std
        else:
            scale = std
        scale = max(scale, 1e-3)
        task_scales[task_name] = scale
        if task_name == "delta_total":
            task_static_weights[task_name] = delta_task_weight
        elif task_name in component_tasks:
            task_static_weights[task_name] = 0.25
        else:
            task_static_weights[task_name] = 1.0
    return active, task_scales, task_static_weights, task_target_clips


def _default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _flatten_metrics(prefix: str, payload: dict, out: dict[str, float]) -> None:
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_metrics(name, value, out)
        elif isinstance(value, (int, float, bool)):
            out[name] = value


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True))


def _write_rows_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _history_rows(history: list[dict]) -> tuple[list[dict[str, float]], list[str]]:
    rows: list[dict[str, float]] = []
    dynamic_headers: set[str] = {"epoch"}
    for item in history:
        row: dict[str, float] = {"epoch": item["epoch"]}
        _flatten_metrics("train", item.get("train", {}), row)
        _flatten_metrics("val", item.get("val", {}), row)
        rows.append(row)
        dynamic_headers.update(row.keys())
    preferred = [
        "epoch",
        "train_loss",
        "train_lr",
        "val_contact_total_pearson_r",
        "val_contact_total_spearman_r",
        "val_contact_total_mae",
        "val_delta_total_mae",
        "val_fg_rank_correlation",
    ]
    remaining = sorted(header for header in dynamic_headers if header not in preferred)
    headers = [header for header in preferred if header in dynamic_headers] + remaining
    return rows, headers


def _write_history_csv(path: Path, history: list[dict]) -> None:
    rows, headers = _history_rows(history)
    _write_rows_csv(path, rows, headers)


def _safe_metric_value(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _json_sanitize(payload):
    if isinstance(payload, dict):
        return {key: _json_sanitize(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_sanitize(value) for value in payload]
    return _safe_metric_value(payload)


def _best_epoch_summary(history: list[dict]) -> dict:
    best_epoch = None
    best_score = float("-inf")
    for item in history:
        score = item.get("val", {}).get("contact_total", {}).get("pearson_r", float("nan"))
        if isinstance(score, (int, float)) and score == score and score > best_score:
            best_score = float(score)
            best_epoch = item
    return {
        "best_val_contact_total_pearson_r": _safe_metric_value(best_score if best_epoch is not None else float("nan")),
        "best_epoch": best_epoch["epoch"] if best_epoch is not None else None,
        "best_epoch_metrics": _json_sanitize(best_epoch) if best_epoch is not None else None,
    }


def _write_training_summary(
    path: Path,
    history: list[dict],
    test_metrics: dict,
    run_name: str,
    train_count: int,
    val_count: int,
    test_count: int,
    test_checkpoint: dict | None = None,
) -> None:
    payload = {
        "run_name": run_name,
        "num_epochs_completed": len(history),
        "dataset_counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "final_epoch": _json_sanitize(history[-1]) if history else None,
        "best": _best_epoch_summary(history),
        "test_metrics": _json_sanitize(test_metrics),
        "test_checkpoint": _json_sanitize(test_checkpoint) if test_checkpoint is not None else None,
    }
    _write_json(path, payload)
    return payload


def _metric_from_path(payload: dict, path: str):
    value = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _aggregate_numeric(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = statistics.mean(finite)
    std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    return {
        "count": len(finite),
        "mean": mean,
        "std": std,
        "min": min(finite),
        "max": max(finite),
    }


def _split_summary_row(summary: dict, split_seed: int, repeat_index: int, run_dir: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "repeat_index": repeat_index,
        "split_seed": split_seed,
        "run_name": summary.get("run_name"),
        "run_dir": str(run_dir),
        "num_epochs_completed": summary.get("num_epochs_completed"),
        "best_epoch": _metric_from_path(summary, "best.best_epoch"),
        "train_count": _metric_from_path(summary, "dataset_counts.train"),
        "val_count": _metric_from_path(summary, "dataset_counts.val"),
        "test_count": _metric_from_path(summary, "dataset_counts.test"),
        "best_val_contact_total_pearson_r": _metric_from_path(summary, "best.best_val_contact_total_pearson_r"),
        "best_val_contact_total_std_ratio": _metric_from_path(
            summary,
            "best.best_epoch_metrics.val.diagnostics.contact_total.pred_target_std_ratio",
        ),
        "test_contact_total_pearson_r": _metric_from_path(summary, "test_metrics.contact_total.pearson_r"),
        "test_contact_total_mae": _metric_from_path(summary, "test_metrics.contact_total.mae"),
        "test_contact_components_sum_pearson_r": _metric_from_path(
            summary,
            "test_metrics.contact_components_sum.pearson_r",
        ),
        "test_contact_total_std_ratio": _metric_from_path(
            summary,
            "test_metrics.diagnostics.contact_total.pred_target_std_ratio",
        ),
        "test_contact_components_sum_std_ratio": _metric_from_path(
            summary,
            "test_metrics.diagnostics.contact_components_sum.pred_target_std_ratio",
        ),
        "test_contact_total_mae_improvement_vs_baseline": _metric_from_path(
            summary,
            "test_metrics.diagnostics.contact_total_mae_improvement_vs_baseline",
        ),
        "test_collapsed_component_channels": _metric_from_path(
            summary,
            "test_metrics.diagnostics.collapsed_component_channels_below_0_1_ratio",
        ),
        "test_delta_total_pearson_r": _metric_from_path(summary, "test_metrics.delta_total.pearson_r"),
        "test_delta_total_mae": _metric_from_path(summary, "test_metrics.delta_total.mae"),
        "test_delta_total_mae_improvement_vs_baseline": _metric_from_path(
            summary,
            "test_metrics.diagnostics.delta_total_mae_improvement_vs_baseline",
        ),
    }
    return row


def _aggregate_split_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int | None]]:
    metric_keys = [
        "best_val_contact_total_pearson_r",
        "best_val_contact_total_std_ratio",
        "test_contact_total_pearson_r",
        "test_contact_total_mae",
        "test_contact_components_sum_pearson_r",
        "test_contact_total_std_ratio",
        "test_contact_components_sum_std_ratio",
        "test_contact_total_mae_improvement_vs_baseline",
        "test_collapsed_component_channels",
        "test_delta_total_pearson_r",
        "test_delta_total_mae",
        "test_delta_total_mae_improvement_vs_baseline",
        "num_epochs_completed",
        "best_epoch",
        "train_count",
        "val_count",
        "test_count",
    ]
    return {
        key: _aggregate_numeric([row.get(key) for row in rows])  # type: ignore[arg-type]
        for key in metric_keys
    }


def _extract_series(rows: list[dict[str, float]], key: str) -> list[float]:
    series: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            series.append(float(value))
        elif isinstance(value, (int, float)):
            series.append(float(value))
        else:
            series.append(float("nan"))
    return series


def _plot_history_graphs(metrics_dir: Path, history: list[dict], fg_rows: list[dict], complex_rows: list[dict]) -> None:
    if not history:
        return
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="model1_mpl_"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = metrics_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    applied_styles = apply_plot_styles(getattr(_plot_history_graphs, "_style_spec", "science+ieee"))

    rows, _headers = _history_rows(history)
    epochs = _extract_series(rows, "epoch")

    def save_line_plot(path: Path, title: str, y_label: str, series_map: dict[str, list[float]]) -> None:
        plt.figure(figsize=(8, 5))
        plotted = 0
        for label, values in series_map.items():
            if any(value == value for value in values):
                plt.plot(epochs, values, marker="o", label=label)
                plotted += 1
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        if plotted > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    save_line_plot(
        plots_dir / "loss_curve.png",
        "Training Loss",
        "Loss",
        {"train_loss": _extract_series(rows, "train_loss")},
    )
    save_line_plot(
        plots_dir / "learning_rate_curve.png",
        "Learning Rate",
        "LR",
        {"train_lr": _extract_series(rows, "train_lr")},
    )
    save_line_plot(
        plots_dir / "val_contact_correlation.png",
        "Validation Contact Correlation",
        "Correlation",
        {
            "pearson_r": _extract_series(rows, "val_contact_total_pearson_r"),
            "spearman_r": _extract_series(rows, "val_contact_total_spearman_r"),
        },
    )
    save_line_plot(
        plots_dir / "val_contact_mae.png",
        "Validation Contact MAE",
        "MAE",
        {"contact_total_mae": _extract_series(rows, "val_contact_total_mae")},
    )

    train_task_keys = sorted(
        key for key in rows[0].keys() if key.startswith("train_loss_") and key not in {"train_loss"}
    )
    if train_task_keys:
        save_line_plot(
            plots_dir / "train_task_losses.png",
            "Per-Task Training Losses",
            "Loss",
            {key.removeprefix("train_loss_"): _extract_series(rows, key) for key in train_task_keys},
        )

    def save_scatter(path: Path, title: str, x_label: str, y_label: str, xs: list[float], ys: list[float]) -> None:
        valid = [
            (x, y)
            for x, y in zip(xs, ys)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x == x and y == y
        ]
        if not valid:
            return
        xs_valid = [point[0] for point in valid]
        ys_valid = [point[1] for point in valid]
        plt.figure(figsize=(6, 6))
        plt.scatter(xs_valid, ys_valid, alpha=0.7)
        min_val = min(xs_valid + ys_valid)
        max_val = max(xs_valid + ys_valid)
        if min_val != max_val:
            plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    save_scatter(
        plots_dir / "test_fg_contact_total_pred_vs_target.png",
        "Test FG Contact Total Prediction vs Target",
        "Target Contact Total",
        "Predicted Contact Total",
        [row["target_contact_total"] for row in fg_rows if row.get("mask_contact_total")],
        [row["pred_contact_total"] for row in fg_rows if row.get("mask_contact_total")],
    )
    save_scatter(
        plots_dir / "test_fg_total_pred_vs_target.png",
        "Test FG Total Prediction vs Target",
        "Target FG Total",
        "Predicted FG Total",
        [
            (
                row["target_vdw"]
                + row["target_elec"]
                + row["target_polar"]
                + row["target_nonpolar"]
                + row["target_preorg"]
            )
            for row in fg_rows
            if row.get("mask_contact") and row.get("mask_preorg")
        ],
        [row["pred_total"] for row in fg_rows if row.get("mask_contact") and row.get("mask_preorg")],
    )
    save_scatter(
        plots_dir / "test_complex_delta_total_pred_vs_target.png",
        "Test Complex Delta Total Prediction vs Target",
        "Target Delta Total",
        "Predicted Delta Total",
        [row["target_delta_total"] for row in complex_rows if row.get("mask_delta_total")],
        [row["pred_delta_total"] for row in complex_rows if row.get("mask_delta_total")],
    )
    _write_json(
        metrics_dir / "plot_config.json",
        {
            "requested_plot_style": getattr(_plot_history_graphs, "_style_spec", "science+ieee"),
            "applied_plot_styles": applied_styles,
        },
    )
    diag_ratio_key = "val_diagnostics_contact_total_pred_target_std_ratio"
    if diag_ratio_key in rows[0]:
        save_line_plot(
            plots_dir / "val_contact_std_ratio.png",
            "Validation Contact Prediction Std Ratio",
            "Pred/Target Std Ratio",
            {
                "contact_total": _extract_series(rows, diag_ratio_key),
                "components_sum": _extract_series(rows, "val_diagnostics_contact_components_sum_pred_target_std_ratio"),
            },
        )


def _build_configs(
    args: argparse.Namespace,
    *,
    split_seed: int,
) -> tuple[ModelConfig, TrainingConfig, DataConfig]:
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        ligand_edge_dim=args.ligand_edge_dim,
        fg_edge_dim=args.fg_edge_dim,
        num_ligand_gnn_layers=args.num_ligand_gnn_layers,
        num_fg_attn_layers=args.num_fg_attn_layers,
        num_fg_global_attn_layers=args.num_fg_global_attn_layers,
        num_cross_attn_layers=args.num_cross_attn_layers,
        num_heads=args.num_heads,
        num_fg_graph_heads=args.num_fg_graph_heads,
        num_cross_attn_heads=args.num_cross_attn_heads,
        dropout=args.dropout,
        aev_rc=args.aev_rc,
        aev_eta=args.aev_eta,
        aev_num_rs=args.aev_num_rs,
        aev_num_ecif_types=args.aev_num_ecif_types,
        aev_dim=args.aev_dim,
        fg_edge_cutoff=args.fg_edge_cutoff,
        cross_attn_cutoff=args.cross_attn_cutoff,
        num_rbf_kernels=args.num_rbf_kernels,
        rbf_d_min=args.rbf_d_min,
        rbf_d_max=args.rbf_d_max,
        use_aev=args.use_aev,
        use_cross_attention=args.use_cross_attention,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        scheduler_type=args.scheduler_type,
        min_lr=args.min_lr,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_threshold=args.plateau_threshold,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_mode=args.early_stopping_mode,
        freeze_modules=tuple(args.freeze_module),
        freeze_epochs=args.freeze_epochs,
        print_batch_every=args.print_batch_every,
        max_abs_delta_total=args.max_abs_delta_total,
    )
    data_config = DataConfig(
        data_root=args.data_root,
        aev_rc=args.aev_rc,
        aev_eta=args.aev_eta,
        fg_edge_cutoff=args.fg_edge_cutoff,
        cross_attn_cutoff=args.cross_attn_cutoff,
        split_type=args.split_type,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        split_seed=split_seed,
        cache_dir=args.cache_dir,
        case_roots=tuple(args.case_root),
        require_baseline=args.require_baseline,
        min_fg_count=args.min_fg_count,
        exclude_zero_fg=args.exclude_zero_fg,
        max_cases=args.max_cases,
    )
    return model_config, training_config, data_config


def _run_single_training(
    args: argparse.Namespace,
    *,
    run_name: str,
    run_dir: Path,
    split_seed: int,
    repeat_index: int | None = None,
) -> dict[str, object]:
    set_global_seed(int(args.seed))
    model_config, training_config, data_config = _build_configs(args, split_seed=split_seed)
    train_loader, val_loader, test_loader = build_dataloaders(data_config, training_config)
    task_stats = estimate_task_statistics(
        train_loader,
        model_config.cross_attn_cutoff,
        DEFAULT_TASK_NAMES,
        max_abs_delta_total=training_config.max_abs_delta_total,
    )
    active_tasks, task_scales, task_static_weights, task_target_clips = select_task_setup(
        args.task_mode,
        task_stats,
        args.degenerate_std_threshold,
        args.delta_task_weight,
    )
    for task_name in ["vdw", "elec", "polar", "nonpolar"]:
        if task_name in active_tasks and args.contact_target_clip > 0.0:
            task_target_clips[task_name] = args.contact_target_clip
    if "delta_total" in active_tasks and args.delta_target_clip > 0.0:
        task_target_clips["delta_total"] = args.delta_target_clip
    print(
        "task_setup "
        f"run={run_name} "
        f"split_seed={split_seed} "
        f"mode={args.task_mode} "
        f"active={','.join(active_tasks)} "
        f"scales={json.dumps(task_scales, sort_keys=True)} "
        f"clips={json.dumps(task_target_clips, sort_keys=True)}",
        flush=True,
    )
    model = build_model(model_config)
    loss_fn = MultiTaskUncertaintyLoss(
        active_tasks,
        task_scales=task_scales,
        task_static_weights=task_static_weights,
        task_target_clips=task_target_clips,
        loss_type=args.loss_type,
        huber_beta=args.huber_beta,
        charbonnier_eps=args.charbonnier_eps,
        tukey_c=args.tukey_c,
        quantile_q=args.quantile_q,
        rank_loss_weight=args.rank_loss_weight,
        rank_loss_margin=args.rank_loss_margin,
        variance_penalty_weight=args.variance_penalty_weight,
    )
    optimizer = build_optimizer(model, training_config, loss_fn=loss_fn if args.train_log_vars else None)
    total_steps = max(len(train_loader) * training_config.num_epochs, 1)
    scheduler = build_scheduler(optimizer, training_config, total_steps)
    baseline_means = {task_name: float(stats.get("mean", 0.0)) for task_name, stats in task_stats.items()}
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        loss_fn,
        training_config,
        args.device,
        baseline_means=baseline_means,
    )

    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    manifest_dir = run_dir / "manifests"
    for directory in (run_dir, checkpoint_dir, metrics_dir, predictions_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_json(
        manifest_dir / "run_config.json",
        {
            "args": vars(args),
            "resolved": {
                "run_name": run_name,
                "split_seed": split_seed,
                "repeat_index": repeat_index,
                "model_seed": int(args.seed),
            },
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "data_config": asdict(data_config),
            "task_setup": {
                "task_mode": args.task_mode,
                "active_tasks": active_tasks,
                "task_scales": task_scales,
                "task_static_weights": task_static_weights,
                "task_target_clips": task_target_clips,
                "loss_type": args.loss_type,
                "huber_beta": args.huber_beta,
                "charbonnier_eps": args.charbonnier_eps,
                "tukey_c": args.tukey_c,
                "quantile_q": args.quantile_q,
                "variance_penalty_weight": args.variance_penalty_weight,
                "degenerate_std_threshold": args.degenerate_std_threshold,
                "task_stats": task_stats,
                "baseline_means": baseline_means,
            },
        },
    )
    _write_json(
        manifest_dir / "splits.json",
        {
            "train_ids": train_loader.dataset.pdb_ids,
            "val_ids": val_loader.dataset.pdb_ids,
            "test_ids": test_loader.dataset.pdb_ids,
            "counts": {
                "train": len(train_loader.dataset),
                "val": len(val_loader.dataset),
                "test": len(test_loader.dataset),
            },
        },
    )

    history = trainer.train(train_loader, val_loader, training_config.num_epochs, checkpoint_dir=checkpoint_dir)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    test_checkpoint = {
        "path": str(best_checkpoint_path),
        "source": "best" if best_checkpoint_path.is_file() else "current",
        "epoch": None,
        "metrics": None,
    }
    if best_checkpoint_path.is_file():
        best_epoch, best_metrics = trainer.load_checkpoint(str(best_checkpoint_path))
        test_checkpoint["epoch"] = best_epoch
        test_checkpoint["metrics"] = best_metrics
    test_metrics = trainer.evaluate(test_loader, split="test")
    fg_rows, complex_rows = trainer.collect_prediction_rows(test_loader)
    _write_json(manifest_dir / "test_checkpoint.json", test_checkpoint)

    _write_json(metrics_dir / "history.json", history)
    _write_history_csv(metrics_dir / "epoch_summary.csv", history)
    _write_json(metrics_dir / "test_metrics.json", test_metrics)
    training_summary = _write_training_summary(
        metrics_dir / "training_summary.json",
        history,
        test_metrics,
        run_name,
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        test_checkpoint=test_checkpoint,
    )
    _write_rows_csv(
        predictions_dir / "test_per_fg_predictions.csv",
        fg_rows,
        [
            "pdb_id",
            "ligand_id",
            "smiles",
            "fg_index_in_complex",
            "fg_global_index",
            "fg_type_id",
            "fg_parent_type_id",
            "pred_vdw",
            "pred_elec",
            "pred_polar",
            "pred_nonpolar",
            "pred_contact_total",
            "pred_contact_components_sum",
            "pred_preorg",
            "pred_total",
            "target_vdw",
            "target_elec",
            "target_polar",
            "target_nonpolar",
            "target_contact_total",
            "target_preorg",
            "target_aux_elec",
            "target_aux_vdw",
            "mask_fg",
            "mask_contact",
            "mask_contact_total",
            "mask_preorg",
            "mask_aux_elec",
            "mask_aux_vdw",
        ],
    )
    _write_rows_csv(
        predictions_dir / "test_per_complex_predictions.csv",
        complex_rows,
        [
            "pdb_id",
            "ligand_id",
            "smiles",
            "n_fg",
            "pred_delta_total",
            "target_delta_total",
            "mask_delta_total",
            "pred_contact_sum",
            "pred_contact_components_sum",
            "pred_preorg_sum",
        ],
    )
    _plot_history_graphs._style_spec = args.plot_style
    _plot_history_graphs(metrics_dir, history, fg_rows, complex_rows)
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "split_seed": split_seed,
        "repeat_index": repeat_index,
        "history": history,
        "test": test_metrics,
        "training_summary": training_summary,
    }


def _write_repeated_split_summary(
    run_dir: Path,
    base_run_name: str,
    args: argparse.Namespace,
    split_rows: list[dict[str, object]],
) -> dict[str, object]:
    metrics_dir = run_dir / "metrics"
    manifest_dir = run_dir / "manifests"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate_split_rows(split_rows)
    payload: dict[str, object] = {
        "run_name": base_run_name,
        "num_splits": len(split_rows),
        "split_type": args.split_type,
        "model_seed": int(args.seed),
        "split_seed_base": _resolve_split_seed(args),
        "split_seed_stride": int(args.split_seed_stride),
        "splits": split_rows,
        "aggregate": aggregate,
    }
    _write_json(metrics_dir / "repeated_split_summary.json", payload)
    csv_headers = [
        "repeat_index",
        "split_seed",
        "run_name",
        "run_dir",
        "num_epochs_completed",
        "best_epoch",
        "train_count",
        "val_count",
        "test_count",
        "best_val_contact_total_pearson_r",
        "best_val_contact_total_std_ratio",
        "test_contact_total_pearson_r",
        "test_contact_total_mae",
        "test_contact_components_sum_pearson_r",
        "test_contact_total_std_ratio",
        "test_contact_components_sum_std_ratio",
        "test_contact_total_mae_improvement_vs_baseline",
        "test_collapsed_component_channels",
        "test_delta_total_pearson_r",
        "test_delta_total_mae",
        "test_delta_total_mae_improvement_vs_baseline",
    ]
    _write_rows_csv(metrics_dir / "repeated_split_summary.csv", split_rows, csv_headers)
    _write_json(
        manifest_dir / "repeated_split_config.json",
        {
            "args": vars(args),
            "resolved_split_seeds": _split_seed_sequence(args),
            "base_run_name": base_run_name,
        },
    )
    return payload


def main(args: argparse.Namespace):
    if getattr(args, "list_plot_styles", False):
        for style_name in available_plot_styles():
            print(style_name)
        return {"plot_styles": available_plot_styles()}

    base_run_name = args.run_name or _default_run_name()
    root_run_dir = Path(args.output_root) / "runs" / base_run_name
    split_seeds = _split_seed_sequence(args)
    if len(split_seeds) == 1:
        result = _run_single_training(
            args,
            run_name=base_run_name,
            run_dir=root_run_dir,
            split_seed=split_seeds[0],
            repeat_index=None,
        )
        return {
            "run_dir": result["run_dir"],
            "history": result["history"],
            "test": result["test"],
        }

    split_rows: list[dict[str, object]] = []
    split_results: list[dict[str, object]] = []
    for repeat_index, split_seed in enumerate(split_seeds):
        child_run_name = f"{base_run_name}__split{repeat_index:02d}_ss{split_seed}"
        child_run_dir = root_run_dir / "splits" / f"split_{repeat_index:02d}_seed_{split_seed}"
        print(
            f"repeat_split {repeat_index + 1}/{len(split_seeds)} "
            f"run={child_run_name} split_seed={split_seed}",
            flush=True,
        )
        result = _run_single_training(
            args,
            run_name=child_run_name,
            run_dir=child_run_dir,
            split_seed=split_seed,
            repeat_index=repeat_index,
        )
        split_results.append(result)
        split_rows.append(
            _split_summary_row(
                result["training_summary"],  # type: ignore[arg-type]
                split_seed,
                repeat_index,
                child_run_dir,
            )
        )

    summary = _write_repeated_split_summary(root_run_dir, base_run_name, args, split_rows)
    return {
        "run_dir": str(root_run_dir),
        "splits": split_results,
        "aggregate": summary,
    }


if __name__ == "__main__":
    main(parse_args())
