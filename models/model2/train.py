# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Training entry point for the counterfactual FG-importance model (model2).

Usage mirrors ``models/model1/train.py``: dataloader configuration and CLI
shape are intentionally kept compatible so the same launcher / sweep scripts
work for both models. The differences are:

- task names (``importance_ghost``, ``importance_charge``, ``importance_lj``,
  ``hotspot``)
- the early-stopping monitor defaults to
  ``importance_ghost.per_complex_spearman_r``
- prediction CSVs export per-FG importance predictions and hotspot scores
"""

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
from .model.model import FGImportanceModel
from .training.loss import ImportanceMultiTaskLoss
from .training.trainer import Trainer

REGRESSION_TASK_NAMES = ["importance_ghost", "importance_charge", "importance_lj"]
DEFAULT_TASK_NAMES = REGRESSION_TASK_NAMES + ["hotspot"]


def _compute_freq_bins(values: torch.Tensor, num_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(bin_edges[B+1], bin_weights[B])`` for inverse-frequency weighting.

    Bins are train-set quantiles so each bin has roughly equal sample count;
    weights are ``mean_count / bin_count`` so the bins with rare extreme values
    get a higher per-sample weight. Empty bins clamp to weight 1.0.
    """
    values = values.float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() < num_bins * 2 or num_bins <= 1:
        return torch.tensor([float("-inf"), float("inf")]), torch.tensor([1.0])
    qs = torch.linspace(0.0, 1.0, num_bins + 1)
    edges = torch.quantile(values, qs)
    edges[0] = float("-inf")
    edges[-1] = float("inf")
    counts = torch.zeros(num_bins, dtype=torch.float32)
    idx = torch.bucketize(values, edges[1:-1], right=True).clamp(min=0, max=num_bins - 1)
    counts.index_add_(0, idx.long(), torch.ones_like(values, dtype=torch.float32))
    mean_count = counts[counts > 0].mean() if (counts > 0).any() else torch.tensor(1.0)
    weights = torch.where(counts > 0, mean_count / counts.clamp_min(1.0), torch.tensor(1.0))
    # Cap to avoid runaway gradients on a single rare sample.
    weights = weights.clamp(max=10.0)
    return edges, weights


def parse_args() -> argparse.Namespace:
    register_scienceplots()
    model_defaults = ModelConfig()
    training_defaults = TrainingConfig()
    data_defaults = DataConfig(data_root="dataloader/_export_check2")
    parser = argparse.ArgumentParser(description="Train the model2 counterfactual FG-importance model")
    parser.add_argument("--data-root", default=data_defaults.data_root)
    parser.add_argument("--cache-dir", default=data_defaults.cache_dir)
    parser.add_argument("--case-root", action="append", default=list(data_defaults.case_roots))
    parser.add_argument(
        "--output-root",
        default="outputs/model2",
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
        choices=["encoders", "ligand_gnn", "fg_pool", "fg_context", "cross_attention",
                 "importance_head", "hotspot_head", "contact_aux_head"],
    )
    parser.add_argument("--freeze-epochs", type=int, default=training_defaults.freeze_epochs)
    parser.add_argument("--print-batch-every", type=int, default=training_defaults.print_batch_every)
    parser.add_argument(
        "--loss-type",
        choices=["huber", "mse", "mae", "log_cosh", "charbonnier", "tukey", "quantile", "ccc", "frequency_huber"],
        default="huber",
    )
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--tukey-c", type=float, default=4.685)
    parser.add_argument("--quantile-q", type=float, default=0.5)
    parser.add_argument("--hotspot-pos-weight", type=float, default=5.0)
    parser.add_argument(
        "--variance-penalty-weight",
        type=float,
        default=0.0,
        help="Anti-collapse penalty on predicted ghost importance.",
    )
    parser.add_argument(
        "--regression-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply per-task robust_std scaling to pred/target *inside the loss*. "
            "Default ON (safer, faster convergence). Flip OFF to keep gradient on "
            "high-magnitude FGs un-squashed -- helps the regression head escape "
            "mean-collapse but typically needs more epochs and gentler grad-clip."
        ),
    )
    parser.add_argument(
        "--frequency-num-bins",
        type=int,
        default=10,
        help=(
            "Number of quantile bins for inverse-frequency reweighting (used "
            "when --loss-type=frequency_huber). Bin edges and per-sample weights "
            "are computed from the train set once."
        ),
    )
    parser.add_argument(
        "--sum-constraint-weight",
        type=float,
        default=0.05,
        help=(
            "Soft penalty weight: Σ pred_ghost over FGs in a complex should "
            "track that complex's baseline DELTA TOTAL. 0 disables."
        ),
    )
    parser.add_argument(
        "--sum-constraint-beta",
        type=float,
        default=5.0,
        help="Huber beta (kcal/mol) for the sum-constraint penalty.",
    )
    parser.add_argument(
        "--aux-contact-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the auxiliary model1-style 4-channel soft contact "
            "decomposition loss. Auto-enables --use-contact-aux-head and the "
            "model1 soft-partition target builder when > 0."
        ),
    )
    parser.add_argument(
        "--use-elec-aev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concatenate a 2-channel Coulombic AEV to the ECIF AEV.",
    )
    parser.add_argument(
        "--use-fg-env-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concatenate a 5-channel per-FG local-pocket summary to the head inputs.",
    )
    parser.add_argument(
        "--use-pair-importance-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use model1-style FG-residue pair scorer for the importance head "
            "instead of the per-FG MLP. Stronger physical inductive bias; slower."
        ),
    )
    parser.add_argument(
        "--use-contact-aux-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the auxiliary 4-channel contact decomposition head. Auto-on "
            "when --aux-contact-weight > 0; off otherwise."
        ),
    )
    parser.add_argument(
        "--use-fg-elec-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add per-FG electrostatic descriptors (net q, dipole, q-variance, min/max q) to the FG encoder.",
    )
    parser.add_argument(
        "--use-fg-residue-coulomb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a 2-channel per-FG Coulombic interaction with the surrounding pocket residues to the head inputs.",
    )
    parser.add_argument(
        "--use-ratio-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a 4th regression head predicting the electrostatic fraction "
            "|charge|/(|charge|+|lj|+eps) per FG. Trained with BCE-with-logits "
            "against the target ratio. Enable in conjunction with --ratio-head-weight > 0."
        ),
    )
    parser.add_argument("--ratio-head-weight", type=float, default=0.0,
                        help="Loss weight on the ratio head (only used when --use-ratio-head).")
    parser.add_argument(
        "--use-variant-aware-charge-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply HID/HIE/HIP/ASH/GLH/LYN/CYM protonation-state corrections in the pocket charge prior.",
    )
    parser.add_argument(
        "--task-static-weight",
        action="append",
        default=[],
        metavar="TASK=WEIGHT",
        help=(
            "Override the static loss weight for a task, e.g. "
            "--task-static-weight importance_charge=1.0. Can be repeated. "
            "Tasks not overridden use the auto-selected defaults."
        ),
    )
    parser.add_argument(
        "--per-task-loss-type",
        action="append",
        default=[],
        metavar="TASK=TYPE",
        help=(
            "Override the loss type for a specific task, e.g. "
            "--per-task-loss-type importance_charge=quantile. Can be repeated. "
            "Tasks not overridden use --loss-type."
        ),
    )
    parser.add_argument(
        "--per-task-quantile-q",
        action="append",
        default=[],
        metavar="TASK=Q",
        help=(
            "Per-task quantile-q override (only meaningful when that task uses "
            "loss-type=quantile). e.g. importance_charge=0.7."
        ),
    )
    parser.add_argument(
        "--signed-log-task",
        action="append",
        default=[],
        metavar="TASK",
        help=(
            "For these regression tasks, transform pred and target with "
            "sign(y)*log1p(|y|) inside the loss. Repeatable. Typical use: "
            "--signed-log-task importance_ghost --signed-log-task importance_charge."
        ),
    )
    parser.add_argument(
        "--train-log-vars",
        action="store_true",
        default=False,
    )
    parser.add_argument("--importance-target-clip", type=float, default=0.0)
    parser.add_argument(
        "--task-mode",
        choices=["full", "regression_only", "ghost_only", "auto"],
        default="full",
        help=(
            "full: ghost+charge+lj regression + hotspot. "
            "regression_only: drop hotspot. "
            "ghost_only: ghost regression only. "
            "auto: drop tasks whose train-set target std is degenerate."
        ),
    )
    parser.add_argument(
        "--degenerate-std-threshold",
        type=float,
        default=1e-4,
        help="Drop auxiliary tasks in auto mode when train-set target std is below this.",
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
    parser.add_argument("--hotspot-top-k", type=int, default=model_defaults.hotspot_top_k)
    parser.add_argument("--hotspot-min-abs", type=float, default=model_defaults.hotspot_min_abs)
    parser.add_argument("--split-type", choices=["pocket_hash", "ligand"], default=data_defaults.split_type)
    parser.add_argument("--val-frac", type=float, default=data_defaults.val_frac)
    parser.add_argument("--test-frac", type=float, default=data_defaults.test_frac)
    parser.add_argument("--seed", type=int, default=data_defaults.seed)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
    )
    parser.add_argument("--repeat-splits", type=int, default=1)
    parser.add_argument("--split-seed-stride", type=int, default=1)
    parser.add_argument("--min-fg-count", type=int, default=data_defaults.min_fg_count)
    parser.add_argument(
        "--exclude-zero-fg",
        action=argparse.BooleanOptionalAction,
        default=data_defaults.exclude_zero_fg,
    )
    parser.add_argument("--max-cases", type=int, default=data_defaults.max_cases)
    parser.add_argument(
        "--max-abs-importance",
        type=float,
        default=training_defaults.max_abs_importance,
    )
    parser.add_argument(
        "--outlier-zscore-threshold",
        type=float,
        default=10.0,
        help=(
            "Per-mode outlier cutoff in robust z-units: any target with "
            "|target - median| > k * (1.4826 * MAD) is masked out as invalid "
            "before the loss sees it. Bounds are computed once from the train "
            "set and applied to all splits. 0 disables. Default 10.0 -- only "
            "kills pathological MM-GBSA values; the natural high tail of FG "
            "importance (FGs with large but legitimate ΔΔ_total) carries "
            "the most ranking signal, so we deliberately don't filter it. "
            "Drop to ~5-6 only if you're seeing the regression head pinned "
            "to the mean and don't care about ranking."
        ),
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
    )
    parser.add_argument(
        "--list-plot-styles",
        action="store_true",
    )
    return parser.parse_args()


def build_dataloaders(data_config: DataConfig, training_config: TrainingConfig):
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


def build_model(config: ModelConfig) -> FGImportanceModel:
    return FGImportanceModel(config)


def build_optimizer(model: FGImportanceModel, config: TrainingConfig, loss_fn: torch.nn.Module | None = None):
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
    if task_name == "importance_ghost":
        return targets["importance_ghost"], masks["importance_ghost"]
    if task_name == "importance_charge":
        return targets["importance_charge"], masks["importance_charge"]
    if task_name == "importance_lj":
        return targets["importance_lj"], masks["importance_lj"]
    if task_name == "hotspot":
        return targets["hotspot"], masks["hotspot"]
    raise KeyError(task_name)


def estimate_task_statistics(
    loader,
    cutoff: float,
    task_names: list[str],
    *,
    max_abs_importance: float = 0.0,
    hotspot_top_k: int = 3,
    hotspot_min_abs: float = 0.5,
) -> dict[str, dict[str, float]]:
    values: dict[str, list[torch.Tensor]] = {name: [] for name in task_names}
    for batch in loader:
        targets, masks = build_targets_from_batch(
            batch,
            cutoff,
            max_abs_importance=max_abs_importance,
            hotspot_top_k=hotspot_top_k,
            hotspot_min_abs=hotspot_min_abs,
        )
        for task_name in task_names:
            target, mask = _task_target_and_mask(task_name, targets, masks)
            mask = mask.bool()
            if not mask.any():
                continue
            selected = target[mask]
            if selected.numel() == 0:
                continue
            values[task_name].append(selected.detach().cpu().reshape(-1))
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
        x = torch.cat(values[task_name]).float()
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
) -> tuple[list[str], dict[str, float], dict[str, float], dict[str, float]]:
    if task_mode == "full":
        active = ["importance_ghost", "importance_charge", "importance_lj", "hotspot"]
    elif task_mode == "regression_only":
        active = ["importance_ghost", "importance_charge", "importance_lj"]
    elif task_mode == "ghost_only":
        active = ["importance_ghost"]
    elif task_mode == "auto":
        active = ["importance_ghost"]
        for task_name in ["importance_charge", "importance_lj"]:
            stats = task_stats.get(task_name, {})
            count = int(stats.get("count", 0))
            std = float(stats.get("std", 0.0))
            if count > 0 and std > degenerate_std_threshold:
                active.append(task_name)
        active.append("hotspot")
    else:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    task_scales: dict[str, float] = {}
    task_static_weights: dict[str, float] = {}
    task_target_clips: dict[str, float] = {}
    for task_name in active:
        if task_name == "hotspot":
            task_scales[task_name] = 1.0
            task_static_weights[task_name] = 0.5
            continue
        stats = task_stats.get(task_name, {})
        std = float(stats.get("std", 0.0))
        robust_std = float(stats.get("robust_std", 0.0))
        scale = robust_std if robust_std > 0.0 else std
        scale = max(scale, 1e-3)
        task_scales[task_name] = scale
        if task_name == "importance_ghost":
            task_static_weights[task_name] = 1.0
        else:
            task_static_weights[task_name] = 0.5
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
        "val_importance_ghost_per_complex_spearman_r",
        "val_importance_ghost_pearson_r",
        "val_importance_ghost_mae",
        "val_importance_ghost_top1_hit_rate",
        "val_importance_ghost_top3_hit_rate",
        "val_hotspot_roc_auc",
        "val_hotspot_pr_auc",
    ]
    remaining = sorted(h for h in dynamic_headers if h not in preferred)
    headers = [h for h in preferred if h in dynamic_headers] + remaining
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


def _best_epoch_summary(history: list[dict], monitor_path: str, mode: str) -> dict:
    best_epoch = None
    best_score = float("-inf") if mode == "max" else float("inf")
    for item in history:
        value: object = item.get("val", {})
        for key in monitor_path.split("."):
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if not isinstance(value, (int, float)) or value != value:
            continue
        score = float(value)
        if mode == "max" and score > best_score:
            best_score = score
            best_epoch = item
        elif mode == "min" and score < best_score:
            best_score = score
            best_epoch = item
    return {
        "best_score": _safe_metric_value(best_score if best_epoch is not None else float("nan")),
        "best_epoch": best_epoch["epoch"] if best_epoch is not None else None,
        "best_epoch_metrics": _json_sanitize(best_epoch) if best_epoch is not None else None,
        "monitor": monitor_path,
        "mode": mode,
    }


def _write_training_summary(
    path: Path,
    history: list[dict],
    test_metrics: dict,
    run_name: str,
    train_count: int,
    val_count: int,
    test_count: int,
    *,
    monitor_path: str,
    monitor_mode: str,
    test_checkpoint: dict | None = None,
) -> dict:
    payload = {
        "run_name": run_name,
        "num_epochs_completed": len(history),
        "dataset_counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "final_epoch": _json_sanitize(history[-1]) if history else None,
        "best": _best_epoch_summary(history, monitor_path, monitor_mode),
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
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = statistics.mean(finite)
    std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    return {"count": len(finite), "mean": mean, "std": std, "min": min(finite), "max": max(finite)}


def _split_summary_row(summary: dict, split_seed: int, repeat_index: int, run_dir: Path) -> dict[str, object]:
    return {
        "repeat_index": repeat_index,
        "split_seed": split_seed,
        "run_name": summary.get("run_name"),
        "run_dir": str(run_dir),
        "num_epochs_completed": summary.get("num_epochs_completed"),
        "best_epoch": _metric_from_path(summary, "best.best_epoch"),
        "train_count": _metric_from_path(summary, "dataset_counts.train"),
        "val_count": _metric_from_path(summary, "dataset_counts.val"),
        "test_count": _metric_from_path(summary, "dataset_counts.test"),
        "best_val_score": _metric_from_path(summary, "best.best_score"),
        "test_importance_ghost_per_complex_spearman_r": _metric_from_path(
            summary, "test_metrics.importance_ghost.per_complex_spearman_r"
        ),
        "test_importance_ghost_pearson_r": _metric_from_path(summary, "test_metrics.importance_ghost.pearson_r"),
        "test_importance_ghost_mae": _metric_from_path(summary, "test_metrics.importance_ghost.mae"),
        "test_importance_ghost_top1_hit_rate": _metric_from_path(
            summary, "test_metrics.importance_ghost.top1_hit_rate"
        ),
        "test_importance_ghost_top3_hit_rate": _metric_from_path(
            summary, "test_metrics.importance_ghost.top3_hit_rate"
        ),
        "test_hotspot_roc_auc": _metric_from_path(summary, "test_metrics.hotspot.roc_auc"),
        "test_hotspot_pr_auc": _metric_from_path(summary, "test_metrics.hotspot.pr_auc"),
    }


def _aggregate_split_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int | None]]:
    metric_keys = [
        "best_val_score",
        "test_importance_ghost_per_complex_spearman_r",
        "test_importance_ghost_pearson_r",
        "test_importance_ghost_mae",
        "test_importance_ghost_top1_hit_rate",
        "test_importance_ghost_top3_hit_rate",
        "test_hotspot_roc_auc",
        "test_hotspot_pr_auc",
        "num_epochs_completed",
        "best_epoch",
        "train_count",
        "val_count",
        "test_count",
    ]
    return {key: _aggregate_numeric([row.get(key) for row in rows]) for key in metric_keys}  # type: ignore[arg-type]


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
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="model2_mpl_"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = metrics_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    applied_styles = apply_plot_styles(getattr(_plot_history_graphs, "_style_spec", "science+ieee"))

    rows, _headers = _history_rows(history)
    epochs = _extract_series(rows, "epoch")

    def save_line(path: Path, title: str, y_label: str, series_map: dict[str, list[float]]) -> None:
        plt.figure(figsize=(8, 5))
        plotted = 0
        for label, values in series_map.items():
            if any(v == v for v in values):
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

    save_line(plots_dir / "loss_curve.png", "Training Loss", "Loss",
              {"train_loss": _extract_series(rows, "train_loss")})
    save_line(plots_dir / "learning_rate_curve.png", "Learning Rate", "LR",
              {"train_lr": _extract_series(rows, "train_lr")})
    save_line(
        plots_dir / "val_ghost_correlation.png",
        "Validation Ghost Importance Correlations",
        "Correlation",
        {
            "per_complex_spearman": _extract_series(rows, "val_importance_ghost_per_complex_spearman_r"),
            "pearson_pooled": _extract_series(rows, "val_importance_ghost_pearson_r"),
            "spearman_pooled": _extract_series(rows, "val_importance_ghost_spearman_r"),
        },
    )
    save_line(
        plots_dir / "val_ghost_hit_rates.png",
        "Validation Top-K Hit Rate (Ghost)",
        "Hit Rate",
        {
            "top1": _extract_series(rows, "val_importance_ghost_top1_hit_rate"),
            "top3": _extract_series(rows, "val_importance_ghost_top3_hit_rate"),
        },
    )
    save_line(
        plots_dir / "val_hotspot_auc.png",
        "Validation Hotspot AUC",
        "AUC",
        {
            "roc_auc": _extract_series(rows, "val_hotspot_roc_auc"),
            "pr_auc": _extract_series(rows, "val_hotspot_pr_auc"),
        },
    )
    save_line(
        plots_dir / "val_ghost_mae.png",
        "Validation Ghost MAE",
        "MAE",
        {"mae": _extract_series(rows, "val_importance_ghost_mae")},
    )

    train_task_keys = sorted(
        key for key in rows[0].keys() if key.startswith("train_loss_") and key != "train_loss"
    )
    if train_task_keys:
        save_line(
            plots_dir / "train_task_losses.png",
            "Per-Task Training Losses",
            "Loss",
            {key.removeprefix("train_loss_"): _extract_series(rows, key) for key in train_task_keys},
        )

    def save_scatter(path: Path, title: str, x_label: str, y_label: str, xs: list[float], ys: list[float]) -> None:
        pairs = [(x, y) for x, y in zip(xs, ys) if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x == x and y == y]
        if not pairs:
            return
        xs_v = [p[0] for p in pairs]
        ys_v = [p[1] for p in pairs]
        plt.figure(figsize=(6, 6))
        plt.scatter(xs_v, ys_v, alpha=0.7)
        lo = min(xs_v + ys_v)
        hi = max(xs_v + ys_v)
        if lo != hi:
            plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    # Use math-mode LaTeX so the scienceplots style (text.usetex=True) renders cleanly;
    # raw unicode Δ in titles trips the LaTeX driver.
    save_scatter(
        plots_dir / "test_fg_ghost_pred_vs_target.png",
        r"Test FG Ghost $\Delta\Delta_{\mathrm{total}}$: Predicted vs Target",
        r"Target ghost $\Delta\Delta_{\mathrm{total}}$ (kcal/mol)",
        r"Predicted ghost $\Delta\Delta_{\mathrm{total}}$ (kcal/mol)",
        [row["target_importance_ghost"] for row in fg_rows if row.get("mask_importance_ghost")],
        [row["pred_importance_ghost"] for row in fg_rows if row.get("mask_importance_ghost")],
    )
    _write_json(
        metrics_dir / "plot_config.json",
        {
            "requested_plot_style": getattr(_plot_history_graphs, "_style_spec", "science+ieee"),
            "applied_plot_styles": applied_styles,
        },
    )


def _build_configs(
    args: argparse.Namespace,
    *,
    split_seed: int,
) -> tuple[ModelConfig, TrainingConfig, DataConfig]:
    # Auto-enable the contact aux head when its loss weight is non-zero unless
    # the user explicitly overrode the flag.
    use_contact_aux_head = args.use_contact_aux_head
    if use_contact_aux_head is None:
        use_contact_aux_head = args.aux_contact_weight > 0.0
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
        hotspot_top_k=args.hotspot_top_k,
        hotspot_min_abs=args.hotspot_min_abs,
        use_elec_aev=args.use_elec_aev,
        use_fg_env_summary=args.use_fg_env_summary,
        use_pair_importance_head=args.use_pair_importance_head,
        use_contact_aux_head=bool(use_contact_aux_head),
        use_fg_elec_features=args.use_fg_elec_features,
        use_fg_residue_coulomb=args.use_fg_residue_coulomb,
        use_ratio_head=args.use_ratio_head,
        use_variant_aware_charge_prior=args.use_variant_aware_charge_prior,
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
        max_abs_importance=args.max_abs_importance,
        sum_constraint_weight=args.sum_constraint_weight,
        aux_contact_weight=args.aux_contact_weight,
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
        max_abs_importance=training_config.max_abs_importance,
        hotspot_top_k=model_config.hotspot_top_k,
        hotspot_min_abs=model_config.hotspot_min_abs,
    )
    active_tasks, task_scales, task_static_weights, task_target_clips = select_task_setup(
        args.task_mode,
        task_stats,
        args.degenerate_std_threshold,
    )
    for task_name in REGRESSION_TASK_NAMES:
        if task_name in active_tasks and args.importance_target_clip > 0.0:
            task_target_clips[task_name] = args.importance_target_clip

    # Per-mode outlier bounds derived from the train-set distribution. The
    # label builder will mask any target outside (median - k*robust_std,
    # median + k*robust_std) as invalid. Robust std is 1.4826*MAD; falls
    # back to ordinary std if MAD is zero (constant column).
    outlier_k = float(args.outlier_zscore_threshold)
    outlier_bounds: dict[str, tuple[float, float]] = {}
    if outlier_k > 0.0:
        for task_name in REGRESSION_TASK_NAMES:
            stats = task_stats.get(task_name, {})
            count = int(stats.get("count", 0))
            if count <= 0:
                continue
            median = float(stats.get("median", 0.0))
            robust = float(stats.get("robust_std", 0.0))
            if robust <= 0.0:
                robust = float(stats.get("std", 0.0))
            if robust <= 0.0:
                continue
            outlier_bounds[task_name] = (median - outlier_k * robust, median + outlier_k * robust)

    # Frequency-bin edges + per-bin weights for inverse-frequency loss. Built
    # once from the train set; the loss looks each sample up via bucketize.
    freq_bin_edges: dict[str, torch.Tensor] = {}
    freq_bin_weights: dict[str, torch.Tensor] = {}
    if args.loss_type == "frequency_huber":
        # Collect train-set targets (post outlier_bounds) once.
        collected: dict[str, list[torch.Tensor]] = {name: [] for name in REGRESSION_TASK_NAMES}
        for batch in train_loader:
            targets, masks = build_targets_from_batch(
                batch,
                model_config.cross_attn_cutoff,
                max_abs_importance=training_config.max_abs_importance,
                hotspot_top_k=model_config.hotspot_top_k,
                hotspot_min_abs=model_config.hotspot_min_abs,
                outlier_bounds=outlier_bounds or None,
            )
            for name in REGRESSION_TASK_NAMES:
                m = masks[name].bool()
                if m.any():
                    collected[name].append(targets[name][m].detach().cpu().reshape(-1))
        for name in REGRESSION_TASK_NAMES:
            if not collected[name]:
                continue
            edges, weights = _compute_freq_bins(torch.cat(collected[name]), args.frequency_num_bins)
            freq_bin_edges[name] = edges
            freq_bin_weights[name] = weights
    print(
        "task_setup "
        f"run={run_name} "
        f"split_seed={split_seed} "
        f"mode={args.task_mode} "
        f"active={','.join(active_tasks)} "
        f"scales={json.dumps(task_scales, sort_keys=True)} "
        f"clips={json.dumps(task_target_clips, sort_keys=True)} "
        f"outlier_k={outlier_k} "
        f"outlier_bounds={json.dumps({k: [round(lo, 3), round(hi, 3)] for k, (lo, hi) in outlier_bounds.items()}, sort_keys=True)}",
        flush=True,
    )
    # Parse per-task CLI overrides (task=value strings, repeatable).
    static_weight_overrides: dict[str, float] = {}
    for spec in args.task_static_weight:
        k, _, v = spec.partition("=")
        if k and v:
            static_weight_overrides[k.strip()] = float(v)
    for k, v in static_weight_overrides.items():
        task_static_weights[k] = v
    per_task_loss_type: dict[str, str] = {}
    for spec in args.per_task_loss_type:
        k, _, v = spec.partition("=")
        if k and v:
            per_task_loss_type[k.strip()] = v.strip()
    per_task_quantile_q: dict[str, float] = {}
    for spec in args.per_task_quantile_q:
        k, _, v = spec.partition("=")
        if k and v:
            per_task_quantile_q[k.strip()] = float(v)
    signed_log_tasks = tuple(s.strip() for s in args.signed_log_task if s.strip())

    model = build_model(model_config)
    loss_fn = ImportanceMultiTaskLoss(
        active_tasks,
        task_scales=task_scales,
        task_static_weights=task_static_weights,
        task_target_clips=task_target_clips,
        loss_type=args.loss_type,
        huber_beta=args.huber_beta,
        charbonnier_eps=args.charbonnier_eps,
        tukey_c=args.tukey_c,
        quantile_q=args.quantile_q,
        hotspot_pos_weight=args.hotspot_pos_weight,
        variance_penalty_weight=args.variance_penalty_weight,
        regression_normalize=args.regression_normalize,
        sum_constraint_weight=args.sum_constraint_weight,
        sum_constraint_beta=args.sum_constraint_beta,
        aux_contact_weight=args.aux_contact_weight,
        loss_freq_bin_edges=freq_bin_edges or None,
        loss_freq_bin_weights=freq_bin_weights or None,
        per_task_loss_type=per_task_loss_type,
        per_task_quantile_q=per_task_quantile_q,
        signed_log_tasks=signed_log_tasks,
        ratio_head_weight=args.ratio_head_weight,
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
        outlier_bounds=outlier_bounds,
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
                "hotspot_pos_weight": args.hotspot_pos_weight,
                "variance_penalty_weight": args.variance_penalty_weight,
                "regression_normalize": args.regression_normalize,
                "sum_constraint_weight": args.sum_constraint_weight,
                "sum_constraint_beta": args.sum_constraint_beta,
                "aux_contact_weight": args.aux_contact_weight,
                "use_elec_aev": args.use_elec_aev,
                "use_fg_env_summary": args.use_fg_env_summary,
                "use_pair_importance_head": args.use_pair_importance_head,
                "use_contact_aux_head": bool(model_config.use_contact_aux_head),
                "degenerate_std_threshold": args.degenerate_std_threshold,
                "task_stats": task_stats,
                "baseline_means": baseline_means,
                "outlier_zscore_threshold": outlier_k,
                "outlier_bounds": {k: list(v) for k, v in outlier_bounds.items()},
                "freq_bin_edges": {k: v.tolist() for k, v in freq_bin_edges.items()},
                "freq_bin_weights": {k: v.tolist() for k, v in freq_bin_weights.items()},
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
        monitor_path=training_config.early_stopping_metric,
        monitor_mode=training_config.early_stopping_mode,
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
            "pred_importance_ghost",
            "pred_importance_charge",
            "pred_importance_lj",
            "pred_hotspot_logit",
            "pred_hotspot_prob",
            "target_importance_ghost",
            "target_importance_charge",
            "target_importance_lj",
            "target_abs_importance_ghost",
            "target_hotspot",
            "mask_importance_ghost",
            "mask_importance_charge",
            "mask_importance_lj",
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
            "n_fg_valid_ghost",
            "pred_ghost_sum",
            "target_ghost_sum",
            "target_max_abs_ghost",
            "pred_max_abs_ghost",
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
        "best_val_score",
        "test_importance_ghost_per_complex_spearman_r",
        "test_importance_ghost_pearson_r",
        "test_importance_ghost_mae",
        "test_importance_ghost_top1_hit_rate",
        "test_importance_ghost_top3_hit_rate",
        "test_hotspot_roc_auc",
        "test_hotspot_pr_auc",
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
        return {"run_dir": result["run_dir"], "history": result["history"], "test": result["test"]}

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
            _split_summary_row(result["training_summary"], split_seed, repeat_index, child_run_dir)  # type: ignore[arg-type]
        )

    summary = _write_repeated_split_summary(root_run_dir, base_run_name, args, split_rows)
    return {"run_dir": str(root_run_dir), "splits": split_results, "aggregate": summary}


if __name__ == "__main__":
    main(parse_args())
