# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
from torch import Tensor


def _filter(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    pred = pred.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if mask is not None:
        valid = valid & mask.detach().bool().reshape(-1)
    return pred[valid], target[valid]


def pearson_r(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    pred, target = _filter(pred, target, mask)
    if pred.numel() < 2:
        return float("nan")
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm()
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((pred @ target / denom).item())


def _rank(x: Tensor) -> Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32, device=x.device)
    return ranks


def spearman_r(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    pred, target = _filter(pred, target, mask)
    if pred.numel() < 2:
        return float("nan")
    return pearson_r(_rank(pred), _rank(target))


def mae(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    pred, target = _filter(pred, target, mask)
    if pred.numel() == 0:
        return float("nan")
    return float((pred - target).abs().mean().item())


def compute_per_task_metrics(preds: dict, targets: dict, masks: dict) -> dict:
    contact_pred = preds["contact"]
    contact_total_pred = preds.get("contact_total", contact_pred.sum(dim=-1))
    contact_target = targets["contact"]
    contact_total_target = targets.get("contact_total", contact_target.sum(dim=-1))
    contact_mask = masks.get("contact", masks.get("fg"))
    contact_total_mask = masks.get("contact_total", contact_mask)
    preorg_mask = masks.get("preorg", masks.get("fg"))
    delta_mask = masks.get("delta_total", masks.get("complex"))
    metrics = {}
    channel_names = ["vdw", "elec", "polar", "nonpolar"]
    for idx, channel_name in enumerate(channel_names):
        metrics[channel_name] = {
            "pearson_r": pearson_r(contact_pred[:, idx], contact_target[:, idx], contact_mask),
            "spearman_r": spearman_r(contact_pred[:, idx], contact_target[:, idx], contact_mask),
            "mae": mae(contact_pred[:, idx], contact_target[:, idx], contact_mask),
        }
    metrics["contact_total"] = {
        "pearson_r": pearson_r(contact_total_pred, contact_total_target, contact_total_mask),
        "spearman_r": spearman_r(contact_total_pred, contact_total_target, contact_total_mask),
        "mae": mae(contact_total_pred, contact_total_target, contact_total_mask),
    }
    pred_components_total = contact_pred.sum(dim=-1)
    metrics["contact_components_sum"] = {
        "pearson_r": pearson_r(pred_components_total, contact_total_target, contact_mask),
        "spearman_r": spearman_r(pred_components_total, contact_total_target, contact_mask),
        "mae": mae(pred_components_total, contact_total_target, contact_mask),
    }
    metrics["preorg"] = {
        "pearson_r": pearson_r(preds["preorg"], targets["preorg"], preorg_mask),
        "spearman_r": spearman_r(preds["preorg"], targets["preorg"], preorg_mask),
        "mae": mae(preds["preorg"], targets["preorg"], preorg_mask),
    }
    metrics["delta_total"] = {
        "pearson_r": pearson_r(preds["delta_total"], targets["delta_total"], delta_mask),
        "spearman_r": spearman_r(preds["delta_total"], targets["delta_total"], delta_mask),
        "mae": mae(preds["delta_total"], targets["delta_total"], delta_mask),
    }
    return metrics


def _std_summary(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> dict[str, float]:
    pred_f, target_f = _filter(pred, target, mask)
    if pred_f.numel() == 0:
        return {
            "pred_std": float("nan"),
            "target_std": float("nan"),
            "pred_target_std_ratio": float("nan"),
        }
    pred_std = float(pred_f.std(unbiased=False).item()) if pred_f.numel() > 0 else float("nan")
    target_std = float(target_f.std(unbiased=False).item()) if target_f.numel() > 0 else float("nan")
    if math.isnan(target_std) or target_std == 0.0:
        ratio = float("nan")
    else:
        ratio = pred_std / target_std
    return {
        "pred_std": pred_std,
        "target_std": target_std,
        "pred_target_std_ratio": ratio,
    }


def _constant_baseline_mae(target: Tensor, baseline: float, mask: Tensor | None = None) -> float:
    target_f, _ = _filter(target, target, mask)
    if target_f.numel() == 0:
        return float("nan")
    pred = torch.full_like(target_f, float(baseline))
    return float((pred - target_f).abs().mean().item())


def compute_prediction_diagnostics(
    preds: dict,
    targets: dict,
    masks: dict,
    baseline_means: dict[str, float] | None = None,
) -> dict:
    baseline_means = baseline_means or {}
    contact_pred = preds["contact"]
    contact_target = targets["contact"]
    contact_total_pred = preds.get("contact_total", contact_pred.sum(dim=-1))
    contact_total_target = targets.get("contact_total", contact_target.sum(dim=-1))
    contact_mask = masks.get("contact", masks.get("fg"))
    contact_total_mask = masks.get("contact_total", contact_mask)
    delta_mask = masks.get("delta_total", masks.get("complex"))

    component_stats: dict[str, dict[str, float]] = {}
    collapsed_component_channels = 0
    for idx, channel_name in enumerate(["vdw", "elec", "polar", "nonpolar"]):
        stats = _std_summary(contact_pred[:, idx], contact_target[:, idx], contact_mask)
        component_stats[channel_name] = stats
        ratio = stats.get("pred_target_std_ratio", float("nan"))
        if isinstance(ratio, float) and ratio == ratio and ratio < 0.1:
            collapsed_component_channels += 1

    diagnostics = {
        "contact_total": _std_summary(contact_total_pred, contact_total_target, contact_total_mask),
        "contact_components_sum": _std_summary(contact_pred.sum(dim=-1), contact_total_target, contact_mask),
        "component_channels": component_stats,
        "collapsed_component_channels_below_0_1_ratio": collapsed_component_channels,
    }

    contact_baseline = baseline_means.get("contact_total")
    if contact_baseline is not None:
        baseline_mae = _constant_baseline_mae(contact_total_target, contact_baseline, contact_total_mask)
        diagnostics["contact_total_mean_baseline_mae"] = baseline_mae
        diagnostics["contact_total_mae_improvement_vs_baseline"] = baseline_mae - mae(
            contact_total_pred,
            contact_total_target,
            contact_total_mask,
        )

    delta_baseline = baseline_means.get("delta_total")
    if delta_baseline is not None:
        baseline_mae = _constant_baseline_mae(targets["delta_total"], delta_baseline, delta_mask)
        diagnostics["delta_total_mean_baseline_mae"] = baseline_mae
        diagnostics["delta_total_mae_improvement_vs_baseline"] = baseline_mae - mae(
            preds["delta_total"],
            targets["delta_total"],
            delta_mask,
        )

    return diagnostics


def fg_rank_correlation(
    contact_preds: Tensor,
    contact_targets: Tensor,
    complex_ids: Tensor,
    mask: Tensor | None = None,
) -> float:
    scores: list[float] = []
    for complex_id in torch.unique(complex_ids.long()):
        local_mask = complex_ids.long() == complex_id
        if mask is not None:
            local_mask = local_mask & mask.bool()
        score = spearman_r(contact_preds[local_mask], contact_targets[local_mask])
        if not math.isnan(score):
            scores.append(score)
    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


def per_chemotype_metrics(
    preds: Tensor,
    targets: Tensor,
    fg_types: Tensor,
    mask: Tensor | None = None,
) -> dict:
    result = {}
    for fg_type in torch.unique(fg_types.long()):
        fg_mask = fg_types.long() == fg_type
        if mask is not None:
            fg_mask = fg_mask & mask.bool()
        if not fg_mask.any():
            continue
        result[int(fg_type.item())] = {
            "pearson_r": pearson_r(preds[fg_mask], targets[fg_mask]),
            "spearman_r": spearman_r(preds[fg_mask], targets[fg_mask]),
            "mae": mae(preds[fg_mask], targets[fg_mask]),
        }
    return result
