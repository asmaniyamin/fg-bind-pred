# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the counterfactual FG-importance model (model2).

Headline metric: per-complex Spearman of predicted vs. true ghost ΔΔ_total
(``importance_ghost.per_complex_spearman_r``). This is what tells us "did the
model figure out which FG is most important in this ligand".

We also report:

- pooled Pearson / Spearman / MAE for each regression head
- Top-K hit rate: in each complex, did the model rank the true top-K FGs
  (by |ghost ΔΔ|) in its own top-K?
- Hotspot ROC AUC and PR AUC, computed pooled across the dataset
- Std-ratio diagnostics (catch prediction collapse)
- Mean-baseline MAE comparison
"""

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


def per_complex_correlation(
    pred: Tensor,
    target: Tensor,
    fg_batch: Tensor,
    mask: Tensor | None = None,
    *,
    method: str = "spearman",
) -> float:
    """Mean within-complex correlation, ignoring complexes with <2 valid FGs."""
    scores: list[float] = []
    for complex_id in torch.unique(fg_batch.long()):
        local_mask = fg_batch.long() == complex_id
        if mask is not None:
            local_mask = local_mask & mask.bool()
        if int(local_mask.sum().item()) < 2:
            continue
        if method == "spearman":
            score = spearman_r(pred[local_mask], target[local_mask])
        elif method == "pearson":
            score = pearson_r(pred[local_mask], target[local_mask])
        else:
            raise ValueError(method)
        if not math.isnan(score):
            scores.append(score)
    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


def top_k_hit_rate(
    pred: Tensor,
    abs_target: Tensor,
    fg_batch: Tensor,
    mask: Tensor | None = None,
    *,
    k: int = 3,
    score_kind: str = "abs",
) -> float:
    """Per-complex top-K hit rate: |predicted top-K| ∩ |target top-K| / K.

    ``score_kind`` controls how the predictions are ranked:

    - ``abs``   : rank FGs by ``|pred|`` -- matches the |ghost ΔΔ| target
    - ``raw``   : rank FGs by ``pred`` directly -- use with ``rank_score``
    """
    hits: list[float] = []
    for complex_id in torch.unique(fg_batch.long()):
        local_mask = fg_batch.long() == complex_id
        if mask is not None:
            local_mask = local_mask & mask.bool()
        if int(local_mask.sum().item()) < 1:
            continue
        local_target = abs_target[local_mask]
        local_pred = pred[local_mask]
        if local_target.numel() == 0:
            continue
        k_local = min(int(k), int(local_target.numel()))
        if k_local <= 0:
            continue
        target_top = torch.topk(local_target, k=k_local).indices.tolist()
        if score_kind == "abs":
            pred_score = local_pred.abs()
        elif score_kind == "raw":
            pred_score = local_pred
        else:
            raise ValueError(score_kind)
        pred_top = torch.topk(pred_score, k=k_local).indices.tolist()
        overlap = len(set(target_top) & set(pred_top))
        hits.append(overlap / float(k_local))
    if not hits:
        return float("nan")
    return float(sum(hits) / len(hits))


def _binary_metrics(scores: Tensor, labels: Tensor) -> tuple[float, float]:
    """Return ROC AUC and PR AUC. Pure-torch, fine for ~10^5 samples."""
    if scores.numel() == 0:
        return float("nan"), float("nan")
    labels = labels.float()
    if labels.sum() == 0 or labels.sum() == labels.numel():
        return float("nan"), float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    total_pos = float(sorted_labels.sum().item())
    total_neg = float((1.0 - sorted_labels).sum().item())
    if total_pos <= 0.0 or total_neg <= 0.0:
        return float("nan"), float("nan")
    tpr = tp / total_pos
    fpr = fp / total_neg
    tpr_padded = torch.cat([scores.new_zeros((1,)), tpr])
    fpr_padded = torch.cat([scores.new_zeros((1,)), fpr])
    roc_auc = float(torch.trapezoid(tpr_padded, fpr_padded).item())
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tpr
    recall_padded = torch.cat([scores.new_zeros((1,)), recall])
    precision_padded = torch.cat([precision.new_ones((1,)), precision])
    pr_auc = float(torch.trapezoid(precision_padded, recall_padded).item())
    return roc_auc, pr_auc


def hotspot_metrics(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor | None = None,
) -> dict[str, float]:
    logits, labels = _filter(logits, labels, mask)
    if logits.numel() == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "positive_rate": float("nan")}
    roc, pr = _binary_metrics(logits, labels)
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "positive_rate": float(labels.mean().item()),
    }


def _std_summary(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> dict[str, float]:
    pred_f, target_f = _filter(pred, target, mask)
    if pred_f.numel() == 0:
        return {"pred_std": float("nan"), "target_std": float("nan"), "pred_target_std_ratio": float("nan")}
    pred_std = float(pred_f.std(unbiased=False).item())
    target_std = float(target_f.std(unbiased=False).item())
    if math.isnan(target_std) or target_std == 0.0:
        ratio = float("nan")
    else:
        ratio = pred_std / target_std
    return {"pred_std": pred_std, "target_std": target_std, "pred_target_std_ratio": ratio}


def _constant_baseline_mae(target: Tensor, baseline: float, mask: Tensor | None = None) -> float:
    target_f, _ = _filter(target, target, mask)
    if target_f.numel() == 0:
        return float("nan")
    pred = torch.full_like(target_f, float(baseline))
    return float((pred - target_f).abs().mean().item())


def compute_per_task_metrics(preds: dict, targets: dict, masks: dict) -> dict:
    fg_batch = targets["fg_batch"]
    metrics: dict[str, dict] = {}

    for mode in ("ghost", "charge", "lj"):
        pred_key = f"importance_{mode}"
        target_key = f"importance_{mode}"
        mask = masks.get(pred_key, masks.get("fg"))
        metrics[pred_key] = {
            "pearson_r": pearson_r(preds[pred_key], targets[target_key], mask),
            "spearman_r": spearman_r(preds[pred_key], targets[target_key], mask),
            "mae": mae(preds[pred_key], targets[target_key], mask),
            "per_complex_spearman_r": per_complex_correlation(
                preds[pred_key], targets[target_key], fg_batch, mask, method="spearman"
            ),
            "per_complex_pearson_r": per_complex_correlation(
                preds[pred_key], targets[target_key], fg_batch, mask, method="pearson"
            ),
        }

    abs_target = targets.get(
        "abs_importance",
        targets["importance_ghost"].abs(),
    )
    ghost_mask = masks.get("importance_ghost", masks.get("fg"))
    metrics["importance_ghost"]["top1_hit_rate"] = top_k_hit_rate(
        preds["importance_ghost"], abs_target, fg_batch, ghost_mask, k=1, score_kind="abs"
    )
    metrics["importance_ghost"]["top3_hit_rate"] = top_k_hit_rate(
        preds["importance_ghost"], abs_target, fg_batch, ghost_mask, k=3, score_kind="abs"
    )
    metrics["hotspot"] = hotspot_metrics(preds["hotspot"], targets["hotspot"], masks.get("hotspot"))
    # Sum-constraint diagnostic: per-complex Pearson between Σ pred_ghost and target ΔG.
    if "delta_total" in targets and "fg_batch" in targets:
        metrics["delta_total"] = _per_complex_sum_metrics(
            preds["importance_ghost"],
            targets.get("delta_total"),
            targets["fg_batch"],
            masks.get("delta_total"),
            masks.get("importance_ghost"),
        )
    return metrics


def _per_complex_sum_metrics(
    pred_per_fg: Tensor,
    delta_total_target: Tensor | None,
    fg_batch: Tensor,
    delta_total_mask: Tensor | None,
    ghost_mask: Tensor | None,
) -> dict:
    """Pooled metrics comparing Σ pred_ghost_per_fg in complex vs baseline ΔG.

    This is the diagnostic that pairs with the sum-constraint regularizer.
    """
    if delta_total_target is None or fg_batch.numel() == 0:
        return {"pearson_r": float("nan"), "spearman_r": float("nan"), "mae": float("nan")}
    fg_batch = fg_batch.long()
    n_complex = int(delta_total_target.numel())
    gm = ghost_mask.bool() if ghost_mask is not None else torch.ones_like(pred_per_fg, dtype=torch.bool)
    contrib = torch.where(gm, pred_per_fg, pred_per_fg.new_zeros(()))
    per_complex = pred_per_fg.new_zeros((n_complex,))
    per_complex.index_add_(0, fg_batch, contrib)
    return {
        "pearson_r": pearson_r(per_complex, delta_total_target, delta_total_mask),
        "spearman_r": spearman_r(per_complex, delta_total_target, delta_total_mask),
        "mae": mae(per_complex, delta_total_target, delta_total_mask),
    }


def compute_prediction_diagnostics(
    preds: dict,
    targets: dict,
    masks: dict,
    baseline_means: dict[str, float] | None = None,
) -> dict:
    baseline_means = baseline_means or {}
    diagnostics: dict[str, object] = {}
    for mode in ("ghost", "charge", "lj"):
        key = f"importance_{mode}"
        mask = masks.get(key, masks.get("fg"))
        diagnostics[key] = _std_summary(preds[key], targets[key], mask)

    ghost_mask = masks.get("importance_ghost", masks.get("fg"))
    ghost_baseline = baseline_means.get("importance_ghost")
    if ghost_baseline is not None:
        baseline_mae = _constant_baseline_mae(targets["importance_ghost"], ghost_baseline, ghost_mask)
        diagnostics["importance_ghost_mean_baseline_mae"] = baseline_mae
        diagnostics["importance_ghost_mae_improvement_vs_baseline"] = baseline_mae - mae(
            preds["importance_ghost"], targets["importance_ghost"], ghost_mask
        )
    return diagnostics


def per_chemotype_metrics(
    preds: Tensor,
    targets: Tensor,
    fg_types: Tensor,
    mask: Tensor | None = None,
) -> dict:
    result: dict[int, dict[str, float]] = {}
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
