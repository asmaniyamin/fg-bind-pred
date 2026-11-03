# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Multi-task loss for the counterfactual FG-importance model (model2).

Active regression+classification tasks:

- ``importance_ghost``  : per-FG regression of ghost-mode ΔΔ_total (primary)
- ``importance_charge`` : per-FG regression of charge_only-mode ΔΔ_total
- ``importance_lj``     : per-FG regression of lj_only-mode ΔΔ_total
- ``hotspot``           : per-FG BCE against per-complex top-K hotspot label

Optional regularizers applied to the Kendall-weighted task sum:

- Sum-constraint (``sum_constraint_weight > 0``): the per-complex sum of
  predicted ghost ΔΔ should track the complex's baseline DELTA TOTAL. Cheap
  Huber penalty; couples the per-FG outputs to a known constraint without
  changing the targets themselves.

- Auxiliary contact decomposition (``aux_contact_weight > 0``): model1's
  noisy soft-partitioned 4-channel contact targets, used purely as a
  representation-learning regularizer. Reads ``preds["contact_aux"]`` and
  ``targets["contact_aux"]`` produced by the model and label builder when
  the corresponding flags are on.

- Variance penalty (``variance_penalty_weight > 0``): explicit anti-
  collapse on predicted ghost importance.

Loss types: 'mse', 'huber', 'mae', 'log_cosh', 'charbonnier', 'tukey',
'quantile', 'ccc', and 'frequency_huber' which buckets train-set
targets into quantile bins and weights each sample by 1/bin_freq. The
freq weights are passed in via ``loss_freq_weights``; the loss falls back
to plain Huber if no bins are configured for the task.

A ``regression_normalize`` toggle (default OFF) controls whether the per-task
robust_std-based scaling is applied to predictions and targets *inside the
loss*. With it off, gradients on high-magnitude FGs are not squashed, which
empirically lets the regression head escape mean-collapse on the ghost task.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

SUPPORTED_LOSS_TYPES = (
    "mse",
    "huber",
    "mae",
    "log_cosh",
    "charbonnier",
    "tukey",
    "quantile",
    "ccc",
    "frequency_huber",
)
REGRESSION_TASKS = ("importance_ghost", "importance_charge", "importance_lj")


def _signed_log_transform(x: Tensor) -> Tensor:
    """``sign(x) * log1p(|x|)`` -- compresses long tails while preserving sign."""
    return torch.sign(x) * torch.log1p(x.abs())


def _combine_mask(mask: Tensor | None, target: Tensor) -> Tensor:
    finite = torch.isfinite(target)
    while finite.dim() > 1:
        finite = finite.all(dim=-1)
    if mask is None:
        return finite
    return mask.bool() & finite


def masked_mse(pred, target, mask, weight=None):
    mask = _combine_mask(mask, target)
    if not mask.any():
        return pred.new_zeros(())
    diff = (pred - torch.nan_to_num(target, nan=0.0))[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    sq = diff.square()
    if weight is not None:
        w = weight[mask]
        return (sq * w).sum() / w.sum().clamp_min(1e-6)
    return sq.mean()


def masked_huber(pred, target, mask, beta: float = 1.0, weight=None):
    mask = _combine_mask(mask, target)
    if not mask.any():
        return pred.new_zeros(())
    p = pred[mask]
    t = torch.nan_to_num(target, nan=0.0)[mask]
    if p.numel() == 0:
        return pred.new_zeros(())
    per = F.smooth_l1_loss(p, t, beta=beta, reduction="none")
    if weight is not None:
        w = weight[mask]
        return (per * w).sum() / w.sum().clamp_min(1e-6)
    return per.mean()


def masked_mae(pred, target, mask, weight=None):
    mask = _combine_mask(mask, target)
    if not mask.any():
        return pred.new_zeros(())
    diff = (pred - torch.nan_to_num(target, nan=0.0))[mask].abs()
    if diff.numel() == 0:
        return pred.new_zeros(())
    if weight is not None:
        w = weight[mask]
        return (diff * w).sum() / w.sum().clamp_min(1e-6)
    return diff.mean()


def masked_log_cosh(pred, target, mask):
    mask = _combine_mask(mask, target)
    if not mask.any():
        return pred.new_zeros(())
    diff = (pred - torch.nan_to_num(target, nan=0.0))[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    return (diff + F.softplus(-2.0 * diff) - math.log(2.0)).mean()


def masked_charbonnier(pred, target, mask, eps: float = 1e-3):
    mask = _combine_mask(mask, target)
    if not mask.any():
        return pred.new_zeros(())
    diff = (pred - torch.nan_to_num(target, nan=0.0))[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    return (torch.sqrt(diff.square() + eps * eps) - eps).mean()


def masked_tukey(pred, target, mask, c: float = 4.685):
    mask = _combine_mask(mask, target)
    if not mask.any():
        return pred.new_zeros(())
    diff = (pred - torch.nan_to_num(target, nan=0.0))[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    abs_diff = diff.abs()
    inside = abs_diff <= c
    u = diff / c
    rho_in = (c * c / 6.0) * (1.0 - (1.0 - u.square()).clamp(min=0.0).pow(3))
    rho_out = torch.full_like(diff, c * c / 6.0)
    rho = torch.where(inside, rho_in, rho_out)
    return rho.mean()


def masked_quantile(pred, target, mask, q: float = 0.5):
    """Pinball / quantile loss. q > 0.5 penalizes under-prediction more,
    which is the right setting for fighting a mean-conservative regressor
    on a heavy right tail (ghost ΔΔ_total)."""
    mask = _combine_mask(mask, target)
    if not mask.any():
        return pred.new_zeros(())
    diff = (torch.nan_to_num(target, nan=0.0) - pred)[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    return torch.maximum(q * diff, (q - 1.0) * diff).mean()


def masked_ccc(pred, target, mask, eps: float = 1e-6):
    mask = _combine_mask(mask, target)
    if not mask.any():
        return pred.new_zeros(())
    p = pred[mask]
    t = torch.nan_to_num(target, nan=0.0)[mask]
    if p.numel() < 2:
        return pred.new_zeros(())
    p_mean = p.mean()
    t_mean = t.mean()
    p_var = ((p - p_mean) ** 2).mean()
    t_var = ((t - t_mean) ** 2).mean()
    cov = ((p - p_mean) * (t - t_mean)).mean()
    denom = p_var + t_var + (p_mean - t_mean).square() + eps
    ccc = 2.0 * cov / denom
    return 1.0 - ccc


def masked_bce_with_logits(pred, target, mask, pos_weight: float | None = None):
    mask = _combine_mask(mask, target)
    if not mask.any():
        return pred.new_zeros(())
    p = pred[mask]
    t = torch.nan_to_num(target, nan=0.0)[mask].float()
    if p.numel() == 0:
        return pred.new_zeros(())
    pos_w = None if pos_weight is None else pred.new_tensor(float(pos_weight))
    return F.binary_cross_entropy_with_logits(p, t, pos_weight=pos_w, reduction="mean")


def _freq_weights_for_targets(
    target: Tensor,
    bin_edges: Tensor | None,
    bin_weights: Tensor | None,
) -> Tensor | None:
    """Per-sample weight = ``bin_weight`` of the bin the target falls into.

    Bins are quantile-based train-set deciles; weights are 1/bin_freq so the
    rare high-importance FGs contribute proportionally more loss. Falls back
    to uniform if no bins were provided.
    """
    if bin_edges is None or bin_weights is None or bin_edges.numel() <= 1:
        return None
    edges = bin_edges.to(device=target.device, dtype=target.dtype)
    weights = bin_weights.to(device=target.device, dtype=target.dtype)
    idx = torch.bucketize(target, edges[1:-1], right=True).clamp(min=0, max=int(weights.numel() - 1))
    return weights[idx]


def _regression_loss(
    loss_type: str,
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    huber_beta: float,
    charbonnier_eps: float,
    tukey_c: float,
    quantile_q: float,
    freq_bin_edges: Tensor | None = None,
    freq_bin_weights: Tensor | None = None,
) -> Tensor:
    if loss_type == "mse":
        return masked_mse(pred, target, mask)
    if loss_type == "huber":
        return masked_huber(pred, target, mask, beta=huber_beta)
    if loss_type == "mae":
        return masked_mae(pred, target, mask)
    if loss_type == "log_cosh":
        return masked_log_cosh(pred, target, mask)
    if loss_type == "charbonnier":
        return masked_charbonnier(pred, target, mask, eps=charbonnier_eps)
    if loss_type == "tukey":
        return masked_tukey(pred, target, mask, c=tukey_c)
    if loss_type == "quantile":
        return masked_quantile(pred, target, mask, q=quantile_q)
    if loss_type == "ccc":
        return masked_ccc(pred, target, mask)
    if loss_type == "frequency_huber":
        w = _freq_weights_for_targets(target, freq_bin_edges, freq_bin_weights)
        return masked_huber(pred, target, mask, beta=huber_beta, weight=w)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


class ImportanceMultiTaskLoss(nn.Module):
    """Kendall-style multi-task loss with optional aux regularizers."""

    def __init__(
        self,
        task_names: list[str],
        *,
        task_scales: dict[str, float] | None = None,
        task_static_weights: dict[str, float] | None = None,
        task_target_clips: dict[str, float] | None = None,
        loss_type: str = "huber",
        huber_beta: float = 1.0,
        charbonnier_eps: float = 1e-3,
        tukey_c: float = 4.685,
        quantile_q: float = 0.5,
        hotspot_pos_weight: float = 5.0,
        variance_penalty_weight: float = 0.0,
        regression_normalize: bool = False,
        sum_constraint_weight: float = 0.0,
        sum_constraint_beta: float = 1.0,
        aux_contact_weight: float = 0.0,
        loss_freq_bin_edges: dict[str, Tensor] | None = None,
        loss_freq_bin_weights: dict[str, Tensor] | None = None,
        per_task_loss_type: dict[str, str] | None = None,
        per_task_quantile_q: dict[str, float] | None = None,
        signed_log_tasks: tuple[str, ...] = (),
        ratio_head_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if loss_type not in SUPPORTED_LOSS_TYPES:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose one of {SUPPORTED_LOSS_TYPES}")
        self.task_names = list(task_names)
        self.log_vars = nn.Parameter(torch.zeros(len(self.task_names)))
        self.loss_type = loss_type
        self.huber_beta = float(huber_beta)
        self.charbonnier_eps = float(charbonnier_eps)
        self.tukey_c = float(tukey_c)
        self.quantile_q = float(quantile_q)
        self.hotspot_pos_weight = float(hotspot_pos_weight)
        self.variance_penalty_weight = float(variance_penalty_weight)
        self.regression_normalize = bool(regression_normalize)
        self.sum_constraint_weight = float(sum_constraint_weight)
        self.sum_constraint_beta = float(sum_constraint_beta)
        self.aux_contact_weight = float(aux_contact_weight)
        # Per-task loss-type overrides. Look up via name; fall back to ``loss_type``.
        self.per_task_loss_type = dict(per_task_loss_type or {})
        self.per_task_quantile_q = dict(per_task_quantile_q or {})
        # Tasks whose targets/predictions are transformed by signed-log inside the loss.
        # Predictions stay in raw kcal/mol externally; the transform is applied only for
        # the loss computation, so metrics on the model output remain in raw units.
        self.signed_log_tasks = set(signed_log_tasks)
        # Optional ratio head: predicts logit(p) where p ≈ charge / (|charge| + |lj| + eps).
        # Loss is BCEWithLogits against the target ratio.
        self.ratio_head_weight = float(ratio_head_weight)
        scales = [float((task_scales or {}).get(name, 1.0)) for name in self.task_names]
        weights = [float((task_static_weights or {}).get(name, 1.0)) for name in self.task_names]
        clips = [float((task_target_clips or {}).get(name, 0.0)) for name in self.task_names]
        self.register_buffer("task_scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("task_static_weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("task_target_clips", torch.tensor(clips, dtype=torch.float32))
        # Frequency-weighting buckets stored as non-persistent buffers per task.
        # We keep the set of task names that have buckets so we can re-resolve
        # the buffers via ``getattr`` at forward time; storing dict refs to the
        # buffers at __init__ time would leave them on CPU even after .to(cuda).
        self._freq_tasks: set[str] = set()
        edges = loss_freq_bin_edges or {}
        bw = loss_freq_bin_weights or {}
        for name in self.task_names:
            if name in edges and name in bw:
                self.register_buffer(f"freq_edges_{name}", edges[name].float(), persistent=False)
                self.register_buffer(f"freq_weights_{name}", bw[name].float(), persistent=False)
                self._freq_tasks.add(name)

    def _freq_bins_for(self, task_name: str) -> tuple[Tensor | None, Tensor | None]:
        if task_name not in self._freq_tasks:
            return None, None
        return getattr(self, f"freq_edges_{task_name}"), getattr(self, f"freq_weights_{task_name}")

    def forward(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        per_task_losses: dict[str, Tensor] = {}
        total_loss = preds["importance_ghost"].new_zeros(())

        task_payloads: dict[str, tuple[Tensor, Tensor, Tensor | None]] = {
            "importance_ghost": (preds["importance_ghost"], targets["importance_ghost"], masks.get("importance_ghost")),
            "importance_charge": (preds["importance_charge"], targets["importance_charge"], masks.get("importance_charge")),
            "importance_lj": (preds["importance_lj"], targets["importance_lj"], masks.get("importance_lj")),
            "hotspot": (preds["hotspot"], targets["hotspot"], masks.get("hotspot")),
        }

        for idx, task_name in enumerate(self.task_names):
            if task_name not in task_payloads:
                raise KeyError(f"Task '{task_name}' requested but the model did not provide matching predictions")
            pred, target, mask = task_payloads[task_name]
            clip_value = float(self.task_target_clips[idx].item())
            if task_name in REGRESSION_TASKS:
                if clip_value > 0.0:
                    target = torch.clamp(target, min=-clip_value, max=clip_value)
                # Optional signed-log target transform: compresses heavy tails by
                # mapping y -> sign(y)*log1p(|y|). Applied symmetrically to pred and target.
                if task_name in self.signed_log_tasks:
                    pred = _signed_log_transform(pred)
                    target = _signed_log_transform(target)
                fe, fw = self._freq_bins_for(task_name)
                if self.regression_normalize:
                    norm = torch.clamp(self.task_scales[idx].to(pred.device), min=1e-6)
                    pred_use = pred / norm
                    target_use = target / norm
                    freq_edges = fe / norm if fe is not None else None
                    freq_weights = fw
                else:
                    pred_use = pred
                    target_use = target
                    freq_edges = fe
                    freq_weights = fw
                # Resolve per-task loss-type override (e.g. {'importance_charge': 'quantile'})
                eff_loss_type = self.per_task_loss_type.get(task_name, self.loss_type)
                eff_quantile = float(self.per_task_quantile_q.get(task_name, self.quantile_q))
                base_loss = _regression_loss(
                    eff_loss_type,
                    pred_use,
                    target_use,
                    mask,
                    huber_beta=self.huber_beta,
                    charbonnier_eps=self.charbonnier_eps,
                    tukey_c=self.tukey_c,
                    quantile_q=eff_quantile,
                    freq_bin_edges=freq_edges,
                    freq_bin_weights=freq_weights,
                )
            elif task_name == "hotspot":
                base_loss = masked_bce_with_logits(pred, target, mask, pos_weight=self.hotspot_pos_weight)
            else:
                raise KeyError(f"Unknown task '{task_name}'")

            scaled = self.task_static_weights[idx].to(pred.device) * base_loss
            precision = torch.exp(-self.log_vars[idx])
            task_loss = 0.5 * precision * scaled + 0.5 * self.log_vars[idx]
            per_task_losses[task_name] = scaled.detach()
            total_loss = total_loss + task_loss

        if self.variance_penalty_weight > 0.0:
            mask = masks.get("importance_ghost")
            if mask is not None:
                m = _combine_mask(mask, targets["importance_ghost"])
                pred_g = preds["importance_ghost"][m]
                if pred_g.numel() >= 2:
                    pred_std = pred_g.std(unbiased=False).clamp_min(1e-6)
                    var_penalty = -torch.log(pred_std).mean()
                    per_task_losses["variance_penalty"] = var_penalty.detach()
                    total_loss = total_loss + self.variance_penalty_weight * var_penalty

        if self.sum_constraint_weight > 0.0 and "delta_total" in targets and "fg_batch" in targets:
            sum_loss = self._sum_constraint_loss(
                preds["importance_ghost"],
                targets["delta_total"],
                masks.get("delta_total"),
                masks.get("importance_ghost"),
                targets["fg_batch"],
            )
            per_task_losses["sum_constraint"] = sum_loss.detach()
            total_loss = total_loss + self.sum_constraint_weight * sum_loss

        if self.aux_contact_weight > 0.0 and "contact_aux" in preds and "contact_aux" in targets:
            aux_loss = masked_huber(
                preds["contact_aux"],
                targets["contact_aux"],
                masks.get("contact_aux"),
                beta=self.huber_beta,
            )
            per_task_losses["contact_aux"] = aux_loss.detach()
            total_loss = total_loss + self.aux_contact_weight * aux_loss

        if (
            self.ratio_head_weight > 0.0
            and "importance_ratio_logit" in preds
            and "importance_ratio" in targets
        ):
            ratio_loss = masked_bce_with_logits(
                preds["importance_ratio_logit"],
                targets["importance_ratio"],
                masks.get("importance_ratio"),
                pos_weight=None,
            )
            per_task_losses["importance_ratio"] = ratio_loss.detach()
            total_loss = total_loss + self.ratio_head_weight * ratio_loss

        return total_loss, per_task_losses

    def _sum_constraint_loss(
        self,
        pred_ghost: Tensor,
        delta_total_target: Tensor,
        delta_total_mask: Tensor | None,
        ghost_mask: Tensor | None,
        fg_batch: Tensor,
    ) -> Tensor:
        """Per-complex constraint: Σ ghost ΔΔ_FG ≈ -baseline_ΔG_bind.

        Physical reasoning: if we ghosted EVERY FG, binding contacts would
        vanish and ΔG_bind would go to ~0; the cumulative ΔΔ across all FGs
        therefore approximates ``-ΔG_bind_baseline``. Baseline DELTA TOTAL
        is negative for a bound complex (~-50 kcal/mol), so the sum target
        is its sign-flip (positive). This is a soft regularizer because
        ghost-mask effects are not strictly additive (FG synergy).
        """
        if delta_total_target.numel() == 0:
            return pred_ghost.new_zeros(())
        device = pred_ghost.device
        fg_batch = fg_batch.long()
        n_complex = int(delta_total_target.numel())
        gm = (ghost_mask.bool() if ghost_mask is not None else torch.ones_like(pred_ghost, dtype=torch.bool))
        contrib = torch.where(gm, pred_ghost, pred_ghost.new_zeros(()))
        per_complex_sum = pred_ghost.new_zeros((n_complex,))
        per_complex_sum.index_add_(0, fg_batch, contrib)
        target = -torch.nan_to_num(delta_total_target.to(device), nan=0.0)  # sign flip
        dm = (delta_total_mask.bool() if delta_total_mask is not None else torch.ones_like(target, dtype=torch.bool))
        if not dm.any():
            return pred_ghost.new_zeros(())
        diff = per_complex_sum - target
        return F.smooth_l1_loss(diff[dm], torch.zeros_like(diff[dm]), beta=self.sum_constraint_beta, reduction="mean")

    def get_loss_weights(self) -> dict[str, float]:
        with torch.no_grad():
            precision = 0.5 * torch.exp(-self.log_vars)
            return {name: float(weight.item()) for name, weight in zip(self.task_names, precision)}
