# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

SUPPORTED_LOSS_TYPES = ("mse", "huber", "mae", "log_cosh", "charbonnier", "tukey", "quantile", "ccc")


def _combine_mask(mask: Tensor | None, target: Tensor) -> Tensor:
    finite = torch.isfinite(target)
    while finite.dim() > 1:
        finite = finite.all(dim=-1)
    if mask is None:
        return finite
    return mask.bool() & finite


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = _combine_mask(mask, target)
    if mask.numel() == 0 or not mask.any():
        return pred.new_zeros(())
    if pred.dim() > mask.dim():
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
    diff = (pred - torch.nan_to_num(target, nan=0.0))[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    return diff.square().mean()


def masked_huber(pred: Tensor, target: Tensor, mask: Tensor, beta: float = 1.0) -> Tensor:
    mask = _combine_mask(mask, target)
    if mask.numel() == 0 or not mask.any():
        return pred.new_zeros(())
    if pred.dim() > mask.dim():
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
    pred_masked = pred[mask]
    target_masked = torch.nan_to_num(target, nan=0.0)[mask]
    if pred_masked.numel() == 0:
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred_masked, target_masked, beta=beta, reduction="mean")


def masked_mae(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = _combine_mask(mask, target)
    if mask.numel() == 0 or not mask.any():
        return pred.new_zeros(())
    if pred.dim() > mask.dim():
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
    diff = (pred - torch.nan_to_num(target, nan=0.0))[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    return diff.abs().mean()


def masked_log_cosh(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = _combine_mask(mask, target)
    if mask.numel() == 0 or not mask.any():
        return pred.new_zeros(())
    if pred.dim() > mask.dim():
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
    diff = (pred - torch.nan_to_num(target, nan=0.0))[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    return (diff + F.softplus(-2.0 * diff) - math.log(2.0)).mean()


def masked_charbonnier(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-3) -> Tensor:
    mask = _combine_mask(mask, target)
    if mask.numel() == 0 or not mask.any():
        return pred.new_zeros(())
    if pred.dim() > mask.dim():
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
    diff = (pred - torch.nan_to_num(target, nan=0.0))[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    return (torch.sqrt(diff.square() + eps * eps) - eps).mean()


def masked_tukey(pred: Tensor, target: Tensor, mask: Tensor, c: float = 4.685) -> Tensor:
    mask = _combine_mask(mask, target)
    if mask.numel() == 0 or not mask.any():
        return pred.new_zeros(())
    if pred.dim() > mask.dim():
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
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


def masked_quantile(pred: Tensor, target: Tensor, mask: Tensor, q: float = 0.5) -> Tensor:
    mask = _combine_mask(mask, target)
    if mask.numel() == 0 or not mask.any():
        return pred.new_zeros(())
    if pred.dim() > mask.dim():
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
    diff = (torch.nan_to_num(target, nan=0.0) - pred)[mask]
    if diff.numel() == 0:
        return pred.new_zeros(())
    return torch.maximum(q * diff, (q - 1.0) * diff).mean()


def masked_ccc(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Lin's concordance correlation coefficient (CCC) loss. Returns 1 - CCC.

    CCC = 2 * cov(pred, target) / (var(pred) + var(target) + (mean(pred) - mean(target))^2)
    Optimizes correlation directly (with a mean-offset penalty), which matches the
    evaluation metric better than element-wise regression losses.
    """
    mask = _combine_mask(mask, target)
    if mask.numel() == 0 or not mask.any():
        return pred.new_zeros(())
    if pred.dim() > mask.dim():
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
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


class MultiTaskUncertaintyLoss(nn.Module):
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
        rank_loss_weight: float = 0.0,
        rank_loss_margin: float = 0.0,
        variance_penalty_weight: float = 0.0,
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
        self.rank_loss_weight = float(rank_loss_weight)
        self.rank_loss_margin = float(rank_loss_margin)
        self.variance_penalty_weight = float(variance_penalty_weight)
        scales = [float((task_scales or {}).get(name, 1.0)) for name in self.task_names]
        weights = [float((task_static_weights or {}).get(name, 1.0)) for name in self.task_names]
        clips = [float((task_target_clips or {}).get(name, 0.0)) for name in self.task_names]
        self.register_buffer("task_scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("task_static_weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("task_target_clips", torch.tensor(clips, dtype=torch.float32))

    def forward(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        per_task_losses: dict[str, Tensor] = {}
        total_loss = preds["contact"].new_zeros(())
        contact = preds["contact"]
        contact_total = preds.get("contact_total", contact.sum(dim=-1))
        preorg = preds["preorg"].squeeze(-1)
        delta_total = preds["delta_total"]
        aux_elec_pred = preds.get("aux_elec")
        aux_vdw_pred = preds.get("aux_vdw")

        task_payloads = {
            "contact_total": (
                contact_total,
                targets["contact_total"],
                masks.get("contact_total", masks.get("contact", masks.get("fg"))),
            ),
            "vdw": (contact[:, 0], targets["contact"][:, 0], masks.get("contact", masks.get("fg"))),
            "elec": (contact[:, 1], targets["contact"][:, 1], masks.get("contact", masks.get("fg"))),
            "polar": (contact[:, 2], targets["contact"][:, 2], masks.get("contact", masks.get("fg"))),
            "nonpolar": (contact[:, 3], targets["contact"][:, 3], masks.get("contact", masks.get("fg"))),
            "preorg": (preorg, targets["preorg"].squeeze(-1), masks.get("preorg", masks.get("fg"))),
            "delta_total": (delta_total, targets["delta_total"], masks.get("delta_total", masks.get("complex"))),
        }
        if aux_elec_pred is not None:
            task_payloads["aux_elec"] = (aux_elec_pred, targets["aux_elec"], masks.get("aux_elec", masks.get("fg")))
        if aux_vdw_pred is not None:
            task_payloads["aux_vdw"] = (aux_vdw_pred, targets["aux_vdw"], masks.get("aux_vdw", masks.get("fg")))
        task_scale = {"aux_elec": 0.25, "aux_vdw": 0.25}

        for idx, task_name in enumerate(self.task_names):
            if task_name not in task_payloads:
                raise KeyError(f"Task '{task_name}' requested but the model did not provide matching predictions")
            pred, target, mask = task_payloads[task_name]
            norm = torch.clamp(self.task_scales[idx].to(pred.device), min=1e-6)
            clip_value = float(self.task_target_clips[idx].item())
            if clip_value > 0.0:
                target = torch.clamp(target, min=-clip_value, max=clip_value)
            pred_norm = pred / norm
            target_norm = target / norm
            if self.loss_type == "mse":
                base_loss = masked_mse(pred_norm, target_norm, mask)
            elif self.loss_type == "huber":
                base_loss = masked_huber(pred_norm, target_norm, mask, beta=self.huber_beta)
            elif self.loss_type == "mae":
                base_loss = masked_mae(pred_norm, target_norm, mask)
            elif self.loss_type == "log_cosh":
                base_loss = masked_log_cosh(pred_norm, target_norm, mask)
            elif self.loss_type == "charbonnier":
                base_loss = masked_charbonnier(pred_norm, target_norm, mask, eps=self.charbonnier_eps)
            elif self.loss_type == "tukey":
                base_loss = masked_tukey(pred_norm, target_norm, mask, c=self.tukey_c)
            elif self.loss_type == "quantile":
                base_loss = masked_quantile(pred_norm, target_norm, mask, q=self.quantile_q)
            elif self.loss_type == "ccc":
                base_loss = masked_ccc(pred_norm, target_norm, mask)
            else:
                raise ValueError(f"Unsupported loss_type: {self.loss_type}")
            scaled = task_scale.get(task_name, 1.0) * self.task_static_weights[idx].to(pred.device) * base_loss
            precision = torch.exp(-self.log_vars[idx])
            task_loss = 0.5 * precision * scaled + 0.5 * self.log_vars[idx]
            per_task_losses[task_name] = scaled.detach()
            total_loss = total_loss + task_loss
        if self.variance_penalty_weight > 0.0:
            contact_mask = masks.get("contact_total", masks.get("contact", masks.get("fg")))
            if contact_mask is not None:
                m = _combine_mask(contact_mask, targets["contact_total"])
                pred_c = contact_total[m]
                if pred_c.numel() >= 2:
                    pred_std = pred_c.std(unbiased=False).clamp_min(1e-6)
                    var_penalty = -torch.log(pred_std).mean()
                    per_task_losses["variance_penalty"] = var_penalty.detach()
                    total_loss = total_loss + self.variance_penalty_weight * var_penalty
        if self.rank_loss_weight > 0.0 and "fg_batch" in targets:
            rank_loss = self._pairwise_rank_loss(
                contact_total,
                targets["contact_total"],
                masks.get("contact_total", masks.get("contact", masks.get("fg"))),
                targets["fg_batch"],
            )
            per_task_losses["rank_contact"] = rank_loss.detach()
            total_loss = total_loss + self.rank_loss_weight * rank_loss
        return total_loss, per_task_losses

    def get_loss_weights(self) -> dict[str, float]:
        with torch.no_grad():
            precision = 0.5 * torch.exp(-self.log_vars)
            return {name: float(weight.item()) for name, weight in zip(self.task_names, precision)}

    def _pairwise_rank_loss(
        self,
        pred_total: Tensor,
        target_total: Tensor,
        mask: Tensor | None,
        fg_batch: Tensor,
    ) -> Tensor:
        if mask is None:
            mask = torch.ones_like(pred_total, dtype=torch.bool)
        mask = _combine_mask(mask, target_total)
        losses: list[Tensor] = []
        for complex_id in torch.unique(fg_batch.long()):
            local = (fg_batch.long() == complex_id) & mask
            if int(local.sum().item()) < 2:
                continue
            local_pred = pred_total[local]
            local_target = target_total[local]
            diff_target = local_target.unsqueeze(1) - local_target.unsqueeze(0)
            diff_pred = local_pred.unsqueeze(1) - local_pred.unsqueeze(0)
            pair_mask = diff_target.abs() > 1e-6
            if not pair_mask.any():
                continue
            sign = diff_target.sign()
            margin_term = self.rank_loss_margin - sign * diff_pred
            losses.append(torch.relu(margin_term[pair_mask]).mean())
        if not losses:
            return pred_total.new_zeros(())
        return torch.stack(losses).mean()
