# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Targets for the counterfactual FG-importance task (model2).

For each functional group we read the per-FG global delta produced by the
single-FG-mask MM-GBSA runs and ask "if we delete (or partially mask) this
functional group, how much does the whole-complex DELTA TOTAL change?".

Three masking modes are exposed by the dataset:

- ``ghost``       -> full FG removal       -> total counterfactual importance
- ``charge_only`` -> partial-charge removal -> electrostatic importance
- ``lj_only``     -> LJ-parameter removal  -> vdW importance

This is a direct supervised signal: there is no soft residue-to-FG partition
step, so the primary targets do not require geometric energy allocation.

The model is trained to predict the ghost delta (and the two component
deltas as auxiliary heads), and also a hotspot classification target
("is this FG one of the top-K most important FGs in its own ligand?").

The label_builder also returns a ranking-friendly view: for each complex it
exposes the per-FG ghost ΔΔ together with a complex id so the trainer can
compute per-complex Spearman correlations directly.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from dataloader.tensor_schema import MASKED_MODE_ORDER, MMGBSA_COMPONENT_ORDER, MMGBSA_ENTITY_ORDER

from .graph_builder import build_cross_attn_mask


# Order is preserved between targets, predictions, and the importance head outputs.
IMPORTANCE_MODE_ORDER: tuple[str, ...] = ("ghost", "charge_only", "lj_only")


def _delta_total_idx() -> int:
    return MMGBSA_COMPONENT_ORDER.index("DELTA TOTAL")


def _mode_indices() -> list[int]:
    return [MASKED_MODE_ORDER.index(name) for name in IMPORTANCE_MODE_ORDER]


def _per_complex_top_k_hotspot(
    abs_importance: Tensor,
    valid_mask: Tensor,
    fg_ptr: Tensor,
    k: int,
    min_abs: float,
) -> Tensor:
    """For each complex, mark the top-K FGs by |ghost ΔΔ_total| as positives.

    FGs without a valid ghost target stay negative (not labeled positive).
    Ties are broken by index order; ``min_abs`` enforces a magnitude floor
    so that complexes whose strongest FG is essentially zero do not get
    spurious positives.
    """
    labels = torch.zeros_like(valid_mask, dtype=torch.bool)
    if abs_importance.numel() == 0:
        return labels
    n_complex = int(fg_ptr.numel() - 1)
    k = max(int(k), 0)
    for b_idx in range(n_complex):
        lo = int(fg_ptr[b_idx].item())
        hi = int(fg_ptr[b_idx + 1].item())
        if hi <= lo or k == 0:
            continue
        local_vals = abs_importance[lo:hi].clone()
        local_valid = valid_mask[lo:hi]
        local_vals = torch.where(local_valid, local_vals, local_vals.new_full(local_vals.shape, float("-inf")))
        k_local = min(k, int(local_valid.sum().item()))
        if k_local <= 0:
            continue
        topk = torch.topk(local_vals, k=k_local, largest=True).indices
        topk_global = topk + lo
        if min_abs > 0.0:
            keep = abs_importance[topk_global] >= float(min_abs)
            topk_global = topk_global[keep]
        labels[topk_global] = True
    return labels


def _compute_residue_to_fg_partition_weights(
    fg_to_residue_dists: Tensor,
    cutoff: float,
    residue_valid_mask: Tensor,
    *,
    temperature: float = 1.5,
) -> tuple[Tensor, Tensor]:
    """Same soft partition that model1 uses for its noisy contact targets.

    Returns ``(weights[n_fg, n_res], fg_valid[n_fg])``. Used only by the
    auxiliary contact-decomposition target.
    """
    fg_to_residue_dists = fg_to_residue_dists.float()
    n_fg, n_res = fg_to_residue_dists.shape
    weights = fg_to_residue_dists.new_zeros((n_fg, n_res))
    fg_valid = torch.zeros((n_fg,), dtype=torch.bool, device=fg_to_residue_dists.device)
    if n_fg == 0 or n_res == 0 or not residue_valid_mask.any():
        return weights, fg_valid
    scores = torch.exp(-fg_to_residue_dists / max(float(temperature), 1e-6))
    scores = scores * (fg_to_residue_dists <= float(cutoff)).float()
    scores = scores * residue_valid_mask.unsqueeze(0).float()
    uncovered = (scores.sum(dim=0) <= 0) & residue_valid_mask
    if uncovered.any():
        masked = fg_to_residue_dists.masked_fill(~residue_valid_mask.unsqueeze(0), float("inf"))
        nearest_fg = masked[:, uncovered].argmin(dim=0)
        residue_idx = torch.nonzero(uncovered, as_tuple=False).squeeze(-1)
        scores[:, residue_idx] = 0.0
        scores[nearest_fg, residue_idx] = 1.0
    denom = scores.sum(dim=0, keepdim=True).clamp_min(1e-6)
    weights = scores / denom
    fg_valid = weights.sum(dim=1) > 0
    return weights, fg_valid


def _build_contact_aux_targets(
    fg_ptr: Tensor,
    res_ptr: Tensor,
    fg_pos: Tensor,
    res_pos: Tensor,
    baseline_residue: Tensor,
    baseline_residue_valid: Tensor,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """Soft-partitioned per-FG 4-channel contact targets, model1-style.

    Returns ``(contact[N_fg, 4], mask[N_fg])``. Each FG's contact value is a
    distance-weighted distribution of nearby valid residue energies across
    the four MM-GBSA residue channels (vdw / elec / polar / nonpolar).
    Noisier than the counterfactual ΔΔ targets but useful as a smoothing
    auxiliary loss.
    """
    total_fg = int(fg_pos.shape[0])
    out = fg_pos.new_zeros((total_fg, 4))
    fg_valid = torch.zeros((total_fg,), dtype=torch.bool, device=fg_pos.device)
    for b_idx in range(int(fg_ptr.numel() - 1)):
        fg_lo = int(fg_ptr[b_idx].item())
        fg_hi = int(fg_ptr[b_idx + 1].item())
        res_lo = int(res_ptr[b_idx].item())
        res_hi = int(res_ptr[b_idx + 1].item())
        n_res = res_hi - res_lo
        if fg_hi <= fg_lo or n_res <= 0:
            continue
        local_fg = fg_pos[fg_lo:fg_hi]
        local_res = res_pos[res_lo:res_hi]
        dists = torch.cdist(local_fg, local_res)
        residue_channels = baseline_residue[b_idx, :n_res].float()
        if residue_channels.numel() == 0:
            continue
        channel_finite = torch.isfinite(residue_channels[..., 1:5]).all(dim=-1)
        valid = baseline_residue_valid[b_idx, :n_res].bool() & channel_finite
        weights, fg_valid_local = _compute_residue_to_fg_partition_weights(dists, cutoff, valid)
        if not fg_valid_local.any():
            continue
        residue_values = torch.nan_to_num(residue_channels[..., 1:5], nan=0.0)
        block = weights @ residue_values
        out[fg_lo:fg_hi] = block
        fg_valid[fg_lo:fg_hi] = fg_valid_local
    return out, fg_valid


def build_targets_from_batch(
    batch: dict[str, Any],
    cutoff: float,
    *,
    max_abs_importance: float = 0.0,
    hotspot_top_k: int = 3,
    hotspot_min_abs: float = 0.5,
    outlier_bounds: dict[str, tuple[float, float]] | None = None,
    include_contact_aux: bool = False,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build counterfactual FG-importance targets.

    Parameters
    ----------
    max_abs_importance
        Hard ceiling. Any target with ``|value| > max_abs_importance`` is
        masked out. ``0`` disables. Cheap safety net for pathological
        MM-GBSA values.
    outlier_bounds
        Optional per-task ``{task_name: (lower, upper)}`` map. When supplied
        (typically derived from train-set ``median ± k * robust_std``),
        any target outside its mode's interval is masked out before the
        loss sees it. This is what fights the regression head's
        mean-collapse when a handful of FGs with huge ΔΔ values dominate
        gradients. Predictions for those FGs are still computed by the
        model; only the supervised gradient is dropped.

    Returns
    -------
    targets: dict
        ``importance``        : ``[N_fg, 3]`` ΔΔ_total per (FG, masked mode)
        ``importance_ghost``  : ``[N_fg]`` ghost-mode ΔΔ_total (primary signal)
        ``importance_charge`` : ``[N_fg]`` charge_only-mode ΔΔ_total
        ``importance_lj``     : ``[N_fg]`` lj_only-mode ΔΔ_total
        ``abs_importance``    : ``[N_fg]`` |ghost ΔΔ_total|, used for ranking diagnostics
        ``hotspot``           : ``[N_fg]`` 0/1 hotspot label (per-complex top-K)
        ``fg_batch``          : ``[N_fg]`` complex id for each FG
    masks: dict
        Per-task validity masks. ``importance_ghost`` is the headline mask.
    """
    mode_to_task_name = {
        0: "importance_ghost",
        1: "importance_charge",
        2: "importance_lj",
    }
    fg_block = batch["functional_groups"]
    pocket_block = batch["pocket"]
    mmgbsa = batch["mmgbsa"]

    fg_ptr = fg_block["fg_ptr"]
    fg_batch = fg_block["fg_batch"]
    total_fg = int(fg_ptr[-1].item()) if fg_ptr.numel() else 0
    batch_size = int(fg_ptr.numel() - 1) if fg_ptr.numel() else 0
    device = fg_ptr.device if fg_ptr.numel() else torch.device("cpu")

    importance = torch.full((total_fg, len(IMPORTANCE_MODE_ORDER)), float("nan"), device=device)
    importance_mask = torch.zeros((total_fg, len(IMPORTANCE_MODE_ORDER)), dtype=torch.bool, device=device)

    if total_fg > 0:
        global_delta = mmgbsa["masked_global_delta"].to(device).float()
        global_delta_mask = mmgbsa["masked_global_delta_mask"].to(device).bool()
        mode_available = mmgbsa["masked_mode_available"].to(device).bool()
        delta_idx = _delta_total_idx()
        mode_idxs = _mode_indices()
        for col, mode_idx in enumerate(mode_idxs):
            raw = global_delta[:, mode_idx, delta_idx]
            valid = (
                global_delta_mask[:, mode_idx, delta_idx]
                & mode_available[:, mode_idx]
                & torch.isfinite(raw)
            )
            if max_abs_importance > 0.0:
                valid = valid & (raw.abs() <= float(max_abs_importance))
            if outlier_bounds:
                bounds = outlier_bounds.get(mode_to_task_name[col])
                if bounds is not None:
                    lo, hi = bounds
                    if math.isfinite(lo) and math.isfinite(hi):
                        valid = valid & (raw >= float(lo)) & (raw <= float(hi))
            importance[:, col] = torch.where(valid, raw, importance[:, col])
            importance_mask[:, col] = valid

    importance_ghost = importance[:, 0]
    importance_charge = importance[:, 1]
    importance_lj = importance[:, 2]

    ghost_mask = importance_mask[:, 0]
    charge_mask = importance_mask[:, 1]
    lj_mask = importance_mask[:, 2]

    abs_importance = torch.where(
        ghost_mask,
        importance_ghost.abs(),
        torch.zeros_like(importance_ghost),
    )

    # Electrostatic fraction target: |charge| / (|charge| + |lj| + eps). Bounded in [0, 1],
    # interpretation = "what fraction of this FG's effect is electrostatic". Valid when
    # both charge AND lj have valid targets. The eps avoids div-by-zero on FGs whose
    # |charge| + |lj| is essentially 0.
    importance_ratio = torch.full_like(importance_ghost, float("nan"))
    importance_ratio_mask = charge_mask & lj_mask
    if importance_ratio_mask.any():
        c = importance_charge.abs()
        l = importance_lj.abs()
        importance_ratio = torch.where(
            importance_ratio_mask,
            c / (c + l + 1e-6),
            importance_ratio,
        )

    hotspot = _per_complex_top_k_hotspot(
        abs_importance,
        ghost_mask,
        fg_ptr,
        hotspot_top_k,
        hotspot_min_abs,
    )

    # Complex-level DELTA TOTAL passthrough -- used by the sum-constraint
    # regularizer (Σ pred_ghost in complex ≈ baseline DELTA TOTAL).
    delta_total = torch.full((batch_size,), float("nan"), device=device)
    complex_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    if batch_size > 0 and "baseline_global_components" in mmgbsa:
        diff_idx = MMGBSA_ENTITY_ORDER.index("differences")
        delta_idx = _delta_total_idx()
        baseline = mmgbsa["baseline_global_components"].to(device).float()
        baseline_mask = mmgbsa["baseline_global_component_mask"].to(device).bool()
        raw = baseline[:, diff_idx, delta_idx]
        ok = baseline_mask[:, diff_idx, delta_idx] & torch.isfinite(raw)
        delta_total = torch.where(ok, raw, delta_total)
        complex_mask = ok

    # Auxiliary contact decomposition (model1-style soft partition). Only
    # built when explicitly requested -- it adds N_fg * N_res distance work
    # per batch.
    contact_aux = torch.full((total_fg, 4), float("nan"), device=device)
    contact_aux_mask = torch.zeros((total_fg,), dtype=torch.bool, device=device)
    if include_contact_aux and total_fg > 0 and "baseline_residue_energy" in mmgbsa:
        res_ptr = pocket_block["residue_ptr"]
        fg_pos = fg_block["fg_centroid_pos"].to(device).float()
        res_pos = pocket_block["residue_pos"].to(device).float()
        baseline_res = mmgbsa["baseline_residue_energy"].to(device).float()
        baseline_res_valid = (
            mmgbsa["baseline_residue_present"].to(device).bool()
            & mmgbsa["baseline_residue_alignment_mask"].to(device).bool()
        )
        contact_aux, contact_aux_mask = _build_contact_aux_targets(
            fg_ptr,
            res_ptr,
            fg_pos,
            res_pos,
            baseline_res,
            baseline_res_valid,
            float(cutoff),
        )

    targets = {
        "importance": importance,
        "importance_ghost": importance_ghost,
        "importance_charge": importance_charge,
        "importance_lj": importance_lj,
        "abs_importance": abs_importance,
        "hotspot": hotspot.float(),
        "fg_batch": fg_batch,
        "fg_type": fg_block["fg_type"],
        "delta_total": delta_total,
        "contact_aux": contact_aux,
        "importance_ratio": importance_ratio,
    }
    masks = {
        "fg": ghost_mask,
        "importance": importance_mask,
        "importance_ghost": ghost_mask,
        "importance_charge": charge_mask,
        "importance_lj": lj_mask,
        "hotspot": ghost_mask,
        "complex": complex_mask,
        "delta_total": complex_mask,
        "contact_aux": contact_aux_mask,
        "importance_ratio": importance_ratio_mask,
    }
    return targets, masks


def build_cross_attention_masks_from_batch(batch: dict[str, Any], cutoff: float) -> list[Tensor]:
    fg_ptr = batch["functional_groups"]["fg_ptr"]
    res_ptr = batch["pocket"]["residue_ptr"]
    fg_pos = batch["functional_groups"]["fg_centroid_pos"]
    residue_pos = batch["pocket"]["residue_pos"]
    masks: list[Tensor] = []
    for idx in range(int(fg_ptr.numel() - 1)):
        fg_slice = slice(int(fg_ptr[idx].item()), int(fg_ptr[idx + 1].item()))
        res_slice = slice(int(res_ptr[idx].item()), int(res_ptr[idx + 1].item()))
        masks.append(build_cross_attn_mask(fg_pos[fg_slice], residue_pos[res_slice], cutoff))
    return masks
