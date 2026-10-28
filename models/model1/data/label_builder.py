# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from dataloader.tensor_schema import MASKED_MODE_ORDER, MMGBSA_COMPONENT_ORDER, MMGBSA_ENTITY_ORDER

from .graph_builder import build_cross_attn_mask


def aggregate_residue_channels_to_fg(
    residue_channels: Tensor,
    fg_to_residue_dists: Tensor,
    cutoff: float,
    residue_valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    residue_channels = residue_channels.float()
    fg_to_residue_dists = fg_to_residue_dists.float()
    if residue_channels.dim() == 2:
        residue_channels = residue_channels.unsqueeze(0).expand(fg_to_residue_dists.shape[0], -1, -1)
    if residue_channels.shape[0] != fg_to_residue_dists.shape[0]:
        raise ValueError("FG axis mismatch between residue channels and FG-residue distances")
    if residue_channels.shape[1] != fg_to_residue_dists.shape[1]:
        raise ValueError("Residue axis mismatch between residue channels and FG-residue distances")
    if residue_channels.numel() == 0:
        return residue_channels.new_full((fg_to_residue_dists.shape[0], 4), float("nan")), residue_channels.new_zeros(
            (fg_to_residue_dists.shape[0],), dtype=torch.bool
        )

    channel_finite = torch.isfinite(residue_channels[..., 1:5]).all(dim=-1)
    if residue_valid_mask is None:
        residue_valid_mask = channel_finite
    else:
        residue_valid_mask = residue_valid_mask.bool() & channel_finite
        if residue_valid_mask.dim() == 1:
            residue_valid_mask = residue_valid_mask.unsqueeze(0).expand(fg_to_residue_dists.shape[0], -1)
    valid = (fg_to_residue_dists <= float(cutoff)) & residue_valid_mask

    if fg_to_residue_dists.shape[1]:
        masked_dists = fg_to_residue_dists.masked_fill(~residue_valid_mask, float("inf"))
        nearest_valid = masked_dists.argmin(dim=-1)
        has_any_valid_residue = residue_valid_mask.any(dim=-1)
        empty = ~valid.any(dim=-1) & has_any_valid_residue
        if empty.any():
            valid[empty] = False
            valid[empty, nearest_valid[empty]] = True

    fg_valid = valid.any(dim=-1)
    masked = torch.where(valid.unsqueeze(-1), torch.nan_to_num(residue_channels, nan=0.0), torch.zeros_like(residue_channels))
    # residue channels are [internal, vdw, elec, polar, nonpolar, total]
    out = residue_channels.new_full((fg_to_residue_dists.shape[0], 4), float("nan"))
    if fg_valid.any():
        out[fg_valid] = masked[..., 1:5].sum(dim=1)[fg_valid]
    return out, fg_valid


def compute_residue_to_fg_partition_weights(
    fg_to_residue_dists: Tensor,
    cutoff: float,
    residue_valid_mask: Tensor | None = None,
    *,
    temperature: float = 1.5,
) -> tuple[Tensor, Tensor]:
    fg_to_residue_dists = fg_to_residue_dists.float()
    n_fg, n_res = fg_to_residue_dists.shape
    weights = fg_to_residue_dists.new_zeros((n_fg, n_res))
    fg_valid = torch.zeros((n_fg,), dtype=torch.bool, device=fg_to_residue_dists.device)
    if n_fg == 0 or n_res == 0:
        return weights, fg_valid

    if residue_valid_mask is None:
        residue_valid_mask = torch.ones((n_res,), dtype=torch.bool, device=fg_to_residue_dists.device)
    else:
        residue_valid_mask = residue_valid_mask.bool().to(fg_to_residue_dists.device)
        if residue_valid_mask.dim() != 1 or residue_valid_mask.shape[0] != n_res:
            raise ValueError("residue_valid_mask must be 1D with one entry per residue")

    valid_residues = residue_valid_mask.any()
    if not bool(valid_residues.item()):
        return weights, fg_valid

    cutoff_scores = torch.exp(-fg_to_residue_dists / max(float(temperature), 1e-6))
    cutoff_scores = cutoff_scores * (fg_to_residue_dists <= float(cutoff)).float()
    cutoff_scores = cutoff_scores * residue_valid_mask.unsqueeze(0).float()

    uncovered = cutoff_scores.sum(dim=0) <= 0
    uncovered = uncovered & residue_valid_mask
    if uncovered.any():
        masked_dists = fg_to_residue_dists.masked_fill(~residue_valid_mask.unsqueeze(0), float("inf"))
        nearest_fg = masked_dists[:, uncovered].argmin(dim=0)
        residue_idx = torch.nonzero(uncovered, as_tuple=False).squeeze(-1)
        cutoff_scores[:, residue_idx] = 0.0
        cutoff_scores[nearest_fg, residue_idx] = 1.0

    denom = cutoff_scores.sum(dim=0, keepdim=True).clamp_min(1e-6)
    weights = cutoff_scores / denom
    fg_valid = weights.sum(dim=1) > 0
    return weights, fg_valid


def aggregate_residue_channels_to_fg_partitioned(
    residue_channels: Tensor,
    fg_to_residue_dists: Tensor,
    cutoff: float,
    residue_valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    residue_channels = residue_channels.float()
    fg_to_residue_dists = fg_to_residue_dists.float()
    if residue_channels.dim() != 2:
        raise ValueError("partitioned aggregation expects [n_residue, n_channel] residue_channels")
    if residue_channels.shape[0] != fg_to_residue_dists.shape[1]:
        raise ValueError("Residue axis mismatch between residue channels and FG-residue distances")
    if residue_channels.numel() == 0:
        return residue_channels.new_zeros((fg_to_residue_dists.shape[0], 4)), residue_channels.new_zeros(
            (fg_to_residue_dists.shape[0],), dtype=torch.bool
        )

    channel_finite = torch.isfinite(residue_channels[..., 1:5]).all(dim=-1)
    if residue_valid_mask is None:
        residue_valid_mask = channel_finite
    else:
        residue_valid_mask = residue_valid_mask.bool() & channel_finite

    weights, fg_valid = compute_residue_to_fg_partition_weights(
        fg_to_residue_dists,
        cutoff,
        residue_valid_mask=residue_valid_mask,
    )
    out = residue_channels.new_zeros((fg_to_residue_dists.shape[0], 4))
    if fg_valid.any():
        residue_values = torch.nan_to_num(residue_channels[..., 1:5], nan=0.0)
        out = weights @ residue_values
    return out, fg_valid


def compute_masked_mode_aux_targets(
    ghost: Tensor,
    charge_only: Tensor,
    lj_only: Tensor,
) -> tuple[Tensor, Tensor]:
    aux_elec = charge_only[..., 1] - ghost[..., 1]
    aux_vdw = lj_only[..., 0] - ghost[..., 0]
    return aux_elec, aux_vdw


def compute_preorg_label(
    mmgbsa_components: Tensor,
    entity_order: list[str],
    component_order: list[str],
) -> Tensor:
    if mmgbsa_components.dim() == 1:
        row = mmgbsa_components
    else:
        row = mmgbsa_components[entity_order.index("differences")]
    idxs = [component_order.index(name) for name in ("BOND", "ANGLE", "DIHED")]
    return row[idxs].sum().reshape(1)


def compute_delta_total_label(
    mmgbsa_components: Tensor,
    entity_order: list[str],
    component_order: list[str],
) -> Tensor:
    if mmgbsa_components.dim() == 1:
        row = mmgbsa_components
    else:
        row = mmgbsa_components[entity_order.index("differences")]
    return row[component_order.index("DELTA TOTAL")].reshape(1)


def build_fg_label_mask(
    labeled_fg_indices: list[int],
    total_fg_count: int,
) -> Tensor:
    mask = torch.zeros((int(total_fg_count),), dtype=torch.bool)
    if labeled_fg_indices:
        mask[torch.as_tensor(labeled_fg_indices, dtype=torch.long)] = True
    return mask


def build_targets_from_batch(
    batch: dict[str, Any],
    cutoff: float,
    *,
    max_abs_delta_total: float = 0.0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    fg_block = batch["functional_groups"]
    pocket_block = batch["pocket"]
    mmgbsa = batch["mmgbsa"]

    fg_ptr = fg_block["fg_ptr"]
    res_ptr = pocket_block["residue_ptr"]
    fg_pos = fg_block["fg_centroid_pos"].float()
    residue_pos = pocket_block["residue_pos"].float()
    batch_size = int(fg_ptr.numel() - 1)
    total_fg = int(fg_pos.shape[0])

    contact = fg_pos.new_zeros((total_fg, 4))
    preorg = fg_pos.new_full((total_fg, 1), float("nan"))
    aux_elec = fg_pos.new_full((total_fg,), float("nan"))
    aux_vdw = fg_pos.new_full((total_fg,), float("nan"))
    delta_total = fg_pos.new_full((batch_size,), float("nan"))

    fg_mask = torch.zeros((total_fg,), dtype=torch.bool, device=fg_pos.device)
    preorg_mask = torch.zeros_like(fg_mask)
    aux_elec_mask = torch.zeros_like(fg_mask)
    aux_vdw_mask = torch.zeros_like(fg_mask)
    complex_mask = torch.zeros((batch_size,), dtype=torch.bool, device=fg_pos.device)

    ghost_idx = MASKED_MODE_ORDER.index("ghost")
    charge_idx = MASKED_MODE_ORDER.index("charge_only")
    lj_idx = MASKED_MODE_ORDER.index("lj_only")
    delta_total_idx = MMGBSA_COMPONENT_ORDER.index("DELTA TOTAL")
    diff_idx = MMGBSA_ENTITY_ORDER.index("differences")

    for b_idx in range(batch_size):
        fg_lo = int(fg_ptr[b_idx].item())
        fg_hi = int(fg_ptr[b_idx + 1].item())
        res_lo = int(res_ptr[b_idx].item())
        res_hi = int(res_ptr[b_idx + 1].item())
        n_res = res_hi - res_lo
        fg_slice = slice(fg_lo, fg_hi)
        res_slice = slice(res_lo, res_hi)

        sample_fg_pos = fg_pos[fg_slice]
        sample_res_pos = residue_pos[res_slice]
        if sample_fg_pos.numel() == 0:
            continue
        if sample_res_pos.numel() == 0:
            dists = sample_fg_pos.new_zeros((sample_fg_pos.shape[0], 0))
        else:
            dists = torch.cdist(sample_fg_pos, sample_res_pos)

        baseline_residue = mmgbsa["baseline_residue_energy"][b_idx, :n_res]
        baseline_residue_valid = (
            mmgbsa["baseline_residue_present"][b_idx, :n_res]
            & mmgbsa["baseline_residue_alignment_mask"][b_idx, :n_res]
        )
        sample_contact, sample_contact_mask = aggregate_residue_channels_to_fg_partitioned(
            baseline_residue,
            dists,
            cutoff,
            residue_valid_mask=baseline_residue_valid,
        )
        contact[fg_slice] = sample_contact
        fg_mask[fg_slice] = sample_contact_mask

        masked_delta = mmgbsa["masked_residue_delta"][fg_slice, :, :n_res]
        masked_residue_present = mmgbsa["masked_residue_present"][fg_slice, :, :n_res]
        masked_residue_alignment = mmgbsa["masked_residue_alignment_mask"][fg_slice, :, :n_res]
        ghost_contact, ghost_contact_mask = aggregate_residue_channels_to_fg(
            masked_delta[:, ghost_idx],
            dists,
            cutoff,
            residue_valid_mask=masked_residue_present[:, ghost_idx] & masked_residue_alignment[:, ghost_idx],
        )
        charge_contact, charge_contact_mask = aggregate_residue_channels_to_fg(
            masked_delta[:, charge_idx],
            dists,
            cutoff,
            residue_valid_mask=masked_residue_present[:, charge_idx] & masked_residue_alignment[:, charge_idx],
        )
        lj_contact, lj_contact_mask = aggregate_residue_channels_to_fg(
            masked_delta[:, lj_idx],
            dists,
            cutoff,
            residue_valid_mask=masked_residue_present[:, lj_idx] & masked_residue_alignment[:, lj_idx],
        )
        sample_aux_elec, sample_aux_vdw = compute_masked_mode_aux_targets(ghost_contact, charge_contact, lj_contact)

        ghost_available = mmgbsa["masked_mode_available"][fg_slice, ghost_idx]
        charge_available = mmgbsa["masked_mode_available"][fg_slice, charge_idx]
        lj_available = mmgbsa["masked_mode_available"][fg_slice, lj_idx]

        ghost_global_delta = mmgbsa["masked_global_delta"][fg_slice, ghost_idx]
        ghost_global_mask = mmgbsa["masked_global_delta_mask"][fg_slice, ghost_idx]
        if ghost_global_delta.numel():
            comp_idx = [MMGBSA_COMPONENT_ORDER.index(name) for name in ("BOND", "ANGLE", "DIHED")]
            valid_preorg = ghost_available & ghost_global_mask[:, comp_idx].all(dim=-1)
            preorg_mask[fg_slice] = valid_preorg
            if valid_preorg.any():
                preorg_values = preorg[fg_slice].clone()
                preorg_values[valid_preorg] = ghost_global_delta[valid_preorg][:, comp_idx].sum(dim=-1, keepdim=True)
                preorg[fg_slice] = preorg_values

        valid_aux_elec = ghost_available & charge_available & ghost_contact_mask & charge_contact_mask
        valid_aux_vdw = ghost_available & lj_available & ghost_contact_mask & lj_contact_mask
        aux_elec_mask[fg_slice] = valid_aux_elec
        aux_vdw_mask[fg_slice] = valid_aux_vdw
        aux_elec_values = aux_elec[fg_slice].clone()
        aux_vdw_values = aux_vdw[fg_slice].clone()
        aux_elec_values[valid_aux_elec] = sample_aux_elec[valid_aux_elec]
        aux_vdw_values[valid_aux_vdw] = sample_aux_vdw[valid_aux_vdw]
        aux_elec[fg_slice] = aux_elec_values
        aux_vdw[fg_slice] = aux_vdw_values

        baseline_delta_mask = mmgbsa["baseline_global_component_mask"][b_idx, diff_idx, delta_total_idx]
        if bool(baseline_delta_mask.item()):
            raw_delta_total = mmgbsa["baseline_global_components"][b_idx, diff_idx, delta_total_idx]
            delta_total[b_idx] = raw_delta_total
            if max_abs_delta_total > 0.0 and abs(float(raw_delta_total.item())) > float(max_abs_delta_total):
                continue
            complex_mask[b_idx] = True

    contact_total = contact.sum(dim=-1)

    targets = {
        "contact": contact,
        "contact_total": contact_total,
        "preorg": preorg,
        "aux_elec": aux_elec,
        "aux_vdw": aux_vdw,
        "delta_total": delta_total,
        "fg_batch": fg_block["fg_batch"],
    }
    masks = {
        "fg": fg_mask,
        "contact": fg_mask,
        "contact_total": fg_mask,
        "preorg": preorg_mask,
        "aux_elec": aux_elec_mask,
        "aux_vdw": aux_vdw_mask,
        "complex": complex_mask,
        "delta_total": complex_mask,
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
