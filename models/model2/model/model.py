# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Counterfactual FG-importance model (model2).

The encoder, GNN, and attention stack produce per-FG representations
``h_fg_final`` for the prediction heads:

- ``FGImportanceHead`` (default): MLP regression to ΔΔ_total per masked mode
- ``FGResiduePairImportanceHead`` (opt-in, ``use_pair_importance_head``):
  model1-style FG-residue pair scorer producing the same 3-mode output
- ``FGHotspotHead``: per-FG binary hotspot logit
- ``ContactDecompositionHead`` (opt-in, ``use_contact_aux_head``): auxiliary
  4-channel soft contact decomposition for representation regularization
- Per-FG local environment summary (always computed when
  ``fg_env_summary_dim > 0``) - explicit geometric features (residue count
  / mean residue charge / mean residue distance within cutoff) concatenated
  to the head inputs.
- Per-ligand-atom Coulombic AEV channel (always computed when
  ``model_config.use_elec_aev``) concatenated alongside the ECIF AEV inside
  the ligand atom encoder.

"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..config import ModelConfig
from ..data.graph_builder import (
    ANI2X_RS,
    compute_ecif_types_from_pocket_features,
    compute_fg_electrostatic_features,
    compute_fg_residue_coulomb,
    compute_ligand_aev,
    compute_ligand_electrostatic_aev,
    radius_graph_from_positions,
)
from ..data.label_builder import build_cross_attention_masks_from_batch
from ..utils.rbf import get_rbf_centers, get_rbf_widths, rbf_expand
from .attention import FGMoleculeSelfAttention, FGPocketCrossAttention, FGSelfAttentionGraph
from .encoders import FGNodeEncoder, LigandAtomEncoder, LigandBondEncoder, PocketResidueEncoder
from .gnn import LigandGNN
from .heads import (
    ContactDecompositionHead,
    FGHotspotHead,
    FGImportanceHead,
    FGResiduePairImportanceHead,
)
from .pooling import FGAttentionPool


# Width of the per-FG local-environment summary tensor when the feature is
# enabled. See `_compute_fg_env_summary` for the exact channels.
FG_ENV_SUMMARY_DIM = 5


class FGImportanceModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        elec_aev_dim = 2 if bool(getattr(config, "use_elec_aev", False)) else 0
        self.atom_encoder = LigandAtomEncoder(config.hidden_dim, elec_aev_dim=elec_aev_dim)
        self.bond_encoder = LigandBondEncoder(config.ligand_edge_dim)
        fg_elec_dim = 5 if bool(getattr(config, "use_fg_elec_features", False)) else 0
        self.fg_encoder = FGNodeEncoder(config.hidden_dim, extra_elec_dim=fg_elec_dim)
        self.fg_elec_dim = fg_elec_dim
        self.residue_encoder = PocketResidueEncoder(
            config.hidden_dim,
            config.num_rbf_kernels,
            config.rbf_d_min,
            config.rbf_d_max,
        )
        self.ligand_gnn = LigandGNN(
            config.hidden_dim,
            config.ligand_edge_dim,
            config.num_ligand_gnn_layers,
            config.num_heads,
            config.dropout,
        )
        self.fg_pool = FGAttentionPool(config.hidden_dim)
        self.fg_context = FGSelfAttentionGraph(
            config.hidden_dim,
            config.num_rbf_kernels,
            config.fg_edge_dim,
            config.num_fg_attn_layers,
            config.num_fg_graph_heads,
            config.dropout,
        )
        self.fg_global_context = FGMoleculeSelfAttention(
            config.hidden_dim,
            config.num_fg_global_attn_layers,
            config.num_cross_attn_heads,
            config.dropout,
        )
        self.cross_attention = FGPocketCrossAttention(
            config.hidden_dim,
            config.num_cross_attn_layers,
            config.num_cross_attn_heads,
            config.dropout,
        )

        # Per-FG extras concatenated to the importance / hotspot / aux contact head
        # inputs. = env summary (5 ch) + FG-residue Coulomb (2 ch) when enabled.
        self.fg_env_summary_dim = FG_ENV_SUMMARY_DIM if bool(getattr(config, "use_fg_env_summary", False)) else 0
        self.fg_residue_coulomb_dim = 2 if bool(getattr(config, "use_fg_residue_coulomb", False)) else 0
        extra = self.fg_env_summary_dim + self.fg_residue_coulomb_dim

        self.use_pair_head = bool(getattr(config, "use_pair_importance_head", False))
        if self.use_pair_head:
            self.importance_head = FGResiduePairImportanceHead(
                config.hidden_dim, config.num_rbf_kernels, config.num_importance_modes
            )
        else:
            self.importance_head = FGImportanceHead(
                config.hidden_dim, config.num_importance_modes, extra_input_dim=extra
            )
        self.hotspot_head = FGHotspotHead(config.hidden_dim, extra_input_dim=extra)

        self.use_contact_aux = bool(getattr(config, "use_contact_aux_head", False))
        if self.use_contact_aux:
            self.contact_aux_head = ContactDecompositionHead(config.hidden_dim, extra_input_dim=extra)

        # Optional 4th regression head: ratio = charge / (charge + lj + eps)
        self.use_ratio_head = bool(getattr(config, "use_ratio_head", False))
        if self.use_ratio_head:
            self.ratio_head = FGImportanceHead(config.hidden_dim, num_modes=1, extra_input_dim=extra)

        self.register_buffer(
            "fg_rbf_centers",
            get_rbf_centers(config.num_rbf_kernels, config.rbf_d_min, config.rbf_d_max),
            persistent=False,
        )
        self.register_buffer(
            "fg_rbf_widths",
            get_rbf_widths(config.num_rbf_kernels, config.rbf_d_min, config.rbf_d_max),
            persistent=False,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        h_atom, h_bond, h_fg_init, h_res = self._encode_inputs(batch)
        h_atom = self._run_ligand_gnn(h_atom, batch["ligand"]["edge_index"], h_bond)
        h_fg = self.fg_pool(
            h_atom,
            batch["functional_groups"]["fg_membership_index"],
            h_fg_init,
            h_fg_init.shape[0],
        )
        fg_slices = self._fg_slices(batch)
        fg_edge_index, fg_edge_attr = self._build_fg_edges(batch)
        h_fg_ctx = self.fg_context(h_fg, fg_edge_index, fg_edge_attr)
        h_fg_cross = self._cross_attend(h_fg_ctx, h_res, batch, fg_slices=fg_slices)
        h_fg_final = self.fg_global_context(h_fg_cross, fg_slices)

        # Build the per-FG extras tensor (env summary + FG-residue Coulomb) once.
        extras_parts: list[Tensor] = []
        if self.fg_env_summary_dim > 0:
            extras_parts.append(self._compute_fg_env_summary(batch))
        if self.fg_residue_coulomb_dim > 0:
            extras_parts.append(self._compute_fg_residue_coulomb(batch))
        extras = torch.cat(extras_parts, dim=-1) if extras_parts else None

        if self.use_pair_head:
            importance = self._predict_pair_importance(h_fg_final, h_res, batch)
        else:
            importance = self.importance_head(h_fg_final, extras)

        hotspot_logits = self.hotspot_head(h_fg_final, extras)

        out: dict[str, Tensor] = {
            "importance": importance,
            "importance_ghost": importance[:, 0],
            "importance_charge": importance[:, 1] if importance.shape[-1] > 1 else importance[:, 0],
            "importance_lj": importance[:, 2] if importance.shape[-1] > 2 else importance[:, 0],
            "hotspot": hotspot_logits,
        }
        if self.use_contact_aux:
            out["contact_aux"] = self.contact_aux_head(h_fg_final, extras)
        if self.use_ratio_head:
            out["importance_ratio_logit"] = self.ratio_head(h_fg_final, extras).squeeze(-1)
        return out

    def _encode_inputs(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ligand = batch["ligand"]
        pocket = batch["pocket"]
        fg = batch["functional_groups"]

        aev = self._compute_batch_aev(batch)
        elec_aev = None
        if bool(getattr(self.config, "use_elec_aev", False)):
            elec_aev = self._compute_batch_elec_aev(batch)
        ligand_x = ligand["x"]
        h_lig_atom = self.atom_encoder(
            atomic_num=ligand_x[:, 0],
            formal_charge=ligand_x[:, 1],
            partial_charge=ligand_x[:, 2],
            is_aromatic=ligand_x[:, 3],
            hybridization=ligand_x[:, 4],
            chiral_tag=ligand_x[:, 5],
            explicit_h=ligand_x[:, 6],
            implicit_h=ligand_x[:, 7],
            isotope=ligand_x[:, 8],
            aev=aev,
            elec_aev=elec_aev,
        )

        edge_attr = ligand["edge_attr"]
        h_bond = self.bond_encoder(
            bond_type=edge_attr[:, 0] if edge_attr.numel() else edge_attr.new_zeros((0,)),
            is_conjugated=edge_attr[:, 1] if edge_attr.numel() else edge_attr.new_zeros((0,)),
            is_in_ring=edge_attr[:, 2] if edge_attr.numel() else edge_attr.new_zeros((0,)),
            bond_stereo=edge_attr[:, 3] if edge_attr.numel() else edge_attr.new_zeros((0,)),
            bond_dir=edge_attr[:, 4] if edge_attr.numel() else edge_attr.new_zeros((0,)),
        )

        fg_features = fg["fg_features"]
        extra_elec = None
        if self.fg_elec_dim > 0:
            extra_elec = compute_fg_electrostatic_features(
                ligand_pos=ligand["pos"],
                ligand_partial_charge=ligand_x[:, 2] if ligand_x.numel() else ligand_x.new_zeros((0,)),
                fg_centroid_pos=fg["fg_centroid_pos"],
                fg_membership_index=fg["fg_membership_index"],
            )
        h_fg_init = self.fg_encoder(
            num_atoms=fg_features[:, 0] if fg_features.numel() else fg_features.new_zeros((0,)),
            total_partial_charge=fg_features[:, 1] if fg_features.numel() else fg_features.new_zeros((0,)),
            max_radial_extent=fg_features[:, 2] if fg_features.numel() else fg_features.new_zeros((0,)),
            mean_partial_charge=fg_features[:, 3] if fg_features.numel() else fg_features.new_zeros((0,)),
            fg_type=fg["fg_type"],
            fg_parent_type=fg["fg_parent_type"],
            extra_elec=extra_elec,
        )

        residue_x = pocket["residue_x"]
        h_res = self.residue_encoder(
            residue_type_id=residue_x[:, 0] if residue_x.numel() else residue_x.new_zeros((0,)),
            residue_base_type_id=residue_x[:, 1] if residue_x.numel() else residue_x.new_zeros((0,)),
            residue_variant_id=residue_x[:, 2] if residue_x.numel() else residue_x.new_zeros((0,)),
            atom_count=residue_x[:, 3] if residue_x.numel() else residue_x.new_zeros((0,)),
            min_ligand_dist=residue_x[:, 4] if residue_x.numel() else residue_x.new_zeros((0,)),
        )
        return h_lig_atom, h_bond, h_fg_init, h_res

    def _compute_batch_aev(self, batch: dict[str, Any]) -> Tensor:
        ligand = batch["ligand"]
        pocket = batch["pocket"]
        ligand_ptr = ligand["ptr"]
        pocket_ptr = pocket["ptr"]
        ligand_pos = ligand["pos"]
        pocket_pos = pocket["pos"]
        pocket_x = pocket["x"]
        aev = ligand_pos.new_zeros((ligand_pos.shape[0], self.config.aev_dim))
        if not self.config.use_aev:
            return aev
        for idx in range(int(ligand_ptr.numel() - 1)):
            lig_slice = slice(int(ligand_ptr[idx].item()), int(ligand_ptr[idx + 1].item()))
            poc_slice = slice(int(pocket_ptr[idx].item()), int(pocket_ptr[idx + 1].item()))
            lig_coords = ligand_pos[lig_slice]
            poc_coords = pocket_pos[poc_slice]
            if lig_coords.numel() == 0:
                continue
            if poc_coords.numel() == 0:
                aev[lig_slice] = lig_coords.new_zeros((lig_coords.shape[0], self.config.aev_dim))
                continue
            poc_x = pocket_x[poc_slice]
            poc_types = compute_ecif_types_from_pocket_features(
                atomic_num=poc_x[:, 0],
                residue_base_type_id=poc_x[:, 2],
                is_backbone=poc_x[:, 5],
                is_water=poc_x[:, 7],
            )
            aev[lig_slice] = compute_ligand_aev(
                lig_coords,
                poc_coords,
                poc_types,
                self.config.aev_rc,
                ANI2X_RS.to(lig_coords.device),
                self.config.aev_eta,
            )
        return aev

    def _compute_batch_elec_aev(self, batch: dict[str, Any]) -> Tensor:
        """Coulombic electrostatic AEV per ligand atom. 2 channels (signed, abs density).

        Pocket partial charges live in ``pocket['x'][:, ?]`` ... but the
        tensorized pocket atom feature schema only has atomic_num and
        residue ids - no explicit partial charge. As a stand-in, we use a
        type-conditioned charge prior approximated from `residue_base_type_id`
        and `is_backbone` flags via a fixed lookup table, see
        ``_pocket_charge_prior`` below. This is intentionally crude; the
        head learns to scale it.
        """
        ligand = batch["ligand"]
        pocket = batch["pocket"]
        ligand_ptr = ligand["ptr"]
        pocket_ptr = pocket["ptr"]
        ligand_pos = ligand["pos"]
        pocket_pos = pocket["pos"]
        ligand_x = ligand["x"]
        pocket_x = pocket["x"]
        rc = float(self.config.aev_rc)
        out = ligand_pos.new_zeros((ligand_pos.shape[0], 2))
        for idx in range(int(ligand_ptr.numel() - 1)):
            lig_slice = slice(int(ligand_ptr[idx].item()), int(ligand_ptr[idx + 1].item()))
            poc_slice = slice(int(pocket_ptr[idx].item()), int(pocket_ptr[idx + 1].item()))
            lig_coords = ligand_pos[lig_slice]
            poc_coords = pocket_pos[poc_slice]
            if lig_coords.numel() == 0 or poc_coords.numel() == 0:
                continue
            lig_q = ligand_x[lig_slice, 2]
            poc_q = _pocket_charge_prior(
                atomic_num=pocket_x[poc_slice, 0],
                residue_base_type_id=pocket_x[poc_slice, 2],
                is_backbone=pocket_x[poc_slice, 5],
                residue_variant_id=pocket_x[poc_slice, 3]
                    if bool(getattr(self.config, "use_variant_aware_charge_prior", True)) else None,
            )
            out[lig_slice] = compute_ligand_electrostatic_aev(
                lig_coords, lig_q, poc_coords, poc_q, rc
            )
        return out

    def _compute_fg_residue_coulomb(self, batch: dict[str, Any]) -> Tensor:
        """Per-FG signed Coulombic interaction with the surrounding pocket residues.

        Two channels per FG (see ``compute_fg_residue_coulomb``):
        ``[Σ q_FG·q_residue/r_eff, Σ |q_residue|/r_eff]``. Residue charges
        come from a base-type + variant lookup that knows about charged
        residues (ASP, GLU, ARG, LYS, HIS) and their protonation variants
        (ASH/GLH/LYN/HIP/HID/HIE). FG net charge is summed from the per-
        atom partial charges of the FG's member atoms.
        """
        fg = batch["functional_groups"]
        pocket = batch["pocket"]
        ligand = batch["ligand"]
        fg_ptr = fg["fg_ptr"]
        res_ptr = pocket["residue_ptr"]
        fg_pos = fg["fg_centroid_pos"]
        res_pos = pocket["residue_pos"]
        residue_x = pocket["residue_x"]
        ligand_pos = ligand["pos"]
        ligand_q = ligand["x"][:, 2] if ligand["x"].numel() else ligand_pos.new_zeros((0,))
        membership = fg["fg_membership_index"]
        n_fg_total = int(fg_pos.shape[0])
        out = fg_pos.new_zeros((n_fg_total, 2))
        if n_fg_total == 0 or membership.numel() == 0:
            return out
        # FG net charge from ligand partial charges via membership.
        fg_net_q = fg_pos.new_zeros((n_fg_total,))
        atom_idx = membership[0].long()
        fg_idx = membership[1].long()
        fg_net_q.index_add_(0, fg_idx, ligand_q.float()[atom_idx])
        cutoff = float(self.config.cross_attn_cutoff)
        for b in range(int(fg_ptr.numel() - 1)):
            fg_lo = int(fg_ptr[b].item())
            fg_hi = int(fg_ptr[b + 1].item())
            res_lo = int(res_ptr[b].item())
            res_hi = int(res_ptr[b + 1].item())
            if fg_hi <= fg_lo or res_hi <= res_lo:
                continue
            res_q = _residue_effective_charge(
                base=residue_x[res_lo:res_hi, 1],
                variant=residue_x[res_lo:res_hi, 2]
                    if bool(getattr(self.config, "use_variant_aware_charge_prior", True))
                    else torch.zeros_like(residue_x[res_lo:res_hi, 2]),
            )
            out[fg_lo:fg_hi] = compute_fg_residue_coulomb(
                fg_centroid_pos=fg_pos[fg_lo:fg_hi],
                fg_net_charge=fg_net_q[fg_lo:fg_hi],
                residue_pos=res_pos[res_lo:res_hi],
                residue_charge=res_q,
                cutoff=cutoff,
            )
        return out

    def _compute_fg_env_summary(self, batch: dict[str, Any]) -> Tensor:
        """Per-FG local pocket-environment summary.

        Five channels, one row per FG (matches ``FG_ENV_SUMMARY_DIM``):

        ``0``: number of residues within ``cross_attn_cutoff`` of the FG centroid
        ``1``: log1p of that count (a soft saturation)
        ``2``: mean residue ``atom_count`` within cutoff
        ``3``: mean ``min_ligand_dist`` within cutoff
        ``4``: number of residues within a tight 4.0 Å shell (close contacts)

        Computed on the fly from the same tensors that the cross-attention
        mask already consumes, so no new batch keys are needed.
        """
        fg = batch["functional_groups"]
        pocket = batch["pocket"]
        fg_ptr = fg["fg_ptr"]
        res_ptr = pocket["residue_ptr"]
        fg_pos = fg["fg_centroid_pos"].float()
        res_pos = pocket["residue_pos"].float()
        residue_x = pocket["residue_x"].float()
        cutoff = float(self.config.cross_attn_cutoff)
        close_cutoff = 4.0
        n_fg_total = int(fg_pos.shape[0])
        out = fg_pos.new_zeros((n_fg_total, FG_ENV_SUMMARY_DIM))
        for idx in range(int(fg_ptr.numel() - 1)):
            fg_lo = int(fg_ptr[idx].item())
            fg_hi = int(fg_ptr[idx + 1].item())
            res_lo = int(res_ptr[idx].item())
            res_hi = int(res_ptr[idx + 1].item())
            if fg_hi <= fg_lo or res_hi <= res_lo:
                continue
            local_fg = fg_pos[fg_lo:fg_hi]
            local_res = res_pos[res_lo:res_hi]
            local_atom_count = residue_x[res_lo:res_hi, 3]
            local_min_dist = residue_x[res_lo:res_hi, 4]
            dists = torch.cdist(local_fg, local_res)
            within = (dists <= cutoff).float()
            close = (dists <= close_cutoff).float()
            n_within = within.sum(dim=-1)
            n_close = close.sum(dim=-1)
            denom = n_within.clamp_min(1.0)
            mean_atoms = (within * local_atom_count.unsqueeze(0)).sum(dim=-1) / denom
            mean_min_dist = (within * local_min_dist.unsqueeze(0)).sum(dim=-1) / denom
            block = torch.stack(
                [
                    n_within,
                    torch.log1p(n_within),
                    mean_atoms,
                    mean_min_dist,
                    n_close,
                ],
                dim=-1,
            )
            out[fg_lo:fg_hi] = block
        return out

    def _run_ligand_gnn(self, h_atom: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return h_atom
        return self.ligand_gnn(h_atom, edge_index, edge_attr)

    def _build_fg_edges(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        fg_pos = batch["functional_groups"]["fg_centroid_pos"]
        fg_ptr = batch["functional_groups"]["fg_ptr"]
        edge_parts: list[Tensor] = []
        attr_parts: list[Tensor] = []
        for idx in range(int(fg_ptr.numel() - 1)):
            lo = int(fg_ptr[idx].item())
            hi = int(fg_ptr[idx + 1].item())
            pos = fg_pos[lo:hi]
            edge_index, distances = radius_graph_from_positions(pos, self.config.fg_edge_cutoff)
            if edge_index.numel():
                edge_parts.append(edge_index + lo)
                attr_parts.append(rbf_expand(distances, self.fg_rbf_centers, self.fg_rbf_widths))
        if edge_parts:
            return torch.cat(edge_parts, dim=1), torch.cat(attr_parts, dim=0)
        return fg_pos.new_zeros((2, 0), dtype=torch.long), fg_pos.new_zeros((0, self.config.num_rbf_kernels))

    def _fg_slices(self, batch: dict[str, Any]) -> list[slice]:
        fg_ptr = batch["functional_groups"]["fg_ptr"]
        return [slice(int(fg_ptr[i].item()), int(fg_ptr[i + 1].item())) for i in range(int(fg_ptr.numel() - 1))]

    def _cross_attend(
        self,
        h_fg_ctx: Tensor,
        h_res: Tensor,
        batch: dict[str, Any],
        *,
        fg_slices: list[slice] | None = None,
    ) -> Tensor:
        if not self.config.use_cross_attention:
            return h_fg_ctx
        res_ptr = batch["pocket"]["residue_ptr"]
        masks = build_cross_attention_masks_from_batch(batch, self.config.cross_attn_cutoff)
        fg_slices = fg_slices or self._fg_slices(batch)
        res_slices = [slice(int(res_ptr[i].item()), int(res_ptr[i + 1].item())) for i in range(int(res_ptr.numel() - 1))]
        return self.cross_attention(h_fg_ctx, h_res, masks, fg_slices, res_slices)

    def _predict_pair_importance(
        self,
        h_fg_final: Tensor,
        h_res: Tensor,
        batch: dict[str, Any],
    ) -> Tensor:
        fg_pos = batch["functional_groups"]["fg_centroid_pos"]
        residue_pos = batch["pocket"]["residue_pos"]
        fg_slices = self._fg_slices(batch)
        res_ptr = batch["pocket"]["residue_ptr"]
        res_slices = [slice(int(res_ptr[i].item()), int(res_ptr[i + 1].item())) for i in range(int(res_ptr.numel() - 1))]
        pair_masks = build_cross_attention_masks_from_batch(batch, self.config.cross_attn_cutoff)
        out = h_fg_final.new_zeros((h_fg_final.shape[0], self.config.num_importance_modes))
        for fg_slice, res_slice, pair_mask in zip(fg_slices, res_slices, pair_masks):
            local_fg = h_fg_final[fg_slice]
            local_res = h_res[res_slice]
            if local_fg.numel() == 0 or local_res.numel() == 0:
                continue
            dists = torch.cdist(fg_pos[fg_slice].float(), residue_pos[res_slice].float())
            out[fg_slice] = self.importance_head(
                local_fg,
                local_res,
                dists,
                pair_mask.to(local_fg.device),
                self.fg_rbf_centers,
                self.fg_rbf_widths,
                self.config.cross_attn_cutoff,
            )
        return out

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def module_groups(self) -> dict[str, list[nn.Module]]:
        groups: dict[str, list[nn.Module]] = {
            "encoders": [self.atom_encoder, self.bond_encoder, self.fg_encoder, self.residue_encoder],
            "ligand_gnn": [self.ligand_gnn],
            "fg_pool": [self.fg_pool],
            "fg_context": [self.fg_context, self.fg_global_context],
            "cross_attention": [self.cross_attention],
            "importance_head": [self.importance_head],
            "hotspot_head": [self.hotspot_head],
        }
        if self.use_contact_aux:
            groups["contact_aux_head"] = [self.contact_aux_head]
        return groups


def _residue_effective_charge(base: Tensor, variant: Tensor) -> Tensor:
    """Residue-level effective net charge at neutral pH using the
    ``residue_base_type_id`` + ``residue_variant_id`` vocab.

    Defaults: ASP/GLU = -1, ARG/LYS = +1, HIS = +0.5 (mixed protonation).
    Variant corrections: ASH/GLH → 0, LYN → 0, HIP → +1, HID/HIE → 0.
    Everything else gets 0. Used by the FG-residue Coulombic interaction
    feature, where signed q matters but the magnitude only needs to be
    correct to within order-of-magnitude.
    """
    base = base.long()
    variant = variant.long()
    q = torch.zeros_like(base, dtype=torch.float32)
    # Base charges. Indices follow STANDARD_RESIDUE_BASE_VOCAB:
    # 0 UNK, 1 ALA, 2 ARG, 3 ASN, 4 ASP, 5 CYS, 6 GLN, 7 GLU, 8 GLY, 9 HIS,
    # 10 ILE, 11 LEU, 12 LYS, 13 MET, 14 PHE, 15 PRO, 16 SER, 17 THR, 18 TRP,
    # 19 TYR, 20 VAL, ...
    q = torch.where(base == 4, torch.full_like(q, -1.0), q)  # ASP
    q = torch.where(base == 7, torch.full_like(q, -1.0), q)  # GLU
    q = torch.where(base == 2, torch.full_like(q, 1.0), q)   # ARG
    q = torch.where(base == 12, torch.full_like(q, 1.0), q)  # LYS
    q = torch.where(base == 9, torch.full_like(q, 0.5), q)   # HIS (mixed)
    # Variant overrides. Indices follow STANDARD_RESIDUE_VARIANT_VOCAB:
    # 0 NONE, 1 HID, 2 HIE, 3 HIP, 4 ASH, 5 GLH, 6 LYN, 7 CYM, 8 CYX, 9 CYN, 10 MSE
    q = torch.where((base == 9) & ((variant == 1) | (variant == 2)), torch.zeros_like(q), q)  # HID/HIE -> 0
    q = torch.where((base == 9) & (variant == 3), torch.ones_like(q), q)  # HIP -> +1
    q = torch.where((base == 4) & (variant == 4), torch.zeros_like(q), q)  # ASH -> 0
    q = torch.where((base == 7) & (variant == 5), torch.zeros_like(q), q)  # GLH -> 0
    q = torch.where((base == 12) & (variant == 6), torch.zeros_like(q), q)  # LYN -> 0
    q = torch.where((base == 5) & (variant == 7), torch.full_like(q, -1.0), q)  # CYM -> -1
    return q


def _pocket_charge_prior(
    atomic_num: Tensor,
    residue_base_type_id: Tensor,
    is_backbone: Tensor,
    residue_variant_id: Tensor | None = None,
) -> Tensor:
    """Per-atom partial-charge prior for pocket atoms.

    The tensorized cache doesn't carry explicit partial charges for pocket
    atoms, so we synthesize a reasonable AMBER-ff14SB-like approximation
    from atomic number + residue base type + backbone flag + (optionally)
    the explicit protonation state (``residue_variant_id``). Values are not
    learned - the projection inside ``LigandAtomEncoder`` learns the scale.

    Protonation-state corrections when ``residue_variant_id`` is provided
    (vocab indices follow STANDARD_RESIDUE_VARIANT_VOCAB):

    - ``HID/HIE`` (1, 2): neutral histidine, ε- or δ-protonated. Net 0
      across the sidechain N's instead of the +0.4 we apply for default HIS.
    - ``HIP`` (3): doubly-protonated histidine, +1 charge.
    - ``ASH`` (4): protonated ASP, neutral carboxyl (drop the -0.8 carboxylate O charge).
    - ``GLH`` (5): protonated GLU, neutral carboxyl.
    - ``LYN`` (6): deprotonated LYS, neutral amine (drop the +0.4 N).
    - ``CYM`` (7): deprotonated CYS sulfur, -1 on S.
    - ``CYX`` (8): disulfide CYS, neutral S.
    - ``CYN`` (9): unspecified CYS protonation, default thiol.
    - ``MSE`` (10): selenomethionine, treat as MET.
    """
    atomic_num = atomic_num.long()
    res_id = residue_base_type_id.long()
    bb = is_backbone.bool()
    rv = residue_variant_id.long() if residue_variant_id is not None else None
    # Default fallback charges by atomic number.
    q = torch.zeros_like(atomic_num, dtype=torch.float32)
    q = torch.where(atomic_num == 6, torch.full_like(q, 0.10), q)
    q = torch.where(atomic_num == 7, torch.full_like(q, -0.40), q)
    q = torch.where(atomic_num == 8, torch.full_like(q, -0.55), q)
    q = torch.where(atomic_num == 16, torch.full_like(q, -0.20), q)
    q = torch.where(atomic_num == 1, torch.full_like(q, 0.10), q)
    # Backbone tweak.
    q = torch.where(bb & (atomic_num == 6), torch.full_like(q, 0.50), q)
    q = torch.where(bb & (atomic_num == 7), torch.full_like(q, -0.50), q)
    q = torch.where(bb & (atomic_num == 8), torch.full_like(q, -0.55), q)
    # Acidic sidechains carry -1 over carboxylate O (base vocab: ASP=4, GLU=7).
    acidic = torch.isin(res_id, torch.tensor([4, 7], device=res_id.device))
    q = torch.where(acidic & ~bb & (atomic_num == 8), torch.full_like(q, -0.80), q)
    # Basic sidechains carry +1 distributed on terminal N's (base vocab: ARG=2, LYS=12; HIS=9).
    basic = torch.isin(res_id, torch.tensor([2, 9, 12], device=res_id.device))
    q = torch.where(basic & ~bb & (atomic_num == 7), torch.full_like(q, 0.40), q)

    if rv is not None:
        # Protonation-state corrections. Indices match
        # dataloader.tensor_schema.STANDARD_RESIDUE_VARIANT_VOCAB:
        # 0 NONE, 1 HID, 2 HIE, 3 HIP, 4 ASH, 5 GLH, 6 LYN, 7 CYM, 8 CYX, 9 CYN, 10 MSE
        # HID / HIE: neutral histidine -> remove the default +0.4 we gave the sidechain N's
        neutral_his = (res_id == 9) & ((rv == 1) | (rv == 2))
        q = torch.where(neutral_his & ~bb & (atomic_num == 7), torch.full_like(q, -0.20), q)
        # HIP: doubly-protonated -> sidechain N's go to +0.5 instead of the default +0.4
        hip = (res_id == 9) & (rv == 3)
        q = torch.where(hip & ~bb & (atomic_num == 7), torch.full_like(q, 0.50), q)
        # ASH: protonated ASP -> carboxyl O goes back to neutral -0.55 (no -0.80)
        ash = (res_id == 4) & (rv == 4)
        q = torch.where(ash & ~bb & (atomic_num == 8), torch.full_like(q, -0.55), q)
        # GLH: protonated GLU -> same neutralization
        glh = (res_id == 7) & (rv == 5)
        q = torch.where(glh & ~bb & (atomic_num == 8), torch.full_like(q, -0.55), q)
        # LYN: deprotonated LYS -> N goes back to neutral -0.40 (no +0.40)
        lyn = (res_id == 12) & (rv == 6)
        q = torch.where(lyn & ~bb & (atomic_num == 7), torch.full_like(q, -0.40), q)
        # CYM: deprotonated CYS -> -1 on S
        cym = (res_id == 5) & (rv == 7)
        q = torch.where(cym & ~bb & (atomic_num == 16), torch.full_like(q, -1.00), q)
    return q
