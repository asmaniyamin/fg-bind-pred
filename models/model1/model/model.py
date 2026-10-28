# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..config import ModelConfig
from ..data.graph_builder import (
    ANI2X_RS,
    compute_ecif_types_from_pocket_features,
    compute_ligand_aev,
    radius_graph_from_positions,
)
from ..data.label_builder import build_cross_attention_masks_from_batch
from ..utils.rbf import get_rbf_centers, get_rbf_widths, rbf_expand
from .attention import FGMoleculeSelfAttention, FGPocketCrossAttention, FGSelfAttentionGraph
from .encoders import FGNodeEncoder, LigandAtomEncoder, LigandBondEncoder, PocketResidueEncoder
from .gnn import LigandGNN
from .heads import DeltaTotalHead, FGResidueContactHead, PreorgHead
from .pooling import FGAttentionPool


class PerFGModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_encoder = LigandAtomEncoder(config.hidden_dim)
        self.bond_encoder = LigandBondEncoder(config.ligand_edge_dim)
        self.fg_encoder = FGNodeEncoder(config.hidden_dim)
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
        self.contact_pair_head = FGResidueContactHead(config.hidden_dim, config.num_rbf_kernels)
        self.preorg_head = PreorgHead(config.hidden_dim)
        self.delta_total_head = DeltaTotalHead(config.hidden_dim * 4 + 5, config.hidden_dim)

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
        h_fg = self._pool_to_fg(
            h_atom,
            batch["functional_groups"]["fg_membership_index"],
            h_fg_init,
            batch["functional_groups"]["fg_ptr"],
        )
        fg_slices = self._fg_slices(batch)
        fg_edge_index, fg_edge_attr = self._build_fg_edges(batch)
        h_fg_ctx = self._fg_context(h_fg, fg_edge_index, fg_edge_attr)
        h_fg_cross = self._cross_attend(h_fg_ctx, h_res, batch, fg_slices=fg_slices)
        h_fg_final = self.fg_global_context(h_fg_cross, fg_slices)
        contact = self._predict_fg_contacts(h_fg_final, h_res, batch)
        contact_total = contact.sum(dim=-1)
        preorg = self.preorg_head(h_fg_ctx)
        delta_total = self._predict_delta_total(
            h_atom,
            h_res,
            h_fg_final,
            contact_total,
            preorg,
            batch,
        )
        return {
            "contact": contact,
            "contact_total": contact_total,
            "preorg": preorg,
            "delta_total": delta_total,
        }

    def _encode_inputs(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ligand = batch["ligand"]
        pocket = batch["pocket"]
        fg = batch["functional_groups"]

        aev = self._compute_batch_aev(batch)
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
        h_fg_init = self.fg_encoder(
            num_atoms=fg_features[:, 0] if fg_features.numel() else fg_features.new_zeros((0,)),
            total_partial_charge=fg_features[:, 1] if fg_features.numel() else fg_features.new_zeros((0,)),
            max_radial_extent=fg_features[:, 2] if fg_features.numel() else fg_features.new_zeros((0,)),
            mean_partial_charge=fg_features[:, 3] if fg_features.numel() else fg_features.new_zeros((0,)),
            fg_type=fg["fg_type"],
            fg_parent_type=fg["fg_parent_type"],
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

    def _run_ligand_gnn(self, h_atom: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return h_atom
        return self.ligand_gnn(h_atom, edge_index, edge_attr)

    def _pool_to_fg(
        self,
        h_atom: Tensor,
        fg_membership: Tensor,
        h_fg_init: Tensor,
        n_fg_per_graph: Tensor,
    ) -> Tensor:
        del n_fg_per_graph
        return self.fg_pool(h_atom, fg_membership, h_fg_init, h_fg_init.shape[0])

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

    def _fg_context(self, h_fg: Tensor, fg_edge_index: Tensor, fg_edge_attr: Tensor) -> Tensor:
        return self.fg_context(h_fg, fg_edge_index, fg_edge_attr)

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
        fg_ptr = batch["functional_groups"]["fg_ptr"]
        res_ptr = batch["pocket"]["residue_ptr"]
        masks = build_cross_attention_masks_from_batch(batch, self.config.cross_attn_cutoff)
        fg_slices = fg_slices or self._fg_slices(batch)
        res_slices = [slice(int(res_ptr[i].item()), int(res_ptr[i + 1].item())) for i in range(int(res_ptr.numel() - 1))]
        return self.cross_attention(h_fg_ctx, h_res, masks, fg_slices, res_slices)

    def _predict_fg_contacts(
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
        out = h_fg_final.new_zeros((h_fg_final.shape[0], 4))
        for fg_slice, res_slice, pair_mask in zip(fg_slices, res_slices, pair_masks):
            local_fg = h_fg_final[fg_slice]
            local_res = h_res[res_slice]
            if local_fg.numel() == 0 or local_res.numel() == 0:
                continue
            dists = torch.cdist(fg_pos[fg_slice].float(), residue_pos[res_slice].float())
            out[fg_slice] = self.contact_pair_head(
                local_fg,
                local_res,
                dists,
                pair_mask.to(local_fg.device),
                self.fg_rbf_centers,
                self.fg_rbf_widths,
                self.config.cross_attn_cutoff,
            )
        return out

    def _pool_by_ptr(self, x: Tensor, ptr: Tensor) -> Tensor:
        summed, counts = self._sum_by_ptr(x, ptr)
        return summed / counts.clamp_min(1.0).unsqueeze(-1)

    def _sum_by_ptr(self, x: Tensor, ptr: Tensor) -> tuple[Tensor, Tensor]:
        n_graphs = int(ptr.numel() - 1)
        if n_graphs <= 0:
            return x.new_zeros((0, x.shape[-1])), x.new_zeros((0,))
        out = x.new_zeros((n_graphs, x.shape[-1]))
        counts = x.new_zeros((n_graphs,))
        for idx in range(n_graphs):
            lo = int(ptr[idx].item())
            hi = int(ptr[idx + 1].item())
            if hi > lo:
                out[idx] = x[lo:hi].sum(dim=0)
                counts[idx] = float(hi - lo)
        return out, counts

    def _pool_by_batch(self, x: Tensor, batch_index: Tensor, num_graphs: int) -> Tensor:
        summed, counts = self._sum_by_batch(x, batch_index, num_graphs)
        return summed / counts.clamp_min(1.0).unsqueeze(-1)

    def _sum_by_batch(self, x: Tensor, batch_index: Tensor, num_graphs: int) -> tuple[Tensor, Tensor]:
        if num_graphs <= 0:
            return x.new_zeros((0, x.shape[-1])), x.new_zeros((0,))
        out = x.new_zeros((num_graphs, x.shape[-1]))
        counts = x.new_zeros((num_graphs,))
        if batch_index.numel():
            out.index_add_(0, batch_index.long(), x)
            counts.index_add_(0, batch_index.long(), torch.ones_like(batch_index, dtype=x.dtype))
        return out, counts

    def _predict_delta_total(
        self,
        h_atom: Tensor,
        h_res: Tensor,
        h_fg_final: Tensor,
        contact_total: Tensor,
        preorg: Tensor,
        batch: dict[str, Any],
    ) -> Tensor:
        num_complexes = int(batch["functional_groups"]["fg_ptr"].numel() - 1)
        if num_complexes <= 0:
            return h_fg_final.new_zeros((0,))
        fg_mean = self._pool_by_batch(
            h_fg_final,
            batch["functional_groups"]["fg_batch"],
            num_complexes,
        )
        fg_sum, fg_counts = self._sum_by_batch(
            h_fg_final,
            batch["functional_groups"]["fg_batch"],
            num_complexes,
        )
        ligand_repr = self._pool_by_ptr(h_atom, batch["ligand"]["ptr"])
        residue_repr = self._pool_by_ptr(h_res, batch["pocket"]["residue_ptr"])
        _, ligand_counts = self._sum_by_ptr(h_atom, batch["ligand"]["ptr"])
        _, residue_counts = self._sum_by_ptr(h_res, batch["pocket"]["residue_ptr"])
        contact_sum, _ = self._sum_by_batch(
            contact_total.unsqueeze(-1),
            batch["functional_groups"]["fg_batch"],
            num_complexes,
        )
        preorg_sum, _ = self._sum_by_batch(
            preorg,
            batch["functional_groups"]["fg_batch"],
            num_complexes,
        )
        size_features = torch.stack(
            [
                torch.log1p(fg_counts),
                torch.log1p(ligand_counts),
                torch.log1p(residue_counts),
            ],
            dim=-1,
        )
        complex_repr = torch.cat(
            [
                fg_mean,
                fg_sum,
                ligand_repr,
                residue_repr,
                size_features,
                contact_sum,
                preorg_sum,
            ],
            dim=-1,
        )
        return self.delta_total_head(complex_repr).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def module_groups(self) -> dict[str, list[nn.Module]]:
        return {
            "encoders": [self.atom_encoder, self.bond_encoder, self.fg_encoder, self.residue_encoder],
            "ligand_gnn": [self.ligand_gnn],
            "fg_pool": [self.fg_pool],
            "fg_context": [self.fg_context, self.fg_global_context],
            "cross_attention": [self.cross_attention],
            "contact_head": [self.contact_pair_head],
            "preorg_head": [self.preorg_head],
            "delta_head": [self.delta_total_head],
        }
