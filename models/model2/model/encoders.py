# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.rbf import get_rbf_centers, get_rbf_widths, rbf_expand


def _clean_scalar(x: Tensor) -> Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)


class LigandAtomEncoder(nn.Module):
    """Atom encoder. ``elec_aev_dim`` is the width of an optional Coulombic
    electrostatic-AEV channel concatenated alongside the ECIF AEV; pass 0 to
    keep the model1-compatible 405-wide input."""

    def __init__(self, hidden_dim: int, elec_aev_dim: int = 0) -> None:
        super().__init__()
        self.atomic_num_emb = nn.Embedding(119, 32)
        self.formal_charge_emb = nn.Embedding(9, 4)
        self.hybridization_emb = nn.Embedding(8, 8)
        self.chiral_tag_emb = nn.Embedding(4, 4)
        self.elec_aev_dim = int(elec_aev_dim)
        in_dim = 405 + self.elec_aev_dim
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atomic_num: Tensor,
        formal_charge: Tensor,
        partial_charge: Tensor,
        is_aromatic: Tensor,
        hybridization: Tensor,
        chiral_tag: Tensor,
        explicit_h: Tensor,
        implicit_h: Tensor,
        isotope: Tensor,
        aev: Tensor,
        elec_aev: Tensor | None = None,
    ) -> Tensor:
        parts = [
            self.atomic_num_emb(atomic_num.long().clamp(min=0, max=118)),
            self.formal_charge_emb((formal_charge.long() + 4).clamp(min=0, max=8)),
            _clean_scalar(partial_charge),
            _clean_scalar(is_aromatic),
            self.hybridization_emb(hybridization.long().clamp(min=0, max=7)),
            self.chiral_tag_emb(chiral_tag.long().clamp(min=0, max=3)),
            _clean_scalar(explicit_h),
            _clean_scalar(implicit_h),
            _clean_scalar(isotope),
            torch.nan_to_num(aev.float(), nan=0.0, posinf=0.0, neginf=0.0),
        ]
        if self.elec_aev_dim > 0:
            if elec_aev is None:
                elec_aev = aev.new_zeros((aev.shape[0], self.elec_aev_dim))
            parts.append(torch.nan_to_num(elec_aev.float(), nan=0.0, posinf=0.0, neginf=0.0))
        x = torch.cat(parts, dim=-1)
        return self.norm(self.proj(x))


class LigandBondEncoder(nn.Module):
    def __init__(self, edge_dim: int) -> None:
        super().__init__()
        self.bond_type_emb = nn.Embedding(6, 16)
        self.bond_stereo_emb = nn.Embedding(6, 8)
        self.bond_dir_emb = nn.Embedding(6, 8)
        self.proj = nn.Linear(34, edge_dim)
        self.norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        bond_type: Tensor,
        is_conjugated: Tensor,
        is_in_ring: Tensor,
        bond_stereo: Tensor,
        bond_dir: Tensor,
    ) -> Tensor:
        parts = [
            self.bond_type_emb(bond_type.long().clamp(min=0, max=5)),
            _clean_scalar(is_conjugated),
            _clean_scalar(is_in_ring),
            self.bond_stereo_emb(bond_stereo.long().clamp(min=0, max=5)),
            self.bond_dir_emb(bond_dir.long().clamp(min=0, max=5)),
        ]
        x = torch.cat(parts, dim=-1)
        return self.norm(self.proj(x))


class FGNodeEncoder(nn.Module):
    """Encodes each FG token from the schema scalars + categorical embeddings.

    ``extra_elec_dim``: width of an optional per-FG electrostatic-features
    block (net charge, dipole magnitude, charge variance, min/max partial
    charge) concatenated alongside the existing scalars. Pass 0 to keep the
    pre-improvement 28-wide input.
    """

    def __init__(self, hidden_dim: int, extra_elec_dim: int = 0) -> None:
        super().__init__()
        self.fg_type_emb = nn.Embedding(1024, 16)
        self.fg_parent_type_emb = nn.Embedding(256, 8)
        self.extra_elec_dim = int(extra_elec_dim)
        in_dim = 28 + self.extra_elec_dim
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        num_atoms: Tensor,
        total_partial_charge: Tensor,
        max_radial_extent: Tensor,
        mean_partial_charge: Tensor,
        fg_type: Tensor,
        fg_parent_type: Tensor,
        extra_elec: Tensor | None = None,
    ) -> Tensor:
        parts = [
            _clean_scalar(num_atoms),
            _clean_scalar(total_partial_charge),
            _clean_scalar(max_radial_extent),
            _clean_scalar(mean_partial_charge),
            self.fg_type_emb(fg_type.long().clamp(min=0, max=self.fg_type_emb.num_embeddings - 1)),
            self.fg_parent_type_emb(fg_parent_type.long().clamp(min=0, max=self.fg_parent_type_emb.num_embeddings - 1)),
        ]
        if self.extra_elec_dim > 0:
            if extra_elec is None:
                extra_elec = num_atoms.new_zeros((num_atoms.shape[0], self.extra_elec_dim), dtype=torch.float32)
            parts.append(torch.nan_to_num(extra_elec.float(), nan=0.0, posinf=0.0, neginf=0.0))
        x = torch.cat(parts, dim=-1)
        return self.norm(self.proj(x))


class PocketResidueEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf_kernels: int, rbf_d_min: float, rbf_d_max: float) -> None:
        super().__init__()
        self.residue_type_emb = nn.Embedding(128, 32)
        self.residue_base_type_emb = nn.Embedding(128, 16)
        self.residue_variant_emb = nn.Embedding(32, 8)
        self.register_buffer("rbf_centers", get_rbf_centers(num_rbf_kernels, rbf_d_min, rbf_d_max), persistent=False)
        self.register_buffer("rbf_widths", get_rbf_widths(num_rbf_kernels, rbf_d_min, rbf_d_max), persistent=False)
        self.proj = nn.Linear(32 + 16 + 8 + 1 + num_rbf_kernels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        residue_type_id: Tensor,
        residue_base_type_id: Tensor,
        residue_variant_id: Tensor,
        atom_count: Tensor,
        min_ligand_dist: Tensor,
    ) -> Tensor:
        dist_rbf = rbf_expand(min_ligand_dist.float(), self.rbf_centers, self.rbf_widths)
        parts = [
            self.residue_type_emb(residue_type_id.long().clamp(min=0, max=self.residue_type_emb.num_embeddings - 1)),
            self.residue_base_type_emb(residue_base_type_id.long().clamp(min=0, max=self.residue_base_type_emb.num_embeddings - 1)),
            self.residue_variant_emb(residue_variant_id.long().clamp(min=0, max=self.residue_variant_emb.num_embeddings - 1)),
            _clean_scalar(atom_count),
            dist_rbf,
        ]
        x = torch.cat(parts, dim=-1)
        return self.norm(self.proj(x))
