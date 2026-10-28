# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.rbf import cosine_cutoff, rbf_expand


class FGResidueContactHead(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf_kernels: int) -> None:
        super().__init__()
        pair_dim = hidden_dim * 4 + num_rbf_kernels + 1
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.channel_out = nn.Linear(hidden_dim // 2, 4)
        self.gate_out = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        h_fg: Tensor,
        h_res: Tensor,
        pair_dists: Tensor,
        pair_mask: Tensor,
        rbf_centers: Tensor,
        rbf_widths: Tensor,
        cutoff: float,
    ) -> Tensor:
        n_fg = h_fg.shape[0]
        out = h_fg.new_zeros((n_fg, 4))
        if n_fg == 0 or h_res.shape[0] == 0 or not pair_mask.any():
            return out

        fg_idx, res_idx = torch.nonzero(pair_mask, as_tuple=True)
        fg_repr = h_fg[fg_idx]
        res_repr = h_res[res_idx]
        dists = pair_dists[fg_idx, res_idx]
        dist_rbf = rbf_expand(dists, rbf_centers, rbf_widths)
        pair_features = torch.cat(
            [
                fg_repr,
                res_repr,
                fg_repr * res_repr,
                (fg_repr - res_repr).abs(),
                dist_rbf,
                dists.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.pair_mlp(pair_features)
        gate = torch.sigmoid(self.gate_out(hidden)) * cosine_cutoff(dists, cutoff).unsqueeze(-1)
        contrib = self.channel_out(hidden) * gate
        out.index_add_(0, fg_idx.long(), contrib)
        return out


class ContactTotalHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h_fg_final: Tensor) -> Tensor:
        return self.net(h_fg_final)


class PreorgHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h_fg_ctx: Tensor) -> Tensor:
        return self.net(h_fg_ctx)


class DeltaTotalHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, complex_repr: Tensor) -> Tensor:
        return self.net(complex_repr)
