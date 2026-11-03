# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import softmax


class FGAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_scorer = nn.Linear(hidden_dim, 1)
        self.merge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        h_atom: Tensor,
        fg_membership_index: Tensor,
        h_fg_init: Tensor,
        n_fg: int,
    ) -> Tensor:
        if n_fg == 0:
            return h_fg_init
        if fg_membership_index.numel() == 0:
            pooled = h_fg_init.new_zeros((n_fg, h_atom.shape[-1]))
            return self.merge_mlp(torch.cat([pooled, h_fg_init], dim=-1))

        atom_idx = fg_membership_index[0].long()
        fg_idx = fg_membership_index[1].long()
        atom_repr = h_atom[atom_idx]
        logits = self.attn_scorer(atom_repr).squeeze(-1)
        alpha = softmax(logits, fg_idx, num_nodes=n_fg)
        pooled = h_fg_init.new_zeros((n_fg, h_atom.shape[-1]))
        pooled.index_add_(0, fg_idx, atom_repr * alpha.unsqueeze(-1))
        return self.merge_mlp(torch.cat([pooled, h_fg_init], dim=-1))
