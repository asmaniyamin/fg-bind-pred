# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv


class GATv2Layer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        update = self.conv(x, edge_index, edge_attr)
        return self.norm(self.residual(x) + self.dropout(update))


class LigandGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GATv2Layer(hidden_dim, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x
