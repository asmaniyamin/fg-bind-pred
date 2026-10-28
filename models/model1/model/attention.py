# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .gnn import GATv2Layer


class FGSelfAttentionGraph(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_input_dim: int,
        edge_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_input_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )
        self.layers = nn.ModuleList(
            [
                GATv2Layer(hidden_dim, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, h_fg: Tensor, fg_edge_index: Tensor, fg_edge_attr: Tensor) -> Tensor:
        if fg_edge_attr.numel() == 0:
            return h_fg
        edge_attr = self.edge_proj(fg_edge_attr.float())
        for layer in self.layers:
            h_fg = layer(h_fg, fg_edge_index, edge_attr)
        return h_fg


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        if query.numel() == 0 or key.numel() == 0:
            return query
        disallow = ~mask.bool()
        attn_out, _ = self.attn(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_mask=disallow,
            need_weights=False,
        )
        x = self.attn_norm(query + self.dropout(attn_out.squeeze(0)))
        return self.ffn_norm(x + self.dropout(self.ffn(x)))


class FGPocketCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        h_fg: Tensor,
        h_res: Tensor,
        attn_masks: list[Tensor],
        fg_slices: list[slice],
        res_slices: list[slice],
    ) -> Tensor:
        out = h_fg
        for layer in self.layers:
            updated = out.clone()
            for mask, fg_slice, res_slice in zip(attn_masks, fg_slices, res_slices):
                updated[fg_slice] = layer(out[fg_slice], h_res[res_slice], h_res[res_slice], mask.to(out.device))
            out = updated
        return out


class FGMoleculeSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, h_fg: Tensor, fg_slices: list[slice]) -> Tensor:
        out = h_fg
        for layer in self.layers:
            updated = out.clone()
            for fg_slice in fg_slices:
                x = out[fg_slice]
                if x.numel() == 0:
                    continue
                full_mask = torch.ones((x.shape[0], x.shape[0]), dtype=torch.bool, device=out.device)
                updated[fg_slice] = layer(x, x, x, full_mask)
            out = updated
        return out
