# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Heads for the counterfactual FG-importance model (model2).

The model emits per-FG predictions:

- ``importance`` : ``[N_fg, num_modes]`` regressed ΔΔ_total per masked mode
                   (ghost / charge_only / lj_only)
- ``hotspot``    : ``[N_fg]`` logit; trained with BCE against per-complex top-K labels

When ``--use-pair-importance-head`` is enabled, the importance prediction is
instead produced by a model1-style FG-residue pair scorer
(``FGResiduePairImportanceHead``): per (FG, residue) within cutoff, the model
emits a scalar contribution to each masked mode, gated by a cosine cutoff on
the pair distance; contributions are scatter-summed into per-FG channels.

"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.rbf import cosine_cutoff, rbf_expand


class FGImportanceHead(nn.Module):
    """Per-FG regression head for counterfactual ΔΔ_total values.

    One scalar per masked mode. Output is in raw kcal/mol so its sign mirrors
    the MM-GBSA convention: negative means "removing this FG makes the
    complex less favorable".

    When ``extra_input_dim > 0``, the head accepts an additional per-FG
    feature vector (e.g. a local-pocket environment summary) concatenated
    to the hidden representation before the MLP.
    """

    def __init__(self, hidden_dim: int, num_modes: int, extra_input_dim: int = 0) -> None:
        super().__init__()
        in_dim = hidden_dim + int(extra_input_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_modes),
        )

    def forward(self, h_fg: Tensor, extra: Tensor | None = None) -> Tensor:
        if extra is not None and extra.numel():
            h_fg = torch.cat([h_fg, extra], dim=-1)
        return self.net(h_fg)


class FGHotspotHead(nn.Module):
    """Per-FG binary head: "is this FG a hotspot in its complex?"."""

    def __init__(self, hidden_dim: int, extra_input_dim: int = 0) -> None:
        super().__init__()
        in_dim = hidden_dim + int(extra_input_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_fg: Tensor, extra: Tensor | None = None) -> Tensor:
        if extra is not None and extra.numel():
            h_fg = torch.cat([h_fg, extra], dim=-1)
        return self.net(h_fg).squeeze(-1)


class FGResiduePairImportanceHead(nn.Module):
    """Pair-resolved importance head: per (FG, residue) → ΔΔ_total contribution.

    Mirrors model1's ``FGResidueContactHead`` but instead of decomposing
    contact energy into 4 channels, it emits a contribution to each of the
    ``num_modes`` counterfactual masked modes (ghost / charge_only / lj_only).
    Per-FG importance is the cosine-cutoff-weighted scatter-sum of pair
    contributions, which gives a strong physical inductive bias: only nearby
    residues contribute, and the contribution decays smoothly with distance.
    """

    def __init__(self, hidden_dim: int, num_rbf_kernels: int, num_modes: int) -> None:
        super().__init__()
        pair_dim = hidden_dim * 4 + num_rbf_kernels + 1
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mode_out = nn.Linear(hidden_dim // 2, num_modes)
        self.gate_out = nn.Linear(hidden_dim // 2, 1)
        self.num_modes = int(num_modes)

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
        out = h_fg.new_zeros((n_fg, self.num_modes))
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
        contrib = self.mode_out(hidden) * gate
        out.index_add_(0, fg_idx.long(), contrib)
        return out


class ContactDecompositionHead(nn.Module):
    """Auxiliary head producing the model1-style soft contact decomposition.

    Used only as a representation-learning regularizer when the auxiliary
    contact loss is enabled (CLI ``--aux-contact-weight > 0``). Predicts a
    4-channel vector ``(vdw, elec, polar, nonpolar)`` per FG; targets are
    the soft-partitioned per-FG residue energies from
    ``model2.data.label_builder.aggregate_residue_channels_to_fg_partitioned``.

    The targets are noisier than the counterfactual ΔΔ_total signal but
    learning them in parallel can give the encoder a smoother feature space.
    """

    def __init__(self, hidden_dim: int, extra_input_dim: int = 0) -> None:
        super().__init__()
        in_dim = hidden_dim + int(extra_input_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(self, h_fg: Tensor, extra: Tensor | None = None) -> Tensor:
        if extra is not None and extra.numel():
            h_fg = torch.cat([h_fg, extra], dim=-1)
        return self.net(h_fg)
