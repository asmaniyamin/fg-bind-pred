# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from dataloader.tensor_schema import STANDARD_RESIDUE_BASE_VOCAB

from ..utils.rbf import ANI2X_RS_VALUES, cosine_cutoff, get_rbf_centers, get_rbf_widths, rbf_expand
from .aev_plig import radial_terms


ANI2X_RC: float = 5.1
ANI2X_ETA: float = 19.7
ANI2X_RS: Tensor = ANI2X_RS_VALUES

_RESIDUE_ID = {name: idx for idx, name in enumerate(STANDARD_RESIDUE_BASE_VOCAB)}
_AROMATIC_RESIDUES = {_RESIDUE_ID.get(name, -1) for name in ("PHE", "TYR", "TRP", "HIS")}
_ACIDIC_RESIDUES = {_RESIDUE_ID.get(name, -1) for name in ("ASP", "GLU")}
_BASIC_RESIDUES = {_RESIDUE_ID.get(name, -1) for name in ("ARG", "LYS", "HIS")}
_PDB_ATOM_KEYS_PATH = Path(__file__).with_name("PDB_Atom_Keys.csv")


def compute_ecif_types(
    atomic_num: Tensor,
    explicit_valence: Tensor,
    num_heavy_neighbors: Tensor,
    num_h_neighbors: Tensor,
    is_aromatic: Tensor,
    in_ring: Tensor,
) -> Tensor:
    atomic_num = atomic_num.long()
    explicit_valence = torch.nan_to_num(explicit_valence.float(), nan=0.0).long().clamp(min=0, max=4)
    num_heavy_neighbors = torch.nan_to_num(num_heavy_neighbors.float(), nan=0.0).long().clamp(min=0, max=4)
    num_h_neighbors = torch.nan_to_num(num_h_neighbors.float(), nan=0.0).long().clamp(min=0, max=4)
    is_aromatic = is_aromatic.bool()
    in_ring = in_ring.bool()

    element_bucket = torch.full_like(atomic_num, 7)
    element_bucket[atomic_num == 6] = 0
    element_bucket[atomic_num == 7] = 1
    element_bucket[atomic_num == 8] = 2
    element_bucket[atomic_num == 16] = 3
    element_bucket[atomic_num == 15] = 4
    element_bucket[torch.isin(atomic_num, torch.tensor([9, 17, 35, 53], device=atomic_num.device))] = 5
    element_bucket[torch.isin(atomic_num, torch.tensor([11, 12, 19, 20, 25, 26, 27, 29, 30], device=atomic_num.device))] = 6

    env_bucket = (
        explicit_valence.clamp(max=2)
        + (num_heavy_neighbors >= 3).long()
        + (num_h_neighbors > 0).long()
        + is_aromatic.long()
        + in_ring.long()
    ) % 3
    type_idx = element_bucket * 3 + env_bucket
    return type_idx.clamp(min=0, max=21)


def compute_ecif_types_from_pocket_features(
    atomic_num: Tensor,
    residue_base_type_id: Tensor,
    is_backbone: Tensor,
    is_water: Tensor,
) -> Tensor:
    atomic_num = atomic_num.long()
    residue_base_type_id = residue_base_type_id.long()
    is_backbone = is_backbone.bool()
    is_water = is_water.bool()

    out = torch.full_like(atomic_num, 21)
    out[is_water & (atomic_num == 8)] = 18
    out[torch.isin(atomic_num, torch.tensor([11, 12, 19, 20, 25, 26, 27, 29, 30], device=atomic_num.device))] = 12
    out[torch.isin(atomic_num, torch.tensor([9, 17, 35, 53], device=atomic_num.device))] = 11
    out[atomic_num == 15] = 10
    out[atomic_num == 16] = 9

    out[is_backbone & (atomic_num == 6)] = 13
    out[is_backbone & (atomic_num == 7)] = 14
    out[is_backbone & (atomic_num == 8)] = 15

    sidechain = ~is_backbone & ~is_water
    out[sidechain & (atomic_num == 6)] = 1
    out[sidechain & (atomic_num == 7)] = 5
    out[sidechain & (atomic_num == 8)] = 8

    aromatic_sidechain = sidechain & torch.isin(
        residue_base_type_id,
        torch.tensor(sorted(_AROMATIC_RESIDUES), device=atomic_num.device),
    )
    out[aromatic_sidechain & (atomic_num == 6)] = 0
    out[aromatic_sidechain & (atomic_num == 7)] = 3

    out[sidechain & torch.isin(
        residue_base_type_id,
        torch.tensor(sorted(_BASIC_RESIDUES), device=atomic_num.device),
    ) & (atomic_num == 7)] = 19
    out[sidechain & torch.isin(
        residue_base_type_id,
        torch.tensor(sorted(_ACIDIC_RESIDUES), device=atomic_num.device),
    ) & (atomic_num == 8)] = 20
    out[sidechain & (atomic_num == 7) & (out == 5)] = 4
    out[sidechain & (atomic_num == 8) & (out == 8)] = 7
    out[sidechain & (atomic_num == 6) & (out == 1)] = 2
    out[sidechain & (atomic_num == 7) & aromatic_sidechain] = 3
    out[sidechain & (atomic_num == 8) & torch.isin(
        residue_base_type_id,
        torch.tensor(sorted(_ACIDIC_RESIDUES), device=atomic_num.device),
    )] = 6
    return out.clamp(min=0, max=21)


def compute_ligand_aev(
    ligand_coords: Tensor,
    pocket_coords: Tensor,
    pocket_ecif_types: Tensor,
    rc: float,
    rs_values: Tensor,
    eta: float,
) -> Tensor:
    ligand_coords = ligand_coords.float()
    pocket_coords = pocket_coords.float()
    pocket_ecif_types = pocket_ecif_types.long()
    if ligand_coords.numel() == 0:
        return ligand_coords.new_zeros((0, 22 * int(rs_values.numel())))
    if pocket_coords.numel() == 0:
        return ligand_coords.new_zeros((ligand_coords.shape[0], 22 * int(rs_values.numel())))

    dists = torch.cdist(ligand_coords, pocket_coords)
    cutoff_mask = dists <= float(rc)
    gauss = radial_terms(
        float(rc),
        torch.as_tensor([float(eta)], dtype=dists.dtype, device=dists.device),
        rs_values.to(device=dists.device, dtype=dists.dtype),
        dists[cutoff_mask],
    )
    out = ligand_coords.new_zeros((ligand_coords.shape[0], 22, int(rs_values.numel())))
    if cutoff_mask.any():
        lig_idx, poc_idx = torch.nonzero(cutoff_mask, as_tuple=True)
        flat_index = lig_idx * 22 + pocket_ecif_types[poc_idx].clamp(min=0, max=21)
        out.view(ligand_coords.shape[0] * 22, -1).index_add_(0, flat_index, gauss)
    return out.reshape(ligand_coords.shape[0], -1)


def compute_fg_electrostatic_features(
    ligand_pos: Tensor,
    ligand_partial_charge: Tensor,
    fg_centroid_pos: Tensor,
    fg_membership_index: Tensor,
) -> Tensor:
    """Per-FG electrostatic descriptors (5 channels).

    Output ``[N_fg, 5]`` with columns:

    - ``0``: net charge ``Σ q_i`` over atoms in the FG (signed, in atomic units)
    - ``1``: dipole magnitude ``‖Σ q_i · (r_i - r_centroid)‖`` (e·Å)
    - ``2``: charge variance ``Σ q_i²`` - distinguishes "polar but neutral"
            (high variance, near-zero net) from "purely nonpolar" (low both)
    - ``3``: minimum partial charge on any FG atom (most negative)
    - ``4``: maximum partial charge on any FG atom (most positive)

    These are the chemically meaningful per-FG electrostatic features the FG
    encoder currently lacks: it sees ``total_partial_charge`` and
    ``mean_partial_charge`` as bare scalars but no geometric/distributional
    information about how the charge is laid out within the FG. Sign, dipole
    layout, and the negative/positive extremes are what discriminate
    "salt-bridge donor" from "polarized but neutral" from "hydrophobic".
    """
    n_fg = int(fg_centroid_pos.shape[0])
    out = fg_centroid_pos.new_zeros((n_fg, 5))
    if n_fg == 0 or fg_membership_index.numel() == 0:
        return out
    atom_idx = fg_membership_index[0].long()
    fg_idx = fg_membership_index[1].long()
    q_per_member = ligand_partial_charge.float()[atom_idx]
    pos_per_member = ligand_pos.float()[atom_idx]
    centroid_per_member = fg_centroid_pos.float()[fg_idx]
    delta = pos_per_member - centroid_per_member  # [N_member, 3]
    weighted_delta = q_per_member.unsqueeze(-1) * delta  # [N_member, 3]
    q_sq = q_per_member.square()

    # net charge
    out[:, 0].index_add_(0, fg_idx, q_per_member)
    # dipole vector accumulated then norm taken at the end
    dipole = fg_centroid_pos.new_zeros((n_fg, 3))
    dipole.index_add_(0, fg_idx, weighted_delta)
    out[:, 1] = torch.linalg.norm(dipole, dim=-1)
    # charge variance proxy
    out[:, 2].index_add_(0, fg_idx, q_sq)
    # min / max: scatter via two-pass (no torch.scatter_reduce_ amin/amax until 1.12+; this is safer)
    big = fg_centroid_pos.new_full((1,), float("inf"))
    small = fg_centroid_pos.new_full((1,), float("-inf"))
    qmin = big.expand(n_fg).clone()
    qmax = small.expand(n_fg).clone()
    qmin.scatter_reduce_(0, fg_idx, q_per_member, reduce="amin", include_self=True)
    qmax.scatter_reduce_(0, fg_idx, q_per_member, reduce="amax", include_self=True)
    # Replace untouched (still ±inf) with 0 -- FGs with no member atoms.
    qmin = torch.where(torch.isinf(qmin), torch.zeros_like(qmin), qmin)
    qmax = torch.where(torch.isinf(qmax), torch.zeros_like(qmax), qmax)
    out[:, 3] = qmin
    out[:, 4] = qmax
    return out


def compute_fg_residue_coulomb(
    fg_centroid_pos: Tensor,
    fg_net_charge: Tensor,
    residue_pos: Tensor,
    residue_charge: Tensor,
    cutoff: float,
    *,
    r_min: float = 1.5,
) -> Tensor:
    """Per-FG explicit Coulombic interaction scalar.

    For each FG, returns ``Σ_residues_within_cutoff (q_FG · q_residue) / r_eff``
    where ``r_eff = max(r, r_min)`` is the FG-centroid → residue-centroid
    distance with a short-range clip. Signed; large negative = stabilizing
    electrostatic interaction (opposite charges close together). Two outputs
    per FG: the raw Coulomb sum and an "abs density" channel
    ``Σ |q_residue|/r_eff`` analogous to the per-atom elec AEV. Returns
    ``[N_fg, 2]``.
    """
    n_fg = int(fg_centroid_pos.shape[0])
    out = fg_centroid_pos.new_zeros((n_fg, 2))
    if n_fg == 0 or residue_pos.numel() == 0:
        return out
    dists = torch.cdist(fg_centroid_pos.float(), residue_pos.float())
    within = (dists <= float(cutoff)).float()
    inv_r = within / torch.clamp(dists, min=float(r_min))
    coulomb = (fg_net_charge.float().unsqueeze(-1) * (residue_charge.float().unsqueeze(0) * inv_r)).sum(dim=-1)
    abs_density = (residue_charge.float().abs().unsqueeze(0) * inv_r).sum(dim=-1)
    out[:, 0] = coulomb
    out[:, 1] = abs_density
    return out


def compute_ligand_electrostatic_aev(
    ligand_coords: Tensor,
    ligand_partial_charge: Tensor,
    pocket_coords: Tensor,
    pocket_partial_charge: Tensor,
    rc: float,
    *,
    r_min: float = 1.0,
) -> Tensor:
    """Per-ligand-atom electrostatic environment vector.

    For each ligand atom, accumulates two channels over pocket atoms within
    ``rc``:

    - ``[:, 0]`` : signed Coulombic interaction ``Σ (q_lig · q_poc) / max(r, r_min)``
    - ``[:, 1]`` : magnitude of pocket-side charge weighted by 1/r,
                   ``Σ |q_poc| / max(r, r_min)`` - gives the head a charge-
                   density feature even when local cancellation kills the
                   signed channel.

    The clip at ``r_min`` keeps the 1/r blowup bounded for very-close pairs;
    this matches what physically-motivated GBSA does (short-range exclusion).
    Returns a ``[N_ligand_atom, 2]`` tensor in arbitrary units (the encoder
    learns the scale).
    """
    ligand_coords = ligand_coords.float()
    pocket_coords = pocket_coords.float()
    lig_q = ligand_partial_charge.float().reshape(-1)
    poc_q = pocket_partial_charge.float().reshape(-1)
    n_lig = ligand_coords.shape[0]
    if n_lig == 0:
        return ligand_coords.new_zeros((0, 2))
    if pocket_coords.numel() == 0 or poc_q.numel() == 0:
        return ligand_coords.new_zeros((n_lig, 2))
    dists = torch.cdist(ligand_coords, pocket_coords)
    cutoff_mask = dists <= float(rc)
    inv_r = 1.0 / torch.clamp(dists, min=float(r_min))
    inv_r = inv_r * cutoff_mask.float()
    coulomb = (lig_q.unsqueeze(-1) * (poc_q.unsqueeze(0) * inv_r)).sum(dim=-1)
    abs_density = ((poc_q.abs().unsqueeze(0)) * inv_r).sum(dim=-1)
    return torch.stack([coulomb, abs_density], dim=-1)


def build_ligand_graph(
    atom_features: dict[str, Tensor],
    bond_features: dict[str, Tensor],
    edge_index: Tensor,
    aev: Tensor,
) -> Data:
    atomic_num = atom_features["atomic_num"].long().clamp(min=0, max=118)
    formal_charge = (atom_features["formal_charge"].long() + 4).clamp(min=0, max=8)
    hybridization = atom_features["hybridization"].long().clamp(min=0, max=7)
    chiral_tag = atom_features["chiral_tag"].long().clamp(min=0, max=3)
    atom_x = torch.cat(
        [
            F.one_hot(atomic_num, num_classes=32).float()[:, :32],
            F.one_hot(formal_charge, num_classes=9).float()[:, :4],
            torch.nan_to_num(atom_features["partial_charge"].float(), nan=0.0).unsqueeze(-1),
            atom_features["is_aromatic"].float().unsqueeze(-1),
            F.one_hot(hybridization, num_classes=8).float()[:, :8],
            F.one_hot(chiral_tag, num_classes=4).float(),
            torch.nan_to_num(atom_features["explicit_h"].float(), nan=0.0).unsqueeze(-1),
            torch.nan_to_num(atom_features["implicit_h"].float(), nan=0.0).unsqueeze(-1),
            torch.nan_to_num(atom_features["isotope"].float(), nan=0.0).unsqueeze(-1),
            aev.float(),
        ],
        dim=-1,
    )

    bond_type = bond_features["bond_type"].long().clamp(min=0, max=5)
    bond_stereo = bond_features["bond_stereo"].long().clamp(min=0, max=5)
    bond_dir = bond_features["bond_dir"].long().clamp(min=0, max=5)
    edge_attr = torch.cat(
        [
            F.one_hot(bond_type, num_classes=16).float()[:, :16],
            bond_features["is_conjugated"].float().unsqueeze(-1),
            bond_features["is_in_ring"].float().unsqueeze(-1),
            F.one_hot(bond_stereo, num_classes=8).float()[:, :8],
            F.one_hot(bond_dir, num_classes=8).float()[:, :8],
        ],
        dim=-1,
    )
    return Data(x=atom_x, edge_index=edge_index.long(), edge_attr=edge_attr)


def build_fg_graph(
    fg_features: dict[str, Tensor],
    fg_centroids: Tensor,
    cutoff: float,
    num_rbf_kernels: int,
    rbf_d_min: float,
    rbf_d_max: float,
) -> Data:
    fg_type = fg_features["fg_type"].long().clamp(min=0)
    fg_parent_type = fg_features["fg_parent_type"].long().clamp(min=0)
    x = torch.cat(
        [
            torch.stack(
                [
                    fg_features["num_atoms"].float(),
                    fg_features["total_partial_charge"].float(),
                    fg_features["max_radial_extent"].float(),
                    fg_features["mean_partial_charge"].float(),
                ],
                dim=-1,
            ),
            F.one_hot(fg_type, num_classes=16).float()[:, :16],
            F.one_hot(fg_parent_type, num_classes=8).float()[:, :8],
        ],
        dim=-1,
    )

    edge_index, distances = radius_graph_from_positions(fg_centroids.float(), cutoff)
    centers = get_rbf_centers(num_rbf_kernels, rbf_d_min, rbf_d_max).to(fg_centroids.device)
    widths = get_rbf_widths(num_rbf_kernels, rbf_d_min, rbf_d_max).to(fg_centroids.device)
    edge_attr = rbf_expand(distances, centers, widths) if distances.numel() else fg_centroids.new_zeros((0, num_rbf_kernels))
    return Data(x=x, pos=fg_centroids.float(), edge_index=edge_index, edge_attr=edge_attr)


def compute_fg_centroids(atom_coords: Tensor, fg_membership: Tensor) -> Tensor:
    atom_coords = atom_coords.float()
    if fg_membership.dim() == 2:
        weights = fg_membership.float()
        denom = weights.sum(dim=0, keepdim=True).clamp_min(1.0)
        return (weights.transpose(0, 1) @ atom_coords) / denom.transpose(0, 1)

    fg_membership = fg_membership.long()
    num_fg = int(fg_membership.max().item()) + 1 if fg_membership.numel() else 0
    centroids = atom_coords.new_zeros((num_fg, 3))
    for idx in range(num_fg):
        mask = fg_membership == idx
        if mask.any():
            centroids[idx] = atom_coords[mask].mean(dim=0)
    return centroids


def build_residue_membership(atom_residue_ids: Tensor) -> Tensor:
    _, inverse = torch.unique(atom_residue_ids.long(), sorted=True, return_inverse=True)
    return inverse


def build_cross_attn_mask(
    fg_centroids: Tensor,
    residue_coords: Tensor,
    cutoff: float,
) -> Tensor:
    if fg_centroids.numel() == 0 or residue_coords.numel() == 0:
        return torch.zeros((fg_centroids.shape[0], residue_coords.shape[0]), dtype=torch.bool, device=fg_centroids.device)
    dists = torch.cdist(fg_centroids.float(), residue_coords.float())
    mask = dists <= float(cutoff)
    if residue_coords.shape[0] > 0:
        nearest = dists.argmin(dim=-1)
        empty_rows = ~mask.any(dim=-1)
        if empty_rows.any():
            mask[empty_rows] = False
            mask[empty_rows, nearest[empty_rows]] = True
    return mask


def build_complex_data(
    ligand_data: Data,
    fg_data: Data,
    residue_features: dict[str, Tensor],
    fg_membership: Tensor,
    cross_attn_mask: Tensor,
    labels: dict[str, Tensor],
) -> dict[str, Any]:
    return {
        "ligand": ligand_data,
        "functional_groups": fg_data,
        "residue_features": residue_features,
        "fg_membership": fg_membership,
        "cross_attn_mask": cross_attn_mask,
        "labels": labels,
    }


def radius_graph_from_positions(pos: Tensor, cutoff: float) -> tuple[Tensor, Tensor]:
    pos = pos.float()
    n_nodes = pos.shape[0]
    if n_nodes <= 1:
        return pos.new_zeros((2, 0), dtype=torch.long), pos.new_zeros((0,), dtype=pos.dtype)
    dists = torch.cdist(pos, pos)
    valid = (dists <= float(cutoff)) & ~torch.eye(n_nodes, dtype=torch.bool, device=pos.device)
    src, dst = torch.nonzero(valid, as_tuple=True)
    if src.numel() == 0:
        nearest = dists.masked_fill(torch.eye(n_nodes, dtype=torch.bool, device=pos.device), float("inf")).argmin(dim=-1)
        src = torch.arange(n_nodes, device=pos.device)
        dst = nearest
    edge_index = torch.stack([src, dst], dim=0).long()
    edge_dists = dists[src, dst]
    return edge_index, edge_dists


@lru_cache(maxsize=1)
def load_plig_atom_key_map() -> dict[str, int]:
    atom_types: dict[str, int] = {}
    if not _PDB_ATOM_KEYS_PATH.is_file():
        return atom_types
    with _PDB_ATOM_KEYS_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            atom_type = str(row["ATOM_TYPE"]).strip()
            if atom_type not in atom_types:
                atom_types[atom_type] = len(atom_types)
    return atom_types
