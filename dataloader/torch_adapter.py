# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Training-facing adapter layer.

Converts the numpy tensorized samples produced by :func:`tensorize_sample` into
torch tensors with explicit dtypes, exposes PyTorch Geometric ``HeteroData``
views, supplies a ``DataLoader``-compatible collate, and provides a
:class:`torch.utils.data.Dataset` over either the live ``MMGBSADataset`` or a
directory of pre-exported ``.npz`` shards written by ``collate.export_*``.

Numerics are NumPy-faithful: NaN-valued slots stay NaN, masks stay bool, and
categorical ID columns are surfaced as ``long`` so they slot into embedding
tables without an extra cast.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, HeteroData

from .collate import _flatten_array_tree
from .mmgbsa_dataset import DatasetRoots, MMGBSADataset, build_sample
from .tensor_schema import (
    DEFAULT_TENSOR_VOCAB,
    MASKED_MODE_ORDER,
    MMGBSA_COMPONENT_ORDER,
    MMGBSA_ENTITY_ORDER,
    MMGBSA_RESIDUE_CHANNELS,
    POCKET_ATOM_FEATURE_NAMES,
    TensorVocab,
    build_feature_schema,
)
from .tensorize import tensorize_sample


# ---------------------------------------------------------------------------
# dtype maps
# ---------------------------------------------------------------------------
# Each leaf tensor produced by tensorize.py / collate.py is given an explicit
# torch dtype here. Putting this in one place makes it trivial to audit the
# integer / float / bool split, and avoids implicit promotions later.
_TENSOR_DTYPES: dict[str, torch.dtype] = {
    # ligand graph
    "ligand_x": torch.float32,
    "ligand_pos": torch.float32,
    "ligand_edge_index": torch.long,
    "ligand_edge_attr": torch.float32,
    # functional groups
    "fg_type": torch.long,
    "fg_parent_type": torch.long,
    "fg_membership": torch.float32,
    "fg_atom_index": torch.long,
    "fg_atom_mask": torch.bool,
    "fg_atom_pos": torch.float32,
    "fg_centroid_pos": torch.float32,
    "fg_mask": torch.float32,
    "fg_features": torch.float32,
    # pocket atoms + residues
    "pocket_available": torch.int8,
    "pocket_x": torch.float32,
    "pocket_pos": torch.float32,
    "pocket_edge_index": torch.long,
    "pocket_atom_to_residue": torch.long,
    "residue_x": torch.float32,
    "residue_pos": torch.float32,
    "residue_mask": torch.bool,
    "residue_mmgbsa_alignment_mask": torch.bool,
    # mmgbsa baseline
    "baseline_global_components": torch.float32,
    "baseline_global_component_mask": torch.bool,
    "baseline_residue_energy": torch.float32,
    "baseline_residue_present": torch.bool,
    "baseline_residue_alignment_mask": torch.bool,
    "baseline_ligand_decomp": torch.float32,
    # mmgbsa masked
    "masked_global_components": torch.float32,
    "masked_global_component_mask": torch.bool,
    "masked_global_delta": torch.float32,
    "masked_global_delta_mask": torch.bool,
    "masked_residue_energy": torch.float32,
    "masked_residue_present": torch.bool,
    "masked_residue_alignment_mask": torch.bool,
    "masked_residue_delta": torch.float32,
    "masked_ligand_decomp": torch.float32,
    "masked_mode_available": torch.bool,
}

# Column indices in pocket_x where categorical IDs live (mirrors tensorize.py).
_POCKET_X_CATEGORICAL_COLUMNS = {
    "pocket_residue_type_id": POCKET_ATOM_FEATURE_NAMES.index("residue_type_id"),
    "pocket_residue_base_type_id": POCKET_ATOM_FEATURE_NAMES.index("residue_base_type_id"),
    "pocket_residue_variant_id": POCKET_ATOM_FEATURE_NAMES.index("residue_variant_id"),
    "pocket_atom_name_id": POCKET_ATOM_FEATURE_NAMES.index("atom_name_id"),
}


def _to_tensor(value: np.ndarray, key: str) -> torch.Tensor:
    dtype = _TENSOR_DTYPES.get(key, torch.float32)
    arr = np.asarray(value)
    if dtype == torch.bool:
        return torch.from_numpy(arr.astype(np.bool_, copy=False))
    if dtype.is_floating_point:
        return torch.from_numpy(arr.astype(np.float32, copy=False)).to(dtype)
    if dtype == torch.long:
        return torch.from_numpy(arr.astype(np.int64, copy=False))
    if dtype == torch.int8:
        return torch.from_numpy(arr.astype(np.int8, copy=False))
    return torch.from_numpy(arr).to(dtype)


# ---------------------------------------------------------------------------
# Sample-level conversion
# ---------------------------------------------------------------------------


def to_torch_sample(numpy_sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a numpy tensorized sample into a torch-tensor sample.

    The output dict mirrors the input layout under ``ligand`` /
    ``functional_groups`` / ``pocket`` / ``mmgbsa``, with arrays replaced by
    typed torch tensors. ``residue_keys`` and ``metadata`` are passed through
    unchanged so downstream code can still align to PDB chain/resseq/resname.
    """
    out: dict[str, Any] = {"metadata": numpy_sample["metadata"]}
    for block_key in ("ligand", "functional_groups", "pocket", "mmgbsa"):
        block_out: dict[str, Any] = {}
        for leaf_key, leaf_value in numpy_sample[block_key].items():
            if leaf_key == "residue_keys":
                block_out[leaf_key] = list(leaf_value)
                continue
            block_out[leaf_key] = _to_tensor(leaf_value, leaf_key)
        out[block_key] = block_out
    return out


def numpy_sample_from_torch(torch_sample: dict[str, Any]) -> dict[str, Any]:
    """Inverse helper, primarily useful for debugging."""
    out: dict[str, Any] = {"metadata": torch_sample["metadata"]}
    for block_key in ("ligand", "functional_groups", "pocket", "mmgbsa"):
        block_out: dict[str, Any] = {}
        for leaf_key, leaf_value in torch_sample[block_key].items():
            if isinstance(leaf_value, torch.Tensor):
                block_out[leaf_key] = leaf_value.detach().cpu().numpy()
            else:
                block_out[leaf_key] = leaf_value
        out[block_key] = block_out
    return out


# ---------------------------------------------------------------------------
# PyG HeteroData view (per sample)
# ---------------------------------------------------------------------------


def to_pyg_hetero(
    torch_sample: dict[str, Any],
    *,
    include_targets: bool = True,
) -> HeteroData:
    """Wrap a torch sample as a PyG ``HeteroData`` object.

    Node types: ``ligand`` (atoms), ``fg`` (functional groups), ``pocket_atom``,
    ``residue``. Edge types capture the four graphs we actually have:
    ligand bonds, pocket atom contacts, atom->residue membership, and
    FG->ligand-atom membership.

    Targets (MMGBSA tensors) are attached at the graph level. They are not
    batchable across samples via ``Batch.from_data_list`` because the residue
    axis varies per sample - use :func:`torch_collate_samples` for that. This
    view is most useful when iterating one sample at a time or when feeding
    the structural graph to a PyG model and the targets to a separate head.
    """
    ligand = torch_sample["ligand"]
    fg = torch_sample["functional_groups"]
    pocket = torch_sample["pocket"]
    mmgbsa = torch_sample["mmgbsa"]

    data = HeteroData()

    data["ligand"].x = ligand["ligand_x"]
    data["ligand"].pos = ligand["ligand_pos"]
    data["ligand"].fg_mask = fg["fg_mask"]

    data["fg"].x = fg["fg_features"]
    data["fg"].fg_type = fg["fg_type"]
    data["fg"].fg_parent_type = fg["fg_parent_type"]
    data["fg"].centroid_pos = fg["fg_centroid_pos"]

    data["pocket_atom"].x = pocket["pocket_x"]
    data["pocket_atom"].pos = pocket["pocket_pos"]
    # Surface categorical IDs as long tensors for embedding lookup.
    pocket_x = pocket["pocket_x"]
    for attr_name, col in _POCKET_X_CATEGORICAL_COLUMNS.items():
        data["pocket_atom"][attr_name] = pocket_x[:, col].long()

    data["residue"].x = pocket["residue_x"]
    data["residue"].pos = pocket["residue_pos"]
    data["residue"].mask = pocket["residue_mask"]
    data["residue"].mmgbsa_alignment_mask = pocket["residue_mmgbsa_alignment_mask"]

    # ('ligand','bond','ligand')
    data["ligand", "bond", "ligand"].edge_index = ligand["ligand_edge_index"]
    data["ligand", "bond", "ligand"].edge_attr = ligand["ligand_edge_attr"]

    # ('pocket_atom','contact','pocket_atom')
    data["pocket_atom", "contact", "pocket_atom"].edge_index = pocket["pocket_edge_index"]

    # ('pocket_atom','in_residue','residue') - derived from atom_to_residue
    atom_to_residue = pocket["pocket_atom_to_residue"]
    if atom_to_residue.numel():
        valid = atom_to_residue >= 0
        atom_idx = torch.nonzero(valid, as_tuple=False).flatten().to(torch.long)
        residue_idx = atom_to_residue[valid].to(torch.long)
        atom_to_residue_ei = torch.stack([atom_idx, residue_idx], dim=0)
    else:
        atom_to_residue_ei = torch.zeros((2, 0), dtype=torch.long)
    data["pocket_atom", "in_residue", "residue"].edge_index = atom_to_residue_ei

    # ('fg','contains','ligand') - derived from fg_atom_index/fg_atom_mask
    fg_atom_index = fg["fg_atom_index"]
    fg_atom_mask = fg["fg_atom_mask"]
    if fg_atom_index.numel():
        valid = fg_atom_mask.bool()
        group_idx = torch.nonzero(valid, as_tuple=False)[:, 0].to(torch.long)
        atom_idx = fg_atom_index[valid].to(torch.long)
        fg_to_atom_ei = torch.stack([group_idx, atom_idx], dim=0)
    else:
        fg_to_atom_ei = torch.zeros((2, 0), dtype=torch.long)
    data["fg", "contains", "ligand"].edge_index = fg_to_atom_ei

    # node counts (helpful when a node type has no features)
    data["ligand"].num_nodes = ligand["ligand_x"].shape[0]
    data["fg"].num_nodes = fg["fg_type"].shape[0]
    data["pocket_atom"].num_nodes = pocket["pocket_x"].shape[0]
    data["residue"].num_nodes = pocket["residue_x"].shape[0]

    if include_targets:
        for key, value in mmgbsa.items():
            data[f"mmgbsa__{key}"] = value
        data["metadata"] = torch_sample["metadata"]

    return data


# ---------------------------------------------------------------------------
# Dict-style batching (primary path)
# ---------------------------------------------------------------------------


def _stack(parts: list[torch.Tensor], dtype: torch.dtype, default_shape: tuple[int, ...]) -> torch.Tensor:
    if not parts:
        return torch.zeros((0, *default_shape), dtype=dtype)
    return torch.stack(parts, dim=0)


def _cat(parts: list[torch.Tensor], dim: int, dtype: torch.dtype, default_shape: tuple[int, ...]) -> torch.Tensor:
    if not parts:
        return torch.zeros(default_shape, dtype=dtype)
    return torch.cat(parts, dim=dim)


def _pad_dim(tensor: torch.Tensor, dim: int, length: int, pad_value: float) -> torch.Tensor:
    if tensor.shape[dim] == length:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = length - tensor.shape[dim]
    pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)


def torch_collate_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Batch a list of torch samples into the same dict layout as the batch.

    The semantics match :func:`collate.collate_tensorized_samples` exactly,
    just expressed with torch tensors. Graphs (ligand, pocket) are
    concatenated and ``batch``/``ptr`` indices are produced PyG-style. The
    masked-MMGBSA per-residue tensors are padded to the largest residue count
    in the batch (NaN for energies, False for masks).
    """
    samples = list(samples)
    if not samples:
        raise ValueError("torch_collate_samples requires at least one sample")

    n_modes = len(MASKED_MODE_ORDER)
    n_entities = len(MMGBSA_ENTITY_ORDER)
    n_components = len(MMGBSA_COMPONENT_ORDER)
    n_channels = len(MMGBSA_RESIDUE_CHANNELS)

    # --- ligand graph ---
    ligand_x = [s["ligand"]["ligand_x"] for s in samples]
    ligand_pos = [s["ligand"]["ligand_pos"] for s in samples]
    ligand_edge_attr = [s["ligand"]["ligand_edge_attr"] for s in samples]
    ligand_edge_index_parts: list[torch.Tensor] = []
    ligand_batch_parts: list[torch.Tensor] = []
    ligand_ptr = [0]
    ligand_offset = 0
    for batch_idx, s in enumerate(samples):
        ei = s["ligand"]["ligand_edge_index"]
        n_atoms = s["ligand"]["ligand_x"].shape[0]
        ligand_edge_index_parts.append(ei + ligand_offset if ei.numel() else ei.clone())
        ligand_batch_parts.append(torch.full((n_atoms,), batch_idx, dtype=torch.long))
        ligand_offset += n_atoms
        ligand_ptr.append(ligand_offset)

    ligand_block = {
        "x": _cat(ligand_x, 0, torch.float32, (0, 9)),
        "pos": _cat(ligand_pos, 0, torch.float32, (0, 3)),
        "edge_index": _cat(ligand_edge_index_parts, 1, torch.long, (2, 0)),
        "edge_attr": _cat(ligand_edge_attr, 0, torch.float32, (0, 5)),
        "batch": _cat(ligand_batch_parts, 0, torch.long, (0,)),
        "ptr": torch.tensor(ligand_ptr, dtype=torch.long),
    }

    # --- pocket graph ---
    pocket_x = [s["pocket"]["pocket_x"] for s in samples]
    pocket_pos = [s["pocket"]["pocket_pos"] for s in samples]
    pocket_edge_index_parts: list[torch.Tensor] = []
    pocket_batch_parts: list[torch.Tensor] = []
    pocket_ptr = [0]
    pocket_offset = 0
    pocket_atom_to_residue_parts: list[torch.Tensor] = []
    residue_offset = 0
    for batch_idx, s in enumerate(samples):
        ei = s["pocket"]["pocket_edge_index"]
        n_atoms = s["pocket"]["pocket_x"].shape[0]
        pocket_edge_index_parts.append(ei + pocket_offset if ei.numel() else ei.clone())
        pocket_batch_parts.append(torch.full((n_atoms,), batch_idx, dtype=torch.long))
        pocket_offset += n_atoms
        pocket_ptr.append(pocket_offset)

        local_atr = s["pocket"]["pocket_atom_to_residue"].clone()
        valid = local_atr >= 0
        local_atr[valid] = local_atr[valid] + residue_offset
        pocket_atom_to_residue_parts.append(local_atr)
        residue_offset += s["pocket"]["residue_x"].shape[0]

    pocket_block: dict[str, Any] = {
        "x": _cat(pocket_x, 0, torch.float32, (0, 10)),
        "pos": _cat(pocket_pos, 0, torch.float32, (0, 3)),
        "edge_index": _cat(pocket_edge_index_parts, 1, torch.long, (2, 0)),
        "batch": _cat(pocket_batch_parts, 0, torch.long, (0,)),
        "ptr": torch.tensor(pocket_ptr, dtype=torch.long),
        "atom_to_residue": _cat(pocket_atom_to_residue_parts, 0, torch.long, (0,)),
    }

    # Residue node store (one node per pocket residue, concatenated across batch).
    residue_x_parts = [s["pocket"]["residue_x"] for s in samples]
    residue_pos_parts = [s["pocket"]["residue_pos"] for s in samples]
    residue_mask_parts = [s["pocket"]["residue_mask"] for s in samples]
    residue_align_parts = [s["pocket"]["residue_mmgbsa_alignment_mask"] for s in samples]
    residue_counts = [s["pocket"]["residue_x"].shape[0] for s in samples]
    residue_batch_parts = [
        torch.full((count,), batch_idx, dtype=torch.long)
        for batch_idx, count in enumerate(residue_counts)
    ]
    residue_ptr = [0]
    running = 0
    for count in residue_counts:
        running += count
        residue_ptr.append(running)

    pocket_block.update({
        "residue_x": _cat(residue_x_parts, 0, torch.float32, (0, 5)),
        "residue_pos": _cat(residue_pos_parts, 0, torch.float32, (0, 3)),
        "residue_mask": _cat(residue_mask_parts, 0, torch.bool, (0,)),
        "residue_mmgbsa_alignment_mask": _cat(residue_align_parts, 0, torch.bool, (0,)),
        "residue_batch": _cat(residue_batch_parts, 0, torch.long, (0,)),
        "residue_ptr": torch.tensor(residue_ptr, dtype=torch.long),
        "residue_keys": [s["pocket"]["residue_keys"] for s in samples],
    })

    # --- functional groups ---
    max_fg_atoms = max(
        (s["functional_groups"]["fg_atom_index"].shape[1] for s in samples),
        default=0,
    )
    fg_type_parts: list[torch.Tensor] = []
    fg_parent_parts: list[torch.Tensor] = []
    fg_feature_parts: list[torch.Tensor] = []
    fg_centroid_parts: list[torch.Tensor] = []
    fg_mask_parts: list[torch.Tensor] = []
    fg_batch_parts: list[torch.Tensor] = []
    fg_membership_pairs: list[torch.Tensor] = []
    fg_atom_index_blocks: list[torch.Tensor] = []
    fg_atom_mask_blocks: list[torch.Tensor] = []
    fg_atom_pos_blocks: list[torch.Tensor] = []
    fg_ptr = [0]
    fg_offset = 0
    ligand_running = 0

    for batch_idx, s in enumerate(samples):
        fg_block = s["functional_groups"]
        n_groups = fg_block["fg_type"].shape[0]
        n_atoms = s["ligand"]["ligand_x"].shape[0]

        fg_type_parts.append(fg_block["fg_type"])
        fg_parent_parts.append(fg_block["fg_parent_type"])
        fg_feature_parts.append(fg_block["fg_features"])
        fg_centroid_parts.append(fg_block["fg_centroid_pos"])
        fg_mask_parts.append(fg_block["fg_mask"])
        fg_batch_parts.append(torch.full((n_groups,), batch_idx, dtype=torch.long))

        atom_idx_padded = torch.full((n_groups, max_fg_atoms), -1, dtype=torch.long)
        atom_mask_padded = torch.zeros((n_groups, max_fg_atoms), dtype=torch.bool)
        atom_pos_padded = torch.zeros((n_groups, max_fg_atoms, 3), dtype=torch.float32)
        if n_groups:
            local_width = fg_block["fg_atom_index"].shape[1]
            atom_idx_padded[:, :local_width] = fg_block["fg_atom_index"]
            atom_mask_padded[:, :local_width] = fg_block["fg_atom_mask"]
            atom_pos_padded[:, :local_width] = fg_block["fg_atom_pos"]
            globalized = atom_idx_padded.clone()
            valid = globalized >= 0
            globalized[valid] = globalized[valid] + ligand_running
            fg_atom_index_blocks.append(globalized)
            fg_atom_mask_blocks.append(atom_mask_padded)
            fg_atom_pos_blocks.append(atom_pos_padded)
        else:
            fg_atom_index_blocks.append(atom_idx_padded)
            fg_atom_mask_blocks.append(atom_mask_padded)
            fg_atom_pos_blocks.append(atom_pos_padded)

        membership = fg_block["fg_membership"]
        if membership.numel():
            atom_idx, local_fg_idx = torch.where(membership > 0)
            if atom_idx.numel():
                fg_membership_pairs.append(
                    torch.stack(
                        [
                            atom_idx.to(torch.long) + ligand_running,
                            local_fg_idx.to(torch.long) + fg_offset,
                        ],
                        dim=0,
                    )
                )

        ligand_running += n_atoms
        fg_offset += n_groups
        fg_ptr.append(fg_offset)

    fg_block_out = {
        "fg_type": _cat(fg_type_parts, 0, torch.long, (0,)),
        "fg_parent_type": _cat(fg_parent_parts, 0, torch.long, (0,)),
        "fg_features": _cat(fg_feature_parts, 0, torch.float32, (0, 4)),
        "fg_centroid_pos": _cat(fg_centroid_parts, 0, torch.float32, (0, 3)),
        "fg_batch": _cat(fg_batch_parts, 0, torch.long, (0,)),
        "fg_ptr": torch.tensor(fg_ptr, dtype=torch.long),
        "fg_mask": _cat(fg_mask_parts, 0, torch.float32, (0,)),
        "fg_membership_index": (
            torch.cat(fg_membership_pairs, dim=1)
            if fg_membership_pairs
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        "fg_atom_index_global": _cat(fg_atom_index_blocks, 0, torch.long, (0, max_fg_atoms)),
        "fg_atom_mask": _cat(fg_atom_mask_blocks, 0, torch.bool, (0, max_fg_atoms)),
        "fg_atom_pos": _cat(fg_atom_pos_blocks, 0, torch.float32, (0, max_fg_atoms, 3)),
    }

    # --- mmgbsa ---
    max_residues = max(residue_counts, default=0)

    baseline_global = _stack(
        [s["mmgbsa"]["baseline_global_components"] for s in samples],
        torch.float32, (n_entities, n_components),
    )
    baseline_global_mask = _stack(
        [s["mmgbsa"]["baseline_global_component_mask"] for s in samples],
        torch.bool, (n_entities, n_components),
    )
    baseline_ligand_decomp = _stack(
        [s["mmgbsa"]["baseline_ligand_decomp"] for s in samples],
        torch.float32, (n_channels,),
    )

    def _pad_residue_dim(parts: list[torch.Tensor], pad_value: float, dtype: torch.dtype, axis: int) -> torch.Tensor:
        if not parts:
            return torch.zeros((0,), dtype=dtype)
        padded = [_pad_dim(p, axis, max_residues, pad_value) for p in parts]
        return torch.stack(padded, dim=0)

    baseline_residue_energy = _pad_residue_dim(
        [s["mmgbsa"]["baseline_residue_energy"] for s in samples],
        float("nan"), torch.float32, axis=0,
    )
    baseline_residue_present = _pad_residue_dim(
        [s["mmgbsa"]["baseline_residue_present"] for s in samples],
        0.0, torch.bool, axis=0,
    )
    baseline_residue_alignment = _pad_residue_dim(
        [s["mmgbsa"]["baseline_residue_alignment_mask"] for s in samples],
        0.0, torch.bool, axis=0,
    )

    # masked-mode tensors: per-sample shapes are (n_groups_i, n_modes, ...).
    # n_groups is concatenated across the batch (fg_batch already encodes the
    # mapping back to samples); the residue axis is padded to max_residues.
    masked_global = _cat(
        [s["mmgbsa"]["masked_global_components"] for s in samples],
        0, torch.float32, (0, n_modes, n_entities, n_components),
    )
    masked_global_mask = _cat(
        [s["mmgbsa"]["masked_global_component_mask"] for s in samples],
        0, torch.bool, (0, n_modes, n_entities, n_components),
    )
    masked_global_delta = _cat(
        [s["mmgbsa"]["masked_global_delta"] for s in samples],
        0, torch.float32, (0, n_modes, n_components),
    )
    masked_global_delta_mask = _cat(
        [s["mmgbsa"]["masked_global_delta_mask"] for s in samples],
        0, torch.bool, (0, n_modes, n_components),
    )
    masked_mode_available = _cat(
        [s["mmgbsa"]["masked_mode_available"] for s in samples],
        0, torch.bool, (0, n_modes),
    )
    masked_ligand_decomp = _cat(
        [s["mmgbsa"]["masked_ligand_decomp"] for s in samples],
        0, torch.float32, (0, n_modes, n_channels),
    )

    masked_residue_energy_parts = []
    masked_residue_delta_parts = []
    masked_residue_present_parts = []
    masked_residue_alignment_parts = []
    for s in samples:
        # axis=2 is the residue axis on (n_groups, n_modes, n_residues, n_channels)
        e = _pad_dim(s["mmgbsa"]["masked_residue_energy"], 2, max_residues, float("nan"))
        d = _pad_dim(s["mmgbsa"]["masked_residue_delta"], 2, max_residues, float("nan"))
        # axis=2 on (n_groups, n_modes, n_residues)
        p = _pad_dim(s["mmgbsa"]["masked_residue_present"], 2, max_residues, 0.0)
        a = _pad_dim(s["mmgbsa"]["masked_residue_alignment_mask"], 2, max_residues, 0.0)
        masked_residue_energy_parts.append(e)
        masked_residue_delta_parts.append(d)
        masked_residue_present_parts.append(p)
        masked_residue_alignment_parts.append(a)

    masked_residue_energy = _cat(
        masked_residue_energy_parts, 0, torch.float32, (0, n_modes, max_residues, n_channels),
    )
    masked_residue_delta = _cat(
        masked_residue_delta_parts, 0, torch.float32, (0, n_modes, max_residues, n_channels),
    )
    masked_residue_present = _cat(
        masked_residue_present_parts, 0, torch.bool, (0, n_modes, max_residues),
    )
    masked_residue_alignment = _cat(
        masked_residue_alignment_parts, 0, torch.bool, (0, n_modes, max_residues),
    )

    feature_schema = build_feature_schema()
    metadata = {
        "batch_size": len(samples),
        "pdb_ids": [s["metadata"]["pdb_id"] for s in samples],
        "ligand_ids": [s["metadata"]["ligand_id"] for s in samples],
        "smiles": [s["metadata"]["smiles"] for s in samples],
        "residue_counts": residue_counts,
        "max_residues": max_residues,
        "feature_schema": feature_schema,
    }

    return {
        "metadata": metadata,
        "ligand": ligand_block,
        "functional_groups": fg_block_out,
        "pocket": pocket_block,
        "mmgbsa": {
            "baseline_global_components": baseline_global,
            "baseline_global_component_mask": baseline_global_mask,
            "baseline_ligand_decomp": baseline_ligand_decomp,
            "baseline_residue_energy": baseline_residue_energy,
            "baseline_residue_present": baseline_residue_present,
            "baseline_residue_alignment_mask": baseline_residue_alignment,
            "masked_global_components": masked_global,
            "masked_global_component_mask": masked_global_mask,
            "masked_global_delta": masked_global_delta,
            "masked_global_delta_mask": masked_global_delta_mask,
            "masked_mode_available": masked_mode_available,
            "masked_ligand_decomp": masked_ligand_decomp,
            "masked_residue_energy": masked_residue_energy,
            "masked_residue_delta": masked_residue_delta,
            "masked_residue_present": masked_residue_present,
            "masked_residue_alignment_mask": masked_residue_alignment,
        },
    }


# ---------------------------------------------------------------------------
# PyG Batch (structural graph) + dict targets
# ---------------------------------------------------------------------------


def to_pyg_batch(
    torch_samples: Sequence[dict[str, Any]],
    *,
    include_targets: bool = True,
) -> tuple[Batch, Optional[dict[str, Any]]]:
    """Build a PyG ``Batch`` for the structural graph and return MMGBSA targets.

    The returned ``Batch`` carries the heterogeneous graph (ligand atoms +
    bonds, pocket atoms + contacts, atom->residue, FG->ligand-atom). The
    residue node store carries the baseline per-residue targets so they batch
    naturally as ``(total_residues, n_channels)``. The masked per-residue
    targets - which are ragged along both the FG and residue axes - are
    returned alongside the batch via the dict from
    :func:`torch_collate_samples`, so callers can use both.
    """
    if not torch_samples:
        raise ValueError("to_pyg_batch requires at least one sample")
    data_list: list[HeteroData] = []
    for sample in torch_samples:
        data = to_pyg_hetero(sample, include_targets=False)
        if include_targets:
            mmgbsa = sample["mmgbsa"]
            data["residue"].baseline_energy = mmgbsa["baseline_residue_energy"]
            data["residue"].baseline_present = mmgbsa["baseline_residue_present"]
            data["residue"].baseline_alignment = mmgbsa["baseline_residue_alignment_mask"]
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    targets = torch_collate_samples(torch_samples)["mmgbsa"] if include_targets else None
    return batch, targets


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------


class MMGBSATorchDataset(Dataset):
    """Torch ``Dataset`` that produces ``to_torch_sample(tensorize_sample(...))``.

    Pass ``pdb_ids`` to fix the index ordering; otherwise the discovery rules
    in :class:`MMGBSADataset` apply. ``vocab`` is shared across all samples so
    categorical IDs stay consistent.
    """

    def __init__(
        self,
        pdb_ids: Optional[Iterable[str]] = None,
        *,
        roots: Optional[DatasetRoots] = None,
        vocab: Optional[TensorVocab] = None,
        require_baseline: bool = True,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        self._base = MMGBSADataset(pdb_ids=pdb_ids, roots=roots, require_baseline=require_baseline)
        self.vocab = vocab or DEFAULT_TENSOR_VOCAB
        self.transform = transform

    @property
    def pdb_ids(self) -> list[str]:
        return self._base.pdb_ids

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int | str) -> dict[str, Any]:
        sample = self._base[index]
        tensorized = tensorize_sample(sample, vocab=self.vocab)
        torch_sample = to_torch_sample(tensorized)
        if self.transform is not None:
            torch_sample = self.transform(torch_sample)
        return torch_sample


class CachedMMGBSATorchDataset(Dataset):
    """Torch ``Dataset`` over a directory of pre-exported ``.npz`` shards.

    Supports two on-disk layouts:

    - Per-PDB subdirs (preferred): ``cache_dir/<pdb_id>/<pdb_id>.npz`` +
      ``cache_dir/<pdb_id>/<pdb_id>.metadata.json``. This is what
      :mod:`scripts.tensorize_all_raw` writes.
    - Flat (legacy): ``cache_dir/<pdb_id>.npz`` +
      ``cache_dir/<pdb_id>.metadata.json``. Still accepted for older caches
      produced by :func:`export_torch_cache`.

    Discovery scans subdirs first; on miss, falls back to the flat layout
    so a single cache can mix the two.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        pdb_ids: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"Cache directory does not exist: {self.cache_dir}")
        if pdb_ids is None:
            discovered: set[str] = set()
            for entry in self.cache_dir.iterdir():
                if entry.name.startswith("_"):
                    continue
                if entry.is_dir():
                    if (entry / f"{entry.name}.npz").is_file():
                        discovered.add(entry.name)
                elif entry.is_file() and entry.suffix == ".npz":
                    discovered.add(entry.stem)
            self.pdb_ids = sorted(discovered)
        else:
            self.pdb_ids = list(pdb_ids)
        self.transform = transform

    def _paths_for(self, pdb_id: str) -> tuple[Path, Path]:
        nested_arr = self.cache_dir / pdb_id / f"{pdb_id}.npz"
        if nested_arr.is_file():
            return nested_arr, self.cache_dir / pdb_id / f"{pdb_id}.metadata.json"
        return (
            self.cache_dir / f"{pdb_id}.npz",
            self.cache_dir / f"{pdb_id}.metadata.json",
        )

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, index: int | str) -> dict[str, Any]:
        pdb_id = self.pdb_ids[index] if isinstance(index, int) else str(index)
        arrays_path, metadata_path = self._paths_for(pdb_id)
        metadata = json.loads(metadata_path.read_text())
        with np.load(arrays_path, allow_pickle=False) as npz:
            blocks: dict[str, dict[str, np.ndarray]] = {
                "ligand": {},
                "functional_groups": {},
                "pocket": {},
                "mmgbsa": {},
            }
            for key in npz.files:
                block_key, _, leaf = key.partition("__")
                if block_key not in blocks or not leaf:
                    continue
                blocks[block_key][leaf] = npz[key]
        # residue_keys is metadata-only in the export - restore as tuples.
        blocks["pocket"]["residue_keys"] = [tuple(k) for k in metadata.get("residue_keys", [])]
        numpy_sample = {"metadata": metadata, **blocks}
        torch_sample = to_torch_sample(numpy_sample)
        if self.transform is not None:
            torch_sample = self.transform(torch_sample)
        return torch_sample


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate: str = "dict",
) -> DataLoader:
    """Build a ``DataLoader`` whose ``collate_fn`` matches the requested format.

    ``collate='dict'`` returns the nested-dict batch from
    :func:`torch_collate_samples`; ``collate='pyg'`` returns
    ``(Batch, target_dict)`` from :func:`to_pyg_batch`.
    """
    if collate == "dict":
        collate_fn: Callable[[list[dict[str, Any]]], Any] = torch_collate_samples
    elif collate == "pyg":
        collate_fn = lambda batch: to_pyg_batch(batch)  # noqa: E731
    else:
        raise ValueError(f"Unknown collate mode: {collate}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# Cache export helper that also serializes per-PDB residue_keys
# ---------------------------------------------------------------------------


def export_torch_cache(
    pdb_ids: Iterable[str],
    output_root: str | Path,
    *,
    roots: Optional[DatasetRoots] = None,
    vocab: Optional[TensorVocab] = None,
    overwrite: bool = False,
) -> list[tuple[str, Path, Path]]:
    """Same as :func:`collate.export_tensorized_dataset` but ensures the
    ``.metadata.json`` includes ``residue_keys`` so the cached torch dataset
    can roundtrip pocket residue identifiers without rebuilding the sample.
    """
    from .collate import _jsonify  # local import to avoid a public api dep

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    roots = roots or DatasetRoots()
    vocab = vocab or DEFAULT_TENSOR_VOCAB

    exported: list[tuple[str, Path, Path]] = []
    for pdb_id in pdb_ids:
        arrays_path = output_root / f"{pdb_id}.npz"
        metadata_path = output_root / f"{pdb_id}.metadata.json"
        if not overwrite and arrays_path.exists() and metadata_path.exists():
            exported.append((str(pdb_id), arrays_path, metadata_path))
            continue
        sample = build_sample(str(pdb_id), roots=roots)
        tensorized = tensorize_sample(sample, vocab=vocab)
        arrays: dict[str, np.ndarray] = {}
        for block_key in ("ligand", "functional_groups", "pocket", "mmgbsa"):
            _flatten_array_tree(block_key, tensorized[block_key], arrays)
        np.savez_compressed(arrays_path, **arrays)
        metadata = tensorized["metadata"]
        metadata_path.write_text(json.dumps(_jsonify(metadata), indent=2))
        exported.append((str(pdb_id), arrays_path, metadata_path))

    schema_path = output_root / "_feature_schema.json"
    schema_path.write_text(json.dumps(build_feature_schema(vocab), indent=2))
    return exported
