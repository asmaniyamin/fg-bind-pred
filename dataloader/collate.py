# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .mmgbsa_dataset import DatasetRoots, build_sample
from .tensor_schema import DEFAULT_TENSOR_VOCAB, TensorVocab, build_feature_schema
from .tensorize import tensorize_sample


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _pad_axis0(arrays: list[np.ndarray], pad_value: float = 0.0, dtype=None) -> np.ndarray:
    if not arrays:
        return np.zeros((0,), dtype=dtype or np.float32)
    max_len = max(arr.shape[0] for arr in arrays)
    tail_shape = arrays[0].shape[1:]
    out_dtype = dtype or arrays[0].dtype
    result = np.full((len(arrays), max_len, *tail_shape), pad_value, dtype=out_dtype)
    for idx, arr in enumerate(arrays):
        result[idx, : arr.shape[0]] = arr
    return result


def _concat_graph_block(
    samples: list[dict[str, Any]],
    block_key: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    xs: list[np.ndarray] = []
    poss: list[np.ndarray] = []
    edge_indices: list[np.ndarray] = []
    edge_attrs: list[np.ndarray] = []
    batch_parts: list[np.ndarray] = []
    ptr = [0]
    node_offset = 0
    for batch_idx, sample in enumerate(samples):
        block = sample[block_key]
        x = block["ligand_x"] if block_key == "ligand" else block["pocket_x"]
        pos = block["ligand_pos"] if block_key == "ligand" else block["pocket_pos"]
        edge_index = block["ligand_edge_index"] if block_key == "ligand" else block["pocket_edge_index"]
        edge_attr = block["ligand_edge_attr"] if block_key == "ligand" else None
        xs.append(x)
        poss.append(pos)
        if edge_index.size:
            edge_indices.append(edge_index + node_offset)
        else:
            edge_indices.append(np.zeros((2, 0), dtype=np.int64))
        if edge_attr is not None:
            edge_attrs.append(edge_attr)
        batch_parts.append(np.full((x.shape[0],), batch_idx, dtype=np.int64))
        node_offset += x.shape[0]
        ptr.append(node_offset)
    merged = {
        "x": np.concatenate(xs, axis=0) if xs else np.zeros((0, 0), dtype=np.float32),
        "pos": np.concatenate(poss, axis=0) if poss else np.zeros((0, 3), dtype=np.float32),
        "edge_index": np.concatenate(edge_indices, axis=1) if edge_indices else np.zeros((2, 0), dtype=np.int64),
        "batch": np.concatenate(batch_parts, axis=0) if batch_parts else np.zeros((0,), dtype=np.int64),
        "ptr": np.asarray(ptr, dtype=np.int64),
    }
    if edge_attrs:
        merged["edge_attr"] = np.concatenate(edge_attrs, axis=0)
    return merged, np.asarray(ptr, dtype=np.int64)


def collate_tensorized_samples(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    samples = list(samples)
    if not samples:
        raise ValueError("collate_tensorized_samples requires at least one sample")

    ligand_graph, ligand_ptr = _concat_graph_block(samples, "ligand")
    pocket_graph, pocket_ptr = _concat_graph_block(samples, "pocket")

    ligand_counts = [sample["ligand"]["ligand_x"].shape[0] for sample in samples]
    residue_counts = [sample["pocket"]["residue_x"].shape[0] for sample in samples]
    max_residue_count = max(residue_counts, default=0)
    max_fg_atoms = max(
        (sample["functional_groups"]["fg_atom_index"].shape[1] for sample in samples),
        default=0,
    )

    fg_type_parts: list[np.ndarray] = []
    fg_parent_parts: list[np.ndarray] = []
    fg_feature_parts: list[np.ndarray] = []
    fg_centroid_parts: list[np.ndarray] = []
    fg_batch_parts: list[np.ndarray] = []
    fg_mask_parts: list[np.ndarray] = []
    fg_membership_pairs: list[np.ndarray] = []
    fg_atom_index_blocks: list[np.ndarray] = []
    fg_atom_mask_blocks: list[np.ndarray] = []
    fg_atom_pos_blocks: list[np.ndarray] = []
    fg_ptr = [0]
    fg_offset = 0
    ligand_offset = 0

    baseline_global_parts: list[np.ndarray] = []
    baseline_global_mask_parts: list[np.ndarray] = []
    baseline_ligand_decomp_parts: list[np.ndarray] = []
    baseline_residue_energy_parts: list[np.ndarray] = []
    baseline_residue_present_parts: list[np.ndarray] = []
    baseline_residue_alignment_parts: list[np.ndarray] = []

    masked_global_parts: list[np.ndarray] = []
    masked_global_mask_parts: list[np.ndarray] = []
    masked_global_delta_parts: list[np.ndarray] = []
    masked_global_delta_mask_parts: list[np.ndarray] = []
    masked_mode_available_parts: list[np.ndarray] = []
    masked_ligand_decomp_parts: list[np.ndarray] = []
    masked_residue_energy_parts: list[np.ndarray] = []
    masked_residue_delta_parts: list[np.ndarray] = []
    masked_residue_present_parts: list[np.ndarray] = []
    masked_residue_alignment_parts: list[np.ndarray] = []

    residue_x_parts: list[np.ndarray] = []
    residue_pos_parts: list[np.ndarray] = []
    residue_mask_parts: list[np.ndarray] = []
    residue_batch_parts: list[np.ndarray] = []
    residue_alignment_parts: list[np.ndarray] = []
    residue_key_lists: list[list[Any]] = []
    residue_ptr = [0]
    residue_offset = 0

    for batch_idx, sample in enumerate(samples):
        fg_block = sample["functional_groups"]
        pocket_block = sample["pocket"]
        mmgbsa_block = sample["mmgbsa"]

        n_ligand_atoms = ligand_counts[batch_idx]
        n_groups = fg_block["fg_type"].shape[0]
        n_residues = residue_counts[batch_idx]

        fg_type_parts.append(fg_block["fg_type"])
        fg_parent_parts.append(fg_block["fg_parent_type"])
        fg_feature_parts.append(fg_block["fg_features"])
        fg_centroid_parts.append(fg_block["fg_centroid_pos"])
        fg_mask_parts.append(fg_block["fg_mask"])
        fg_batch_parts.append(np.full((n_groups,), batch_idx, dtype=np.int64))

        fg_atom_index = np.full((n_groups, max_fg_atoms), -1, dtype=np.int64)
        fg_atom_mask = np.zeros((n_groups, max_fg_atoms), dtype=bool)
        fg_atom_pos = np.zeros((n_groups, max_fg_atoms, 3), dtype=np.float32)
        if n_groups:
            local_width = fg_block["fg_atom_index"].shape[1]
            fg_atom_index[:, :local_width] = fg_block["fg_atom_index"]
            fg_atom_mask[:, :local_width] = fg_block["fg_atom_mask"]
            fg_atom_pos[:, :local_width] = fg_block["fg_atom_pos"]
            globalized = fg_atom_index.copy()
            valid = globalized >= 0
            globalized[valid] += ligand_offset
            fg_atom_index_blocks.append(globalized)
            fg_atom_mask_blocks.append(fg_atom_mask)
            fg_atom_pos_blocks.append(fg_atom_pos)
        else:
            fg_atom_index_blocks.append(fg_atom_index)
            fg_atom_mask_blocks.append(fg_atom_mask)
            fg_atom_pos_blocks.append(fg_atom_pos)

        atom_idx, local_fg_idx = np.where(fg_block["fg_membership"] > 0)
        if atom_idx.size:
            fg_membership_pairs.append(
                np.stack(
                    [
                        atom_idx.astype(np.int64) + ligand_offset,
                        local_fg_idx.astype(np.int64) + fg_offset,
                    ],
                    axis=0,
                )
            )

        baseline_global_parts.append(mmgbsa_block["baseline_global_components"])
        baseline_global_mask_parts.append(mmgbsa_block["baseline_global_component_mask"])
        baseline_ligand_decomp_parts.append(mmgbsa_block["baseline_ligand_decomp"])

        baseline_residue_energy_parts.append(mmgbsa_block["baseline_residue_energy"])
        baseline_residue_present_parts.append(mmgbsa_block["baseline_residue_present"])
        baseline_residue_alignment_parts.append(mmgbsa_block["baseline_residue_alignment_mask"])

        masked_global_parts.append(mmgbsa_block["masked_global_components"])
        masked_global_mask_parts.append(mmgbsa_block["masked_global_component_mask"])
        masked_global_delta_parts.append(mmgbsa_block["masked_global_delta"])
        masked_global_delta_mask_parts.append(mmgbsa_block["masked_global_delta_mask"])
        masked_mode_available_parts.append(mmgbsa_block["masked_mode_available"])
        masked_ligand_decomp_parts.append(mmgbsa_block["masked_ligand_decomp"])

        padded_residue_energy = np.full((n_groups, 3, max_residue_count, 6), np.nan, dtype=np.float32)
        padded_residue_delta = np.full((n_groups, 3, max_residue_count, 6), np.nan, dtype=np.float32)
        padded_residue_present = np.zeros((n_groups, 3, max_residue_count), dtype=bool)
        padded_residue_alignment = np.zeros((n_groups, 3, max_residue_count), dtype=bool)
        if n_groups and n_residues:
            padded_residue_energy[:, :, :n_residues, :] = mmgbsa_block["masked_residue_energy"]
            padded_residue_delta[:, :, :n_residues, :] = mmgbsa_block["masked_residue_delta"]
            padded_residue_present[:, :, :n_residues] = mmgbsa_block["masked_residue_present"]
            padded_residue_alignment[:, :, :n_residues] = mmgbsa_block["masked_residue_alignment_mask"]
        masked_residue_energy_parts.append(padded_residue_energy)
        masked_residue_delta_parts.append(padded_residue_delta)
        masked_residue_present_parts.append(padded_residue_present)
        masked_residue_alignment_parts.append(padded_residue_alignment)

        residue_x_parts.append(pocket_block["residue_x"])
        residue_pos_parts.append(pocket_block["residue_pos"])
        residue_mask_parts.append(pocket_block["residue_mask"])
        residue_alignment_parts.append(pocket_block["residue_mmgbsa_alignment_mask"])
        residue_batch_parts.append(np.full((n_residues,), batch_idx, dtype=np.int64))
        residue_key_lists.append(pocket_block["residue_keys"])
        residue_offset += n_residues
        residue_ptr.append(residue_offset)

        ligand_offset += n_ligand_atoms
        fg_offset += n_groups
        fg_ptr.append(fg_offset)

    residue_x = np.concatenate(residue_x_parts, axis=0) if residue_x_parts else np.zeros((0, 0), dtype=np.float32)
    residue_pos = np.concatenate(residue_pos_parts, axis=0) if residue_pos_parts else np.zeros((0, 3), dtype=np.float32)
    residue_mask = np.concatenate(residue_mask_parts, axis=0) if residue_mask_parts else np.zeros((0,), dtype=bool)
    residue_batch = np.concatenate(residue_batch_parts, axis=0) if residue_batch_parts else np.zeros((0,), dtype=np.int64)
    residue_alignment_mask = np.concatenate(residue_alignment_parts, axis=0) if residue_alignment_parts else np.zeros((0,), dtype=bool)

    pocket_atom_to_residue_parts: list[np.ndarray] = []
    running_residue_offset = 0
    for sample, count in zip(samples, residue_counts):
        local = sample["pocket"]["pocket_atom_to_residue"].copy()
        valid = local >= 0
        local[valid] += running_residue_offset
        pocket_atom_to_residue_parts.append(local)
        running_residue_offset += count

    feature_schema = build_feature_schema()
    return {
        "metadata": {
            "batch_size": len(samples),
            "pdb_ids": [sample["metadata"]["pdb_id"] for sample in samples],
            "ligand_ids": [sample["metadata"]["ligand_id"] for sample in samples],
            "smiles": [sample["metadata"]["smiles"] for sample in samples],
            "feature_schema": feature_schema,
        },
        "ligand": ligand_graph,
        "functional_groups": {
            "fg_type": np.concatenate(fg_type_parts, axis=0) if fg_type_parts else np.zeros((0,), dtype=np.int64),
            "fg_parent_type": np.concatenate(fg_parent_parts, axis=0) if fg_parent_parts else np.zeros((0,), dtype=np.int64),
            "fg_features": np.concatenate(fg_feature_parts, axis=0) if fg_feature_parts else np.zeros((0, 0), dtype=np.float32),
            "fg_centroid_pos": np.concatenate(fg_centroid_parts, axis=0) if fg_centroid_parts else np.zeros((0, 3), dtype=np.float32),
            "fg_batch": np.concatenate(fg_batch_parts, axis=0) if fg_batch_parts else np.zeros((0,), dtype=np.int64),
            "fg_ptr": np.asarray(fg_ptr, dtype=np.int64),
            "fg_mask": np.concatenate(fg_mask_parts, axis=0) if fg_mask_parts else np.zeros((0,), dtype=np.float32),
            "fg_membership_index": np.concatenate(fg_membership_pairs, axis=1) if fg_membership_pairs else np.zeros((2, 0), dtype=np.int64),
            "fg_atom_index_global": np.concatenate(fg_atom_index_blocks, axis=0) if fg_atom_index_blocks else np.zeros((0, max_fg_atoms), dtype=np.int64),
            "fg_atom_mask": np.concatenate(fg_atom_mask_blocks, axis=0) if fg_atom_mask_blocks else np.zeros((0, max_fg_atoms), dtype=bool),
            "fg_atom_pos": np.concatenate(fg_atom_pos_blocks, axis=0) if fg_atom_pos_blocks else np.zeros((0, max_fg_atoms, 3), dtype=np.float32),
        },
        "pocket": {
            **pocket_graph,
            "atom_to_residue": np.concatenate(pocket_atom_to_residue_parts, axis=0) if pocket_atom_to_residue_parts else np.zeros((0,), dtype=np.int64),
            "residue_x": residue_x,
            "residue_pos": residue_pos,
            "residue_mask": residue_mask,
            "residue_batch": residue_batch,
            "residue_ptr": np.asarray(residue_ptr, dtype=np.int64),
            "residue_mmgbsa_alignment_mask": residue_alignment_mask,
            "residue_keys": residue_key_lists,
        },
        "mmgbsa": {
            "baseline_global_components": np.stack(baseline_global_parts, axis=0) if baseline_global_parts else np.zeros((0, 4, 15), dtype=np.float32),
            "baseline_global_component_mask": np.stack(baseline_global_mask_parts, axis=0) if baseline_global_mask_parts else np.zeros((0, 4, 15), dtype=bool),
            "baseline_ligand_decomp": np.stack(baseline_ligand_decomp_parts, axis=0) if baseline_ligand_decomp_parts else np.zeros((0, 6), dtype=np.float32),
            "baseline_residue_energy": _pad_axis0(baseline_residue_energy_parts, pad_value=np.nan, dtype=np.float32),
            "baseline_residue_present": _pad_axis0([arr.astype(np.float32)[:, None] for arr in baseline_residue_present_parts], pad_value=0.0, dtype=np.float32)[:, :, 0].astype(bool) if baseline_residue_present_parts else np.zeros((0, 0), dtype=bool),
            "baseline_residue_alignment_mask": _pad_axis0([arr.astype(np.float32)[:, None] for arr in baseline_residue_alignment_parts], pad_value=0.0, dtype=np.float32)[:, :, 0].astype(bool) if baseline_residue_alignment_parts else np.zeros((0, 0), dtype=bool),
            "masked_global_components": np.concatenate(masked_global_parts, axis=0) if masked_global_parts else np.zeros((0, 3, 4, 15), dtype=np.float32),
            "masked_global_component_mask": np.concatenate(masked_global_mask_parts, axis=0) if masked_global_mask_parts else np.zeros((0, 3, 4, 15), dtype=bool),
            "masked_global_delta": np.concatenate(masked_global_delta_parts, axis=0) if masked_global_delta_parts else np.zeros((0, 3, 15), dtype=np.float32),
            "masked_global_delta_mask": np.concatenate(masked_global_delta_mask_parts, axis=0) if masked_global_delta_mask_parts else np.zeros((0, 3, 15), dtype=bool),
            "masked_ligand_decomp": np.concatenate(masked_ligand_decomp_parts, axis=0) if masked_ligand_decomp_parts else np.zeros((0, 3, 6), dtype=np.float32),
            "masked_mode_available": np.concatenate(masked_mode_available_parts, axis=0) if masked_mode_available_parts else np.zeros((0, 3), dtype=bool),
            "masked_residue_energy": np.concatenate(masked_residue_energy_parts, axis=0) if masked_residue_energy_parts else np.zeros((0, 3, max_residue_count, 6), dtype=np.float32),
            "masked_residue_delta": np.concatenate(masked_residue_delta_parts, axis=0) if masked_residue_delta_parts else np.zeros((0, 3, max_residue_count, 6), dtype=np.float32),
            "masked_residue_present": np.concatenate(masked_residue_present_parts, axis=0) if masked_residue_present_parts else np.zeros((0, 3, max_residue_count), dtype=bool),
            "masked_residue_alignment_mask": np.concatenate(masked_residue_alignment_parts, axis=0) if masked_residue_alignment_parts else np.zeros((0, 3, max_residue_count), dtype=bool),
        },
    }


def _flatten_array_tree(prefix: str, value: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(value, np.ndarray):
        out[prefix] = value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}__{key}" if prefix else str(key)
            _flatten_array_tree(child_prefix, child, out)


def export_tensorized_sample(
    sample: dict[str, Any],
    output_dir: str | Path,
    *,
    vocab: Optional[TensorVocab] = None,
    filename_stem: Optional[str] = None,
) -> tuple[Path, Path]:
    tensorized = tensorize_sample(sample, vocab=vocab or DEFAULT_TENSOR_VOCAB)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = filename_stem or sample["pdb_id"]

    arrays: dict[str, np.ndarray] = {}
    for block_key in ("ligand", "functional_groups", "pocket", "mmgbsa"):
        _flatten_array_tree(block_key, tensorized[block_key], arrays)

    arrays_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}.metadata.json"
    np.savez_compressed(arrays_path, **arrays)
    metadata_path.write_text(json.dumps(_jsonify(tensorized["metadata"]), indent=2))
    return arrays_path, metadata_path


def export_tensorized_dataset(
    pdb_ids: Iterable[str],
    output_root: str | Path,
    *,
    roots: Optional[DatasetRoots] = None,
    vocab: Optional[TensorVocab] = None,
    overwrite: bool = False,
) -> list[tuple[str, Path, Path]]:
    roots = roots or DatasetRoots()
    vocab = vocab or DEFAULT_TENSOR_VOCAB
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    exported: list[tuple[str, Path, Path]] = []
    for pdb_id in pdb_ids:
        arrays_path = output_root / f"{pdb_id}.npz"
        metadata_path = output_root / f"{pdb_id}.metadata.json"
        if not overwrite and arrays_path.exists() and metadata_path.exists():
            exported.append((pdb_id, arrays_path, metadata_path))
            continue
        sample = build_sample(str(pdb_id), roots=roots)
        arr_path, meta_path = export_tensorized_sample(
            sample,
            output_root,
            vocab=vocab,
            filename_stem=str(pdb_id),
        )
        exported.append((str(pdb_id), arr_path, meta_path))
    schema_path = output_root / "_feature_schema.json"
    schema_path.write_text(json.dumps(_jsonify(build_feature_schema(vocab)), indent=2))
    return exported
