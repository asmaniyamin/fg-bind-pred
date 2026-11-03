# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
from rdkit import Chem

from chem.build.build_utils import build_edge_index

from .tensor_schema import (
    BACKBONE_ATOM_NAMES,
    DEFAULT_TENSOR_VOCAB,
    FG_FEATURE_NAMES,
    LIGAND_ATOM_FEATURE_NAMES,
    LIGAND_EDGE_FEATURE_NAMES,
    MASKED_MODE_ORDER,
    MMGBSA_COMPONENT_ORDER,
    MMGBSA_ENTITY_ORDER,
    MMGBSA_RESIDUE_CHANNELS,
    POCKET_ATOM_FEATURE_NAMES,
    RESIDUE_FEATURE_NAMES,
    TensorVocab,
    build_feature_schema,
)


PERIODIC_TABLE = Chem.GetPeriodicTable()
RESNAME_NORMALIZATION = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
    "CYM": "CYS",
    "CYX": "CYS",
    "CYN": "CYS",
    "MSE": "MET",
}
RESIDUE_VARIANT_TOKENS = set(RESNAME_NORMALIZATION.keys())


def _zeros(shape, dtype=np.float32) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def _nan_full(shape, dtype=np.float32) -> np.ndarray:
    arr = np.full(shape, np.nan, dtype=dtype)
    return arr


def _optional_array(value: Optional[np.ndarray], length: int, dtype, default=0) -> np.ndarray:
    if value is None:
        return np.full((length,), default, dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _directed_edge_attr_from_bonds(edge_attr_undirected: np.ndarray) -> np.ndarray:
    if edge_attr_undirected.size == 0:
        return edge_attr_undirected.reshape((0, edge_attr_undirected.shape[-1] if edge_attr_undirected.ndim == 2 else 0))
    return np.repeat(edge_attr_undirected, 2, axis=0)


def _pairwise_min_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.full((a.shape[0],), np.nan, dtype=np.float32)
    diff = a[:, None, :] - b[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist.min(axis=1).astype(np.float32)


def _build_radius_graph(pos: np.ndarray, radius: float = 4.5, max_neighbors: int = 24) -> np.ndarray:
    n = pos.shape[0]
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    edges: list[tuple[int, int]] = []
    for i in range(n):
        neighbors = np.where(dist[i] <= radius)[0]
        if neighbors.size == 0:
            neighbors = np.argsort(dist[i])[: min(max_neighbors, max(0, n - 1))]
            neighbors = neighbors[np.isfinite(dist[i, neighbors])]
        else:
            order = np.argsort(dist[i, neighbors])
            neighbors = neighbors[order[:max_neighbors]]
        for j in neighbors:
            edges.append((i, int(j)))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def _atomic_number_from_element(element: str | None) -> int:
    token = (element or "").strip()
    if not token:
        return 0
    try:
        return int(PERIODIC_TABLE.GetAtomicNumber(token.capitalize()))
    except Exception:  # noqa: BLE001
        return 0


def _normalize_resname(resname: str | None) -> str:
    token = (resname or "").strip().upper()
    if not token:
        return "UNK"
    return RESNAME_NORMALIZATION.get(token, token)


def _residue_variant_token(resname: str | None) -> str:
    token = (resname or "").strip().upper()
    if not token:
        return "NONE"
    return token if token in RESIDUE_VARIANT_TOKENS else "NONE"


def _parse_residue_token(token: str | None) -> tuple[str, Optional[int]]:
    compact = str(token or "").strip().replace(" ", "")
    if not compact:
        return "", None
    match = re.match(r"^([A-Za-z][A-Za-z0-9]*?)(-?\d+)?$", compact)
    if not match:
        return compact.upper(), None
    resname = match.group(1).upper()
    resseq = int(match.group(2)) if match.group(2) is not None else None
    return resname, resseq


def _parse_mmgbsa_row_key(row: dict[str, Any]) -> tuple[str, Optional[int], str]:
    residue_token = str(row.get("residue") or "").strip()
    location = str(row.get("location") or "").strip()
    entity = location[:1] if location else ""
    resname, resseq = _parse_residue_token(residue_token)
    if resseq is None and location:
        tail = location[1:].strip()
        loc_resname, loc_resseq = _parse_residue_token(tail)
        if loc_resname:
            resname = loc_resname
        if loc_resseq is not None:
            resseq = loc_resseq
    return resname, resseq, entity


def _build_amber_renum_lookup(renum_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for row in renum_rows or []:
        try:
            amber_resseq = int(row["amber_resseq"])
            lookup[amber_resseq] = row
        except Exception:  # noqa: BLE001
            continue
    return lookup


def _build_receptor_to_pocket_lookup(
    residue_keys: list[tuple[str, int, str, str]],
    prepared_to_receptor_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int, str], list[int]], dict[tuple[str, int, str], list[int]]]:
    pocket_exact_lookup: dict[tuple[str, int, str, str], int] = {}
    pocket_norm_lookup: dict[tuple[str, int, str], list[int]] = {}
    for idx, (chain_id, resseq, _icode, resname) in enumerate(residue_keys):
        pocket_exact_lookup[(str(chain_id).strip(), int(resseq), "", resname.upper())] = idx
        pocket_norm_lookup.setdefault(
            (str(chain_id).strip(), int(resseq), _normalize_resname(resname)),
            [],
        ).append(idx)

    receptor_exact_lookup: dict[tuple[str, int, str], list[int]] = {}
    receptor_norm_lookup: dict[tuple[str, int, str], list[int]] = {}
    for row in prepared_to_receptor_rows or []:
        prepared_key = (
            str(row.get("chain_id") or "").strip(),
            int(row["prepared_resseq"]),
            str(row.get("prepared_insertion_code") or "").strip(),
            str(row.get("prepared_resname") or "").strip().upper(),
        )
        pocket_idx = pocket_exact_lookup.get(prepared_key)
        if pocket_idx is None:
            norm_key = (
                prepared_key[0],
                prepared_key[1],
                _normalize_resname(prepared_key[3]),
            )
            pocket_matches = pocket_norm_lookup.get(norm_key, [])
            if len(pocket_matches) != 1:
                continue
            pocket_idx = pocket_matches[0]
        receptor_exact_lookup.setdefault(
            (
                str(row.get("chain_id") or "").strip(),
                int(row["receptor_resseq"]),
                str(row.get("receptor_resname") or "").strip().upper(),
            ),
            [],
        ).append(pocket_idx)
        receptor_norm_lookup.setdefault(
            (
                str(row.get("chain_id") or "").strip(),
                int(row["receptor_resseq"]),
                _normalize_resname(row.get("receptor_resname")),
            ),
            [],
        ).append(pocket_idx)
    return receptor_exact_lookup, receptor_norm_lookup


def _build_ligand_tensors(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    molgraph = sample["ligand"]["molgraph"]
    if molgraph is None:
        return {
            "ligand_x": _zeros((0, 9), dtype=np.float32),
            "ligand_pos": _zeros((0, 3), dtype=np.float32),
            "ligand_edge_index": np.zeros((2, 0), dtype=np.int64),
            "ligand_edge_attr": _zeros((0, 5), dtype=np.float32),
        }

    arrays = molgraph.arrays
    n_atoms = len(arrays.atomic_num)
    ligand_x = np.stack(
        [
            np.asarray(arrays.atomic_num, dtype=np.float32),
            np.asarray(arrays.formal_charge, dtype=np.float32),
            _optional_array(arrays.partial_charge, n_atoms, np.float32, default=0.0),
            _optional_array(arrays.is_aromatic, n_atoms, np.float32, default=0),
            _optional_array(arrays.hybridization, n_atoms, np.float32, default=0),
            _optional_array(arrays.chiral_tag, n_atoms, np.float32, default=0),
            _optional_array(arrays.explicit_h, n_atoms, np.float32, default=0),
            _optional_array(arrays.implicit_h, n_atoms, np.float32, default=0),
            _optional_array(arrays.isotope, n_atoms, np.float32, default=0),
        ],
        axis=1,
    ).astype(np.float32)

    pos = np.asarray(arrays.pos, dtype=np.float32) if arrays.pos is not None else _zeros((n_atoms, 3))
    edge_index = build_edge_index(arrays)
    m_bonds = arrays.bonds.shape[0]
    edge_attr_undirected = np.stack(
        [
            np.asarray(arrays.bond_type, dtype=np.float32),
            _optional_array(arrays.is_conjugated, m_bonds, np.float32, default=0),
            _optional_array(arrays.is_in_ring, m_bonds, np.float32, default=0),
            _optional_array(arrays.bond_stereo, m_bonds, np.float32, default=0),
            _optional_array(arrays.bond_dir, m_bonds, np.float32, default=0),
        ],
        axis=1,
    ).astype(np.float32) if m_bonds else _zeros((0, 5))
    edge_attr = _directed_edge_attr_from_bonds(edge_attr_undirected).astype(np.float32)

    return {
        "ligand_x": ligand_x,
        "ligand_pos": pos,
        "ligand_edge_index": edge_index,
        "ligand_edge_attr": edge_attr,
    }


def _build_fg_tensors(sample: dict[str, Any], vocab: TensorVocab, ligand_pos: np.ndarray, ligand_partial_charge: np.ndarray) -> dict[str, np.ndarray]:
    fgs = sample["ligand"]["functional_groups"] or []
    n_atoms = ligand_pos.shape[0]
    n_groups = len(fgs)
    if n_groups == 0:
        return {
            "fg_type": np.zeros((0,), dtype=np.int64),
            "fg_parent_type": np.zeros((0,), dtype=np.int64),
            "fg_membership": np.zeros((n_atoms, 0), dtype=np.float32),
            "fg_atom_index": np.zeros((0, 0), dtype=np.int64),
            "fg_atom_mask": np.zeros((0, 0), dtype=bool),
            "fg_atom_pos": _zeros((0, 0, 3), dtype=np.float32),
            "fg_centroid_pos": _zeros((0, 3), dtype=np.float32),
            "fg_mask": np.zeros((n_atoms,), dtype=np.float32),
            "fg_features": _zeros((0, 4), dtype=np.float32),
        }

    max_atoms = max(len(fg.get("atom_indices", [])) for fg in fgs)
    fg_type = np.zeros((n_groups,), dtype=np.int64)
    fg_parent_type = np.zeros((n_groups,), dtype=np.int64)
    fg_membership = np.zeros((n_atoms, n_groups), dtype=np.float32)
    fg_atom_index = np.full((n_groups, max_atoms), -1, dtype=np.int64)
    fg_atom_mask = np.zeros((n_groups, max_atoms), dtype=bool)
    fg_atom_pos = _zeros((n_groups, max_atoms, 3), dtype=np.float32)
    fg_centroid_pos = _zeros((n_groups, 3), dtype=np.float32)
    fg_features = _zeros((n_groups, 4), dtype=np.float32)

    for g_idx, fg in enumerate(fgs):
        atom_indices = [int(i) for i in fg.get("atom_indices", []) if 0 <= int(i) < n_atoms]
        fg_type[g_idx] = vocab.id_for_fg_type(fg.get("name"))
        fg_parent_type[g_idx] = vocab.id_for_fg_parent_type(fg.get("parent"))
        if atom_indices:
            fg_membership[atom_indices, g_idx] = 1.0
            fg_atom_index[g_idx, : len(atom_indices)] = atom_indices
            fg_atom_mask[g_idx, : len(atom_indices)] = True
            fg_atom_pos[g_idx, : len(atom_indices)] = ligand_pos[atom_indices]
            fg_centroid_pos[g_idx] = ligand_pos[atom_indices].mean(axis=0)
            fg_features[g_idx, 0] = float(len(atom_indices))
            fg_features[g_idx, 1] = float(ligand_partial_charge[atom_indices].sum())
            fg_features[g_idx, 2] = float(np.linalg.norm(ligand_pos[atom_indices] - fg_centroid_pos[g_idx], axis=1).max())
            fg_features[g_idx, 3] = float(np.mean(ligand_partial_charge[atom_indices]))

    fg_mask = np.asarray(sample["ligand"]["functional_group_mask"] or [0] * n_atoms, dtype=np.float32)
    return {
        "fg_type": fg_type,
        "fg_parent_type": fg_parent_type,
        "fg_membership": fg_membership,
        "fg_atom_index": fg_atom_index,
        "fg_atom_mask": fg_atom_mask,
        "fg_atom_pos": fg_atom_pos,
        "fg_centroid_pos": fg_centroid_pos,
        "fg_mask": fg_mask,
        "fg_features": fg_features,
    }


def _build_pocket_tensors(sample: dict[str, Any], vocab: TensorVocab, ligand_pos: np.ndarray, fg_centroid_pos: np.ndarray) -> dict[str, Any]:
    selected = sample["pocket"]["selected_pocket"]
    if not selected:
        return {
            "pocket_available": np.asarray(0, dtype=np.int8),
            "pocket_x": _zeros((0, 10), dtype=np.float32),
            "pocket_pos": _zeros((0, 3), dtype=np.float32),
            "pocket_edge_index": np.zeros((2, 0), dtype=np.int64),
            "pocket_atom_to_residue": np.zeros((0,), dtype=np.int64),
            "residue_x": _zeros((0, 5), dtype=np.float32),
            "residue_pos": _zeros((0, 3), dtype=np.float32),
            "residue_mask": np.zeros((0,), dtype=bool),
            "residue_keys": [],
            "residue_mmgbsa_alignment_mask": np.zeros((0,), dtype=bool),
        }

    pocket_atoms = selected.get("pocket_atoms") or []
    pocket_residues = selected.get("pocket_residues") or []
    n_atoms = len(pocket_atoms)
    n_res = len(pocket_residues)

    pocket_pos = np.asarray(
        [[atom["x"], atom["y"], atom["z"]] for atom in pocket_atoms],
        dtype=np.float32,
    ) if pocket_atoms else _zeros((0, 3), dtype=np.float32)
    ligand_centroid = ligand_pos.mean(axis=0) if ligand_pos.size else np.zeros((3,), dtype=np.float32)
    nearest_ligand_dist = _pairwise_min_distance(pocket_pos, ligand_pos) if pocket_atoms and ligand_pos.size else np.full((n_atoms,), np.nan, dtype=np.float32)
    nearest_fg_dist = _pairwise_min_distance(pocket_pos, fg_centroid_pos) if pocket_atoms and fg_centroid_pos.size else np.full((n_atoms,), np.nan, dtype=np.float32)

    residue_keys: list[tuple[str, int, str, str]] = []
    residue_index_lookup: dict[tuple[str, int, str, str], int] = {}
    residue_positions = _zeros((n_res, 3), dtype=np.float32)
    residue_features = _zeros((n_res, 5), dtype=np.float32)
    for idx, residue_entry in enumerate(pocket_residues):
        residue = residue_entry.get("residue", {})
        key = (
            str(residue.get("chain_id") or "").strip(),
            int(residue.get("resseq")),
            str(residue.get("insertion_code") or "").strip(),
            str(residue.get("resname") or "").strip().upper(),
        )
        residue_keys.append(key)
        residue_index_lookup[key] = idx
        atoms = residue_entry.get("atoms", [])
        if atoms:
            coords = np.asarray([[atom["x"], atom["y"], atom["z"]] for atom in atoms], dtype=np.float32)
            residue_positions[idx] = coords.mean(axis=0)
        residue_features[idx, 0] = float(vocab.id_for_residue_type(key[3]))
        residue_features[idx, 1] = float(vocab.id_for_residue_base_type(_normalize_resname(key[3])))
        residue_features[idx, 2] = float(vocab.id_for_residue_variant(_residue_variant_token(key[3])))
        residue_features[idx, 3] = float(residue_entry.get("atom_count", len(atoms)))
        residue_features[idx, 4] = float(residue_entry.get("min_distance", np.nan))

    pocket_atom_to_residue = np.full((n_atoms,), -1, dtype=np.int64)
    pocket_x = _zeros((n_atoms, 10), dtype=np.float32)
    for atom_idx, atom in enumerate(pocket_atoms):
        residue = atom.get("residue", {})
        key = (
            str(residue.get("chain_id") or "").strip(),
            int(residue.get("resseq")),
            str(residue.get("insertion_code") or "").strip(),
            str(residue.get("resname") or "").strip().upper(),
        )
        res_idx = residue_index_lookup.get(key, -1)
        pocket_atom_to_residue[atom_idx] = res_idx
        atom_name = str(atom.get("atom_name") or "").strip().upper()
        resname = key[3]
        pocket_x[atom_idx, 0] = float(_atomic_number_from_element(atom.get("element")))
        pocket_x[atom_idx, 1] = float(vocab.id_for_residue_type(resname))
        pocket_x[atom_idx, 2] = float(vocab.id_for_residue_base_type(_normalize_resname(resname)))
        pocket_x[atom_idx, 3] = float(vocab.id_for_residue_variant(_residue_variant_token(resname)))
        pocket_x[atom_idx, 4] = float(vocab.id_for_pocket_atom_name(atom_name))
        pocket_x[atom_idx, 5] = float(atom_name in BACKBONE_ATOM_NAMES)
        pocket_x[atom_idx, 6] = float(atom_name not in BACKBONE_ATOM_NAMES)
        pocket_x[atom_idx, 7] = float(resname in {"HOH", "WAT"})
        pocket_x[atom_idx, 8] = nearest_ligand_dist[atom_idx]
        pocket_x[atom_idx, 9] = nearest_fg_dist[atom_idx]

    pocket_edge_index = _build_radius_graph(pocket_pos)
    return {
        "pocket_available": np.asarray(1, dtype=np.int8),
        "pocket_x": pocket_x,
        "pocket_pos": pocket_pos,
        "pocket_edge_index": pocket_edge_index,
        "pocket_atom_to_residue": pocket_atom_to_residue,
        "residue_x": residue_features,
        "residue_pos": residue_positions,
        "residue_mask": np.ones((n_res,), dtype=bool),
        "residue_keys": residue_keys,
        "residue_mmgbsa_alignment_mask": np.zeros((n_res,), dtype=bool),
    }


def _build_global_mmgbsa_tensor(mmgbsa_payload: Optional[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    n_entities = len(MMGBSA_ENTITY_ORDER)
    n_components = len(MMGBSA_COMPONENT_ORDER)
    values = _nan_full((n_entities, n_components))
    mask = np.zeros((n_entities, n_components), dtype=bool)
    if not mmgbsa_payload:
        return values, mask
    components = mmgbsa_payload.get("components") or {}
    for e_idx, entity in enumerate(MMGBSA_ENTITY_ORDER):
        entity_payload = (components.get("GB") or {}).get(entity, {})
        for c_idx, component in enumerate(MMGBSA_COMPONENT_ORDER):
            comp = entity_payload.get(component)
            if isinstance(comp, dict) and comp.get("average") is not None:
                values[e_idx, c_idx] = float(comp["average"])
                mask[e_idx, c_idx] = True
    return values, mask


def _align_residue_rows(
    residue_keys: list[tuple[str, int, str, str]],
    decomp_rows: list[dict[str, Any]],
    amber_renum_rows: Optional[list[dict[str, Any]]] = None,
    prepared_to_receptor_rows: Optional[list[dict[str, Any]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_res = len(residue_keys)
    values = _nan_full((n_res, len(MMGBSA_RESIDUE_CHANNELS)))
    present = np.zeros((n_res,), dtype=bool)
    exact_unique = np.zeros((n_res,), dtype=bool)
    if not residue_keys or not decomp_rows:
        return values, present, exact_unique

    exact_lookup: dict[tuple[str, int, str], list[int]] = {}
    norm_lookup: dict[tuple[str, int, str], list[int]] = {}
    pair_to_indices: dict[tuple[str, int], list[int]] = {}
    pair_norm_to_indices: dict[tuple[str, int], list[int]] = {}
    for idx, (chain_id, resseq, _, resname) in enumerate(residue_keys):
        chain = str(chain_id).strip()
        seq = int(resseq)
        exact_lookup.setdefault((chain, seq, resname.upper()), []).append(idx)
        norm_lookup.setdefault((chain, seq, _normalize_resname(resname)), []).append(idx)
        pair_to_indices.setdefault((resname.upper(), seq), []).append(idx)
        pair_norm_to_indices.setdefault((_normalize_resname(resname), seq), []).append(idx)

    amber_lookup = _build_amber_renum_lookup(amber_renum_rows or [])
    receptor_to_pocket_exact, receptor_to_pocket_norm = _build_receptor_to_pocket_lookup(
        residue_keys,
        prepared_to_receptor_rows or [],
    )

    for row in decomp_rows:
        resname, resseq, entity = _parse_mmgbsa_row_key(row)
        if entity != "R" or resseq is None:
            continue
        matched: list[int] = []
        amber_map = amber_lookup.get(int(resseq))
        if amber_map is not None:
            mapped_chain = str(amber_map.get("original_chain") or "").strip()
            mapped_seq = int(amber_map["original_resseq"])
            mapped_resname = str(amber_map.get("original_resname") or "").strip().upper()
            matched = receptor_to_pocket_exact.get((mapped_chain, mapped_seq, mapped_resname), [])
            if len(matched) != 1:
                matched = receptor_to_pocket_norm.get(
                    (mapped_chain, mapped_seq, _normalize_resname(mapped_resname)),
                    [],
                )
            if len(matched) != 1:
                matched = exact_lookup.get((mapped_chain, mapped_seq, mapped_resname), [])
            if len(matched) != 1:
                matched = norm_lookup.get(
                    (mapped_chain, mapped_seq, _normalize_resname(mapped_resname)),
                    [],
                )
        if len(matched) != 1:
            matched = pair_to_indices.get((resname.upper(), int(resseq)), [])
        if len(matched) != 1:
            matched = pair_norm_to_indices.get((_normalize_resname(resname), int(resseq)), [])
        if len(matched) != 1:
            continue
        idx = matched[0]
        values[idx] = np.asarray(
            [row[ch] for ch in MMGBSA_RESIDUE_CHANNELS],
            dtype=np.float32,
        )
        present[idx] = True
        exact_unique[idx] = True
    return values, present, exact_unique


def _extract_ligand_decomp_vector(decomp_rows: list[dict[str, Any]]) -> np.ndarray:
    for row in decomp_rows:
        _, _, entity = _parse_mmgbsa_row_key(row)
        if entity == "L":
            return np.asarray([row[ch] for ch in MMGBSA_RESIDUE_CHANNELS], dtype=np.float32)
    return _nan_full((len(MMGBSA_RESIDUE_CHANNELS),))


def _build_mmgbsa_tensors(
    sample: dict[str, Any],
    residue_keys: list[tuple[str, int, str, str]],
) -> dict[str, Any]:
    baseline = sample["baseline_mmgbsa"]
    amber_renum_rows = baseline.get("receptor_amber_renum", []) or []
    prepared_to_receptor_rows = baseline.get("prepared_to_receptor_residue_map", []) or []
    baseline_global, baseline_global_mask = _build_global_mmgbsa_tensor(baseline if baseline.get("completed") else None)
    baseline_residue_energy, baseline_residue_present, baseline_residue_alignment = _align_residue_rows(
        residue_keys,
        baseline.get("residue_decomposition", []) if baseline.get("completed") else [],
        amber_renum_rows,
        prepared_to_receptor_rows,
    )
    baseline_ligand_decomp = _extract_ligand_decomp_vector(
        baseline.get("residue_decomposition", []) if baseline.get("completed") else []
    )

    groups = sample["masked_mmgbsa"]["groups"]
    fg_list = sample["ligand"]["functional_groups"] or []
    n_groups = len(fg_list)
    n_modes = len(MASKED_MODE_ORDER)
    n_entities = len(MMGBSA_ENTITY_ORDER)
    n_components = len(MMGBSA_COMPONENT_ORDER)
    n_res = len(residue_keys)
    n_channels = len(MMGBSA_RESIDUE_CHANNELS)

    masked_global = _nan_full((n_groups, n_modes, n_entities, n_components))
    masked_global_mask = np.zeros((n_groups, n_modes, n_entities, n_components), dtype=bool)
    masked_global_delta = _nan_full((n_groups, n_modes, n_components))
    masked_global_delta_mask = np.zeros((n_groups, n_modes, n_components), dtype=bool)
    masked_residue_energy = _nan_full((n_groups, n_modes, n_res, n_channels))
    masked_residue_present = np.zeros((n_groups, n_modes, n_res), dtype=bool)
    masked_residue_alignment = np.zeros((n_groups, n_modes, n_res), dtype=bool)
    masked_residue_delta = _nan_full((n_groups, n_modes, n_res, n_channels))
    masked_ligand_decomp = _nan_full((n_groups, n_modes, n_channels))
    masked_mode_available = np.zeros((n_groups, n_modes), dtype=bool)

    baseline_diff = baseline_global[MMGBSA_ENTITY_ORDER.index("differences")]
    baseline_diff_mask = baseline_global_mask[MMGBSA_ENTITY_ORDER.index("differences")]

    group_by_slug = {name: payload for name, payload in groups.items()}
    for g_idx, fg in enumerate(fg_list):
        slug = f"{g_idx + 1:02d}_{str(fg.get('name') or 'group').strip().lower().replace(' ', '_').replace('/', '_')}"
        group_payload = group_by_slug.get(slug)
        if group_payload is None:
            continue
        for m_idx, mode in enumerate(MASKED_MODE_ORDER):
            mode_payload = group_payload.get("modes", {}).get(mode)
            if not mode_payload:
                continue
            masked_mode_available[g_idx, m_idx] = True
            global_values, global_mask = _build_global_mmgbsa_tensor(mode_payload)
            masked_global[g_idx, m_idx] = global_values
            masked_global_mask[g_idx, m_idx] = global_mask
            diff_values = global_values[MMGBSA_ENTITY_ORDER.index("differences")]
            diff_mask = global_mask[MMGBSA_ENTITY_ORDER.index("differences")]
            usable = diff_mask & baseline_diff_mask
            masked_global_delta[g_idx, m_idx, usable] = diff_values[usable] - baseline_diff[usable]
            masked_global_delta_mask[g_idx, m_idx, usable] = True

            residue_values, residue_present, residue_alignment = _align_residue_rows(
                residue_keys,
                mode_payload.get("residue_decomposition", []),
                amber_renum_rows,
                prepared_to_receptor_rows,
            )
            masked_residue_energy[g_idx, m_idx] = residue_values
            masked_residue_present[g_idx, m_idx] = residue_present
            masked_residue_alignment[g_idx, m_idx] = residue_alignment
            both = residue_present & baseline_residue_present
            masked_residue_delta[g_idx, m_idx, both] = residue_values[both] - baseline_residue_energy[both]
            masked_ligand_decomp[g_idx, m_idx] = _extract_ligand_decomp_vector(
                mode_payload.get("residue_decomposition", [])
            )

    return {
        "baseline_global_components": baseline_global,
        "baseline_global_component_mask": baseline_global_mask,
        "baseline_residue_energy": baseline_residue_energy,
        "baseline_residue_present": baseline_residue_present,
        "baseline_residue_alignment_mask": baseline_residue_alignment,
        "baseline_ligand_decomp": baseline_ligand_decomp,
        "masked_global_components": masked_global,
        "masked_global_component_mask": masked_global_mask,
        "masked_global_delta": masked_global_delta,
        "masked_global_delta_mask": masked_global_delta_mask,
        "masked_residue_energy": masked_residue_energy,
        "masked_residue_present": masked_residue_present,
        "masked_residue_alignment_mask": masked_residue_alignment,
        "masked_residue_delta": masked_residue_delta,
        "masked_ligand_decomp": masked_ligand_decomp,
        "masked_mode_available": masked_mode_available,
    }


def tensorize_sample(sample: dict[str, Any], vocab: Optional[TensorVocab] = None) -> dict[str, Any]:
    vocab = vocab or DEFAULT_TENSOR_VOCAB
    ligand = _build_ligand_tensors(sample)
    ligand_partial_charge = ligand["ligand_x"][:, 2] if ligand["ligand_x"].size else np.zeros((0,), dtype=np.float32)
    fg = _build_fg_tensors(sample, vocab, ligand["ligand_pos"], ligand_partial_charge)
    pocket = _build_pocket_tensors(sample, vocab, ligand["ligand_pos"], fg["fg_centroid_pos"])
    mmgbsa = _build_mmgbsa_tensors(sample, pocket["residue_keys"])

    pocket["residue_mmgbsa_alignment_mask"] = mmgbsa["baseline_residue_alignment_mask"]

    return {
        "metadata": {
            "pdb_id": sample["pdb_id"],
            "ligand_id": sample["ligand"]["ligand_id"],
            "smiles": sample["ligand"]["smiles"],
            "pocket_available": bool(int(pocket["pocket_available"])),
            "fg_names": [fg_payload.get("name") for fg_payload in (sample["ligand"]["functional_groups"] or [])],
            "residue_keys": pocket["residue_keys"],
            "mmgbsa_component_order": MMGBSA_COMPONENT_ORDER,
            "mmgbsa_entity_order": MMGBSA_ENTITY_ORDER,
            "mmgbsa_residue_channels": MMGBSA_RESIDUE_CHANNELS,
            "masked_mode_order": MASKED_MODE_ORDER,
            "ligand_atom_feature_names": LIGAND_ATOM_FEATURE_NAMES,
            "ligand_edge_feature_names": LIGAND_EDGE_FEATURE_NAMES,
            "fg_feature_names": FG_FEATURE_NAMES,
            "pocket_atom_feature_names": POCKET_ATOM_FEATURE_NAMES,
            "residue_feature_names": RESIDUE_FEATURE_NAMES,
            "vocab_sizes": {
                "fg_type": len(vocab.fg_type),
                "fg_parent_type": len(vocab.fg_parent_type),
                "residue_type": len(vocab.residue_type),
                "residue_base_type": len(vocab.residue_base_type),
                "residue_variant": len(vocab.residue_variant),
                "pocket_atom_name": len(vocab.pocket_atom_name),
            },
            "feature_schema": build_feature_schema(vocab),
        },
        "ligand": ligand,
        "functional_groups": fg,
        "pocket": pocket,
        "mmgbsa": mmgbsa,
    }
