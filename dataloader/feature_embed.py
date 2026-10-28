# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Training-side feature embedder.

A single ``nn.Module`` that takes either the dict batch from
:func:`torch_collate_samples` or the PyG ``Batch`` from :func:`to_pyg_batch`,
looks up the categorical ID columns in embedding tables, and concatenates the
result with the existing numeric features along the feature dim.

Vocab sizes are pulled from the ``feature_schema`` shipped in
``metadata['feature_schema']`` (so the embedder stays consistent with whatever
vocab a particular dataset shard was tensorized against). Periodic-table /
RDKit-driven IDs (``atomic_num``, ``hybridization``, ``chiral_tag``) are not
schema-tracked and fall back to safe defaults; override via ``vocab_overrides``
when needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from .tensor_schema import (
    FG_FEATURE_NAMES,
    LIGAND_ATOM_FEATURE_NAMES,
    POCKET_ATOM_FEATURE_NAMES,
    RESIDUE_FEATURE_NAMES,
    build_feature_schema,
)


# Safe default caps for vocab sizes not tracked in the schema.
_DEFAULT_ATOMIC_NUM_VOCAB = 120        # periodic table + headroom
_DEFAULT_HYBRIDIZATION_VOCAB = 9       # RDKit HybridizationType (8) + padding
_DEFAULT_CHIRAL_TAG_VOCAB = 9          # RDKit ChiralType (4) + padding


# Column index lookups (kept module-level so the forward path stays tight).
_LIGAND_ATOMIC_NUM_COL = LIGAND_ATOM_FEATURE_NAMES.index("atomic_num")
_LIGAND_HYBRID_COL = LIGAND_ATOM_FEATURE_NAMES.index("hybridization")
_LIGAND_CHIRAL_COL = LIGAND_ATOM_FEATURE_NAMES.index("chiral_tag")
_LIGAND_CATEGORICAL_COLS = {_LIGAND_ATOMIC_NUM_COL, _LIGAND_HYBRID_COL, _LIGAND_CHIRAL_COL}
_LIGAND_NUMERIC_COLS = [
    i for i in range(len(LIGAND_ATOM_FEATURE_NAMES)) if i not in _LIGAND_CATEGORICAL_COLS
]

_POCKET_ATOMIC_NUM_COL = POCKET_ATOM_FEATURE_NAMES.index("atomic_num")
_POCKET_RTYPE_COL = POCKET_ATOM_FEATURE_NAMES.index("residue_type_id")
_POCKET_RBASE_COL = POCKET_ATOM_FEATURE_NAMES.index("residue_base_type_id")
_POCKET_RVAR_COL = POCKET_ATOM_FEATURE_NAMES.index("residue_variant_id")
_POCKET_ATOM_NAME_COL = POCKET_ATOM_FEATURE_NAMES.index("atom_name_id")
_POCKET_CATEGORICAL_COLS = {
    _POCKET_ATOMIC_NUM_COL,
    _POCKET_RTYPE_COL,
    _POCKET_RBASE_COL,
    _POCKET_RVAR_COL,
    _POCKET_ATOM_NAME_COL,
}
_POCKET_NUMERIC_COLS = [
    i for i in range(len(POCKET_ATOM_FEATURE_NAMES)) if i not in _POCKET_CATEGORICAL_COLS
]

_RES_RTYPE_COL = RESIDUE_FEATURE_NAMES.index("residue_type_id")
_RES_RBASE_COL = RESIDUE_FEATURE_NAMES.index("residue_base_type_id")
_RES_RVAR_COL = RESIDUE_FEATURE_NAMES.index("residue_variant_id")
_RES_CATEGORICAL_COLS = {_RES_RTYPE_COL, _RES_RBASE_COL, _RES_RVAR_COL}
_RES_NUMERIC_COLS = [
    i for i in range(len(RESIDUE_FEATURE_NAMES)) if i not in _RES_CATEGORICAL_COLS
]


@dataclass
class EmbeddingConfig:
    atomic_num_dim: int = 16
    hybridization_dim: int = 4
    chiral_tag_dim: int = 4
    residue_type_dim: int = 16
    residue_base_type_dim: int = 12
    residue_variant_dim: int = 4
    pocket_atom_name_dim: int = 8
    fg_type_dim: int = 16
    fg_parent_type_dim: int = 8
    fill_nan: float = 0.0


def _vocab_sizes_from_schema(schema: dict[str, Any]) -> dict[str, int]:
    vocabs = schema.get("categorical_vocabularies", {}) or {}
    # Embedding sizes must be >=1; an empty list would crash nn.Embedding.
    return {name: max(1, len(tokens)) for name, tokens in vocabs.items()}


def _safe_embed(emb: nn.Embedding, indices: torch.Tensor) -> torch.Tensor:
    # Tensorized samples store IDs as float in the *_x tables; cast + clamp
    # so out-of-vocab ids (e.g., an unseen pocket atom name from a future
    # shard) get folded onto the last index instead of crashing.
    upper = emb.num_embeddings - 1
    return emb(indices.long().clamp(min=0, max=upper))


class MMGBSAFeatureEmbedder(nn.Module):
    """Project the categorical ID columns of every stream through embeddings.

    Outputs a dict with keys ``ligand_x``, ``pocket_atom_x``, ``residue_x``,
    ``fg_x``. Each is ``[numeric_features | embedding_features]`` along the
    last dim. ``out_dims`` exposes the resulting feature size per stream for
    sizing downstream layers.

    The module is batch-format agnostic - :meth:`forward` auto-dispatches
    between the dict batch and a PyG ``HeteroData`` / ``Batch``. The PyG path
    can optionally write each result back onto ``store.x_emb`` so downstream
    PyG layers can read ``data['ligand'].x_emb`` etc.
    """

    def __init__(
        self,
        feature_schema: Optional[dict[str, Any]] = None,
        config: Optional[EmbeddingConfig] = None,
        vocab_overrides: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__()
        schema = feature_schema or build_feature_schema()
        schema_vocab = _vocab_sizes_from_schema(schema)
        overrides = vocab_overrides or {}
        cfg = config or EmbeddingConfig()
        self.cfg = cfg

        def _vocab(name: str, default: int) -> int:
            return int(overrides.get(name, schema_vocab.get(name, default)))

        # Periodic-table / RDKit categoricals: not in schema, default-capped.
        self.atomic_num_emb = nn.Embedding(_vocab("atomic_num", _DEFAULT_ATOMIC_NUM_VOCAB), cfg.atomic_num_dim)
        self.hybridization_emb = nn.Embedding(_vocab("hybridization", _DEFAULT_HYBRIDIZATION_VOCAB), cfg.hybridization_dim)
        self.chiral_tag_emb = nn.Embedding(_vocab("chiral_tag", _DEFAULT_CHIRAL_TAG_VOCAB), cfg.chiral_tag_dim)

        # Schema-derived categoricals.
        self.residue_type_emb = nn.Embedding(_vocab("residue_type", 40), cfg.residue_type_dim)
        self.residue_base_type_emb = nn.Embedding(_vocab("residue_base_type", 40), cfg.residue_base_type_dim)
        self.residue_variant_emb = nn.Embedding(_vocab("residue_variant", 16), cfg.residue_variant_dim)
        self.pocket_atom_name_emb = nn.Embedding(_vocab("pocket_atom_name", 64), cfg.pocket_atom_name_dim)
        self.fg_type_emb = nn.Embedding(_vocab("fg_type", 64), cfg.fg_type_dim)
        self.fg_parent_type_emb = nn.Embedding(_vocab("fg_parent_type", 32), cfg.fg_parent_type_dim)

        self.out_dims = {
            "ligand": (
                len(_LIGAND_NUMERIC_COLS)
                + cfg.atomic_num_dim + cfg.hybridization_dim + cfg.chiral_tag_dim
            ),
            "pocket_atom": (
                len(_POCKET_NUMERIC_COLS)
                + cfg.atomic_num_dim + cfg.residue_type_dim
                + cfg.residue_base_type_dim + cfg.residue_variant_dim
                + cfg.pocket_atom_name_dim
            ),
            "residue": (
                len(_RES_NUMERIC_COLS)
                + cfg.residue_type_dim + cfg.residue_base_type_dim + cfg.residue_variant_dim
            ),
            "fg": (
                len(FG_FEATURE_NAMES)
                + cfg.fg_type_dim + cfg.fg_parent_type_dim
            ),
        }

    def _clean_numeric(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            return x.float()
        return torch.nan_to_num(
            x,
            nan=self.cfg.fill_nan,
            posinf=self.cfg.fill_nan,
            neginf=self.cfg.fill_nan,
        )

    def _embed_ligand(self, ligand_x: torch.Tensor) -> torch.Tensor:
        if ligand_x.shape[0] == 0:
            return torch.zeros((0, self.out_dims["ligand"]), dtype=torch.float32, device=ligand_x.device)
        numeric = self._clean_numeric(ligand_x[:, _LIGAND_NUMERIC_COLS])
        return torch.cat(
            [
                numeric,
                _safe_embed(self.atomic_num_emb, ligand_x[:, _LIGAND_ATOMIC_NUM_COL]),
                _safe_embed(self.hybridization_emb, ligand_x[:, _LIGAND_HYBRID_COL]),
                _safe_embed(self.chiral_tag_emb, ligand_x[:, _LIGAND_CHIRAL_COL]),
            ],
            dim=-1,
        )

    def _embed_pocket_atom(self, pocket_x: torch.Tensor) -> torch.Tensor:
        if pocket_x.shape[0] == 0:
            return torch.zeros((0, self.out_dims["pocket_atom"]), dtype=torch.float32, device=pocket_x.device)
        numeric = self._clean_numeric(pocket_x[:, _POCKET_NUMERIC_COLS])
        return torch.cat(
            [
                numeric,
                _safe_embed(self.atomic_num_emb, pocket_x[:, _POCKET_ATOMIC_NUM_COL]),
                _safe_embed(self.residue_type_emb, pocket_x[:, _POCKET_RTYPE_COL]),
                _safe_embed(self.residue_base_type_emb, pocket_x[:, _POCKET_RBASE_COL]),
                _safe_embed(self.residue_variant_emb, pocket_x[:, _POCKET_RVAR_COL]),
                _safe_embed(self.pocket_atom_name_emb, pocket_x[:, _POCKET_ATOM_NAME_COL]),
            ],
            dim=-1,
        )

    def _embed_residue(self, residue_x: torch.Tensor) -> torch.Tensor:
        if residue_x.shape[0] == 0:
            return torch.zeros((0, self.out_dims["residue"]), dtype=torch.float32, device=residue_x.device)
        numeric = self._clean_numeric(residue_x[:, _RES_NUMERIC_COLS])
        return torch.cat(
            [
                numeric,
                _safe_embed(self.residue_type_emb, residue_x[:, _RES_RTYPE_COL]),
                _safe_embed(self.residue_base_type_emb, residue_x[:, _RES_RBASE_COL]),
                _safe_embed(self.residue_variant_emb, residue_x[:, _RES_RVAR_COL]),
            ],
            dim=-1,
        )

    def _embed_fg(
        self,
        fg_features: torch.Tensor,
        fg_type: torch.Tensor,
        fg_parent_type: torch.Tensor,
    ) -> torch.Tensor:
        if fg_features.shape[0] == 0:
            return torch.zeros((0, self.out_dims["fg"]), dtype=torch.float32, device=fg_features.device)
        return torch.cat(
            [
                self._clean_numeric(fg_features),
                _safe_embed(self.fg_type_emb, fg_type),
                _safe_embed(self.fg_parent_type_emb, fg_parent_type),
            ],
            dim=-1,
        )

    # --- public api ---

    def embed_dict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Embed a dict batch from :func:`torch_collate_samples`."""
        fg = batch["functional_groups"]
        return {
            "ligand_x": self._embed_ligand(batch["ligand"]["x"]),
            "pocket_atom_x": self._embed_pocket_atom(batch["pocket"]["x"]),
            "residue_x": self._embed_residue(batch["pocket"]["residue_x"]),
            "fg_x": self._embed_fg(fg["fg_features"], fg["fg_type"], fg["fg_parent_type"]),
        }

    def embed_pyg(self, batch, *, inplace: bool = True):
        """Embed a PyG HeteroData/Batch.

        When ``inplace`` is True (default), writes ``x_emb`` onto each node
        store and returns the same object. Otherwise returns a dict matching
        :meth:`embed_dict`'s keys.
        """
        result = {
            "ligand_x": self._embed_ligand(batch["ligand"].x),
            "pocket_atom_x": self._embed_pocket_atom(batch["pocket_atom"].x),
            "residue_x": self._embed_residue(batch["residue"].x),
            "fg_x": self._embed_fg(batch["fg"].x, batch["fg"].fg_type, batch["fg"].fg_parent_type),
        }
        if inplace:
            batch["ligand"].x_emb = result["ligand_x"]
            batch["pocket_atom"].x_emb = result["pocket_atom_x"]
            batch["residue"].x_emb = result["residue_x"]
            batch["fg"].x_emb = result["fg_x"]
            return batch
        return result

    def forward(self, batch):
        # Heuristic: dict batches are plain dicts whose 'ligand' value is also
        # a dict; the PyG Batch / HeteroData expose attribute-style access.
        if isinstance(batch, dict) and isinstance(batch.get("ligand"), dict):
            return self.embed_dict(batch)
        return self.embed_pyg(batch, inplace=False)
