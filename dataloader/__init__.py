# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
_TOOLKIT_DIR = _REPO_ROOT / "small-molecule-mod-toolkit"
for _directory in (_REPO_ROOT, _TOOLKIT_DIR):
    if str(_directory) not in _sys.path:
        _sys.path.insert(0, str(_directory))

from .collate import collate_tensorized_samples, export_tensorized_dataset, export_tensorized_sample
from .feature_embed import EmbeddingConfig, MMGBSAFeatureEmbedder
from .mmgbsa_dataset import DatasetRoots, MMGBSADataset, build_ligand_molgraph, build_sample
from .tensor_schema import DEFAULT_TENSOR_VOCAB, TensorVocab, build_feature_schema
from .tensorize import tensorize_sample
from .torch_adapter import (
    CachedMMGBSATorchDataset,
    MMGBSATorchDataset,
    export_torch_cache,
    make_dataloader,
    numpy_sample_from_torch,
    to_pyg_batch,
    to_pyg_hetero,
    to_torch_sample,
    torch_collate_samples,
)

__all__ = [
    "DatasetRoots",
    "MMGBSADataset",
    "build_sample",
    "build_ligand_molgraph",
    "TensorVocab",
    "DEFAULT_TENSOR_VOCAB",
    "build_feature_schema",
    "tensorize_sample",
    "collate_tensorized_samples",
    "export_tensorized_sample",
    "export_tensorized_dataset",
    "to_torch_sample",
    "numpy_sample_from_torch",
    "to_pyg_hetero",
    "to_pyg_batch",
    "torch_collate_samples",
    "MMGBSATorchDataset",
    "CachedMMGBSATorchDataset",
    "make_dataloader",
    "export_torch_cache",
    "EmbeddingConfig",
    "MMGBSAFeatureEmbedder",
]
