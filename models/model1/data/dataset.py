# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from dataloader.mmgbsa_dataset import DatasetRoots, build_sample
from dataloader.tensorize import tensorize_sample
from dataloader.torch_adapter import (
    CachedMMGBSATorchDataset,
    MMGBSATorchDataset,
    export_torch_cache,
    make_dataloader,
    to_torch_sample,
)

from ..config import DataConfig
from .splits import ligand_split, pocket_hash_split


class PerFGDataset(Dataset):
    def __init__(
        self,
        root: str,
        config: DataConfig,
        split: str,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.split = split
        self.transform = transform

        self.cache_dir = Path(config.cache_dir or root)
        self._use_cache = self._looks_like_cache(self.cache_dir)
        self.roots = self._make_roots(config)
        all_ids = self._discover_ids()
        all_ids = self._filter_ids(all_ids)
        max_cases = int(getattr(config, "max_cases", 0) or 0)
        if max_cases > 0:
            all_ids = all_ids[:max_cases]
        train_ids, val_ids, test_ids = self._split_ids(all_ids)
        split_to_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
        self.pdb_ids = split_to_ids[split]
        self._dataset = self._build_backend()

    def _make_roots(self, config: DataConfig) -> DatasetRoots:
        case_roots = [Path(p) for p in config.case_roots]
        return DatasetRoots(case_roots=case_roots if case_roots else None)

    @staticmethod
    def _looks_like_cache(path: Path) -> bool:
        if not path.exists():
            return False
        return any(path.glob("*.npz")) or any(path.glob("*/*.npz"))

    def _discover_ids(self) -> list[str]:
        if self._use_cache:
            ids: set[str] = set()
            for entry in self.cache_dir.iterdir():
                if entry.is_dir() and (entry / f"{entry.name}.npz").is_file():
                    ids.add(entry.name)
                elif entry.is_file() and entry.suffix == ".npz":
                    ids.add(entry.stem)
            return sorted(ids)
        return self.roots.discover_pdb_ids(require_baseline=self.config.require_baseline)

    def _load_cached_metadata(self, pdb_id: str) -> dict[str, Any]:
        nested = self.cache_dir / pdb_id / f"{pdb_id}.metadata.json"
        flat = self.cache_dir / f"{pdb_id}.metadata.json"
        metadata_path = nested if nested.is_file() else flat
        return json.loads(metadata_path.read_text())

    def _split_ids(self, pdb_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        split_seed = int(getattr(self.config, "split_seed", getattr(self.config, "seed", 42)) or 42)
        if self.config.split_type == "ligand":
            smiles = {}
            if self._use_cache:
                for pdb_id in pdb_ids:
                    smiles[pdb_id] = self._load_cached_metadata(pdb_id).get("smiles") or pdb_id
            return ligand_split(pdb_ids, smiles, self.config.val_frac, self.config.test_frac, split_seed)

        pocket_sequences = {}
        if self._use_cache:
            for pdb_id in pdb_ids:
                metadata = self._load_cached_metadata(pdb_id)
                residue_keys = metadata.get("residue_keys", [])
                pocket_sequences[pdb_id] = "-".join(str(key[3]) for key in residue_keys)
        return pocket_hash_split(pdb_ids, pocket_sequences, self.config.val_frac, self.config.test_frac, split_seed)

    def _build_backend_for_ids(self, pdb_ids: list[str]):
        if self._use_cache:
            return CachedMMGBSATorchDataset(self.cache_dir, pdb_ids=pdb_ids, transform=self.transform)
        return MMGBSATorchDataset(
            pdb_ids=pdb_ids,
            roots=self.roots,
            require_baseline=self.config.require_baseline,
            transform=self.transform,
        )

    def _build_backend(self):
        return self._build_backend_for_ids(self.pdb_ids)

    def _filter_ids(self, pdb_ids: list[str]) -> list[str]:
        min_fg_count = max(int(getattr(self.config, "min_fg_count", 0)), 0)
        exclude_zero_fg = bool(getattr(self.config, "exclude_zero_fg", False))
        if min_fg_count <= 0 and not exclude_zero_fg:
            return pdb_ids
        backend = self._build_backend_for_ids(pdb_ids)
        filtered_ids: list[str] = []
        for idx, pdb_id in enumerate(pdb_ids):
            sample = backend[idx]
            fg_count = int(sample["functional_groups"]["fg_type"].shape[0])
            if exclude_zero_fg and fg_count == 0:
                continue
            if fg_count < min_fg_count:
                continue
            filtered_ids.append(pdb_id)
        return filtered_ids

    def len(self) -> int:
        return len(self._dataset)

    def __len__(self) -> int:
        return self.len()

    def get(self, idx: int):
        return self._dataset[idx]

    def __getitem__(self, idx: int):
        return self.get(idx)

    def process(self):
        if self._use_cache:
            return
        export_torch_cache(self.pdb_ids, self.cache_dir, roots=self.roots, overwrite=False)

    def _load_raw(self, pdb_id: str) -> dict[str, Any]:
        return build_sample(pdb_id, roots=self.roots)

    def _process_single(self, pdb_id: str):
        return to_torch_sample(tensorize_sample(self._load_raw(pdb_id)))

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{pdb_id}.raw" for pdb_id in self.pdb_ids]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{pdb_id}.npz" for pdb_id in self.pdb_ids]
