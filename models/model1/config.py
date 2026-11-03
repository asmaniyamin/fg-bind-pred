# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelConfig:
    hidden_dim: int = 128
    ligand_edge_dim: int = 64
    fg_edge_dim: int = 64
    num_ligand_gnn_layers: int = 5
    num_fg_attn_layers: int = 2
    num_fg_global_attn_layers: int = 1
    num_cross_attn_layers: int = 2
    num_heads: int = 3
    num_fg_graph_heads: int = 4
    num_cross_attn_heads: int = 4
    dropout: float = 0.1

    aev_rc: float = 5.1
    aev_eta: float = 19.7
    aev_num_rs: int = 16
    aev_num_ecif_types: int = 22
    aev_dim: int = 352

    fg_edge_cutoff: float = 5.0
    cross_attn_cutoff: float = 6.0
    num_rbf_kernels: int = 16
    rbf_d_min: float = 1.0
    rbf_d_max: float = 8.0
    use_aev: bool = True
    use_cross_attention: bool = True


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 200
    warmup_steps: int = 500
    grad_clip: float = 1.0
    early_stopping_patience: int = 20
    num_workers: int = 0
    pin_memory: bool = False
    gradient_accumulation_steps: int = 1
    scheduler_type: str = "cosine_warmup"
    scheduler_step_on: str = "batch"
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1e-4
    min_lr: float = 1e-6
    early_stopping_metric: str = "contact_total.pearson_r"
    early_stopping_mode: str = "max"
    freeze_modules: tuple[str, ...] = field(default_factory=tuple)
    freeze_epochs: int = 0
    print_batch_every: int = 0
    max_abs_delta_total: float = 1000.0


@dataclass(slots=True)
class DataConfig:
    data_root: str
    aev_rc: float = 5.1
    aev_eta: float = 19.7
    fg_edge_cutoff: float = 5.0
    cross_attn_cutoff: float = 6.0
    split_type: str = "pocket_hash"
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42
    split_seed: int | None = None
    cache_dir: str | None = None
    case_roots: tuple[str, ...] = field(default_factory=tuple)
    require_baseline: bool = True
    min_fg_count: int = 0
    exclude_zero_fg: bool = False
    max_cases: int = 0  # 0 == unlimited; cap discovered PDB IDs before splitting
