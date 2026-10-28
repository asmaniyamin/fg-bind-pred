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

    # number of counterfactual masked modes predicted by the importance head.
    # Order matches data.label_builder.IMPORTANCE_MODE_ORDER.
    num_importance_modes: int = 3
    # number of FGs labeled positive per complex for the hotspot classification task.
    hotspot_top_k: int = 3
    # minimum |ghost ΔΔ_total| (kcal/mol) for an FG to count as a hotspot positive,
    # in addition to being inside the per-complex top-K. A value of 0 disables the threshold.
    hotspot_min_abs: float = 0.5

    # ---- Optional input features and prediction heads ----
    # Concatenate a 2-channel Coulombic AEV (signed + abs-density) to the ECIF AEV.
    use_elec_aev: bool = True
    # Concatenate a 5-channel per-FG local-pocket-environment summary to the head inputs.
    use_fg_env_summary: bool = True
    # Predict importance with a model1-style FG-residue pair scorer instead of a per-FG MLP.
    # Strong physical inductive bias (cosine cutoff, distance RBF) but slower.
    use_pair_importance_head: bool = False
    # Add a 4-channel soft contact decomposition head (model1 targets) as a representation
    # regularizer. Loss weight set via TrainingConfig.aux_contact_weight.
    use_contact_aux_head: bool = False
    # Concatenate 5-channel per-FG electrostatic features (net q, dipole, q variance,
    # min/max q) to the FG node encoder input. Helps the charge head.
    use_fg_elec_features: bool = True
    # Concatenate 2-channel per-FG Coulombic interaction scalar (FG net q · pocket q / r)
    # to the importance/hotspot head inputs alongside the env summary. Helps the charge head.
    use_fg_residue_coulomb: bool = True
    # Add a 4th regression output: charge / (charge + lj + eps) -- a bounded ratio
    # describing what fraction of an FG's binding effect is electrostatic. Trained
    # against the same ratio computed from the targets.
    use_ratio_head: bool = False
    # Apply HID/HIE/HIP/ASH/GLH/LYN/CYM protonation-state corrections inside
    # `_pocket_charge_prior` (and the residue effective-charge lookup used by
    # the FG-residue Coulomb feature). Default ON; flip OFF to ablate the
    # variant-awareness while keeping the base atomic-number + residue prior.
    use_variant_aware_charge_prior: bool = True


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
    # default monitor: per-complex Spearman ranking of FG importance.
    # this is the headline metric: "which FG is most important in this ligand".
    early_stopping_metric: str = "importance_ghost.per_complex_spearman_r"
    early_stopping_mode: str = "max"
    freeze_modules: tuple[str, ...] = field(default_factory=tuple)
    freeze_epochs: int = 0
    print_batch_every: int = 0
    # clip absurd per-FG ΔΔ targets (rare instability in MM-GBSA decomp). 0 disables.
    max_abs_importance: float = 200.0
    # Soft penalty on Σ pred_ghost_per_FG_in_complex vs the complex DELTA TOTAL. 0 disables.
    sum_constraint_weight: float = 0.05
    # Auxiliary contact-decomposition (model1-style) loss weight. 0 disables.
    aux_contact_weight: float = 0.0


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
    max_cases: int = 0
