# Per-FG binding contribution model — interface reference

## Project layout
```
project/
├── config.py
├── train.py
├── utils/rbf.py
├── data/
│   ├── dataset.py
│   ├── graph_builder.py
│   ├── label_builder.py
│   └── splits.py
├── model/
│   ├── encoders.py
│   ├── gnn.py
│   ├── pooling.py
│   ├── attention.py
│   ├── heads.py
│   └── model.py
└── training/
    ├── loss.py
    ├── metrics.py
    └── trainer.py
```

---

## config.py

```python
@dataclass
class ModelConfig:
    hidden_dim: int = 128           # d — all node hidden dims
    ligand_edge_dim: int = 64       # d_e — ligand bond edge dim
    fg_edge_dim: int = 64           # d_e — FG-FG centroid edge dim
    num_ligand_gnn_layers: int = 5  # 5 layers (Valsson et al. 2025)
    num_fg_attn_layers: int = 2
    num_cross_attn_layers: int = 2
    num_heads: int = 3              # 3 heads (Valsson et al. 2025)
    dropout: float = 0.1
    # AEV — ANI-2x params (Valsson et al. 2025, Table 3)
    aev_rc: float = 5.1             # Å — cutoff radius
    aev_eta: float = 19.7
    aev_num_rs: int = 16            # 16 Rs values (0.80–4.83 Å)
    aev_num_ecif_types: int = 22    # 22 ECIF protein atom types
    aev_dim: int = 352              # 22 × 16 × 1
    fg_edge_cutoff: float = 5.0     # Å — FG centroid edge threshold
    cross_attn_cutoff: float = 6.0  # Å — FG-residue mask threshold
    num_rbf_kernels: int = 16
    rbf_d_min: float = 1.0
    rbf_d_max: float = 8.0

@dataclass
class TrainingConfig:
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 200
    warmup_steps: int = 500
    grad_clip: float = 1.0
    early_stopping_patience: int = 20

@dataclass
class DataConfig:
    data_root: str
    aev_rc: float = 5.1
    aev_eta: float = 19.7
    fg_edge_cutoff: float = 5.0
    cross_attn_cutoff: float = 6.0
    split_type: str = 'pocket_hash'
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42
```

---

## utils/rbf.py

```python
# Used for: pocket residue min_ligand_dist, FG centroid edge distances.
# NOT used for AEV (AEV has its own Gaussian kernel formula).

ANI2X_RS_VALUES: Tensor  # [16] — 0.80,1.07,1.34,...,4.83 Å  (Table 3, Valsson et al.)

def get_rbf_centers(num_kernels: int, d_min: float, d_max: float) -> Tensor
    # → [num_kernels]  evenly spaced centers

def get_rbf_widths(num_kernels: int, d_min: float, d_max: float) -> Tensor
    # → [num_kernels]

def rbf_expand(
    distances: Tensor,  # [N]
    centers: Tensor,    # [K]
    widths: Tensor      # [K]
) -> Tensor             # [N, K]
```

---

## data/graph_builder.py

```python
# ANI-2x AEV constants (no learnable params — computed at graph construction time)
ANI2X_RC: float = 5.1
ANI2X_ETA: float = 19.7
ANI2X_RS: Tensor  # [16]

def compute_ecif_types(
    atomic_num: Tensor,            # [N_poc]  int
    explicit_valence: Tensor,      # [N_poc]  int
    num_heavy_neighbors: Tensor,   # [N_poc]  int
    num_h_neighbors: Tensor,       # [N_poc]  int
    is_aromatic: Tensor,           # [N_poc]  bool
    in_ring: Tensor                # [N_poc]  bool
) -> Tensor                        # [N_poc]  int 0-21, ECIF type index

def compute_ligand_aev(
    ligand_coords: Tensor,         # [N_lig, 3]   float — docked pose coordinates
    pocket_coords: Tensor,         # [N_poc, 3]   float
    pocket_ecif_types: Tensor,     # [N_poc]       int 0-21
    rc: float,                     # 5.1 Å
    rs_values: Tensor,             # [16]          ANI-2x Rs values
    eta: float                     # 19.7
) -> Tensor                        # [N_lig, 352]  float — 22 × 16 × 1

def build_ligand_graph(
    atom_features: dict,           # keys: atomic_num[N], formal_charge[N], partial_charge[N],
                                   #   is_aromatic[N], hybridization[N], chiral_tag[N],
                                   #   explicit_h[N], implicit_h[N], isotope[N]
    bond_features: dict,           # keys: bond_type[E], is_conjugated[E], is_in_ring[E],
                                   #   bond_stereo[E], bond_dir[E]
    edge_index: Tensor,            # [2, N_bonds]
    aev: Tensor                    # [N_lig, 352]  precomputed AEV
) -> Data
    # Data.x:          [N_lig, 405]   raw concat before encoder (53 feats + 352 AEV)
    # Data.edge_index: [2, N_bonds]
    # Data.edge_attr:  [N_bonds, 34]  raw bond feats before encoder

def build_fg_graph(
    fg_features: dict,             # keys: num_atoms[N_fg], total_partial_charge[N_fg],
                                   #   max_radial_extent[N_fg], mean_partial_charge[N_fg],
                                   #   fg_type[N_fg], fg_parent_type[N_fg]
    fg_centroids: Tensor,          # [N_fg, 3]
    cutoff: float,                 # 5.0 Å
    num_rbf_kernels: int,          # 16
    rbf_d_min: float,
    rbf_d_max: float
) -> Data
    # Data.x:          [N_fg, 28]     raw scalar + cat feats
    # Data.edge_index: [2, N_fg_edges]
    # Data.edge_attr:  [N_fg_edges, 16]  RBF(centroid dist)

def compute_fg_centroids(
    atom_coords: Tensor,           # [N_lig, 3]
    fg_membership: Tensor          # [N_lig]  int — which FG each atom belongs to
) -> Tensor                        # [N_fg, 3]

def build_residue_membership(
    atom_residue_ids: Tensor       # [N_poc]  int
) -> Tensor                        # [N_poc]  0-indexed residue assignment

def build_cross_attn_mask(
    fg_centroids: Tensor,          # [N_fg, 3]
    residue_coords: Tensor,        # [N_res, 3]  representative residue coordinate
    cutoff: float                  # 6.0 Å
) -> Tensor                        # [N_fg, N_res]  bool

def build_complex_data(
    ligand_data: Data,
    fg_data: Data,
    residue_features: dict,        # keys: residue_type_id[N_res], residue_base_type_id[N_res],
                                   #   residue_variant_id[N_res], atom_count[N_res],
                                   #   min_ligand_dist[N_res]
    fg_membership: Tensor,         # [N_lig]  int
    cross_attn_mask: Tensor,       # [N_fg, N_res]  bool
    labels: dict                   # y_contact[N_fg,4], y_preorg[N_fg,1], y_aux_elec[N_fg],
                                   #   y_aux_vdw[N_fg], y_delta_total[1], fg_label_mask[N_fg]
) -> Data
```

---

## data/label_builder.py

```python
def aggregate_residue_channels_to_fg(
    residue_channels: Tensor,       # [N_res, 6]  (internal/vdw/elec/polar/nonpolar/total)
    fg_to_residue_dists: Tensor,    # [N_fg, N_res]  float — centroid-to-residue distances
    cutoff: float                   # 6.0 Å — matches cross_attn_cutoff
) -> Tensor                         # [N_fg, 4]  (vdw, elec, polar, nonpolar)

def compute_masked_mode_aux_targets(
    ghost: Tensor,                  # [N_fg, 4]  MMGBSA channels under ghost masking
    charge_only: Tensor,            # [N_fg, 4]
    lj_only: Tensor                 # [N_fg, 4]
) -> Tuple[Tensor, Tensor]
    # aux_elec: [N_fg]  charge_only.elec − ghost.elec
    # aux_vdw:  [N_fg]  lj_only.vdw − ghost.vdw

def compute_preorg_label(
    mmgbsa_components: Tensor,      # [num_entities, num_components]
    entity_order: List[str],        # ['complex','receptor','ligand','differences']
    component_order: List[str]      # ['BOND','ANGLE','DIHED','VDWAALS',...]
) -> Tensor                         # [1]  scalar — BOND+ANGLE+DIHED of ligand differences

def compute_delta_total_label(
    mmgbsa_components: Tensor,      # [num_entities, num_components]
    entity_order: List[str],
    component_order: List[str]
) -> Tensor                         # [1]  scalar — DELTA TOTAL from differences row

def build_fg_label_mask(
    labeled_fg_indices: List[int],  # which FGs have masked MMGBSA data (5 per structure)
    total_fg_count: int
) -> Tensor                         # [N_fg]  bool
```

---

## data/splits.py

```python
def pocket_hash_split(
    pdb_ids: List[str],
    pocket_sequences: dict,        # pdb_id → pocket residue sequence string
    val_frac: float,
    test_frac: float,
    seed: int
) -> Tuple[List[str], List[str], List[str]]   # (train_ids, val_ids, test_ids)

def time_split(
    pdb_ids: List[str],
    deposition_dates: dict,        # pdb_id → datetime
    cutoff_date: str
) -> Tuple[List[str], List[str]]              # (train_ids, test_ids)

def ligand_split(
    pdb_ids: List[str],
    ligand_smiles: dict,           # pdb_id → SMILES
    val_frac: float,
    test_frac: float,
    seed: int
) -> Tuple[List[str], List[str], List[str]]
```

---

## data/dataset.py

```python
class PerFGDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        config: DataConfig,
        split: str,                # 'train' | 'val' | 'test'
        transform = None
    )

    def len(self) -> int

    def get(self, idx: int) -> Data

    def process(self)
        # Calls _process_single for each pdb_id and saves to disk

    def _load_raw(self, pdb_id: str) -> dict
        # Loads precomputed feature dict from HDF5/pickle
        # Returns: {atom_features, bond_features, fg_features, residue_features,
        #           atom_coords, pocket_coords, fg_membership, mmgbsa, labeled_fg_indices}

    def _process_single(self, pdb_id: str) -> Data
        # Calls: compute_ecif_types → compute_ligand_aev
        #        → build_ligand_graph → build_fg_graph
        #        → build_cross_attn_mask → all label_builder fns
        #        → build_complex_data

    @property
    def raw_file_names(self) -> List[str]

    @property
    def processed_file_names(self) -> List[str]
```

---

## model/encoders.py

```python
class LigandAtomEncoder(nn.Module):
    # Embeds categoricals, normalizes scalars, concatenates with AEV
    # concat dim: atomic_num(32)+formal_charge(4)+partial_charge(1)+is_aromatic(1)
    #             +hybridization(8)+chiral_tag(4)+explicit_h(1)+implicit_h(1)+isotope(1)
    #             + AEV(352) = 405 → Linear(405,128)+LN
    def __init__(self, hidden_dim: int)      # hidden_dim=128
    def forward(
        self,
        atomic_num: Tensor,                  # [N]  int → Embedding(119,32)
        formal_charge: Tensor,               # [N]  int → Embedding(9,4)
        partial_charge: Tensor,              # [N]  float → BatchNorm
        is_aromatic: Tensor,                 # [N]  bool → scalar
        hybridization: Tensor,               # [N]  int → Embedding(7,8)
        chiral_tag: Tensor,                  # [N]  int → Embedding(4,4)
        explicit_h: Tensor,                  # [N]  int → scalar (norm)
        implicit_h: Tensor,                  # [N]  int → scalar (norm)
        isotope: Tensor,                     # [N]  int → scalar (norm)
        aev: Tensor                          # [N, 352]  float — precomputed, no grad
    ) -> Tensor                              # [N, 128]

class LigandBondEncoder(nn.Module):
    # concat dim: bond_type(16)+is_conjugated(1)+is_in_ring(1)+bond_stereo(8)+bond_dir(8) = 34
    def __init__(self, edge_dim: int)        # edge_dim=64
    def forward(
        self,
        bond_type: Tensor,                   # [E]  int → Embedding(6,16)
        is_conjugated: Tensor,               # [E]  bool → scalar
        is_in_ring: Tensor,                  # [E]  bool → scalar
        bond_stereo: Tensor,                 # [E]  int → Embedding(6,8)
        bond_dir: Tensor                     # [E]  int → Embedding(6,8)
    ) -> Tensor                              # [E, 64]  — Linear(34,64)+LN

class FGNodeEncoder(nn.Module):
    # concat dim: num_atoms(1)+total_partial_charge(1)+max_radial_extent(1)
    #             +mean_partial_charge(1)+fg_type(16)+fg_parent_type(8) = 28
    # Note: fg_type vocab=["UNK"] — trivial until populated
    def __init__(self, hidden_dim: int)      # hidden_dim=128
    def forward(
        self,
        num_atoms: Tensor,                   # [N_fg]  int → scalar (norm)
        total_partial_charge: Tensor,        # [N_fg]  float → scalar (norm)
        max_radial_extent: Tensor,           # [N_fg]  float → scalar (norm)
        mean_partial_charge: Tensor,         # [N_fg]  float → scalar (norm)
        fg_type: Tensor,                     # [N_fg]  int → Embedding(1,16)
        fg_parent_type: Tensor               # [N_fg]  int → Embedding(2,8)
    ) -> Tensor                              # [N_fg, 128]  — Linear(28,128)+LN

class PocketResidueEncoder(nn.Module):
    # concat dim: residue_type_id(32)+residue_base_type_id(16)+residue_variant_id(8)
    #             +atom_count(1)+min_ligand_dist_RBF(16) = 73
    def __init__(self, hidden_dim: int, num_rbf_kernels: int,
                 rbf_d_min: float, rbf_d_max: float)
    def forward(
        self,
        residue_type_id: Tensor,             # [N_res]  int → Embedding(40,32)
        residue_base_type_id: Tensor,        # [N_res]  int → Embedding(37,16)
        residue_variant_id: Tensor,          # [N_res]  int → Embedding(11,8)
        atom_count: Tensor,                  # [N_res]  int → scalar (norm)
        min_ligand_dist: Tensor              # [N_res]  float → RBF(16 kernels)
    ) -> Tensor                              # [N_res, 128]  — Linear(73,128)+LN
```

---

## model/gnn.py

```python
class GATv2Layer(nn.Module):
    # Per-layer params: lin_l(128×96)+lin_r(128×96)+lin_edge(64×96)+att(96)+bias(96)
    #                   +residual_linear(128×128+128)+LN(128) ≈ 58K per layer
    def __init__(
        self,
        in_dim: int,          # 128
        out_dim: int,         # 128
        heads: int,           # 3  (Valsson et al. 2025)
        edge_dim: int,        # 64
        dropout: float        # 0.1
    )
    def forward(
        self,
        x: Tensor,            # [N, 128]
        edge_index: Tensor,   # [2, E]
        edge_attr: Tensor     # [E, 64]
    ) -> Tensor               # [N, 128]

class LigandGNN(nn.Module):
    # 5-layer GATv2 on ligand covalent graph
    # AEV already baked into x from LigandAtomEncoder — no separate pocket GNN needed
    # Total: 5 × 58K ≈ 290K params
    def __init__(
        self,
        hidden_dim: int,      # 128
        edge_dim: int,        # 64
        num_layers: int,      # 5
        heads: int,           # 3
        dropout: float        # 0.1
    )
    def forward(
        self,
        x: Tensor,            # [N_lig_total, 128]  — LigandAtomEncoder output
        edge_index: Tensor,   # [2, N_bonds_total]
        edge_attr: Tensor     # [N_bonds_total, 64]  — LigandBondEncoder output
    ) -> Tensor               # [N_lig_total, 128]   h_atom
```

---

## model/pooling.py

```python
class FGAttentionPool(nn.Module):
    # attn_scorer: Linear(128,1)+bias → 129 params
    # merge_mlp:   Linear(256,128)+ReLU+Linear(128,128)+LN → ~49K params
    # Total: ~50K params
    def __init__(self, hidden_dim: int)   # hidden_dim=128
    def forward(
        self,
        h_atom: Tensor,           # [N_lig_total, 128]  — from LigandGNN
        fg_membership: Tensor,    # [N_lig_total]  int — FG index per atom (within-graph)
        h_fg_init: Tensor,        # [N_fg_total, 128]  — from FGNodeEncoder
        n_fg: int                 # total number of FG nodes across batch
    ) -> Tensor                   # [N_fg_total, 128]
    # Implementation:
    #   α_i = softmax(Linear(h_atom_i))  over atoms i in each FG
    #   h_fg_atoms = Σ α_i h_atom_i
    #   return MLP(concat(h_fg_atoms, h_fg_init))  — [N_fg_total, 128]
```

---

## model/attention.py

```python
class FGSelfAttentionGraph(nn.Module):
    # Intra-ligand context: FG nodes attend to neighbouring FGs (centroid dist ≤ 5Å)
    # Edge projection: Linear(16,64)+LN → 1,216 params
    # 2 × GATv2Layer(128,128,heads=4,edge_dim=64) → 2 × ~58K ≈ 116K
    # Total: ~117K params
    def __init__(
        self,
        hidden_dim: int,      # 128
        edge_dim: int,        # 64
        num_layers: int,      # 2
        heads: int,           # 4
        dropout: float        # 0.1
    )
    def forward(
        self,
        h_fg: Tensor,              # [N_fg_total, 128]  — from FGAttentionPool
        fg_edge_index: Tensor,     # [2, N_fg_edges_total]
        fg_edge_attr: Tensor       # [N_fg_edges_total, 16]  — RBF(centroid dist)
    ) -> Tensor                    # [N_fg_total, 128]  — h_fg_ctx

class CrossAttentionLayer(nn.Module):
    # Standard multi-head cross-attention + FFN
    # W_Q,W_K,W_V,W_O: Linear(128,128) each → 4 × 16,512 = 66,048
    # FFN: Linear(128,256)+ReLU+Linear(256,128) + 2×LN → ~66K
    # Per layer: ~132K params
    def __init__(
        self,
        hidden_dim: int,      # 128
        heads: int,           # 4
        dropout: float        # 0.1
    )
    def forward(
        self,
        query: Tensor,        # [N_fg, 128]
        key: Tensor,          # [N_res, 128]
        value: Tensor,        # [N_res, 128]
        mask: Tensor          # [N_fg, N_res]  bool — True where attention is allowed
    ) -> Tensor               # [N_fg, 128]

class FGPocketCrossAttention(nn.Module):
    # FG queries attend to pocket residue keys/values
    # Mask derived from nearest_fg_centroid_dist ≤ 6Å
    # 2 × CrossAttentionLayer → ~265K params
    def __init__(
        self,
        hidden_dim: int,      # 128
        num_layers: int,      # 2
        heads: int,           # 4
        dropout: float        # 0.1
    )
    def forward(
        self,
        h_fg: Tensor,         # [N_fg_total, 128]   h_fg_ctx (from FGSelfAttentionGraph)
        h_res: Tensor,        # [N_res_total, 128]  h_res (from PocketResidueEncoder, cached)
        attn_mask: Tensor     # [N_fg_total, N_res_max]  bool (padded per batch)
    ) -> Tensor               # [N_fg_total, 128]   h_fg_final
```

---

## model/heads.py

```python
class ContactHead(nn.Module):
    # Predicts per-FG contact energy decomposition
    # Labels: mmgbsa residue_channels (vdw,elec,polar,nonpolar) summed over nearby residues
    # Linear(128,64)+ReLU+Linear(64,4) → ~8.5K params
    def __init__(self, hidden_dim: int)   # hidden_dim=128
    def forward(
        self,
        h_fg_final: Tensor                # [N_fg_total, 128]  — post cross-attention
    ) -> Tensor                           # [N_fg_total, 4]  [vdw, elec, polar, nonpolar]

class PreorgHead(nn.Module):
    # Predicts per-FG intramolecular strain — pocket-INDEPENDENT
    # Uses h_fg_ctx (before cross-attention) — no pocket context
    # Labels: BOND+ANGLE+DIHED from ligand entity differences in mmgbsa_components
    # Linear(128,64)+ReLU+Linear(64,1) → ~8.3K params
    def __init__(self, hidden_dim: int)   # hidden_dim=128
    def forward(
        self,
        h_fg_ctx: Tensor                  # [N_fg_total, 128]  — pre cross-attention
    ) -> Tensor                           # [N_fg_total, 1]
```

---

## model/model.py

```python
class PerFGModel(nn.Module):
    # Total params: ~557K
    # Components: LigandAtomEncoder(~35K) + LigandBondEncoder(~2K) + FGNodeEncoder(~4K)
    #             + PocketResidueEncoder(~10K) + LigandGNN(~290K) + FGAttentionPool(~50K)
    #             + FGSelfAttentionGraph(~117K) + FGPocketCrossAttention(~265K)
    #             + ContactHead(~8.5K) + PreorgHead(~8.3K) + loss_weights(8)

    def __init__(self, config: ModelConfig)

    def forward(self, batch: Data) -> dict
    # Returns: {
    #   'contact':     [N_fg_total, 4]   vdw/elec/polar/nonpolar per FG
    #   'preorg':      [N_fg_total, 1]   intramolecular strain per FG
    #   'delta_total': [B]               sum of all FG contributions per complex
    # }

    def _encode_inputs(
        self, batch: Data
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    # h_lig_atom: [N_lig_total, 128]   (LigandAtomEncoder — includes AEV)
    # h_bond:     [N_bonds_total, 64]  (LigandBondEncoder)
    # h_fg_init:  [N_fg_total, 128]    (FGNodeEncoder)
    # h_res:      [N_res_total, 128]   (PocketResidueEncoder — cached per target)

    def _run_ligand_gnn(
        self,
        h_atom: Tensor,        # [N_lig_total, 128]
        edge_index: Tensor,    # [2, N_bonds_total]
        edge_attr: Tensor      # [N_bonds_total, 64]
    ) -> Tensor                # [N_lig_total, 128]

    def _pool_to_fg(
        self,
        h_atom: Tensor,            # [N_lig_total, 128]
        fg_membership: Tensor,     # [N_lig_total]  int
        h_fg_init: Tensor,         # [N_fg_total, 128]
        n_fg_per_graph: Tensor     # [B]  — needed for within-graph pooling indices
    ) -> Tensor                    # [N_fg_total, 128]

    def _fg_context(
        self,
        h_fg: Tensor,              # [N_fg_total, 128]
        fg_edge_index: Tensor,     # [2, N_fg_edges_total]
        fg_edge_attr: Tensor       # [N_fg_edges_total, 16]
    ) -> Tensor                    # [N_fg_total, 128]  — h_fg_ctx

    def _cross_attend(
        self,
        h_fg_ctx: Tensor,          # [N_fg_total, 128]
        h_res: Tensor,             # [N_res_total, 128]
        attn_mask: Tensor,         # [N_fg_total, N_res_max]  bool
        res_batch: Tensor          # [N_res_total]  — complex index per residue
    ) -> Tensor                    # [N_fg_total, 128]  — h_fg_final

    def _aggregate_whole_ligand(
        self,
        contact_pred: Tensor,      # [N_fg_total, 4]
        preorg_pred: Tensor,       # [N_fg_total, 1]
        fg_batch: Tensor           # [N_fg_total]  — complex index per FG
    ) -> Tensor                    # [B]  sum(vdw+elec+polar+nonpolar+preorg) per complex

    def count_parameters(self) -> int
```

---

## training/loss.py

```python
class MultiTaskUncertaintyLoss(nn.Module):
    # Kendall et al. 2018: L = Σ_t [1/(2σ_t²) · MSE_t + log σ_t]
    # Tasks: vdw, elec, polar, nonpolar (per-FG, [N_fg] each)
    #        preorg (per-FG, [N_fg])
    #        aux_elec, aux_vdw (per-FG, from masked modes, weight=0.25)
    #        delta_total (per-complex, [B])
    # Learnable: log_vars [8] — one per task
    def __init__(self, task_names: List[str])
    # task_names: ['vdw','elec','polar','nonpolar','preorg','aux_elec','aux_vdw','delta_total']

    def forward(
        self,
        preds: dict,       # 'contact':[N_fg,4], 'preorg':[N_fg,1], 'delta_total':[B]
        targets: dict,     # 'contact':[N_fg,4], 'preorg':[N_fg,1],
                           #   'aux_elec':[N_fg], 'aux_vdw':[N_fg], 'delta_total':[B]
        masks: dict        # 'fg':[N_fg] bool (which FGs have labels),
                           #   'complex':[B] bool (all True if MMGBSA was run)
    ) -> Tuple[Tensor, dict]   # (total_loss [1], per_task_losses dict)

    def get_loss_weights(self) -> dict   # 1/(2σ_t²) per task for logging

def masked_mse(
    pred: Tensor,    # [N, ...]
    target: Tensor,  # [N, ...]
    mask: Tensor     # [N]  bool
) -> Tensor          # scalar

def masked_mae(
    pred: Tensor,
    target: Tensor,
    mask: Tensor
) -> Tensor          # scalar
```

---

## training/metrics.py

```python
def pearson_r(pred: Tensor, target: Tensor, mask: Tensor = None) -> float
def spearman_r(pred: Tensor, target: Tensor, mask: Tensor = None) -> float
def mae(pred: Tensor, target: Tensor, mask: Tensor = None) -> float

def compute_per_task_metrics(
    preds: dict,     # {'contact':[N,4], 'preorg':[N,1], 'delta_total':[B]}
    targets: dict,
    masks: dict
) -> dict            # {task_name: {'pearson_r':float, 'spearman_r':float, 'mae':float}}

def fg_rank_correlation(
    contact_preds: Tensor,    # [N_fg]  — summed contact per FG (vdw+elec+polar+nonpolar)
    contact_targets: Tensor,  # [N_fg]
    complex_ids: Tensor       # [N_fg]  — which complex each FG belongs to
) -> float                    # mean within-complex Spearman rank correlation

def per_chemotype_metrics(
    preds: Tensor,            # [N_fg, 4]
    targets: Tensor,          # [N_fg, 4]
    fg_types: Tensor,         # [N_fg]  int — fg_type index
    mask: Tensor = None       # [N_fg]  bool
) -> dict                     # {fg_type_id: {task: metric}}
```

---

## training/trainer.py

```python
class Trainer:
    def __init__(
        self,
        model: PerFGModel,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fn: MultiTaskUncertaintyLoss,
        config: TrainingConfig,
        device: str
    )

    def train_epoch(self, loader: DataLoader) -> dict
    # Runs one epoch; returns {'loss':float, per_task_losses..., 'lr':float}

    def evaluate(self, loader: DataLoader, split: str = 'val') -> dict
    # Runs full evaluation; returns all metrics from compute_per_task_metrics
    # + fg_rank_correlation + per_chemotype_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    )
    # Main loop with early stopping on val pearson_r (contact head)

    def save_checkpoint(self, path: str, epoch: int, metrics: dict)

    def load_checkpoint(self, path: str) -> Tuple[int, dict]
    # Returns (epoch, metrics)

    def _move_to_device(self, batch: Data) -> Data

    def _collect_all_predictions(
        self, loader: DataLoader
    ) -> Tuple[dict, dict, dict]
    # Returns (all_preds, all_targets, all_masks) over full dataset for offline metrics
```

---

## train.py

```python
def parse_args() -> argparse.Namespace

def build_dataloaders(
    data_config: DataConfig,
    training_config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]   # train, val, test

def build_model(config: ModelConfig) -> PerFGModel

def build_optimizer(
    model: PerFGModel,
    config: TrainingConfig
) -> Optimizer

def build_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    num_training_steps: int
) -> LRScheduler

def main(args: argparse.Namespace)
```

---

## Key dimension summary

| Tensor | Shape | Notes |
|--------|-------|-------|
| Ligand atom raw feats | `[N_lig, 53]` | before AEV concat |
| AEV | `[N_lig, 352]` | 22 ECIF types × 16 Rs, η=19.7, Rc=5.1Å |
| Ligand atom encoder input | `[N_lig, 405]` | 53 + 352 |
| h_atom (after GNN) | `[N_lig, 128]` | d=128 |
| Bond encoder input | `[N_bonds, 34]` | 5 bond feats → embeddings |
| h_bond | `[N_bonds, 64]` | d_e=64 |
| FG encoder input | `[N_fg, 28]` | 4 scalar + 2 cat → embed |
| h_fg_init | `[N_fg, 128]` | d=128 |
| Residue encoder input | `[N_res, 73]` | 3 embed + 1 scalar + RBF(16) |
| h_res | `[N_res, 128]` | d=128, cached per target |
| h_fg (after pool) | `[N_fg, 128]` | |
| h_fg_ctx (after FG self-attn) | `[N_fg, 128]` | branches to preorg head |
| h_fg_final (after cross-attn) | `[N_fg, 128]` | feeds contact head |
| y_contact | `[N_fg, 4]` | vdw, elec, polar, nonpolar |
| y_preorg | `[N_fg, 1]` | intramolecular strain |
| y_delta_total | `[B]` | whole-ligand DELTA TOTAL |
| fg_label_mask | `[N_fg]` | bool — 5/N_fg FGs labeled |
