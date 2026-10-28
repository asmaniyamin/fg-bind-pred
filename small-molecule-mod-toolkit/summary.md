# Project Summary: small-molecule-mod-toolkit
# 04/18/2026

This toolkit provides a NumPy-first, RDKit-backed pipeline for representing, building, editing, cleaning, merging, hashing, and canonicalizing small molecules as graph objects (`MolGraph`). The two root namespaces are `core/` (data structures and graph-level operations) and `chem/` (chemistry-domain operations: build, clean, edit, merge, dummy atoms, functional groups, and pipeline runners).

---

## `core/`

### `core/structs.py`

The single source of truth for all data structures. Contains only data (no methods). All logic lives in other modules.

**Enum classes (int-coded):**

| Class | Purpose |
|---|---|
| `AtomHybridization` | SP, SP2, SP3, SP3D, SP3D2 as ints |
| `AtomChiralTag` | UNSPECIFIED, CW, CCW, OTHER |
| `AtomCIPCode` | NONE, R, S, UNKNOWN |
| `BondType` | SINGLE, DOUBLE, TRIPLE, AROMATIC as ints |
| `BondDir` | NONE, BEGINWEDGE, BEGINDASH, ENDDOWNRIGHT, ENDUPRIGHT, UNKNOWN |
| `BondStereo` | NONE, E, Z, CIS, TRANS, UP, DOWN, UNKNOWN |
| `BondResonanceType` | NONE, LOCALIZED, DELOCALIZED, AROMATIC |
| `AttachmentKind` | DUMMY, H_SUB, OPEN_VALENCE |

**Dataclasses:**

#### `MolArrays`
The canonical chemistry state. N = number of atoms, M = number of undirected bonds.
- **Required fields:** `atomic_num` (int16, N), `formal_charge` (int8, N), `bonds` (int32, M×2), `bond_type` (int8, M)
- **Optional atom fields:** `isotope`, `is_aromatic`, `hybridization`, `chiral_tag`, `cip_code`, `atom_map`, `attachment_label`, `explicit_h`, `implicit_h`, `partial_charge`
- **Optional geometry fields:** `pos` (float32, (N,3) or (K,N,3) for conformer ensembles), `coord_frame` (string identifier), `coord_valid` (bool mask, invalidated after structural mutations)
- **Optional bond fields:** `is_conjugated`, `is_in_ring`, `bond_dir`, `bond_stereo`, `bond_resonance_type`
- **Globals:** `total_charge` (int), `multiplicity` (optional int)

#### `AttachmentPointArray`
Vectorized attachment points (derived, not canonical).
- Fields: `idx` (atom indices), `kind`, `target` (heavy neighbor index, -1 if none), `label_id`, `is_insertion`, `insertion_anchors`

#### `Editability`
Mutation policy scores per atom and bond.
- Fields: `atom_scores` (ndarray), `bond_scores` (ndarray)

#### `Fragment`
A named subset of atoms with attachment indices.
- Fields: `fragment_id`, `atom_indices`, `attachment_indices`, optional `role`, `origin`, `smiles`, `metadata`

#### `ResonanceSystem`
Optional rich view for a delocalized electron system.
- Fields: `resonance_system_id`, `atom_indices`, `bond_indices`, `smarts`, `label`, `role`, `origin`, `metadata`

#### `DerivedCaches`
Non-canonical tensors for ML/graph export.
- Fields: `edge_index` (directed PyG-style (2, 2M)), `x` (node features), `edge_attr` (edge features), `bond_pair_to_index` (lookup dict)

#### `DirtyFlags`
Tracks whether caches need rebuilding.
- Fields: `structure_dirty` (bool), `features_dirty` (bool)

#### `MolGraph`
Top-level molecular graph object.
- Fields: `arrays` (MolArrays, canonical truth), `attachments` (derived), `editability`, `fragments`, `resonance_systems`, `cache`, `dirty`, `meta` (dict for SMILES, op_log, etc.), `rdkit_mol` (optional, not repr'd)

---

### `core/structs_helper.py`
**NOTE: Marked deprecated** (wrong dimensions for ensemble pos). Kept for reference.

Helpers to validate and invalidate `MolGraph`/`MolArrays`.

#### `_require_1d(name, a, n, dtype)`
Raises `ValueError` if optional array `a` is not shape `(n,)` or has wrong dtype.

#### `_require_bond_1d(name, a, m)`
Raises `ValueError` if optional bond array `a` is not shape `(m,)`.

#### `validate_molarrays(arr: MolArrays)`
Checks that required arrays have consistent shapes; checks optional arrays via `_require_1d`/`_require_bond_1d`; validates `pos` shape as `(N,2)` or `(N,3)`.

#### `validate_attachment_points(ap: AttachmentPointArray, N)`
Validates that `idx`/`kind`/`target` are all shape `(P,)`, that `idx` and `target` are in-bounds atom indices, and optional arrays have correct length.

#### `validate_molgraph(g: MolGraph)`
Calls `validate_molarrays(g.arrays)` and `validate_attachment_points(g.attachments, N)` if present.

#### `invalidate(g: MolGraph, structure, features)`
Manually sets dirty flags and clears relevant caches (`edge_index`, `bond_pair_to_index`, `edge_attr`, `x`).

---

### `core/molgraph/canonicalize.py`

Canonical atom ordering via Weisfeiler-Lehman (WL) graph refinement over `MolGraph`.

#### `_h(x: bytes) -> bytes`
Returns SHA-256 digest of bytes. Used as a hash combiner throughout this module.

#### `_build_adj(bonds, n_atoms) -> list[list[(v, bidx)]]`
Builds an adjacency list from the bond array. Returns a list of length `n_atoms`, each containing `(neighbor_idx, bond_idx)` tuples.

#### `_atom_seed(g, i, stereo) -> bytes`
Generates a per-atom seed hash from: atomic_num, formal_charge, aromaticity, isotope, chiral_tag (if `stereo=True`), explicit_h, implicit_h.

#### `_bond_seed(g, bidx, stereo) -> bytes`
Generates a per-bond seed hash from: bond_type, is_conjugated, is_in_ring, bond_dir and bond_stereo (if `stereo=True`), bond_resonance_type.

#### `compute_canonical_order(g, iters=4, stereo=True) -> np.ndarray`
- **Input:** `MolGraph`, number of WL iterations, whether to include stereo
- **Output:** `new_to_old` permutation array (length N) giving canonical atom order
- **How:** Initializes per-atom and per-bond seed hashes; runs `iters` rounds of WL neighborhood aggregation (sorted neighbor messages → hashed); sorts atoms by `(final_hash, degree, neighbor_signature, original_index)` for a deterministic total order.

#### `canonicalize(g, iters=4, stereo=True) -> (g2, old_to_new, new_to_old, bond_perm)`
- **Input:** `MolGraph`
- **Output:** Reordered `MolGraph`, plus three index maps
- **How:** Calls `compute_canonical_order`, then delegates all remapping to `permute_atoms` from `chem.edit.atoms`. Sets `meta["canonicalized"] = True`.

---

### `core/molgraph/hash.py`

Deterministic SHA-256 hash of a `MolGraph`. **Recommended usage:** canonicalize first, then hash.

#### Byte-encoding helpers: `_i8`, `_u8`, `_i16`, `_u16`, `_i32`, `_u32`
Convert Python ints to little-endian byte sequences of the given width/signedness.

#### `hash_topology(mg) -> str`
- **Input:** `MolGraph` (expected to be in canonical order)
- **Output:** Hex SHA-256 string
- **How:** Serializes atoms (atomic_num, formal_charge, aromaticity, isotope, explicit_h, implicit_h) and bonds (u, v, bond_type, is_conjugated, is_in_ring, bond_resonance_type) into a byte blob with prefix `b"TOPOv1"` and hashes. Does not include stereo fields.

#### `hash_stereo(mg) -> str`
Same as `hash_topology` but adds `chiral_tag`, `bond_dir`, `bond_stereo` per atom/bond. Uses prefix `b"STEREOv1"`.

#### `hash_molgraph(mg, mode="stereo") -> str`
- **Input:** `MolGraph`, `mode` = `"topology"` or `"stereo"`
- **Output:** Hex hash string
- **How:** Dispatches to `hash_topology` or `hash_stereo`. Raises `ValueError` for unknown mode.

---

### `core/molgraph/validate.py`

Structural invariant checking for `MolGraph`.

#### `MolGraphValidationError(ValueError)`
Custom exception raised on failed validation in strict mode.

#### `validate_molgraph(mg, strict=True) -> List[str]`
- **Input:** `MolGraph`, strict flag
- **Output:** List of error/warning strings (empty if valid)
- **How:** Checks atom arrays (not None, correct shape), bond arrays (shape (M,2), indices in bounds, no self-bonds, u < v normalized), coordinate shapes for single `(N,3)` and ensemble `(K,N,3)`, `coord_valid` shape consistency, and fragment atom indices in bounds. If `strict=True`, raises `MolGraphValidationError` on first failure batch.

---

### `core/molgraph/ops/canonicalize.py`

Transform wrapper for canonicalization.

#### `Canonicalize(Transform)`
- **Fields:** `iters=4`, `stereo=True`, `name="canonicalize"`
- `apply(mg) -> (mg2, OpInfo)`: Calls `canonicalize()`, records `old_to_new`, `new_to_old`, `bond_perm` in `OpInfo`.

---

### `core/molgraph/ops/hash.py`

Transform wrapper for hashing.

#### `HashMolGraph(Transform)`
- **Fields:** `mode="stereo"`, `write_to_meta=True`, `meta_key=None`, `name="hash_molgraph"`
- `apply(mg) -> (mg, OpInfo)`: Computes hash via `hash_molgraph()`, optionally writes it to `mg.meta[meta_key]`. No structural changes; `n_atoms` and `n_bonds` unchanged.

---

### `core/molgraph/ops/validate.py`

Transform wrapper for validation.

#### `Validate(Transform)`
- **Fields:** `strict=True`, `name="validate"`
- `apply(mg) -> (mg, OpInfo)`: Calls `validate_molgraph()`; collects warnings in non-strict mode. No structural changes.

---

## `chem/`

---

### `chem/build/`

Converts between SMILES/RDKit `Mol` objects and `MolGraph`.

---

#### `chem/build/build_utils.py`

Internal helpers for RDKit → `MolGraph` conversion.

**Enum maps (module-level dicts):**
- `HYBRIDIZATION_MAP`, `CHIRAL_TAG_MAP`, `BOND_TYPE_MAP`, `BOND_DIR_MAP`, `BOND_STEREO_MAP`: Map RDKit enum values to project int codes.

#### `extract_atom_arrays(mol, compute_charges=True) -> dict`
- **Input:** RDKit `Mol`, optional charge computation flag
- **Output:** Dict with keys `atomic_num`, `formal_charge`, `isotope`, `is_aromatic`, `hybridization`, `chiral_tag`, `atom_map`, `attachment_label`, `explicit_h`, `implicit_h`, `partial_charge`, `pos`
- **How:** Iterates RDKit atoms, applying enum maps. Sets `isotope` and `atom_map` to `None` if all-zero. Extracts conformer positions if present. Calls `compute_gasteiger_charges` if requested.

#### `extract_bond_arrays(mol) -> dict`
- **Input:** RDKit `Mol`
- **Output:** Dict with `bonds` (M×2, normalized min/max), `bond_type`, `is_conjugated`, `is_in_ring`, `bond_dir`, `bond_stereo`, `bond_resonance_type`
- **How:** Iterates RDKit bonds, applies enum maps, infers resonance type per bond. Sets `bond_dir`/`bond_stereo` to `None` if all-zero.

#### `infer_resonance_type(bond) -> int`
- **Input:** RDKit `Bond`
- **Output:** `BondResonanceType` int code
- **How:** Returns AROMATIC if aromatic, DELOCALIZED if conjugated, LOCALIZED if single/double/triple, else NONE.

#### `compute_gasteiger_charges(mol) -> np.ndarray`
- **Input:** RDKit `Mol`
- **Output:** float32 array of Gasteiger partial charges, shape (N,), NaN→0.0
- **How:** If charges already computed (stored as `_GasteigerCharge` property), extracts them. Otherwise computes via `AllChem.ComputeGasteigerCharges`. For molecules with dummy atoms (atomic_num=0), temporarily replaces dummies with H before computing, then transfers charges back to real atoms only.

#### `extract_attachment_points(mol) -> Optional[AttachmentPointArray]`
- **Input:** RDKit `Mol`
- **Output:** `AttachmentPointArray` or `None` if no dummy atoms
- **How:** Finds all atoms with atomic_num=0, records their first non-dummy neighbor as `target`, and collects atom map numbers as labels.

#### `build_edge_index(arrays) -> np.ndarray`
- **Input:** `MolArrays`
- **Output:** Directed edge index array of shape (2, 2M) for PyG
- **How:** Concatenates forward `bonds.T` and backward `bonds[:,::-1].T`.

#### `build_bond_pair_to_index(arrays) -> dict`
- **Input:** `MolArrays`
- **Output:** Dict mapping `(u,v)` and `(v,u)` → bond index
- **How:** Iterates bonds array, inserts both directed entries.

#### `parse_and_clean_smiles(smiles) -> (mol, repair_log)`
- **Input:** SMILES string
- **Output:** Cleaned RDKit `Mol`, list of repair steps
- **How:** Parses without sanitization (`sanitize=False`), runs `MoleculeCleaner`, raises `ValueError` if cleaning fails.

#### `compute_gasteiger_with_capping(mol) -> mol`
- **Input:** RDKit `Mol` (possibly with dummies)
- **Output:** Same `Mol` with `_GasteigerCharge` properties added to all real atoms
- **How:** If dummies present, temporarily swaps them for H (on a copy), computes charges, transfers back to originals. Sets dummy charges to "0.0". If no dummies, computes directly.

#### `prepare_molecule(smiles, add_hs=True, compute_charges=True) -> (mol, repair_log)`
- **Input:** SMILES string, optional flags
- **Output:** Prepared RDKit `Mol`, repair log
- **How:** Calls `parse_and_clean_smiles`, optionally adds explicit H via `Chem.AddHs`, optionally computes charges via `compute_gasteiger_with_capping`.

---

#### `chem/build/create_molgraph.py`

Entry points for converting to `MolGraph`.

#### `generate_coordinates(mol, mode, num_confs, random_seed, optimize, max_iterations) -> (pos, coord_frame, coord_valid)`
- **Input:** RDKit `Mol`, mode (`"canonical"` or `"ensemble"`), embedding parameters
- **Output:** Numpy positions array (N,3) or (K,N,3), coord_frame string, bool validity mask
- **How:** Configures RDKit ETKDGv3 parameters; for canonical mode embeds one conformer; for ensemble mode embeds multiple via `EmbedMultipleConfs`; optionally minimizes with MMFF force field. All coordinates initially valid.

#### `mol_to_molgraph(mol, compute_charges, keep_rdkit_mol, coords, num_confs, optimize_coords, random_seed) -> MolGraph`
- **Input:** RDKit `Mol`, various flags
- **Output:** `MolGraph`
- **How:** Optionally generates 3D coords via `generate_coordinates` (on a copy). Calls `extract_atom_arrays` and `extract_bond_arrays`, overrides `pos` if coordinates were generated, assembles `MolArrays`, calls `extract_attachment_points`, stores SMILES in meta.

#### `smiles_to_molgraph(smiles, add_hs, compute_charges, keep_rdkit_mol, coords, ...) -> Optional[MolGraph]`
- **Input:** SMILES string, optional flags
- **Output:** `MolGraph` or `None` on failure
- **How:** Calls `prepare_molecule` (parse + clean + optional H/charges), then `mol_to_molgraph`. Stores `input_smiles` and `repair_log` in meta.

---

#### `chem/build/molgraph_to_mol.py`

Inverse of `create_molgraph.py`: reconstructs an RDKit `Mol` from a `MolGraph`.

**Reverse enum maps:** `CHIRAL_TAG_TO_RDKIT`, `BOND_TYPE_TO_RDKIT`, `BOND_DIR_TO_RDKIT`, `BOND_STEREO_TO_RDKIT`

#### `_create_rdkit_atom(arrays, idx) -> Chem.Atom`
Creates a single RDKit `Atom` from `MolArrays` at index `idx`. Handles dummy atoms specially (sets attachment label as property). For real atoms, sets formal charge, aromaticity, isotope, chiral tag.

#### `_build_atoms(arrays) -> (RWMol, atom_idx_map)`
Iterates all atoms, calls `_create_rdkit_atom`, adds to an `RWMol`, returns the mol and an identity index map.

#### `_build_bonds(rw_mol, arrays, atom_idx_map)`
Iterates bonds, looks up RDKit bond type, adds bond to `RWMol`, sets bond direction and stereo if arrays are present.

#### `_enforce_bond_aromaticity(mol)`
For each bond between two aromatic atoms, sets it to aromatic type and conjugated.

#### `_validate_aromatic_consistency(mol)`
Raises `ValueError` if any bond between two aromatic atoms is not aromatic type.

#### `_sanitize_mol(mol) -> Optional[Mol]`
Attempts full RDKit sanitization; on failure tries partial sanitization (radicals + conjugation + hybridization + cleanup + properties). Returns `None` if both fail.

#### `_assign_stereochemistry(mol)`
Calls `Chem.AssignStereochemistry(force=True, cleanIt=True)`.

#### `molgraph_to_mol(graph, sanitize=True, remove_hs=False) -> Optional[Chem.Mol]`
- **Input:** `MolGraph`, optional flags
- **Output:** RDKit `Mol` or `None` on failure
- **How:** Builds atoms → bonds → enforces aromaticity → validates consistency → sanitizes → assigns stereo → optionally removes H.

#### `molgraph_to_smiles(graph, canonical=True, isomeric=True) -> Optional[str]`
- **Input:** `MolGraph`, optional flags
- **Output:** SMILES string or `None`
- **How:** Calls `molgraph_to_mol`, then `Chem.MolToSmiles`.

---

### `chem/clean/`

Rule-based molecular repair.

---

#### `chem/clean/detect_tools.py`

Stateless detection functions for structural problems.

#### `_default_valence(atomic_num) -> int` *(cached)*
Returns RDKit periodic table default valence for element Z.

#### `_valence_list(atomic_num) -> tuple` *(cached)*
Returns tuple of all allowed valences for element Z.

#### `detect_dummies_in_aromatic_ring(mol) -> bool`
Returns `True` if any dummy atom (atomic_num=0) is adjacent to an aromatic atom.

#### `detect_unconnected_fragments(mol) -> bool`
Returns `True` if `Chem.GetMolFrags(mol)` finds more than one fragment.

#### `detect_valence_errors(mol) -> bool`
For each non-dummy atom, checks whether `total_valence > max(allowed_valences + formal_charge)`. Returns `True` if any atom exceeds limits.

#### `detect_formal_charge_imbalance(mol, max_charge=2) -> bool`
Returns `True` if any atom has |formal_charge| > max_charge.

#### `detect_missing_stereochemistry(mol) -> bool`
Assigns stereo, then returns `True` if any atom has `_ChiralityPossible` but no `_CIPCode`.

#### `detect_nonstandard_valence_deprecated(mol) -> bool` *(deprecated)*
Old version using `GetExplicitValence` directly.

#### `detect_nonstandard_valence(mol) -> bool`
Returns `True` if any non-dummy atom has explicit valence > its default valence.

#### `detect_kekulization_failure(mol) -> bool`
Attempts `Chem.Kekulize(clearAromaticFlags=True)`; returns `True` if it throws.

#### `detect_rings_broken(mol) -> bool`
Returns `True` if the molecule has zero rings detected.

#### `find_overvalent_atoms(mol) -> List[Chem.Atom]`
Returns list of non-dummy atoms whose explicit valence exceeds their default valence.

---

#### `chem/clean/fatal_repair_tools.py`

Rule-based structural repair functions.

#### `fatal_repair_dummy_in_aromatic(mol) -> Mol`
Moves dummy atoms connected to aromatic atoms to a nearby non-aromatic atom by deleting the dummy and re-adding it bonded to the non-aromatic neighbor. Returns original if no change.

#### `fatal_repair_unconnected_fragments(mol) -> Mol`
Returns the largest connected fragment via `Chem.GetMolFrags(asMols=True)`.

#### `fatal_repair_valence(mol) -> Optional[Mol]`
Attempts `Chem.SanitizeMol`; returns `None` on `MolSanitizeException`.

#### `fatal_repair_remove_excess_hydrogens(mol) -> Mol`
Calls `Chem.RemoveHs(sanitize=False)` to remove explicit H atoms.

#### `fatal_repair_formal_charges(mol) -> Mol`
Deep-copies mol and calls `UpdatePropertyCache(strict=False)` to refresh charge state.

#### `fatal_repair_stereochemistry(mol) -> Mol`
Deep-copies mol and calls `Chem.AssignStereochemistry(cleanIt=True, force=True)`.

#### `fatal_repair_tautomer_normalization(mol) -> Mol`
Normalizes using `rdMolStandardize.Normalizer().normalize(mol)`.

#### `try_neutralise_overvalence(mol, bad_atoms) -> Mol`
For overvalent C/N/O atoms with non-zero formal charge, resets charge to 0 and allows implicit H.

#### `try_promote_hypervalent_atoms(mol, bad_atoms) -> Mol`
For P/S/Cl/Br/I atoms with excess valence, sets formal charge to +1.

#### `try_neutralise_common_hypervalence(mol, max_iter=15) -> Mol`
Applies SMARTS-based pattern replacements iteratively (nitro, sulfone, quaternary N, anionic nitrile, phosphoric acid variants) until convergence or max iterations. Uses `_RESCUE_SMARTS` list of `(smarts, replacement_smiles)` pairs.

#### `clear_nonring_aromaticity(mol)`
Removes aromatic flags from atoms and bonds that are not in any ring.

#### `fatal_repair_all(mol) -> Mol`
Retained for reference (not updated). Runs all repair steps in sequence.

---

#### `chem/clean/kekulization_repair.py`

Modular fallback pipeline for Kekulization failures.

#### `step_with_logging(step_fn, smiles, verbose, **kwargs) -> Optional[Mol]`
Runs a repair step function; catches exceptions and logs if verbose; returns `None` on failure.

#### `try_sanitize_then_kekulize_preserving_aromatic(smiles, **kwargs) -> Mol`
Parses without sanitization, sanitizes except Kekulize, clears non-ring aromaticity, then Kekulizes preserving aromatic flags.

#### `try_sanitize_without_aromatization(smiles, **kwargs) -> Mol`
Parses without sanitization, sanitizes with `aromatizeIfPossible` flag from kwargs.

#### `try_clear_aromaticity_and_sanitize(smiles, **kwargs) -> Mol`
Parses without sanitization, manually clears all aromatic flags, then fully sanitizes.

#### `try_add_hs_and_sanitize(smiles, **kwargs) -> Mol`
Parses without sanitization, adds explicit Hs, then sanitizes.

#### `try_normalize_and_sanitize(smiles, **kwargs) -> Mol`
Parses without sanitization, normalizes with `rdMolStandardize`, then sanitizes.

#### `try_resonance_forms(smiles, **kwargs) -> Optional[Mol]`
Generates resonance forms and returns the first one that sanitizes cleanly.

#### `fix_kekulization_modular(smiles, verbose, extreme, aromatizeIfPossible) -> Mol`
- **Input:** SMILES, flags
- **Output:** A valid `Mol` after Kekulization repair
- **How:** Tries each step function in order via `step_with_logging`. If `extreme=True`, adds `try_clear_aromaticity_and_sanitize` as a final fallback. Raises `ValueError` if all steps fail.

---

#### `chem/clean/clean_molecule.py`

High-level repair interface.

#### `MoleculeCleaner`
Rule-based molecule fixer.
- **`__init__(verbose, return_log, preserveAromaticity)`**: Sets options; initializes log list.
- **`clean_dep(mol)`** *(deprecated)*: Older 5-stage pipeline; does not handle hypervalence or Kekulization.
- **`clean(mol)`**: Main cleaning method.
  - Stage 1: Keeps largest fragment; normalizes charges, tautomers, stereochemistry.
  - Stage 2: Quick exit if no valence errors.
  - Stage 3: Remove excess H atoms; re-check valence.
  - Stage 4 (Kekule): If Kekulization fails, serializes SMILES and calls `fix_kekulization_modular`.
  - Stage 5 (hypervalence): Finds overvalent atoms; applies `try_neutralise_common_hypervalence` and re-checks.
  - Gives up and returns `(None, log)` or `None` if unfixable.

#### `clean_molecule(mol, preserve_aromaticity, verbose) -> Mol` *(in graph_cleaning.py)*
Thin wrapper: creates `MoleculeCleaner` and calls `.clean(mol)`. Raises `ValueError` if cleaning fails.

---

### `chem/dummy/`

Management of dummy atoms (attachment points, atomic_num=0).

**Canonical rules:**
- A dummy is any atom with `arrays.atomic_num == 0`
- Its label lives **only** in `arrays.attachment_label[atom_idx]`
- `MolGraph.attachments` and `meta["label_to_index"]` are derived/cached, rebuilt on demand

---

#### `chem/dummy/query.py`

Read-only helpers.

#### `dummy_indices(g) -> np.ndarray`
Returns indices of all atoms where `atomic_num == 0`.

#### `dummy_labels(g) -> np.ndarray`
Returns object array of labels for each dummy index; unlabeled dummies get `None`.

#### `dummy_by_label(g, label) -> Optional[int]`
Scans dummy atoms for a matching label (string-compared); returns atom index or `None`.

#### `neighbors(g, atom_idx) -> np.ndarray`
Returns neighbor indices by scanning both columns of `bonds`. Works correctly for atoms appearing in either bond direction.

#### `dummy_target(g, dummy_idx) -> Optional[int]`
Returns the single non-dummy neighbor if there is exactly one; else `None`. Used for substituent-type dummies.

#### `is_insertion_dummy(g, dummy_idx) -> bool`
Returns `True` if the dummy has 2+ neighbors (insertion site heuristic).

---

#### `chem/dummy/edit.py`

Canonical mutations of dummy labels.

#### `_ensure_attachment_label_array(g)`
If `arrays.attachment_label` is `None`, creates an object array of `None`s of length N.

#### `set_dummy_label(g, idx, label) -> MolGraph`
Sets `arrays.attachment_label[idx] = label`. Raises if atom is not a dummy.

#### `clear_dummy_label(g, idx) -> MolGraph`
Sets `arrays.attachment_label[idx] = None`.

#### `relabel_dummy(g, old_label, new_label) -> MolGraph`
Finds dummy with `old_label`, checks `new_label` doesn't already exist, calls `set_dummy_label`. Raises on conflicts.

#### `enforce_label_invariant(g) -> MolGraph`
Sets `attachment_label = None` for all non-dummy atoms. Enforces canonical invariant.

---

#### `chem/dummy/derive.py`

Builds derived `AttachmentPointArray` from canonical arrays.

#### `build_attachment_points(g) -> Optional[AttachmentPointArray]`
- **Input:** `MolGraph`
- **Output:** `AttachmentPointArray` or `None` if no dummies
- **How:** Collects dummy indices, builds `kind` array (all DUMMY), computes `target` via `dummy_target` for each dummy (−1 if ambiguous), collects labels from `dummy_labels`, sets `is_insertion` flag via `neighbors(g, di).size >= 2`.

---

#### `chem/dummy/clean.py`

Validation and cleanup helpers.

#### `remove_orphan_dummies_mark_only(g) -> MolGraph`
Finds dummy atoms with degree 0 (not connected to any bond) and records their indices in `g.meta["orphan_dummies"]`. Does not delete anything.

#### `validate_dummy_invariants(g) -> List[str]`
Checks that no non-dummy atoms have attachment labels (label leakage invariant). Returns warning strings.

---

### `chem/edit/`

Low-level structural mutation of `MolGraph` arrays.

---

#### `chem/edit/atoms.py`

Atom-level mutations. All functions invalidate derived caches after mutation.

#### `_invalidate_derived(g) -> MolGraph`
Clears `attachments`, `cache` (edge_index, x, edge_attr, bond_pair_to_index), sets dirty flags, clears `rdkit_mol` and `meta["label_to_index"]`.

#### `_slice_optional(x, mask)`, `_slice_pos(pos, mask)`, `_slice_coord_valid(cv, mask)`
Safe slicing of optional arrays; handle both single-conformer `(N,3)` and ensemble `(K,N,3)` pos shapes.

#### `_permute_pos(pos, new_to_old)`, `_permute_coord_valid(cv, new_to_old)`
Permute pos/coord_valid by a `new_to_old` index array; handle both 2D and 3D pos.

#### `_as_unique_sorted_atom_indices(atom_indices, n_atoms) -> np.ndarray`
Validates and deduplicates atom index list; raises `IndexError` on out-of-bounds.

#### `_compute_old_to_new_index_map(n_atoms, delete_idx) -> (keep_mask, new_index_of_old)`
Returns a boolean keep mask and a map from old atom index → new index (−1 for deleted atoms).

#### `_remap_fragment(f, new_index_of_old) -> Fragment`
Remaps `atom_indices` and `attachment_indices` of a `Fragment` through `new_index_of_old`, dropping deleted atoms.

#### `_remap_resonance_system(rs, new_index_of_old_atoms, new_index_of_old_bonds) -> ResonanceSystem`
Remaps atom and bond indices in a `ResonanceSystem`.

#### `delete_atom(g, atom_index, return_maps=False)`
Single-atom deletion; delegates to `delete_atoms`.

#### `delete_atoms(g, atom_indices, return_maps=False)`
- **Input:** `MolGraph`, list of atom indices
- **Output:** Modified `MolGraph` (and optionally `old_to_new`, `new_to_old` maps)
- **How:** Builds keep mask, remaps all atom-aligned arrays via slicing, filters bonds that touch deleted atoms, remaps bond endpoints, slices optional bond arrays, rebuilds `MolArrays`, remaps editability/fragments/resonance_systems, invalidates caches.

#### `bond_scores_len_ok(editability, m_bonds) -> bool`
Guard: checks whether `editability.bond_scores` has length M (safe for cases where it wasn't set consistently).

#### `permute_atoms(g, new_to_old) -> (g, old_to_new, bond_perm)`
- **Input:** `MolGraph`, permutation array `new_to_old` (length N)
- **Output:** Reindexed `MolGraph`, inverse map, bond sort permutation
- **How:** Validates permutation, computes `old_to_new`, permutes all atom arrays, remaps bond endpoints → normalizes (min,max) → stable sort by (u,v) → `bond_perm`. Permutes all bond arrays. Remaps editability/fragments/resonance_systems.

#### `add_atom(g, *, atomic_num, formal_charge=0, **fields) -> (g, new_idx)`
- **Input:** `MolGraph`, required atom fields, optional kwargs for existing optional arrays
- **Output:** Modified `MolGraph`, new atom index
- **How:** Appends to `atomic_num` and `formal_charge`; for each existing optional array appends with provided value or default (0 for numerics, `None` for labels). Handles pos for both `(N,3)` and `(K,N,3)`. Recomputes `total_charge`. Invalidates caches.

#### `replace_atom(g, idx, *, preserve_attachment_label=True, **fields) -> MolGraph`
Updates atom properties at `idx` in-place. Only updates fields provided in `**fields`. Enforces canonical rule: `attachment_label` is cleared for non-dummy atoms.

#### `set_dummy_label(g, idx, label) -> MolGraph`
Sets canonical dummy attachment label at `idx`. Raises if not a dummy or if `attachment_label` array is `None`.

---

#### `chem/edit/bonds.py`

Bond-level mutations.

#### `_normalize_pair(u, v) -> (a, b)`
Returns `(min, max)` of two atom indices, matching canonical bond storage order.

#### `_find_bond_indices_by_pair(bonds, u, v) -> np.ndarray`
Returns all bond indices matching `(u,v)` or `(v,u)` in both directions (robust match).

#### `_append_optional(x, value, default) -> Optional[np.ndarray]`
Appends one element to an optional bond array; returns `None` if array is `None` (does not auto-create).

#### `delete_bond(g, u, v, undirected=True) -> MolGraph`
Finds bond between atoms u and v, delegates to `delete_bonds`. Raises if bond not found.

#### `delete_bonds(g, bond_indices) -> MolGraph`
- **Input:** `MolGraph`, list of bond indices
- **Output:** Modified `MolGraph`
- **How:** Builds keep mask, slices `bonds`, `bond_type`, and all optional bond arrays, remaps resonance system bond indices, updates editability if aligned, invalidates caches.

#### `add_bond(g, u, v, *, bond_type, **fields) -> MolGraph`
- **Input:** `MolGraph`, atom pair, bond_type code, optional bond field values
- **Output:** Modified `MolGraph`
- **How:** Validates indices, normalizes pair, checks for duplicate bond (via cache or scan), appends to `bonds` and `bond_type`, appends to all existing optional arrays with provided or default values, extends editability bond_scores if aligned.

#### `edit_bond(g, u, v, *, bond_type=None, ...) -> MolGraph`
Updates attributes of an existing bond (u,v) in-place. Only modifies fields provided. Uses cache for fast lookup if available.

---

### `chem/functional_groups/`

Functional group detection via AccFG (532 groups).

---

#### `chem/functional_groups/match.py`

Result dataclasses.

#### `FunctionalGroupMatch`
- **Fields:** `name`, `atom_indices` (tuple of ints), `key_atoms` (dict name→index), `parent` (parent FG name), `children` (list), `smarts`
- `__getattr__`: Allows `match.carbonyl_c` style access into `key_atoms`
- `indices_array -> np.ndarray`: Returns atom_indices as int64 array
- `contains_atom(idx) -> bool`: Checks membership
- `overlaps_with(other) -> bool`: Set intersection of atom_indices

#### `FunctionalGroupResult`
Container for all detection results on a molecule.
- **Fields:** `matches` (list of `FunctionalGroupMatch`), `hierarchy` (AccFG DiGraph, optional), `smiles`
- `by_name(name) -> List[FunctionalGroupMatch]`: Case-insensitive lookup
- `find_one(name, allow_multiple=False) -> FunctionalGroupMatch`: Single-match accessor; raises if 0 or >1
- `has(name) -> bool`: Boolean existence check
- `count(name) -> int`: Count occurrences
- `names -> List[str]` (property): Unique FG names in order of first occurrence
- `atoms_to_groups() -> Dict[int, List[str]]`: Inverts matches to per-atom FG lists
- `non_fg_atoms(n_atoms) -> np.ndarray`: Returns indices of atoms not in any FG

---

#### `chem/functional_groups/detect.py`

Wraps AccFG for detection, with lazy-loaded cached instances.

#### `_get_accfg(lite, user_defined_fgs) -> AccFG`
Returns cached `AccFG` instance (separate caches for lite/full). If `user_defined_fgs` provided, always creates a new instance.

#### `get_accfg_instance(lite=False) -> AccFG`
Public accessor to the raw AccFG instance for advanced usage (inspecting `dict_fgs`, calling `run()` directly, etc.).

#### `get_smarts_for_fg(fg_name, lite=False) -> Optional[str]`
Returns SMARTS pattern for a named functional group.

#### `print_fg_tree(fg_graph, root_names, show_atom_idx=True)`
Prints AccFG hierarchy as ASCII tree. Falls back to simple print if AccFG draw module unavailable.

#### `detect_functional_groups(smiles, *, lite, user_defined_fgs, include_hierarchy, canonical) -> FunctionalGroupResult`
- **Input:** SMILES string, flags
- **Output:** `FunctionalGroupResult`
- **How:** Gets AccFG instance, calls `afg.run(smiles, show_atoms=True, show_graph=True)`, converts output dict `{name: [(atom_tuple,...), ...]}` to `FunctionalGroupMatch` objects with hierarchy info from the DiGraph predecessors/successors.

#### `detect_from_molgraph(mg, *, lite, user_defined_fgs, include_hierarchy) -> FunctionalGroupResult`
Converts `MolGraph` to SMILES (non-canonical to preserve atom ordering), then calls `detect_functional_groups` with `canonical=False`.

#### `detect_from_mol(mol, *, lite, user_defined_fgs, include_hierarchy) -> FunctionalGroupResult`
Uses `afg.run_mol()` directly on an RDKit `Mol`.

#### `compare_molecules(smiles_a, smiles_b, similarity_threshold=0.7) -> Tuple`
Calls `accfg.compare_mols`; returns FG differences between two molecules.

#### `draw_mol_with_fgs(smiles, *, lite, user_defined_fgs, with_legend, with_atom_idx, img_size) -> bytes`
Returns PNG bytes of molecule with highlighted functional groups.

#### `draw_compare_mols(smiles_a, smiles_b, *, lite, img_size) -> List[PIL.Image]`
Returns side-by-side PIL images showing FG differences.

**Convenience helpers (SMILES or MolGraph → List[FunctionalGroupMatch]):**
- `find_carboxylic_acids(smiles_or_mg, *, lite)`: Returns carboxylic acid matches
- `find_amines(smiles_or_mg, *, primary_only, lite)`: Returns amine matches (optionally primary only)
- `find_halides(smiles_or_mg, *, alkyl_only, lite)`: Returns halide matches (alkyl chloride/bromide/iodide/fluoride, optionally aryl too)
- `find_alcohols(smiles_or_mg, *, lite)`: Returns primary/secondary/tertiary hydroxyl matches
- `list_all_functional_groups(lite) -> List[str]`: All 532 FG names
- `search_functional_groups(query, lite) -> List[str]`: Case-insensitive substring search over FG names

---

### `chem/merge/merge.py`

NumPy-first merge of two `MolGraph`s at labeled dummy attachment sites.

**Supports two modes:**
- **Substituent merge:** Each dummy has exactly 1 heavy neighbor; dummies removed, anchor atoms bonded directly
- **Insertion merge:** One side has 2 heavy neighbors (insertion), other is substituent (1 heavy neighbor)

#### `_SiteInfo`
Dataclass (slots) holding resolved site info: `label`, `dummy_idx`, `heavy_neighbors`, `h_neighbors`, `mode`.

#### `resolve_site(g, label) -> _SiteInfo`
- **Input:** `MolGraph`, attachment label
- **Output:** `_SiteInfo`
- **How:** Finds dummy by label, inspects neighbors to classify as substituent (1 heavy) or insertion (2 heavy). Raises for 0 or 3+ heavy neighbors.

#### `_delete_atoms_with_map(g, atom_indices) -> (MolGraph, old_to_new)`
Local atom deletion that also returns the `old_to_new` mapping. Handles slicing all optional arrays, bond filtering, fragment remapping. Sets `pos`/`coord_frame`/`coord_valid` to `None` (geometry is invalidated by structural changes).

#### `_maybe_remove_one_explicit_h(g, anchor_idx) -> (MolGraph, old_to_new)`
If anchor atom has any explicit H node neighbor (atomic_num=1), deletes one such H to make room for the incoming bond.

#### `_union_atom_optional(a, b, n_a, n_b, *, default_value, dtype_if_missing, object_array) -> Optional[np.ndarray]`
Concatenates two optional atom arrays; if one is `None` but the other isn't, fills the missing side with `default_value` to maintain alignment.

#### `_union_bond_optional(a, b, m_a, m_b, *, default_value, dtype_if_missing) -> Optional[np.ndarray]`
Same as above for bond-aligned optional arrays.

#### `merge_by_labels(g_a, g_b, label_a, label_b, bond_type_code, validate_with_rdkit, on_rdkit_fail, coords, ...) -> MolGraph`
*(Main entry point — continued in second half of file not shown in full, but behavior documented here)*
- **Input:** Two `MolGraph`s, labels identifying dummy sites on each, bond type for the new bond, validation/coordinate options
- **Output:** Merged `MolGraph`
- **How:**
  1. Resolves attachment sites on both graphs
  2. Removes dummy atoms and any explicit H on anchors (via `_delete_atoms_with_map` + `_maybe_remove_one_explicit_h`)
  3. Computes atom index offsets for concatenation
  4. Concatenates all MolArrays fields (required + optional) using `_union_atom_optional`/`_union_bond_optional`
  5. Adds the new bond between the two anchor atoms
  6. Handles insertion mode by reconnecting the two anchors if needed
  7. Invalidates all derived caches, writes to `meta["merge_log"]`
  8. Optionally validates with RDKit and computes canonical SMILES

---

### `chem/ops/`

Transform wrappers that make each edit operation loggable and pipeline-composable.

---

#### `chem/ops/base.py`

Infrastructure for the Transform system.

#### `ChangeSummary`
Dataclass: before/after atom and bond counts; optional index-level info (`atoms_removed_old`, `atoms_added_new`, `bonds_removed_old`, `bonds_added_new`); freeform notes.

#### `OpInfo`
Standard payload returned by every `Transform.apply()`. Fields: `op` (name string), `params` (dict), `warnings` (list), `old_to_new`, `new_to_old`, `bond_perm` (optional index maps), `changes` (ChangeSummary), `t_unix` (timestamp).

#### `append_op_log(mg, info)`
Appends an `OpInfo` entry to `mg.meta["op_log"]` (creates list if absent). Converts numpy arrays to int64 for storage.

#### `Transform(ABC)`
Abstract base class. Requires `apply(mg) -> (mg2, OpInfo)`. Optional `inverse(info) -> Optional[Transform]`.

#### `apply_transform(mg, op) -> (mg2, OpInfo)`
Convenience runner: calls `op.apply(mg)`, fills `ChangeSummary` if missing, enforces `info.op == op.name`, appends to `mg2.meta["op_log"]`.

---

#### `chem/ops/atoms.py`

Transform wrappers for atom edits (each is a `@dataclass` subclassing `Transform`).

#### `DeleteAtom(idx)`
Wraps `delete_atoms([idx], return_maps=True)`. Records `atoms_removed_old` in `ChangeSummary`.

#### `AddAtom(atomic_num, formal_charge, isotope, aromatic)`
Wraps `add_atom(...)`. Records `atoms_added_new` with the new atom index.

#### `ReplaceAtom(idx, atomic_num, ...)`
Wraps `replace_atom(...)`. Notes "atom attributes replaced".

#### `SetDummyLabel(idx, label)`
Wraps `set_dummy_label(...)`. Notes "dummy label updated".

#### `PermuteAtoms(new_to_old)`
Wraps `permute_atoms(...)`. Records `old_to_new`, `new_to_old`, `bond_perm` in `OpInfo`.

---

#### `chem/ops/bonds.py`

Transform wrappers for bond edits.

#### `AddBond(u, v, bond_type, is_conjugated, is_in_ring, bond_dir, bond_stereo, bond_resonance_type)`
Wraps `add_bond(...)`. Atom index maps are identity (no atoms removed/added).

#### `DeleteBond(u, v, undirected=True)`
Wraps `delete_bond(u, v)`. Atom maps are identity.

#### `DeleteBonds(bond_indices)`
Wraps `delete_bonds(...)`. Records `bonds_removed_old`.

#### `EditBond(u, v, bond_type, is_conjugated, ...)`
Wraps `edit_bond(...)`. Only modifies provided fields. Atom maps are identity.

---

#### `chem/ops/merge.py`

Transform wrapper for merge.

#### `MergeByLabels(g_b, label_a, label_b, bond_type_code, validate_with_rdkit, on_rdkit_fail, coords, ...)`
- Wraps `merge_by_labels(g_a, g_b, ...)` where `g_a` is the `apply()` input and `g_b` is stored in the op.
- `ChangeSummary.n_atoms_before = nA + nB` (both inputs counted). Index maps left `None` (two-input ops don't produce a single mapping).
- Pulls any warnings from `merged.meta["merge_log"]` into `OpInfo.warnings`.

---

#### `chem/ops/pipeline.py`

Higher-order Transform combinators.

#### `Compose(ops)`
Sequentially applies a list of Transforms via `apply_transform`. Each sub-op logs itself. Returns summary `OpInfo` with list of op names and collected warnings.

#### `Sequential(ops)`
Alias for `Compose` (common ML naming).

#### `Repeat(op, k)`
Applies one Transform `k` times. Raises if `k < 0`.

#### `Maybe(op, p=0.5, seed=None)`
Applies op with probability `p`; otherwise no-op. Uses `np.random.default_rng(seed)` for reproducibility.

#### `RandomChoice(ops, weights=None, seed=None)`
Selects one op uniformly or by normalized weights, applies it. Logs `choice_idx` and `choice_op`.

#### `Conditional(predicate, then_op, else_op=None)`
Calls `predicate(mg)` to decide between `then_op` and `else_op`. Logs the condition result and which op ran.

#### `StopOnWarning(op)`
Runs op; if any warnings were emitted, reverts to input graph and returns a "skipped" OpInfo. Useful for safe augmentation pipelines.
