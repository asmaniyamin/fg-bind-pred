# MM/GBSA preprocessing

Single-snapshot MM/GBSA calculations on prepared CIF complexes using AmberTools.

## Components

- `run_mmpbsa.py`: receptor/ligand preparation, topology generation, baseline MM/GBSA, and energy decomposition. Configured for generalized Born (GB), without a Poisson-Boltzmann calculation.
- `rerun_subset.py`: run selected PDB IDs with configurable thread limits and ligand selection.
- `structure_repair.py`: structure repair helpers used by the pipeline.

## Running calculations

Use a scientific environment with the required AmberTools executables and Python dependencies. Place prepared structures at `mmgbsa/<pdb_id>.cif`, with lowercase PDB IDs.

From the repository root:

```sh
python mmgbsa/rerun_subset.py <pdb_id> --threads-per-complex 2 --max-total-threads 16
```

To process every CIF in `mmgbsa/`:

```sh
python mmgbsa/run_mmpbsa.py --max-total-threads 16
```

Outputs are written under `mmgbsa/mmpbsa_work/<pdb_id>/`, including `FINAL_RESULTS_MMGBSA*` and `FINAL_DECOMP_MMGBSA*` files. Keep prepared structures and generated results outside version control.

For per-functional-group masking, see `scripts/run_per_fg_masked_gbsa_batch.py` and its command-line arguments. No fixed local testing cohort is bundled.
