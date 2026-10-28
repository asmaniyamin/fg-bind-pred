# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/extract/_pocket.py

Pocket geometry, pocket JSON payload, SDF writer, and the shared extraction
orchestrator used by both the unprocessed and processed extractors.

The orchestrator (:func:`run_pocket_extraction`) parameterizes the only two
parts that actually differ between the two paths: the ligand-builder and the
grouping-bond filter. Everything else - bio-assembly expansion, candidate
detection, pocket geometry, SDF writing, JSON/manifest emission - is shared.

Functions are pure except :func:`write_ligand_sdf` and
:func:`run_pocket_extraction`, which do disk I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import traceback
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from rdkit import Chem

from . import _core
from . import _ligands
from ._core import (
    AtomRecord,
    DEFAULT_OUTPUT_SUFFIX,
    LigandGroup,
    ResidueKey,
    StructConnLink,
    collect_residue_atoms,
)
from ._search import (
    PocketSearchIndex,
    build_pocket_search_index,
    find_pocket_residues,
    minimum_distance_between_sets,
)


# Pocket residue search lives in ``_search``. The orchestrator below uses
# the imported ``PocketSearchIndex`` / ``build_pocket_search_index`` /
# ``find_pocket_residues`` symbols re-exported above so existing call sites
# (and external imports of those names from this module) keep working.


# ---------------------------------------------------------------------------
# Pocket JSON payload + ligand SDF writer
# ---------------------------------------------------------------------------

def build_pocket_payload(
    *,
    ligand_group: LigandGroup,
    ligand_mol: Chem.Mol,
    ligand_atom_records: Sequence[AtomRecord],
    pocket_residues: Sequence[dict[str, object]],
    component_metadata: dict[str, dict[str, str]],
    ligand_sdf_path: Path,
    pocket_radius: float,
) -> dict[str, object]:
    """Build a JSON-serializable payload for one extracted ligand pocket."""
    component_names = sorted({record.residue_key.resname for record in ligand_atom_records})
    component_descriptions = {
        name: component_metadata.get(name, {})
        for name in component_names
    }

    return {
        "ligand_id": ligand_group.ligand_id,
        "ligand_residues": [key.to_dict() for key in ligand_group.residue_keys],
        "ligand_components": component_descriptions,
        "ligand_sdf": str(ligand_sdf_path),
        "ligand_atom_count": len(ligand_atom_records),
        "ligand_bond_count": ligand_mol.GetNumBonds(),
        "ligand_atoms": [record.to_dict() for record in ligand_atom_records],
        "ligand_bonds": ligand_bonds_to_dicts(ligand_mol, ligand_atom_records),
        "cofactor_family": ligand_group.cofactor_family,
        "categories": ligand_group.categories,
        "pocket_radius": pocket_radius,
        "pocket_residue_count": len(pocket_residues),
        "pocket_residues": list(pocket_residues),
    }


def ligand_bonds_to_dicts(
    ligand_mol: Chem.Mol,
    ligand_atom_records: Sequence[AtomRecord],
) -> list[dict[str, object]]:
    """Serialize ligand bonds with atom names and bond order."""
    atom_records = list(ligand_atom_records)
    bond_entries: list[dict[str, object]] = []
    for bond in ligand_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        begin_atom = atom_records[begin_idx]
        end_atom = atom_records[end_idx]
        bond_entries.append(
            {
                "begin_atom_index": begin_idx,
                "end_atom_index": end_idx,
                "begin_atom_name": begin_atom.atom_name,
                "end_atom_name": end_atom.atom_name,
                "begin_residue": begin_atom.residue_key.to_dict(),
                "end_residue": end_atom.residue_key.to_dict(),
                "bond_type": str(bond.GetBondType()),
                "aromatic": bool(bond.GetIsAromatic()),
            }
        )
    return bond_entries


def write_ligand_sdf(mol: Chem.Mol, output_path: Path) -> None:
    """Write one ligand molecule to SDF."""
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()


# ---------------------------------------------------------------------------
# Shared extraction orchestrator
# ---------------------------------------------------------------------------

LigandBuilder = Callable[
    [LigandGroup, Sequence[object], Sequence[StructConnLink]],
    tuple[Chem.Mol, list[AtomRecord], str | None],
]
LigandBuilderFactory = Callable[[dict[str, object]], LigandBuilder]


# ---------------------------------------------------------------------------
# Per-ligand error isolation, resume, and failure manifest helpers.
#
# These three helpers exist so the orchestrator's loop body stays small and
# the failure / skip code paths are unit-testable in isolation (no mmCIF
# round-trip required).
# ---------------------------------------------------------------------------

def _build_failure_record(
    ligand_group: LigandGroup,
    exc: BaseException,
) -> dict[str, object]:
    """Capture a per-ligand failure for ``failed_ligands.json``.

    Includes the exception class name, message, and full traceback so a
    failed batch can be triaged without re-running anything. The
    traceback is built from ``exc.__traceback__`` rather than
    ``traceback.format_exc()`` because the latter only works inside the
    active ``except`` frame.
    """
    return {
        "ligand_id": ligand_group.ligand_id,
        "residues": [key.to_dict() for key in ligand_group.residue_keys],
        "cofactor_family": ligand_group.cofactor_family,
        "categories": list(ligand_group.categories),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _build_pipeline_failure_record(
    exc: BaseException,
    phase: str,
) -> dict[str, object]:
    """Capture a whole-pipeline failure (pre-loop or orchestrator-level).

    Records the failing ``phase``, exception type, message, and traceback
    for errors during structure loading, assembly expansion, search-index
    construction, or ligand detection.
    """
    return {
        "phase": phase,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


# ---------------------------------------------------------------------------
# Manifest checksums and tool versions for reproducibility.
#
# Input checksums prevent --resume from reusing results for modified inputs.
# Dependency versions identify the environment that produced the manifest.
# Increment MANIFEST_SCHEMA_VERSION when the manifest structure changes
# incompatibly; adding values within an existing field does not require it.
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = 1

# Tool versions worth recording. Order is the order they appear in the
# manifest; missing modules degrade to ``"not_installed"`` rather than
# being silently dropped, so ``--resume`` can detect a switched-out
# environment.
_VERSIONED_MODULES: tuple[str, ...] = ("rdkit", "scipy", "gemmi", "Bio", "numpy")

# Hash chunk size - 64 KiB. Small enough to keep peak memory flat for
# multi-GB CIFs (cryo-EM bio-assemblies routinely cross 100 MB), large
# enough that the syscall overhead is negligible.
_HASH_CHUNK_BYTES = 1 << 16


def _sha256_file(path: Path) -> str:
    """Stream-hash a file with a fixed buffer so peak memory stays flat."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_tool_versions() -> dict[str, str]:
    """Snapshot the ``__version__`` of every dep that shapes the output.

    Missing modules become ``"not_installed"`` rather than raising or
    being silently dropped - that way a manifest produced without
    openbabel still records *that* fact, and a resume from a different
    environment can see the difference rather than guessing.
    """
    versions: dict[str, str] = {"python": platform.python_version()}
    for mod_name in _VERSIONED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            versions[mod_name] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            versions[mod_name] = "not_installed"
    # OpenBabel exposes its version through a function call rather than
    # ``__version__``; degrade cleanly when the binding is missing.
    try:
        from openbabel import openbabel as ob  # type: ignore[import-not-found]
        versions["openbabel"] = str(ob.OBReleaseVersion())
    except Exception:
        versions["openbabel"] = "not_installed"
    return versions


def _build_environment_metadata(input_path: Path) -> dict[str, object]:
    """One-time manifest envelope: input checksum + tool versions + schema tag.

    Computed once at the top of ``run_pocket_extraction`` and threaded
    into every ``_write_outputs`` call via the orchestrator's closure,
    so the manifest stays self-describing even when a partial run is
    flushed by the ``finally`` clause after a crash.

    ``kdtree_leafsize=16`` mirrors the literal in
    :mod:`protein.extract._search` - recording it here means a future
    leafsize change shows up in the manifest envelope, not just in
    ``git log``.
    """
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "input_checksum_sha256": _sha256_file(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "kdtree_leafsize": 16,
        "versions": _collect_tool_versions(),
    }


def _check_resume_compatibility(
    output_dir: Path,
    environment_metadata: dict[str, object],
    *,
    force_resume: bool,
) -> None:
    """Refuse to ``--resume`` when the input mmCIF has changed since the prior run.

    The whole point of resume is to skip ligands whose pocket JSON +
    SDF already exist on disk and reuse the cached extraction. That's
    only sound if the input the cache was built from is byte-identical
    to the input now sitting at ``args.input_path``. If the user edited
    the CIF (or PrepWizard re-ran with different options and clobbered
    the prepped output), reusing the prior pocket JSONs would silently
    mix two extractions.

    The check compares the prior manifest's ``input_checksum_sha256``
    against the current one. Mismatch → :class:`PocketExtractionError`
    with a clear message and a ``--force-resume`` escape hatch for
    operators who *do* want to reuse stale caches deliberately.

    Tolerates two common cases without raising: (a) no prior manifest
    exists (fresh output_dir), and (b) prior manifest lacks an ``environment`` envelope. Both fall back to "no checksum to
    compare against, proceed."
    """
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt prior manifest is itself a reason to be cautious, but
        # not something to raise about - the resume loop will just see
        # no prior pocket JSONs to skip if they're gone, or cleanly
        # overwrite if they're present. Don't block.
        return

    prior_env = prior_manifest.get("environment")
    if not isinstance(prior_env, dict):
        return  # No environment metadata to compare.
    prior_checksum = prior_env.get("input_checksum_sha256")
    if not prior_checksum:
        return

    current_checksum = environment_metadata.get("input_checksum_sha256")
    if prior_checksum == current_checksum:
        return

    if force_resume:
        return

    raise _core.PocketExtractionError(
        f"--resume refused: input mmCIF checksum changed since the prior run.\n"
        f"  prior:   {prior_checksum}\n"
        f"  current: {current_checksum}\n"
        f"Re-extract from scratch, or pass --force-resume to reuse the prior "
        f"pocket JSONs anyway."
    )


def _write_outputs(
    *,
    output_dir: Path,
    input_path: Path,
    args: argparse.Namespace,
    manifest_entries: Sequence[dict[str, object]],
    failed_entries: Sequence[dict[str, object]],
    skipped_count: int,
    pipeline_failure: dict[str, object] | None,
    environment_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write ``manifest.json`` + ``failed_ligands.json`` to ``output_dir``.

    Idempotent and safe to call repeatedly: every successful ligand,
    once at the end of the loop, and once more in the ``finally`` clause
    of :func:`run_pocket_extraction` all call this helper. The cost is
    one ``json.dumps`` + one ``write_text`` per call - small for a
    typical PDB and bounded by ``ligand_count`` for any batch - and the
    payoff is that a SIGKILL or KeyboardInterrupt mid-loop leaves the
    on-disk manifest reflecting whichever ligands had finished, not
    silently nothing.

    The ``failed_ligands_path`` field in the manifest is intentionally a
    bare filename (sibling of ``manifest.json``), not an absolute path,
    so moving the output directory doesn't break the pointer.
    """
    failed_ligands_path = output_dir / "failed_ligands.json"
    failed_ligands_path.write_text(
        json.dumps(list(failed_entries), indent=2), encoding="utf-8"
    )

    manifest: dict[str, object] = {
        "input_mmcif": str(input_path),
        "output_dir": str(output_dir),
        "model_index": args.model_index,
        "pocket_radius": args.radius,
        "ligand_count": len(manifest_entries),
        "failed_ligand_count": len(failed_entries),
        "skipped_ligand_count": int(skipped_count),
        "failed_ligands_path": failed_ligands_path.name,
        "ligands": list(manifest_entries),
    }
    if environment_metadata is not None:
        manifest["environment"] = dict(environment_metadata)
    if pipeline_failure is not None:
        manifest["pipeline_failure"] = pipeline_failure

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _build_skip_entry(
    ligand_group: LigandGroup,
    pocket_json_path: Path,
    ligand_sdf_path: Path,
) -> dict[str, object]:
    """Reconstruct a manifest entry from an existing pocket JSON for ``--resume``.

    Reads the prior pocket JSON and pulls back the manifest-relevant
    summary fields (counts, cleanup, validation, bond source) so the
    resumed run's manifest is shape-identical to a fresh run, just with
    ``status="skipped"`` flagging that this ligand wasn't re-extracted
    in the current invocation.

    Tolerates missing fields in pocket JSONs without validation metadata - every
    field is read with ``payload.get(...)`` with sensible defaults.
    """
    payload = json.loads(pocket_json_path.read_text(encoding="utf-8"))
    entry: dict[str, object] = {
        "ligand_id": ligand_group.ligand_id,
        "residues": [key.to_dict() for key in ligand_group.residue_keys],
        "ligand_sdf": str(ligand_sdf_path),
        "pocket_json": str(pocket_json_path),
        "pocket_residue_count": payload.get("pocket_residue_count", 0),
        "ligand_atom_count": payload.get("ligand_atom_count", 0),
        "ligand_bond_count": payload.get("ligand_bond_count", 0),
        "cofactor_family": payload.get("cofactor_family"),
        "categories": payload.get("categories", []),
        "ligand_cleanup": payload.get("ligand_cleanup"),
        "validation": payload.get("validation"),
        "status": "skipped",
    }
    if "bond_source" in payload:
        entry["bond_source"] = payload["bond_source"]
    return entry


def _extract_one_ligand(
    *,
    ligand_group: LigandGroup,
    residue_lookup: dict[ResidueKey, object],
    search_index: PocketSearchIndex,
    build_ligand: LigandBuilder,
    struct_conn_links: Sequence[StructConnLink],
    component_metadata: dict[str, dict[str, str]],
    args: argparse.Namespace,
    ligand_sdf_path: Path,
    pocket_json_path: Path,
) -> dict[str, object]:
    """Build → cleanup → validate → pocket → write for one ligand.

    Returns the manifest entry. Raises whatever the underlying steps raise
    (``PocketExtractionError``, RDKit/gemmi exceptions, etc.). The
    orchestrator's try/except decides whether to abort or record-and-continue.

    Side effects: writes ``ligand_sdf_path`` and ``pocket_json_path`` to disk.
    On failure, neither file is guaranteed to exist; the resume path treats
    a missing pocket JSON as "needs re-extraction" and re-runs cleanly.
    """
    ligand_residues = [residue_lookup[key] for key in ligand_group.residue_keys]
    # Reuse the cached per-residue atom records from the search index
    # instead of re-running collect_residue_atoms over the ligand residues.
    ligand_atoms: list[AtomRecord] = []
    for key in ligand_group.residue_keys:
        ligand_atoms.extend(search_index.residue_atoms[key])

    ligand_mol, atom_records, bond_source = build_ligand(
        ligand_group, ligand_residues, struct_conn_links
    )
    ligand_mol, cleanup_info = _ligands.clean_and_validate_ligand(ligand_mol)

    sanitization_info = _core.validate_ligand_sanitization(ligand_mol)
    completeness_info = _core.audit_atom_completeness(ligand_mol, atom_records)
    coord_sanity_info = _core.audit_coordinate_sanity(
        ligand_atoms=atom_records,
        protein_kdtree=search_index.protein_kdtree,
        min_contact_floor=getattr(
            args, "coord_clash_floor", _core.DEFAULT_COORD_CLASH_FLOOR
        ),
        max_contact_ceiling=getattr(
            args, "coord_orphan_ceiling", _core.DEFAULT_COORD_ORPHAN_CEILING
        ),
    )
    validation_info: dict[str, object] = {
        "sanitization": sanitization_info,
        "atom_completeness": completeness_info,
        "coordinate_sanity": coord_sanity_info,
    }
    if getattr(args, "strict_validity", False):
        failures: list[str] = []
        if not sanitization_info["sanitized"]:
            failures.append(
                f"sanitization failed: {sanitization_info['failed_operations']}"
            )
        if not completeness_info["complete"]:
            failures.append(
                f"atom-completeness failed: "
                f"{completeness_info['missing_count']} atoms missing"
            )
        if not coord_sanity_info["coordinate_sanity_ok"]:
            failures.append(
                f"coordinate sanity failed: "
                + "; ".join(coord_sanity_info["issues"])
            )
        if failures:
            raise _core.PocketExtractionError(
                f"Ligand {ligand_group.ligand_id} failed strict validity gate: "
                + "; ".join(failures)
            )

    pocket_residues = find_pocket_residues(
        search_index=search_index,
        ligand_group=ligand_group,
        ligand_atoms=ligand_atoms,
        radius=args.radius,
        include_hetero=args.include_hetero_pocket,
        include_water=args.include_water,
    )

    write_ligand_sdf(ligand_mol, ligand_sdf_path)

    pocket_payload = build_pocket_payload(
        ligand_group=ligand_group,
        ligand_mol=ligand_mol,
        ligand_atom_records=atom_records,
        pocket_residues=pocket_residues,
        component_metadata=component_metadata,
        ligand_sdf_path=ligand_sdf_path,
        pocket_radius=args.radius,
    )
    if bond_source is not None:
        pocket_payload["bond_source"] = bond_source
    pocket_payload["ligand_cleanup"] = cleanup_info
    pocket_payload["validation"] = validation_info
    pocket_json_path.write_text(
        json.dumps(pocket_payload, indent=2), encoding="utf-8"
    )

    entry: dict[str, object] = {
        "ligand_id": ligand_group.ligand_id,
        "residues": [key.to_dict() for key in ligand_group.residue_keys],
        "ligand_sdf": str(ligand_sdf_path),
        "pocket_json": str(pocket_json_path),
        "pocket_residue_count": len(pocket_residues),
        "ligand_atom_count": len(atom_records),
        "ligand_bond_count": ligand_mol.GetNumBonds(),
        "cofactor_family": ligand_group.cofactor_family,
        "categories": ligand_group.categories,
        "ligand_cleanup": cleanup_info,
        "validation": validation_info,
        "status": "extracted",
    }
    if bond_source is not None:
        entry["bond_source"] = bond_source
    return entry


def run_pocket_extraction(
    args: argparse.Namespace,
    *,
    build_ligand_factory: LigandBuilderFactory,
    is_grouping_bond: Callable[[StructConnLink], bool] | None = None,
    output_suffix: str = DEFAULT_OUTPUT_SUFFIX,
) -> dict[str, object]:
    """Shared ligand-pocket extraction pipeline.

    Parameters
    ----------
    args:
        Parsed argparse namespace. Must expose ``input_path``, ``output_dir``,
        ``model_index``, ``radius``, ``include_resname``, ``exclude_resname``,
        ``min_heavy_atoms``, ``include_hetero_pocket``, ``include_water``,
        ``exclude_cofactors``, ``biological_assembly``, ``assembly_id``.
    build_ligand_factory:
        Called once per run with the *post-expansion* ``cif_dict`` and
        returns a :data:`LigandBuilder` closure used for every ligand in
        the run. This ordering lets path-specific setup (e.g. building
        ``_chem_comp_bond`` templates for the unprocessed path) see the
        biological-assembly-expanded CIF rather than the raw input. The
        returned builder's third tuple element is a free-form bond-source
        tag (``None`` for the unprocessed path, ``"struct_conn"`` /
        ``"openbabel_fallback"`` for the processed path); when non-None
        it is copied into ``pocket_payload['bond_source']`` and the
        manifest entry.
    is_grouping_bond:
        Strategy passed through to :func:`_core.detect_ligand_groups` so the
        processed path can treat additional ``_struct_conn`` rows as
        grouping links. ``None`` falls back to the unprocessed default
        (covalent links only).
    output_suffix:
        Suffix appended to ``input_path.stem`` when ``args.output_dir`` is
        not given.
    """
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input mmCIF file was not found: {input_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.with_name(f"{input_path.stem}{output_suffix}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # State accumulators, declared up here so the ``finally`` clause can
    # write whatever progress was made when something blows up.
    manifest_entries: list[dict[str, object]] = []
    failed_entries: list[dict[str, object]] = []
    skipped_count = 0
    pipeline_failure: dict[str, object] | None = None
    strict_batch = bool(getattr(args, "strict_batch", False))
    resume_enabled = bool(getattr(args, "resume", False))
    force_resume = bool(getattr(args, "force_resume", False))

    # Compute the manifest envelope (input checksum + tool versions +
    # schema tag) once, before anything that could mutate the input or
    # blow up. The same dict gets handed to every ``_write_outputs``
    # call via the closure below, so a SIGKILL mid-loop still leaves a
    # self-describing manifest behind.
    environment_metadata = _build_environment_metadata(input_path)

    # Resume guard: if the user asked to resume but the input mmCIF has
    # been edited since the prior run, refuse rather than silently
    # mixing two extractions. Runs *after* the metadata is computed so
    # the comparison has both checksums in hand.
    if resume_enabled:
        _check_resume_compatibility(
            output_dir, environment_metadata, force_resume=force_resume
        )

    # ``phase`` is updated as the orchestrator walks through each setup
    # step, so a pre-loop crash records *which* phase failed in the
    # manifest. An operator triaging a failed batch can then spot e.g.
    # "all 50 PDBs died at structure_load" (network/fixture issue) vs.
    # "one PDB died at ligand_detection" (data issue).
    phase = "init"

    def flush_outputs() -> dict[str, object]:
        """Re-flush manifest + failed_ligands.json from current accumulator state.

        Closes over the local accumulators (``manifest_entries``,
        ``failed_entries``, ``skipped_count``, ``pipeline_failure``) so
        every call site reads the latest values without re-passing them.
        Used after every loop iteration, by the ``finally`` clause, and
        wherever the orchestrator wants to persist progress to disk.
        """
        return _write_outputs(
            output_dir=output_dir,
            input_path=input_path,
            args=args,
            manifest_entries=manifest_entries,
            failed_entries=failed_entries,
            skipped_count=skipped_count,
            pipeline_failure=pipeline_failure,
            environment_metadata=environment_metadata,
        )

    manifest: dict[str, object] = {}

    try:
        phase = "biological_assembly_expansion"
        if getattr(args, "biological_assembly", False):
            assembly_id = getattr(args, "assembly_id", "1")
            expanded_path = output_dir / f"{input_path.stem}_assembly_{assembly_id}.cif"
            cif_path = _core.expand_to_biological_assembly(
                input_path, assembly_id=assembly_id, output_path=expanded_path
            )
        else:
            cif_path = input_path

        phase = "structure_load"
        structure, cif_dict = _core.load_structure_and_cif(cif_path)
        model = structure[args.model_index]

        phase = "metadata_extraction"
        component_metadata = _core.build_component_metadata(cif_dict)
        struct_conn_links = _core.build_struct_conn_links(cif_dict)
        residue_lookup = _core.index_model_residues(model)

        phase = "search_index_build"
        # One-time spatial index for pocket residue search. Hoists the
        # residue sort, per-residue atom collection, and coord-array
        # construction out of the per-ligand loop so each
        # find_pocket_residues call is just a kd-tree query.
        search_index = build_pocket_search_index(residue_lookup)

        phase = "ligand_builder_factory"
        build_ligand = build_ligand_factory(cif_dict)

        phase = "ligand_detection"
        ligand_groups = _core.detect_ligand_groups(
            residue_lookup=residue_lookup,
            component_metadata=component_metadata,
            struct_conn_links=struct_conn_links,
            include_resnames={name.upper() for name in args.include_resname},
            exclude_resnames={name.upper() for name in args.exclude_resname},
            min_heavy_atoms=args.min_heavy_atoms,
            exclude_cofactors=bool(getattr(args, "exclude_cofactors", False)),
            is_grouping_bond=is_grouping_bond,
        )

        phase = "ligand_loop"
        # The loop dispatches to ``_extract_one_ligand`` for every
        # ligand_group. A bad ligand no longer aborts the run by default:
        # the exception is captured into ``failed_entries`` and the loop
        # continues. ``--strict-batch`` opts back into "abort on first".
        # After every successful ligand we re-flush manifest +
        # failed_ligands.json so a SIGKILL / Ctrl+C mid-loop still leaves
        # an on-disk manifest reflecting whichever ligands had finished.
        for ligand_group in ligand_groups:
            ligand_sdf_path = output_dir / f"{ligand_group.ligand_id}.sdf"
            pocket_json_path = output_dir / f"{ligand_group.ligand_id}.json"

            if (
                resume_enabled
                and pocket_json_path.exists()
                and ligand_sdf_path.exists()
            ):
                try:
                    manifest_entries.append(
                        _build_skip_entry(
                            ligand_group, pocket_json_path, ligand_sdf_path
                        )
                    )
                    skipped_count += 1
                except Exception as exc:
                    if strict_batch:
                        raise
                    failed_entries.append(_build_failure_record(ligand_group, exc))
                flush_outputs()
                continue

            try:
                entry = _extract_one_ligand(
                    ligand_group=ligand_group,
                    residue_lookup=residue_lookup,
                    search_index=search_index,
                    build_ligand=build_ligand,
                    struct_conn_links=struct_conn_links,
                    component_metadata=component_metadata,
                    args=args,
                    ligand_sdf_path=ligand_sdf_path,
                    pocket_json_path=pocket_json_path,
                )
            except Exception as exc:
                if strict_batch:
                    raise
                failed_entries.append(_build_failure_record(ligand_group, exc))
                flush_outputs()
                continue

            manifest_entries.append(entry)
            flush_outputs()

        phase = "complete"

    except Exception as exc:
        # Either pre-loop setup failed (mmCIF parse, bio-expansion,
        # search-index build, etc.) or strict_batch=True re-raised a
        # per-ligand failure. Capture the phase + traceback for the
        # manifest, then re-raise so the CLI exits non-zero.
        if pipeline_failure is None:
            pipeline_failure = _build_pipeline_failure_record(exc, phase)
        raise

    finally:
        # Always run, even on KeyboardInterrupt / SystemExit / strict_batch
        # re-raise. Leaves a recoverable manifest + failed_ligands.json on
        # disk no matter how the orchestrator unwinds. The captured value
        # is what the success-path return below hands to the caller.
        manifest = flush_outputs()

    return manifest
