# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/testing/test_batch_robustness.py

Unit tests for the batch-robustness helpers in ``_pocket.py``:

    * ``_build_failure_record`` - builds a ``failed_ligands.json`` entry
      from a caught exception, with full traceback.
    * ``_build_skip_entry`` - reconstructs a manifest entry from an
      existing pocket JSON for the ``--resume`` path.
    * ``--strict-batch`` and ``--resume`` CLI flags on both extractors.

Fast and self-contained: no mmCIF, no RCSB fetch, no full pipeline run.
End-to-end coverage of the orchestration loop sits in
``test_pipeline_e2e.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Let this file run as a script from the repo root without requiring the
# ``protein`` package to be installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protein.testing.runner import run_suite, suite_main  # noqa: E402
from protein.extract import find_pocket_unprocessed as unprocessed  # noqa: E402
from protein.extract import find_pocket_processed as processed  # noqa: E402
from protein.extract._core import (  # noqa: E402
    LigandGroup,
    PocketExtractionError,
    ResidueKey,
)
from protein.extract._pocket import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    _build_environment_metadata,
    _build_failure_record,
    _build_pipeline_failure_record,
    _build_skip_entry,
    _check_resume_compatibility,
    _collect_tool_versions,
    _sha256_file,
    _write_outputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ligand_group(
    ligand_id: str = "LIG_001",
    chain_id: str = "A",
    resseq: int = 100,
    resname: str = "ATP",
    cofactor_family: str | None = None,
) -> LigandGroup:
    return LigandGroup(
        ligand_id=ligand_id,
        residue_keys=(
            ResidueKey(
                chain_id=chain_id,
                resseq=resseq,
                insertion_code="",
                resname=resname,
            ),
        ),
        cofactor_family=cofactor_family,
    )


def _raise_and_capture(exc: BaseException) -> BaseException:
    """Raise then catch ``exc`` so it has a real ``__traceback__`` attached.

    The failure record formats traceback from ``exc.__traceback__``, which
    Python only sets when the exception actually goes through a
    raise/except cycle. Constructing the exception object alone is not
    enough.
    """
    try:
        raise exc
    except BaseException as caught:
        return caught


# ---------------------------------------------------------------------------
# _build_failure_record
# ---------------------------------------------------------------------------

def test_failure_record_captures_exception_type_and_message() -> None:
    exc = _raise_and_capture(RuntimeError("boom"))
    record = _build_failure_record(_ligand_group(), exc)
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "boom"


def test_failure_record_captures_traceback_string() -> None:
    exc = _raise_and_capture(ValueError("boom"))
    record = _build_failure_record(_ligand_group(), exc)
    tb = record["traceback"]
    assert isinstance(tb, str)
    # Must include both the exception class and the marker frame from
    # _raise_and_capture so an operator can localize the failure.
    assert "ValueError" in tb
    assert "_raise_and_capture" in tb


def test_failure_record_includes_ligand_identity() -> None:
    group = _ligand_group(
        ligand_id="ATP_A100",
        chain_id="A",
        resseq=100,
        resname="ATP",
    )
    exc = _raise_and_capture(RuntimeError("boom"))
    record = _build_failure_record(group, exc)
    assert record["ligand_id"] == "ATP_A100"
    assert record["residues"] == [
        {"chain_id": "A", "resseq": 100, "insertion_code": "", "resname": "ATP"}
    ]
    assert record["cofactor_family"] is None
    assert record["categories"] == []


def test_failure_record_includes_cofactor_metadata() -> None:
    group = _ligand_group(
        ligand_id="HEM_A1",
        resname="HEM",
        cofactor_family="heme",
    )
    exc = _raise_and_capture(RuntimeError("boom"))
    record = _build_failure_record(group, exc)
    assert record["cofactor_family"] == "heme"
    assert record["categories"] == ["cofactor"]


def test_failure_record_handles_domain_exception() -> None:
    exc = _raise_and_capture(
        PocketExtractionError("strict-validity gate refused HEM_A1")
    )
    record = _build_failure_record(_ligand_group(), exc)
    assert record["error_type"] == "PocketExtractionError"
    assert "strict-validity" in record["error_message"]


def test_failure_record_json_serializable() -> None:
    """``failed_ligands.json`` is dumped via ``json.dumps`` - every field must round-trip."""
    exc = _raise_and_capture(RuntimeError("boom"))
    record = _build_failure_record(_ligand_group(), exc)
    restored = json.loads(json.dumps(record))
    assert restored == record


# ---------------------------------------------------------------------------
# _build_skip_entry
# ---------------------------------------------------------------------------

def _write_pocket_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_skip_entry_pulls_summary_fields_from_pocket_json() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pocket_json = td_path / "ATP_A100.json"
        sdf_path = td_path / "ATP_A100.sdf"
        sdf_path.touch()
        payload = {
            "ligand_id": "ATP_A100",
            "ligand_residues": [
                {"chain_id": "A", "resseq": 100, "insertion_code": "", "resname": "ATP"}
            ],
            "ligand_atom_count": 31,
            "ligand_bond_count": 33,
            "pocket_residue_count": 18,
            "cofactor_family": None,
            "categories": [],
            "ligand_cleanup": {"changes": [], "warnings": []},
            "validation": {
                "sanitization": {"sanitized": True},
                "atom_completeness": {"complete": True},
            },
        }
        _write_pocket_json(pocket_json, payload)
        entry = _build_skip_entry(_ligand_group(), pocket_json, sdf_path)

    assert entry["status"] == "skipped"
    assert entry["ligand_atom_count"] == 31
    assert entry["ligand_bond_count"] == 33
    assert entry["pocket_residue_count"] == 18
    assert entry["validation"]["sanitization"]["sanitized"] is True


def test_skip_entry_preserves_bond_source_when_present() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pocket_json = td_path / "ATP_A100.json"
        sdf_path = td_path / "ATP_A100.sdf"
        sdf_path.touch()
        payload = {
            "ligand_atom_count": 5,
            "ligand_bond_count": 4,
            "pocket_residue_count": 3,
            "bond_source": "struct_conn",
        }
        _write_pocket_json(pocket_json, payload)
        entry = _build_skip_entry(_ligand_group(), pocket_json, sdf_path)
    assert entry["bond_source"] == "struct_conn"


def test_skip_entry_tolerates_pre_stage8_pocket_json() -> None:
    """Older pocket JSONs lack ``validation`` / ``ligand_cleanup``; defaults must hold."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pocket_json = td_path / "ATP_A100.json"
        sdf_path = td_path / "ATP_A100.sdf"
        sdf_path.touch()
        # Minimal payload without provenance metadata - no validation, no cleanup.
        payload = {
            "ligand_atom_count": 5,
            "ligand_bond_count": 4,
            "pocket_residue_count": 3,
        }
        _write_pocket_json(pocket_json, payload)
        entry = _build_skip_entry(_ligand_group(), pocket_json, sdf_path)
    assert entry["validation"] is None
    assert entry["ligand_cleanup"] is None
    assert entry["categories"] == []


def test_skip_entry_json_serializable() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pocket_json = td_path / "ATP_A100.json"
        sdf_path = td_path / "ATP_A100.sdf"
        sdf_path.touch()
        payload = {
            "ligand_atom_count": 5,
            "ligand_bond_count": 4,
            "pocket_residue_count": 3,
            "validation": {"sanitization": {"sanitized": True}},
        }
        _write_pocket_json(pocket_json, payload)
        entry = _build_skip_entry(_ligand_group(), pocket_json, sdf_path)
    restored = json.loads(json.dumps(entry))
    assert restored == entry


# ---------------------------------------------------------------------------
# _build_pipeline_failure_record
# ---------------------------------------------------------------------------

def test_pipeline_failure_record_captures_phase_and_metadata() -> None:
    exc = _raise_and_capture(RuntimeError("structure load blew up"))
    record = _build_pipeline_failure_record(exc, phase="structure_load")
    assert record["phase"] == "structure_load"
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "structure load blew up"


def test_pipeline_failure_record_captures_traceback() -> None:
    exc = _raise_and_capture(ValueError("boom"))
    record = _build_pipeline_failure_record(exc, phase="ligand_detection")
    tb = record["traceback"]
    assert isinstance(tb, str)
    assert "ValueError" in tb
    assert "_raise_and_capture" in tb


def test_pipeline_failure_record_json_serializable() -> None:
    exc = _raise_and_capture(PocketExtractionError("bad input"))
    record = _build_pipeline_failure_record(exc, phase="metadata_extraction")
    restored = json.loads(json.dumps(record))
    assert restored == record


# ---------------------------------------------------------------------------
# _write_outputs (manifest + failed_ligands.json writer)
# ---------------------------------------------------------------------------

def _outputs_args(model_index: int = 0, radius: float = 10.0) -> argparse.Namespace:
    """Minimal argparse.Namespace shape that ``_write_outputs`` reads from."""
    return argparse.Namespace(model_index=model_index, radius=radius)


def test_write_outputs_creates_both_files() -> None:
    """Manifest + failed_ligands.json both land in output_dir on every call."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        manifest = _write_outputs(
            output_dir=td_path,
            input_path=Path("/fake/input.cif"),
            args=_outputs_args(),
            manifest_entries=[],
            failed_entries=[],
            skipped_count=0,
            pipeline_failure=None,
        )
        assert (td_path / "manifest.json").is_file()
        assert (td_path / "failed_ligands.json").is_file()
        assert manifest["ligand_count"] == 0
        assert manifest["failed_ligand_count"] == 0
        assert manifest["skipped_ligand_count"] == 0


def test_write_outputs_idempotent_on_repeated_call() -> None:
    """Calling twice with the same inputs leaves the same files. Required by
    the incremental-write loop and by the finally clause that always runs."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        kwargs = dict(
            output_dir=td_path,
            input_path=Path("/fake/input.cif"),
            args=_outputs_args(),
            manifest_entries=[{"ligand_id": "L1", "status": "extracted"}],
            failed_entries=[],
            skipped_count=0,
            pipeline_failure=None,
        )
        m1 = _write_outputs(**kwargs)
        m2 = _write_outputs(**kwargs)
        assert m1 == m2
        assert json.loads((td_path / "manifest.json").read_text()) == m1


def test_write_outputs_failed_ligands_path_is_relative_filename() -> None:
    """Manifest's failed_ligands_path is a bare filename, not an absolute path.

    Fixes the gap where moving output_dir broke the manifest's pointer.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        manifest = _write_outputs(
            output_dir=td_path,
            input_path=Path("/fake/input.cif"),
            args=_outputs_args(),
            manifest_entries=[],
            failed_entries=[],
            skipped_count=0,
            pipeline_failure=None,
        )
    # Bare filename - no path components, no leading slash.
    assert manifest["failed_ligands_path"] == "failed_ligands.json"
    assert "/" not in manifest["failed_ligands_path"]


def test_write_outputs_includes_pipeline_failure_when_set() -> None:
    """When pre-loop or strict_batch raised, the manifest carries the failure record."""
    failure = {
        "phase": "structure_load",
        "error_type": "ValueError",
        "error_message": "bad CIF",
        "traceback": "...",
    }
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        manifest = _write_outputs(
            output_dir=td_path,
            input_path=Path("/fake/input.cif"),
            args=_outputs_args(),
            manifest_entries=[],
            failed_entries=[],
            skipped_count=0,
            pipeline_failure=failure,
        )
        assert manifest["pipeline_failure"] == failure
        # Also persisted on disk identically.
        on_disk = json.loads((td_path / "manifest.json").read_text())
        assert on_disk["pipeline_failure"] == failure


def test_write_outputs_omits_pipeline_failure_when_none() -> None:
    """A clean run produces a manifest with NO pipeline_failure key."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        manifest = _write_outputs(
            output_dir=td_path,
            input_path=Path("/fake/input.cif"),
            args=_outputs_args(),
            manifest_entries=[],
            failed_entries=[],
            skipped_count=0,
            pipeline_failure=None,
        )
    assert "pipeline_failure" not in manifest


def test_write_outputs_persists_failed_entries() -> None:
    """failed_ligands.json contains the per-ligand failure list verbatim."""
    failed = [
        {
            "ligand_id": "ATP_A1",
            "residues": [{"chain_id": "A", "resseq": 1, "insertion_code": "", "resname": "ATP"}],
            "cofactor_family": None,
            "categories": [],
            "error_type": "RuntimeError",
            "error_message": "something",
            "traceback": "Traceback ...",
        }
    ]
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _write_outputs(
            output_dir=td_path,
            input_path=Path("/fake/input.cif"),
            args=_outputs_args(),
            manifest_entries=[],
            failed_entries=failed,
            skipped_count=0,
            pipeline_failure=None,
        )
        on_disk = json.loads((td_path / "failed_ligands.json").read_text())
    assert on_disk == failed


def test_write_outputs_round_trips_through_json() -> None:
    """Whole manifest is JSON-clean - no numpy scalars, no path objects in values."""
    failure = _build_pipeline_failure_record(
        _raise_and_capture(RuntimeError("x")), phase="ligand_loop"
    )
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        manifest = _write_outputs(
            output_dir=td_path,
            input_path=Path("/fake/input.cif"),
            args=_outputs_args(),
            manifest_entries=[{"ligand_id": "L1", "status": "extracted"}],
            failed_entries=[
                _build_failure_record(_ligand_group(), _raise_and_capture(RuntimeError("y")))
            ],
            skipped_count=0,
            pipeline_failure=failure,
        )
    restored = json.loads(json.dumps(manifest))
    assert restored == manifest


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

def test_strict_batch_flag_unprocessed() -> None:
    args_on = unprocessed.parse_args(["/tmp/fake.cif", "--strict-batch"])
    assert args_on.strict_batch is True
    args_default = unprocessed.parse_args(["/tmp/fake.cif"])
    assert args_default.strict_batch is False


def test_strict_batch_flag_processed() -> None:
    args_on = processed.parse_args(["/tmp/fake.cif", "--strict-batch"])
    assert args_on.strict_batch is True
    args_default = processed.parse_args(["/tmp/fake.cif"])
    assert args_default.strict_batch is False


def test_resume_flag_unprocessed() -> None:
    args_on = unprocessed.parse_args(["/tmp/fake.cif", "--resume"])
    assert args_on.resume is True
    args_default = unprocessed.parse_args(["/tmp/fake.cif"])
    assert args_default.resume is False


def test_resume_flag_processed() -> None:
    args_on = processed.parse_args(["/tmp/fake.cif", "--resume"])
    assert args_on.resume is True
    args_default = processed.parse_args(["/tmp/fake.cif"])
    assert args_default.resume is False


# ---------------------------------------------------------------------------
# Manifest envelope (checksum + tool versions + schema tag)
# ---------------------------------------------------------------------------

def test_sha256_file_matches_known_content() -> None:
    """Hash a file with known bytes; result must be deterministic."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "input.cif"
        path.write_bytes(b"hello world\n")
        # Independently-computed sha256 of "hello world\n".
        expected = (
            "a948904f2f0f479b8f8197694b30184b"
            "0d2ed1c1cd2a1ec0fb85d299a192a447"
        )
        assert _sha256_file(path) == expected


def test_sha256_file_changes_when_content_changes() -> None:
    """Editing one byte changes the hash. This is the whole point of the check."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "input.cif"
        path.write_bytes(b"original")
        h1 = _sha256_file(path)
        path.write_bytes(b"oRiginal")  # one bit flipped
        h2 = _sha256_file(path)
        assert h1 != h2


def test_collect_tool_versions_includes_python_and_known_modules() -> None:
    """rdkit + scipy are required deps; both must surface a version string."""
    versions = _collect_tool_versions()
    assert "python" in versions and versions["python"]
    assert "rdkit" in versions and versions["rdkit"]
    assert "scipy" in versions and versions["scipy"]
    # Every entry is a string (no numpy scalars, no ints).
    assert all(isinstance(v, str) for v in versions.values())


def test_build_environment_metadata_shape() -> None:
    """The envelope must carry checksum, schema_version, and a versions dict."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "input.cif"
        path.write_bytes(b"data ATOM\n")
        env = _build_environment_metadata(path)
        # Compute the comparison checksum INSIDE the with block - the
        # tempfile is gone the moment we exit it.
        expected_checksum = _sha256_file(path)

    assert env["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert env["input_checksum_sha256"] == expected_checksum
    assert env["input_size_bytes"] == len(b"data ATOM\n")
    assert env["kdtree_leafsize"] == 16
    assert isinstance(env["versions"], dict)
    assert "rdkit" in env["versions"]


def test_build_environment_metadata_json_serializable() -> None:
    """The whole envelope round-trips through json.dumps cleanly."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "input.cif"
        path.write_bytes(b"data ATOM\n")
        env = _build_environment_metadata(path)
    restored = json.loads(json.dumps(env))
    assert restored == env


def test_write_outputs_includes_environment_block_when_provided() -> None:
    """When env metadata is passed, manifest carries the ``environment`` key."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cif = td_path / "input.cif"
        cif.write_bytes(b"data ATOM\n")
        env = _build_environment_metadata(cif)
        manifest = _write_outputs(
            output_dir=td_path,
            input_path=cif,
            args=_outputs_args(),
            manifest_entries=[],
            failed_entries=[],
            skipped_count=0,
            pipeline_failure=None,
            environment_metadata=env,
        )
        # Read back inside the with block - the tempdir is cleaned on exit.
        on_disk = json.loads((td_path / "manifest.json").read_text())

    assert manifest["environment"] == env
    assert on_disk["environment"] == env


def test_write_outputs_omits_environment_block_when_none() -> None:
    """No env metadata → no ``environment`` key. Backwards-compatible default."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        manifest = _write_outputs(
            output_dir=td_path,
            input_path=Path("/fake/input.cif"),
            args=_outputs_args(),
            manifest_entries=[],
            failed_entries=[],
            skipped_count=0,
            pipeline_failure=None,
        )
    assert "environment" not in manifest


# ---------------------------------------------------------------------------
# Resume compatibility guard
# ---------------------------------------------------------------------------

def _seed_prior_manifest(
    output_dir: Path,
    input_bytes: bytes,
    *,
    extra_env: dict | None = None,
) -> Path:
    """Helper: write a prior-run manifest pretending ``input_bytes`` was the input."""
    cif = output_dir / "input.cif"
    cif.write_bytes(input_bytes)
    env = _build_environment_metadata(cif)
    if extra_env:
        env.update(extra_env)
    _write_outputs(
        output_dir=output_dir,
        input_path=cif,
        args=_outputs_args(),
        manifest_entries=[],
        failed_entries=[],
        skipped_count=0,
        pipeline_failure=None,
        environment_metadata=env,
    )
    return cif


def test_resume_compat_passes_when_checksum_matches() -> None:
    """Identical input bytes → checksum match → resume proceeds silently."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cif = _seed_prior_manifest(td_path, b"data ATOM\n")
        current_env = _build_environment_metadata(cif)
        # No exception = pass.
        _check_resume_compatibility(td_path, current_env, force_resume=False)


def test_resume_compat_refuses_when_checksum_mismatches() -> None:
    """Edit the input bytes → checksum mismatch → resume refused (without --force-resume)."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cif = _seed_prior_manifest(td_path, b"data ATOM\n")
        # Mutate the input. Now the on-disk manifest's checksum disagrees
        # with the freshly-computed one.
        cif.write_bytes(b"data ATOM EDITED\n")
        current_env = _build_environment_metadata(cif)

        try:
            _check_resume_compatibility(td_path, current_env, force_resume=False)
        except PocketExtractionError as exc:
            message = str(exc)
            assert "checksum" in message.lower()
            assert "--force-resume" in message
            return
        raise AssertionError("expected PocketExtractionError, none raised")


def test_resume_compat_force_resume_overrides_mismatch() -> None:
    """--force-resume must bypass the refusal even when checksums disagree."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cif = _seed_prior_manifest(td_path, b"data ATOM\n")
        cif.write_bytes(b"data ATOM EDITED\n")
        current_env = _build_environment_metadata(cif)
        # No exception = pass.
        _check_resume_compatibility(td_path, current_env, force_resume=True)


def test_resume_compat_silent_when_no_prior_manifest() -> None:
    """First-time --resume (no prior manifest) → no comparison to make → no raise."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cif = td_path / "input.cif"
        cif.write_bytes(b"data ATOM\n")
        current_env = _build_environment_metadata(cif)
        _check_resume_compatibility(td_path, current_env, force_resume=False)


def test_resume_compat_silent_for_pre_stage8_manifest() -> None:
    """Older manifest with no ``environment`` key → no comparison possible → no raise.

    Backwards tolerance: output directories without provenance metadata should still be
    resumable. The protective guard only kicks in once we have a baseline
    checksum to compare against.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Hand-write a manifest without an environment block.
        manifest = {
            "input_mmcif": "/fake/input.cif",
            "output_dir": str(td_path),
            "model_index": 0,
            "pocket_radius": 10.0,
            "ligand_count": 0,
            "failed_ligand_count": 0,
            "skipped_ligand_count": 0,
            "failed_ligands_path": "failed_ligands.json",
            "ligands": [],
        }
        (td_path / "manifest.json").write_text(json.dumps(manifest))
        cif = td_path / "input.cif"
        cif.write_bytes(b"any bytes")
        current_env = _build_environment_metadata(cif)
        _check_resume_compatibility(td_path, current_env, force_resume=False)


def test_resume_compat_silent_for_corrupt_prior_manifest() -> None:
    """Corrupt prior manifest → don't crash the user; just skip the guard."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "manifest.json").write_text("not valid json {{{")
        cif = td_path / "input.cif"
        cif.write_bytes(b"any bytes")
        current_env = _build_environment_metadata(cif)
        _check_resume_compatibility(td_path, current_env, force_resume=False)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

ALL_TESTS = (
    test_failure_record_captures_exception_type_and_message,
    test_failure_record_captures_traceback_string,
    test_failure_record_includes_ligand_identity,
    test_failure_record_includes_cofactor_metadata,
    test_failure_record_handles_domain_exception,
    test_failure_record_json_serializable,
    test_skip_entry_pulls_summary_fields_from_pocket_json,
    test_skip_entry_preserves_bond_source_when_present,
    test_skip_entry_tolerates_pre_stage8_pocket_json,
    test_skip_entry_json_serializable,
    test_pipeline_failure_record_captures_phase_and_metadata,
    test_pipeline_failure_record_captures_traceback,
    test_pipeline_failure_record_json_serializable,
    test_write_outputs_creates_both_files,
    test_write_outputs_idempotent_on_repeated_call,
    test_write_outputs_failed_ligands_path_is_relative_filename,
    test_write_outputs_includes_pipeline_failure_when_set,
    test_write_outputs_omits_pipeline_failure_when_none,
    test_write_outputs_persists_failed_entries,
    test_write_outputs_round_trips_through_json,
    test_strict_batch_flag_unprocessed,
    test_strict_batch_flag_processed,
    test_resume_flag_unprocessed,
    test_resume_flag_processed,
    test_sha256_file_matches_known_content,
    test_sha256_file_changes_when_content_changes,
    test_collect_tool_versions_includes_python_and_known_modules,
    test_build_environment_metadata_shape,
    test_build_environment_metadata_json_serializable,
    test_write_outputs_includes_environment_block_when_provided,
    test_write_outputs_omits_environment_block_when_none,
    test_resume_compat_passes_when_checksum_matches,
    test_resume_compat_refuses_when_checksum_mismatches,
    test_resume_compat_force_resume_overrides_mismatch,
    test_resume_compat_silent_when_no_prior_manifest,
    test_resume_compat_silent_for_pre_stage8_manifest,
    test_resume_compat_silent_for_corrupt_prior_manifest,
)


def run_all_tests() -> bool:
    """Entry point for the master :mod:`run_all_tests` runner."""
    return run_suite(ALL_TESTS)


if __name__ == "__main__":
    sys.exit(suite_main(ALL_TESTS))
