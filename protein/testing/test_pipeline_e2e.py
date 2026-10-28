# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/testing/test_pipeline_e2e.py

Opt-in end-to-end pipeline test for:
    fetch.py -> preprocess.py -> find_pocket_processed.py

This script is intentionally a runnable integration test rather than a fast
unit test. It:
    1. fetches a structure by PDB ID from RCSB,
    2. preprocesses it with Schrodinger PrepWizard,
    3. extracts ligand-centered pockets,
    4. validates that an SDF plus residue-level pocket metadata were written.

Recommended usage:
    # macOS / Linux
    source .venv/bin/activate
    python protein/testing/test_pipeline_e2e.py --pdb-id 1HVR --host localhost:2

    # Windows
    C:\\Users\\<you>\\anaconda3\\envs\\mlchem\\python.exe ^
        protein\\extract\\testing\\test_pipeline_e2e.py ^
        --pdb-id 1HVR --host localhost:2
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# Single-source Schrodinger discovery lives in protein.clean.preprocess so there
# is one mac/linux/windows-aware resolver for the whole repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from protein.clean.preprocess import (  # noqa: E402
    PreprocessError,
    resolve_schrodinger_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDB_ID = "1HVR"
DEFAULT_RADIUS = 10.0
DEFAULT_HOST = "localhost:2"
MAX_LOCAL_CORES = 2


class E2EPipelineError(RuntimeError):
    """Raised when the end-to-end pipeline test fails."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an end-to-end fetch -> preprocess -> pocket extraction test. "
            "This requires internet access for RCSB fetches and a working "
            "Schrodinger installation for preprocessing."
        )
    )
    parser.add_argument(
        "--pdb-id",
        default=DEFAULT_PDB_ID,
        help=f"PDB ID to test (default: {DEFAULT_PDB_ID}).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS,
        help=f"Pocket radius in angstrom (default: {DEFAULT_RADIUS}).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Schrodinger host string for preprocessing (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python interpreter used to run the project scripts (default: current interpreter).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where the integration test artifacts will be written. "
            "Defaults to protein/testing/_artifacts/<pdb_id>."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before running.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the Maestro intermediate produced during preprocessing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2))
    return 0


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Run the complete fetch -> preprocess -> extract pipeline."""
    pdb_id = args.pdb_id.lower()
    validate_host_limit(args.host)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (REPO_ROOT / "protein" / "extract" / "testing" / "_artifacts" / pdb_id).resolve()
    )
    python_exe = str(Path(args.python_exe).expanduser().resolve())

    if args.clean and output_dir.exists():
        remove_tree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fetched_cif = output_dir / f"{pdb_id}.cif"
    prepared_cif = output_dir / f"{pdb_id}_prepped_pdbx.cif"
    prepared_mae = output_dir / f"{pdb_id}_prepped.maegz"
    pockets_dir = output_dir / f"{pdb_id}_pockets"

    run_command(
        [
            python_exe,
            "protein/fetch/fetch.py",
            args.pdb_id,
            "--output-dir",
            repo_relative(output_dir),
            "--overwrite",
        ],
        description="Fetch raw mmCIF from RCSB",
    )

    ensure_local_job_server()

    preprocess_command = [
        python_exe,
        "protein/clean/preprocess.py",
        repo_relative(fetched_cif),
        repo_relative(prepared_cif),
        "--use-mae-intermediate",
        "--intermediate-output",
        repo_relative(prepared_mae),
        "--overwrite",
        "--assign-all-residues",
        "--re-add-hydrogens",
        "--disulfides",
        "--max-states",
        "1",
        "--epik-ph",
        "7.4",
        "--epik-pht",
        "2.0",
        "--sample-water",
        "--include-epik-states",
        "--propka-ph",
        "7.4",
        "--force-field",
        "S-OPLS",
        "--rmsd",
        "0.3",
        "--water-distance",
        "5.0",
        "--jobname",
        f"{pdb_id}_prep_e2e",
        "--host",
        args.host,
    ]
    if args.keep_intermediate:
        preprocess_command.append("--keep-intermediate")

    run_command(
        preprocess_command,
        description="Preprocess the fetched structure with Schrodinger",
        timeout_seconds=1800,
    )

    run_command(
        [
            python_exe,
            "-m",
            "protein.extract.find_pocket_processed",
            repo_relative(prepared_cif),
            "--output-dir",
            repo_relative(pockets_dir),
            "--radius",
            str(args.radius),
            "--allow-openbabel-fallback",
        ],
        description="Extract ligand pocket(s) from the processed mmCIF",
        timeout_seconds=600,
    )

    summary = validate_outputs(
        pdb_id=pdb_id,
        output_dir=output_dir,
        fetched_cif=fetched_cif,
        prepared_cif=prepared_cif,
        prepared_mae=prepared_mae,
        pockets_dir=pockets_dir,
        radius=args.radius,
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def validate_outputs(
    *,
    pdb_id: str,
    output_dir: Path,
    fetched_cif: Path,
    prepared_cif: Path,
    prepared_mae: Path,
    pockets_dir: Path,
    radius: float,
) -> dict[str, object]:
    """Validate and summarize the artifacts produced by the pipeline."""
    for path in (fetched_cif, prepared_cif, pockets_dir):
        if not path.exists():
            raise E2EPipelineError(f"Expected artifact was not created: {path}")

    manifest_path = pockets_dir / "manifest.json"
    if not manifest_path.is_file():
        raise E2EPipelineError(f"Pocket manifest was not created: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ligands = manifest.get("ligands", [])
    if not ligands:
        raise E2EPipelineError("Pocket extraction finished, but no ligand pockets were found.")

    best_ligand = max(ligands, key=lambda entry: entry.get("pocket_residue_count", 0))
    ligand_json_path = Path(best_ligand["pocket_json"])
    ligand_sdf_path = Path(best_ligand["ligand_sdf"])
    if not ligand_json_path.is_file():
        raise E2EPipelineError(f"Ligand JSON was not created: {ligand_json_path}")
    if not ligand_sdf_path.is_file():
        raise E2EPipelineError(f"Ligand SDF was not created: {ligand_sdf_path}")

    ligand_payload = json.loads(ligand_json_path.read_text(encoding="utf-8"))
    if not ligand_payload.get("ligand_atoms"):
        raise E2EPipelineError("Ligand payload does not contain atom records.")
    if not ligand_payload.get("ligand_bonds"):
        raise E2EPipelineError("Ligand payload does not contain bond records.")
    if not ligand_payload.get("pocket_residues"):
        raise E2EPipelineError("Ligand payload does not contain any pocket residues.")

    first_residue = ligand_payload["pocket_residues"][0]
    first_atom = first_residue["atoms"][0]
    required_atom_keys = {"atom_name", "element", "x", "y", "z", "residue"}
    if not required_atom_keys.issubset(first_atom):
        raise E2EPipelineError(
            "Pocket residue atom records are missing required keys: "
            + ", ".join(sorted(required_atom_keys - set(first_atom)))
        )

    return {
        "pdb_id": pdb_id.upper(),
        "radius": radius,
        "output_dir": str(output_dir),
        "fetched_cif": str(fetched_cif),
        "prepared_cif": str(prepared_cif),
        "prepared_maegz": str(prepared_mae) if prepared_mae.exists() else None,
        "manifest_json": str(manifest_path),
        "ligand_count": manifest.get("ligand_count", 0),
        "selected_ligand_id": best_ligand["ligand_id"],
        "selected_ligand_sdf": str(ligand_sdf_path),
        "selected_ligand_json": str(ligand_json_path),
        "selected_ligand_atom_count": ligand_payload["ligand_atom_count"],
        "selected_ligand_bond_count": ligand_payload["ligand_bond_count"],
        "selected_ligand_bond_source": ligand_payload.get("bond_source"),
        "selected_pocket_residue_count": ligand_payload["pocket_residue_count"],
        "first_pocket_residue": first_residue["residue"],
        "first_pocket_atom": first_atom,
    }


def validate_host_limit(host: str) -> None:
    """Refuse local PrepWizard host settings that exceed the machine-safe core cap."""
    prefix = "localhost:"
    normalized = host.strip()
    if not normalized.lower().startswith(prefix):
        return

    slots_text = normalized[len(prefix):].strip()
    try:
        slots = int(slots_text)
    except ValueError as exc:
        raise E2EPipelineError(
            f"Unsupported local host format: {host!r}. Expected localhost:<cores>."
        ) from exc

    if slots > MAX_LOCAL_CORES:
        raise E2EPipelineError(
            f"Refusing to run with {host!r}: local e2e tests are capped at "
            f"{MAX_LOCAL_CORES} cores to avoid overloading this workstation."
        )


def ensure_local_job_server() -> None:
    """Start Schrodinger's local job server so localhost jobs can submit cleanly."""
    try:
        schrodinger_root = resolve_schrodinger_root()
    except PreprocessError as exc:
        raise E2EPipelineError(str(exc)) from exc

    jsc_name = "jsc.exe" if os.name == "nt" else "jsc"
    jsc = schrodinger_root / jsc_name
    if not jsc.is_file():
        raise E2EPipelineError(f"Schrodinger jsc binary not found: {jsc}")

    env = os.environ.copy()
    env["SCHRODINGER"] = str(schrodinger_root)
    run_command(
        [str(jsc), "local-server-start"],
        description="Ensure Schrodinger local job server is running",
        env=env,
        timeout_seconds=120,
    )


def repo_relative(path: Path) -> str:
    """Return a path relative to the repository root when possible."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def remove_tree(path: Path) -> None:
    """Remove a directory tree, making read-only files writable when needed."""
    def _onerror(func, target, exc_info):
        target_path = Path(target)
        try:
            os.chmod(target_path, stat.S_IWRITE)
        except OSError:
            pass
        func(target)

    shutil.rmtree(path, onerror=_onerror)


def run_command(
    command: Sequence[str],
    *,
    description: str,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 300,
) -> None:
    """Run a subprocess and raise a helpful error if it fails."""
    print(f"\n[step] {description}")
    print(subprocess.list2cmdline(list(command)))
    completed = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    if completed.returncode != 0:
        raise E2EPipelineError(
            f"{description} failed with exit code {completed.returncode}."
        )


def run_all_tests() -> bool:
    """Compatibility hook for the repository's test-runner style."""
    try:
        run_pipeline(parse_args([]))
        return True
    except Exception as exc:
        print(f"E2E pipeline test failed: {exc}")
        return False


if __name__ == "__main__":
    raise SystemExit(main())
