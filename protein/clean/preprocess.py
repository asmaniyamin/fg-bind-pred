# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/clean/preprocess.py

Wrap Schrodinger Protein Preparation Wizard (prepwizard) as a Python API and
CLI.

That help text reports:
    Input file should be in Maestro, PDB, or mmCIF format.
    Output file should be in Maestro, PDB, or mmCIF format.

So this wrapper supports direct ``.cif``/``.mmcif`` input and output. For
workflows that still prefer a Maestro intermediate, use
``--use-mae-intermediate`` so the final structure is converted back to mmCIF
with Schrodinger ``structconvert -PDBx``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SUPPORTED_PREPWIZARD_EXTENSIONS = frozenset({".mae", ".maegz", ".pdb", ".cif", ".mmcif"})
SUPPORTED_MAESTRO_EXTENSIONS = frozenset({".mae", ".maegz"})
SUPPORTED_MMCIF_EXTENSIONS = frozenset({".cif", ".mmcif"})
SCHRODINGER_VERSION_RE = re.compile(r"Schrodinger(?P<year>\d+)-(?P<release>\d+)")


class PreprocessError(RuntimeError):
    """Raised when Schrodinger preprocessing fails or is misconfigured."""


@dataclass(frozen=True, slots=True)
class SchrodingerTools:
    """Resolved Schrodinger tool paths."""

    root: Path
    prepwizard: Path
    structconvert: Path


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """Summary of the preprocessing job that was launched."""

    input_path: Path
    output_path: Path
    prepwizard_command: tuple[str, ...]
    structconvert_command: tuple[str, ...] | None
    used_mae_intermediate: bool
    intermediate_path: Path | None = None


def resolve_schrodinger_tools(
    *,
    schrodinger_root: str | Path | None = None,
    prepwizard_exe: str | Path | None = None,
    structconvert_exe: str | Path | None = None,
) -> SchrodingerTools:
    """Resolve Schrodinger install and utility paths."""
    root = resolve_schrodinger_root(schrodinger_root)

    suffix = ".exe" if os.name == "nt" else ""
    prepwizard = (
        Path(prepwizard_exe)
        if prepwizard_exe is not None
        else root / "utilities" / f"prepwizard{suffix}"
    )
    structconvert = (
        Path(structconvert_exe)
        if structconvert_exe is not None
        else root / "utilities" / f"structconvert{suffix}"
    )

    if not prepwizard.is_file():
        raise PreprocessError(f"Schrodinger prepwizard executable was not found: {prepwizard}")
    if not structconvert.is_file():
        raise PreprocessError(f"Schrodinger structconvert executable was not found: {structconvert}")

    return SchrodingerTools(
        root=root,
        prepwizard=prepwizard,
        structconvert=structconvert,
    )


def preprocess_structure(args: argparse.Namespace) -> PreprocessResult:
    """Run Schrodinger Protein Preparation Wizard using parsed CLI arguments."""
    tools = resolve_schrodinger_tools(
        schrodinger_root=args.schrodinger_root,
        prepwizard_exe=args.prepwizard_exe,
        structconvert_exe=args.structconvert_exe,
    )

    input_path = Path(args.input_path).expanduser()
    output_path = Path(args.output_path).expanduser()

    _validate_structure_path(input_path, SUPPORTED_PREPWIZARD_EXTENSIONS, "Input")
    _validate_structure_path(output_path, SUPPORTED_PREPWIZARD_EXTENSIONS, "Output")

    if input_path.resolve() == output_path.resolve():
        raise PreprocessError("Input and output paths must be different.")

    if not args.dry_run and not input_path.is_file():
        raise FileNotFoundError(f"Input structure file was not found: {input_path}")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output without --overwrite: {output_path}"
        )

    if args.use_mae_intermediate and not _is_mmcif_path(output_path):
        raise PreprocessError(
            "--use-mae-intermediate is only supported when the final output is "
            "mmCIF/CIF."
        )

    if args.use_mae_intermediate and not args.wait:
        raise PreprocessError(
            "A Maestro intermediate requires the Schrodinger job to finish "
            "before structconvert can run, so do not combine "
            "--use-mae-intermediate with --no-wait."
        )

    intermediate_path: Path | None = None
    structconvert_command: list[str] | None = None

    prepwizard_input = input_path
    prepwizard_output = output_path

    if args.use_mae_intermediate:
        intermediate_path = (
            Path(args.intermediate_output).expanduser()
            if args.intermediate_output
            else output_path.parent / f"{output_path.stem}.prepwizard.maegz"
        )
        _validate_structure_path(
            intermediate_path,
            SUPPORTED_MAESTRO_EXTENSIONS,
            "Intermediate Maestro output",
        )

        if intermediate_path.exists() and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing intermediate output without "
                f"--overwrite: {intermediate_path}"
            )

        prepwizard_output = intermediate_path
        structconvert_command = build_structconvert_command(
            tools.structconvert,
            intermediate_path,
            output_path,
        )

    prepwizard_command = build_prepwizard_command(
        tools.prepwizard,
        prepwizard_input,
        prepwizard_output,
        args,
    )

    if args.print_command or args.dry_run:
        print(_format_command(prepwizard_command))
        if structconvert_command is not None:
            print(_format_command(structconvert_command))

    if args.dry_run:
        return PreprocessResult(
            input_path=input_path,
            output_path=output_path,
            prepwizard_command=tuple(prepwizard_command),
            structconvert_command=tuple(structconvert_command) if structconvert_command else None,
            used_mae_intermediate=structconvert_command is not None,
            intermediate_path=intermediate_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if intermediate_path is not None:
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SCHRODINGER"] = str(tools.root)
    if args.debug:
        # Env-var based debug knobs honored by various Schrodinger Python
        # layers. Not every tool respects every one, but setting them is
        # harmless when ignored. Together with -SAVE (added in
        # build_prepwizard_command) these make protassign tracebacks name
        # the offending residue instead of truncating at IndexError.
        env["SCHRODINGER_PYTHON_DEBUG"] = "1"
        env["SCHRODINGER_MMLIBS_DEBUG"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONFAULTHANDLER"] = "1"

    _run_command(prepwizard_command, env=env, stream=args.debug)

    if structconvert_command is not None:
        _run_command(structconvert_command, env=env, stream=args.debug)
        if intermediate_path is not None and intermediate_path.exists() and not args.keep_intermediate:
            intermediate_path.unlink()

    return PreprocessResult(
        input_path=input_path,
        output_path=output_path,
        prepwizard_command=tuple(prepwizard_command),
        structconvert_command=tuple(structconvert_command) if structconvert_command else None,
        used_mae_intermediate=structconvert_command is not None,
        intermediate_path=intermediate_path,
    )


def build_prepwizard_command(
    prepwizard_exe: Path,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Translate CLI arguments into a Schrodinger prepwizard command."""
    command = [str(prepwizard_exe)]

    if args.skip_preprocess:
        command.append("-nopreprocess")
    _append_option(command, "-reference_st_file", args.reference_structure)
    _append_option(command, "-reference_pdbid", args.reference_pdbid)
    if args.no_bond_orders:
        command.append("-nobondorders")
    if args.no_ccd:
        command.append("-noccd")
    if args.assign_all_residues:
        command.append("-assign_all_residues")
    if args.no_hydrogen_treatment:
        command.append("-nohtreat")
    if args.re_add_hydrogens:
        command.append("-rehtreat")
    if args.no_metal_treatment:
        command.append("-nometaltreat")
    if args.disulfides:
        command.append("-disulfides")
    if args.glycosylation:
        command.append("-glycosylation")
    if args.palmitoylation:
        command.append("-palmitoylation")
    _append_option(command, "-antibody_cdr_scheme", args.antibody_cdr_scheme)
    if args.renumber_antibody_residues:
        command.append("-renumber_ab_residues")
    _append_option(command, "-tcr_cdr_scheme", args.tcr_cdr_scheme)
    if args.renumber_tcr_residues:
        command.append("-renumber_tcr_residues")
    if args.mse:
        command.append("-mse")
    _append_option(command, "-preprocess_watdist", args.preprocess_water_distance)
    if args.fill_loops:
        command.append("-fillloops")
    _append_option(command, "-fasta_file", args.fasta_file)
    if args.fill_sidechains:
        command.append("-fillsidechains")
    if args.add_oxt:
        command.append("-addOXT")
    if args.cap_termini:
        command.append("-captermini")
    _append_option(command, "-cap_termini_min_atoms", args.cap_termini_min_atoms)
    if args.no_epik:
        command.append("-noepik")
    _append_option(command, "-epik_pH", args.epik_ph)
    _append_option(command, "-epik_pHt", args.epik_pht)
    _append_option(command, "-max_states", args.max_states)
    if args.no_idealize_htf:
        command.append("-noidealizehtf")

    if args.skip_protassign:
        command.append("-noprotassign")
    if args.sample_water:
        command.append("-samplewater")
    if args.include_epik_states:
        command.append("-include_epik_states")
    if args.use_crystal_symmetry:
        command.append("-xtal")
    if args.no_propka:
        command.append("-nopropka")
    _append_option(command, "-propka_pH", args.propka_ph)
    if args.label_pkas:
        command.append("-label_pkas")
    _append_option(command, "-simplified_pH", args.simplified_ph)
    if args.force_residue:
        for residue, state in args.force_residue:
            command.extend(["-force", residue, state])
    if args.minimize_adjustable_hydrogens:
        command.append("-minimize_adj_h")

    if args.skip_impref:
        command.append("-noimpref")
    _append_option(command, "-rmsd", args.rmsd)
    if args.fix_heavy_atoms:
        command.append("-fix")
    _append_option(command, "-f", args.force_field)
    if args.keep_far_waters:
        command.append("-keepfarwat")
    _append_option(command, "-watdist", args.water_distance)
    _append_option(command, "-delwater_hbond_cutoff", args.delete_water_hbond_cutoff)

    if args.preserve_structure_titles:
        command.append("-preserve_st_titles")
    if args.use_pdb_ph:
        command.append("-use_PDB_pH")

    _append_option(command, "-HOST", args.host)
    if args.wait:
        command.append("-WAIT")
    if args.save_job_archive or args.debug:
        # Debug mode forces -SAVE so the job directory with protassign.log
        # and the full Python traceback survives the run. prepwizard's
        # toplevel.py does NOT accept -DEBUG (unlike some other Schrodinger
        # tools), so verbosity has to come from env vars, not CLI flags.
        command.append("-SAVE")
    if args.nojobid:
        command.append("-NOJOBID")
    _append_option(command, "-JOBNAME", args.jobname)

    command.extend([str(input_path), str(output_path)])
    return command


def build_structconvert_command(
    structconvert_exe: Path,
    input_path: Path,
    output_path: Path,
) -> list[str]:
    """Build the Schrodinger structconvert command for mmCIF output."""
    if not _is_mmcif_path(output_path):
        raise PreprocessError(
            "structconvert back-conversion is only implemented for mmCIF/CIF output."
        )

    return [
        str(structconvert_exe),
        "-PDBx",
        str(input_path),
        str(output_path),
    ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Schrodinger protein preprocessing."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Schrodinger Protein Preparation Wizard. "
            "Local prepwizard --help confirms that mmCIF/CIF is valid input "
            "and valid output."
        )
    )
    parser.set_defaults(wait=True)

    parser.add_argument("input_path", help="Input structure file (.mae/.maegz/.pdb/.cif/.mmcif).")
    parser.add_argument("output_path", help="Output structure file (.mae/.maegz/.pdb/.cif/.mmcif).")

    runtime_group = parser.add_argument_group("Schrodinger Runtime")
    runtime_group.add_argument("--schrodinger-root", help="Explicit Schrodinger installation root.")
    runtime_group.add_argument(
        "--prepwizard-exe",
        help="Explicit path to the prepwizard binary (prepwizard on macOS/Linux, prepwizard.exe on Windows).",
    )
    runtime_group.add_argument(
        "--structconvert-exe",
        help="Explicit path to the structconvert binary (structconvert on macOS/Linux, structconvert.exe on Windows).",
    )
    runtime_group.add_argument(
        "--use-mae-intermediate",
        action="store_true",
        help="Write a Maestro intermediate and convert the final result back to mmCIF/CIF with structconvert -PDBx.",
    )
    runtime_group.add_argument(
        "--intermediate-output",
        help="Explicit Maestro intermediate path used with --use-mae-intermediate.",
    )
    runtime_group.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the Maestro intermediate after converting it back to mmCIF/CIF.",
    )
    runtime_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing output files.",
    )
    runtime_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated Schrodinger commands without executing them.",
    )
    runtime_group.add_argument(
        "--print-command",
        action="store_true",
        help="Print the exact Schrodinger command line(s) before execution.",
    )
    runtime_group.add_argument(
        "--jobserver-dir",
        help="Writable directory to use for Schrodinger's local job server state.",
    )
    runtime_group.add_argument(
        "--schrodinger-home",
        help="Writable HOME directory for Schrodinger subprocess state such as ~/.schrodinger.",
    )
    runtime_group.add_argument(
        "--schrodinger-tmpdir",
        help="Writable directory to use for SCHRODINGER_TMPDIR/TMPDIR/TMP/TEMP.",
    )

    preprocess_group = parser.add_argument_group("Preprocess")
    preprocess_group.add_argument("--skip-preprocess", action="store_true", help="Map to -nopreprocess.")
    preprocess_group.add_argument("--reference-structure", help="Map to -reference_st_file.")
    preprocess_group.add_argument("--reference-pdbid", help="Map to -reference_pdbid.")
    preprocess_group.add_argument("--no-bond-orders", action="store_true", help="Map to -nobondorders.")
    preprocess_group.add_argument("--no-ccd", action="store_true", help="Map to -noccd.")
    preprocess_group.add_argument("--assign-all-residues", action="store_true", help="Map to -assign_all_residues.")
    preprocess_group.add_argument("--no-hydrogen-treatment", action="store_true", help="Map to -nohtreat.")
    preprocess_group.add_argument("--re-add-hydrogens", action="store_true", help="Map to -rehtreat.")
    preprocess_group.add_argument("--no-metal-treatment", action="store_true", help="Map to -nometaltreat.")
    preprocess_group.add_argument("--disulfides", action="store_true", help="Map to -disulfides.")
    preprocess_group.add_argument("--glycosylation", action="store_true", help="Map to -glycosylation.")
    preprocess_group.add_argument("--palmitoylation", action="store_true", help="Map to -palmitoylation.")
    preprocess_group.add_argument(
        "--antibody-cdr-scheme",
        choices=["Chothia", "Kabat", "IMGT", "EnhancedChothia", "AHo", "None"],
        help="Map to -antibody_cdr_scheme.",
    )
    preprocess_group.add_argument(
        "--renumber-antibody-residues",
        action="store_true",
        help="Map to -renumber_ab_residues.",
    )
    preprocess_group.add_argument(
        "--tcr-cdr-scheme",
        choices=["IMGT", "AHo", "None"],
        help="Map to -tcr_cdr_scheme.",
    )
    preprocess_group.add_argument(
        "--renumber-tcr-residues",
        action="store_true",
        help="Map to -renumber_tcr_residues.",
    )
    preprocess_group.add_argument("--mse", action="store_true", help="Map to -mse.")
    preprocess_group.add_argument(
        "--preprocess-water-distance",
        type=float,
        help="Map to -preprocess_watdist.",
    )
    preprocess_group.add_argument("--fill-loops", action="store_true", help="Map to -fillloops.")
    preprocess_group.add_argument("--fasta-file", help="Map to -fasta_file.")
    preprocess_group.add_argument("--fill-sidechains", action="store_true", help="Map to -fillsidechains.")
    preprocess_group.add_argument("--add-oxt", action="store_true", help="Map to -addOXT.")
    preprocess_group.add_argument("--cap-termini", action="store_true", help="Map to -captermini.")
    preprocess_group.add_argument(
        "--cap-termini-min-atoms",
        type=int,
        help="Map to -cap_termini_min_atoms.",
    )
    preprocess_group.add_argument("--no-epik", action="store_true", help="Map to -noepik.")
    preprocess_group.add_argument("--epik-ph", type=float, help="Map to -epik_pH.")
    preprocess_group.add_argument("--epik-pht", type=float, help="Map to -epik_pHt.")
    preprocess_group.add_argument("--max-states", type=int, help="Map to -max_states.")
    preprocess_group.add_argument("--no-idealize-htf", action="store_true", help="Map to -noidealizehtf.")

    protassign_group = parser.add_argument_group("Optimize H-Bond Assignments")
    protassign_group.add_argument("--skip-protassign", action="store_true", help="Map to -noprotassign.")
    protassign_group.add_argument("--sample-water", action="store_true", help="Map to -samplewater.")
    protassign_group.add_argument(
        "--include-epik-states",
        action="store_true",
        help="Map to -include_epik_states.",
    )
    protassign_group.add_argument("--use-crystal-symmetry", action="store_true", help="Map to -xtal.")
    protassign_group.add_argument("--no-propka", action="store_true", help="Map to -nopropka.")
    protassign_group.add_argument("--propka-ph", type=float, help="Map to -propka_pH.")
    protassign_group.add_argument("--label-pkas", action="store_true", help="Map to -label_pkas.")
    protassign_group.add_argument(
        "--simplified-ph",
        choices=["very_low", "low", "neutral", "high"],
        help="Map to -simplified_pH.",
    )
    protassign_group.add_argument(
        "--force-residue",
        nargs=2,
        action="append",
        metavar=("RESIDUE", "STATE"),
        help="Map one residue/state pair to -force. May be repeated.",
    )
    protassign_group.add_argument(
        "--minimize-adjustable-hydrogens",
        action="store_true",
        help="Map to -minimize_adj_h.",
    )

    minimize_group = parser.add_argument_group("Minimize and Delete Waters")
    minimize_group.add_argument("--skip-impref", action="store_true", help="Map to -noimpref.")
    minimize_group.add_argument("--rmsd", type=float, help="Map to -rmsd.")
    minimize_group.add_argument(
        "--fix-heavy-atoms",
        action="store_true",
        help="Map to -fix (minimize hydrogens only).",
    )
    minimize_group.add_argument(
        "--force-field",
        choices=["S-OPLS", "OPLS_2005"],
        help="Map to prepwizard -f.",
    )
    minimize_group.add_argument("--keep-far-waters", action="store_true", help="Map to -keepfarwat.")
    minimize_group.add_argument("--water-distance", type=float, help="Map to -watdist.")
    minimize_group.add_argument(
        "--delete-water-hbond-cutoff",
        type=float,
        help="Map to -delwater_hbond_cutoff.",
    )

    other_group = parser.add_argument_group("Other")
    other_group.add_argument(
        "--preserve-structure-titles",
        action="store_true",
        help="Map to -preserve_st_titles.",
    )
    other_group.add_argument("--use-pdb-ph", action="store_true", help="Map to -use_PDB_pH.")

    job_group = parser.add_argument_group("Job Control")
    job_group.add_argument("--host", help="Map to -HOST, for example localhost:6.")
    job_group.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Do not add -WAIT. The Schrodinger job may continue after this wrapper exits.",
    )
    job_group.add_argument("--save-job-archive", action="store_true", help="Map to -SAVE.")
    job_group.add_argument("--nojobid", action="store_true", help="Map to -NOJOBID.")
    job_group.add_argument("--jobname", help="Map to -JOBNAME.")
    job_group.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable Schrodinger verbose logging. Implies -SAVE (preserves "
            "the job directory so protassign.log and the full Python "
            "traceback survive the run) and exports SCHRODINGER_PYTHON_DEBUG"
            "=1 / PYTHONUNBUFFERED=1 / SCHRODINGER_MMLIBS_DEBUG=1 so the "
            "failing residue is identifiable in the stack trace. "
            "prepwizard's toplevel does not accept -DEBUG as a flag."
        ),
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    result = preprocess_structure(args)

    if args.dry_run:
        print("Dry run completed.")
        return 0

    print(f"Prepared structure written to {result.output_path.resolve()}")
    if result.used_mae_intermediate and result.intermediate_path is not None:
        if args.keep_intermediate:
            print(f"Kept Maestro intermediate at {result.intermediate_path.resolve()}")
        else:
            print("Used a Maestro intermediate and converted the final structure back to mmCIF/CIF.")

    return 0


def resolve_schrodinger_root(explicit_root: str | Path | None = None) -> Path:
    """Find a usable Schrodinger installation root."""
    candidates: list[Path] = []

    if explicit_root is not None:
        candidates.append(Path(explicit_root).expanduser())

    env_root = os.environ.get("SCHRODINGER")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    # Windows: C:\Program Files\Schrodinger*
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    if program_files.is_dir():
        installed_roots = [
            p for p in program_files.glob("Schrodinger*") if p.is_dir()
        ]
        installed_roots.sort(key=_schrodinger_sort_key, reverse=True)
        candidates.extend(installed_roots)

    # macOS / Linux: /opt/schrodinger* and /Applications/SchrodingerSuites*
    for parent in (Path("/opt"), Path("/Applications")):
        if not parent.is_dir():
            continue
        found = [
            p for p in parent.glob("[Ss]chrodinger*") if p.is_dir()
        ] + [
            p for p in parent.glob("SchrodingerSuites*") if p.is_dir()
        ]
        found.sort(key=_schrodinger_sort_key, reverse=True)
        candidates.extend(found)

    prepwizard_name = "prepwizard.exe" if os.name == "nt" else "prepwizard"

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if (normalized / "utilities" / prepwizard_name).is_file():
            return normalized

    raise PreprocessError(
        "Could not find a Schrodinger installation. Set SCHRODINGER or pass "
        "--schrodinger-root."
    )


def _schrodinger_sort_key(path: Path) -> tuple[int, int]:
    """Sort Schrodinger installs newest-first when versioned names are present."""
    match = SCHRODINGER_VERSION_RE.fullmatch(path.name)
    if match is None:
        return (-1, -1)
    return (int(match.group("year")), int(match.group("release")))


def _validate_structure_path(path: Path, allowed_suffixes: Sequence[str], label: str) -> None:
    """Validate file suffixes accepted by this wrapper."""
    suffix = path.suffix.lower()
    if suffix not in allowed_suffixes:
        allowed_text = ", ".join(sorted(allowed_suffixes))
        raise PreprocessError(
            f"{label} file must use one of these extensions: {allowed_text}. Got: {path}"
        )


def _is_mmcif_path(path: Path) -> bool:
    """Return True if the path looks like an mmCIF/CIF file."""
    return path.suffix.lower() in SUPPORTED_MMCIF_EXTENSIONS


def _append_option(command: list[str], flag: str, value: object | None) -> None:
    """Append a flag/value pair when the value is present."""
    if value is not None:
        command.extend([flag, str(value)])


def _run_command(
    command: Sequence[str],
    *,
    env: dict[str, str],
    cwd: str | Path | None = None,
    stream: bool = False,
) -> None:
    """Run a command and raise a helpful error if it fails.

    When ``stream`` is True, the child's stdout/stderr go directly to the
    parent terminal so verbose Schrodinger debug output is visible in real
    time. The error message in that case won't be able to quote the
    captured output, so callers should tell the user to scroll up.
    """
    if stream:
        completed = subprocess.run(
            list(command),
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
        if completed.returncode != 0:
            raise PreprocessError(
                f"Command failed with exit code {completed.returncode}: "
                f"{_format_command(command)}\n\n"
                "Full stdout/stderr streamed above (scroll up). With "
                "--debug, the Schrodinger job directory is preserved in "
                "the current working directory; check *.log and any "
                "*_prep-*/ subdirectory for the traceback."
            )
        return

    completed = subprocess.run(
        list(command),
        env=env,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        message_parts = [
            f"Command failed with exit code {completed.returncode}: {_format_command(command)}"
        ]
        if stdout:
            message_parts.append(f"stdout:\n{stdout}")
        if stderr:
            message_parts.append(f"stderr:\n{stderr}")
        raise PreprocessError("\n\n".join(message_parts))


def _format_command(command: Sequence[str]) -> str:
    """Format a subprocess command for display."""
    return subprocess.list2cmdline(list(command))


if __name__ == "__main__":
    raise SystemExit(main())
