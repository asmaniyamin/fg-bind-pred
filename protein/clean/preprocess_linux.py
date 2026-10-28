# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/clean/preprocess_linux.py

Linux-friendly wrapper around ``preprocess.py``.

This keeps the original CLI surface but updates Schrodinger discovery for Linux
installs and starts the local Schrodinger job server automatically when needed.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

import preprocess as base


PREPWIZARD_NAMES = ("prepwizard", "prepwizard.exe")
STRUCTCONVERT_NAMES = ("structconvert", "structconvert.exe")
SCHRODINGER_VERSION_RE = re.compile(r"schrodinger(?P<year>\d+)-(?P<release>\d+)", re.IGNORECASE)


def resolve_schrodinger_tools(
    *,
    schrodinger_root: str | Path | None = None,
    prepwizard_exe: str | Path | None = None,
    structconvert_exe: str | Path | None = None,
) -> base.SchrodingerTools:
    """Resolve a usable Schrodinger install on Linux."""
    root = _resolve_schrodinger_root(
        schrodinger_root=schrodinger_root,
        prepwizard_exe=prepwizard_exe,
        structconvert_exe=structconvert_exe,
    )

    prepwizard = (
        Path(prepwizard_exe).expanduser().resolve()
        if prepwizard_exe is not None
        else _resolve_utility(root, PREPWIZARD_NAMES, "prepwizard")
    )
    structconvert = (
        Path(structconvert_exe).expanduser().resolve()
        if structconvert_exe is not None
        else _resolve_utility(root, STRUCTCONVERT_NAMES, "structconvert")
    )

    if not prepwizard.is_file():
        raise base.PreprocessError(f"Schrodinger prepwizard executable was not found: {prepwizard}")
    if not structconvert.is_file():
        raise base.PreprocessError(f"Schrodinger structconvert executable was not found: {structconvert}")

    return base.SchrodingerTools(
        root=root,
        prepwizard=prepwizard,
        structconvert=structconvert,
    )


def preprocess_structure(args) -> base.PreprocessResult:
    """Run Schrodinger Protein Preparation Wizard using parsed CLI arguments."""
    tools = resolve_schrodinger_tools(
        schrodinger_root=args.schrodinger_root,
        prepwizard_exe=args.prepwizard_exe,
        structconvert_exe=args.structconvert_exe,
    )

    input_path = Path(args.input_path).expanduser()
    output_path = Path(args.output_path).expanduser()

    base._validate_structure_path(input_path, base.SUPPORTED_PREPWIZARD_EXTENSIONS, "Input")
    base._validate_structure_path(output_path, base.SUPPORTED_PREPWIZARD_EXTENSIONS, "Output")

    if input_path.resolve() == output_path.resolve():
        raise base.PreprocessError("Input and output paths must be different.")

    if not args.dry_run and not input_path.is_file():
        raise FileNotFoundError(f"Input structure file was not found: {input_path}")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output without --overwrite: {output_path}"
        )

    if args.use_mae_intermediate and not base._is_mmcif_path(output_path):
        raise base.PreprocessError(
            "--use-mae-intermediate is only supported when the final output is "
            "mmCIF/CIF."
        )

    if args.use_mae_intermediate and not args.wait:
        raise base.PreprocessError(
            "A Maestro intermediate requires the Schrodinger job to finish "
            "before structconvert can run, so do not combine "
            "--use-mae-intermediate with --no-wait."
        )

    intermediate_path: Path | None = None
    structconvert_command: list[str] | None = None

    prepwizard_output = output_path

    if args.use_mae_intermediate:
        intermediate_path = (
            Path(args.intermediate_output).expanduser()
            if args.intermediate_output
            else output_path.parent / f"{output_path.stem}.prepwizard.maegz"
        )
        base._validate_structure_path(
            intermediate_path,
            base.SUPPORTED_MAESTRO_EXTENSIONS,
            "Intermediate Maestro output",
        )

        if intermediate_path.exists() and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing intermediate output without "
                f"--overwrite: {intermediate_path}"
            )

        prepwizard_output = intermediate_path
        structconvert_command = base.build_structconvert_command(
            tools.structconvert,
            intermediate_path,
            output_path,
        )

    prepwizard_command = base.build_prepwizard_command(
        tools.prepwizard,
        input_path,
        prepwizard_output,
        args,
    )

    if args.print_command or args.dry_run:
        print(base._format_command(prepwizard_command))
        if structconvert_command is not None:
            print(base._format_command(structconvert_command))

    if args.dry_run:
        return base.PreprocessResult(
            input_path=input_path,
            output_path=output_path,
            prepwizard_command=tuple(prepwizard_command),
            structconvert_command=tuple(structconvert_command) if structconvert_command else None,
            used_mae_intermediate=structconvert_command is not None,
            intermediate_path=intermediate_path,
        )

    schrodinger_home = getattr(args, "schrodinger_home", None)
    jobserver_dir = getattr(args, "jobserver_dir", None)
    schrodinger_tmpdir = getattr(args, "schrodinger_tmpdir", None)

    ensure_local_job_server(
        tools.root,
        schrodinger_home=schrodinger_home,
        jobserver_dir=jobserver_dir,
        schrodinger_tmpdir=schrodinger_tmpdir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if intermediate_path is not None:
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)

    env = build_schrodinger_runtime_env(
        tools.root,
        schrodinger_home=schrodinger_home,
        jobserver_dir=jobserver_dir,
        schrodinger_tmpdir=schrodinger_tmpdir,
    )

    launch_dir = intermediate_path.parent if intermediate_path is not None else output_path.parent

    base._run_command(prepwizard_command, env=env, cwd=launch_dir)

    if structconvert_command is not None:
        base._run_command(structconvert_command, env=env, cwd=launch_dir)
        if intermediate_path is not None and intermediate_path.exists() and not args.keep_intermediate:
            intermediate_path.unlink()

    return base.PreprocessResult(
        input_path=input_path,
        output_path=output_path,
        prepwizard_command=tuple(prepwizard_command),
        structconvert_command=tuple(structconvert_command) if structconvert_command else None,
        used_mae_intermediate=structconvert_command is not None,
        intermediate_path=intermediate_path,
    )


def build_schrodinger_runtime_env(
    root: Path,
    *,
    schrodinger_home: str | Path | None = None,
    jobserver_dir: str | Path | None = None,
    schrodinger_tmpdir: str | Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["SCHRODINGER"] = str(root)

    if schrodinger_home is not None:
        home_dir = Path(schrodinger_home).expanduser().resolve()
        home_dir.mkdir(parents=True, exist_ok=True)
        (home_dir / ".schrodinger").mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home_dir)

    if schrodinger_tmpdir is not None:
        tmpdir = Path(schrodinger_tmpdir).expanduser().resolve()
        tmpdir.mkdir(parents=True, exist_ok=True)
        env["SCHRODINGER_TMPDIR"] = str(tmpdir)
        env["TMPDIR"] = str(tmpdir)
        env["TMP"] = str(tmpdir)
        env["TEMP"] = str(tmpdir)

    if jobserver_dir is not None:
        local_server_dir = Path(jobserver_dir).expanduser().resolve()
        local_server_dir.mkdir(parents=True, exist_ok=True)
        env["SCHRODINGER_LOCAL_SERVER_DIR"] = str(local_server_dir)

    return env


def ensure_local_job_server(
    root: Path,
    *,
    schrodinger_home: str | Path | None = None,
    jobserver_dir: str | Path | None = None,
    schrodinger_tmpdir: str | Path | None = None,
) -> None:
    """Start the Schrodinger local job server when it is not already running."""
    jsc = root / "jsc"
    if not jsc.is_file():
        raise base.PreprocessError(f"Schrodinger jsc executable was not found: {jsc}")

    env = build_schrodinger_runtime_env(
        root,
        schrodinger_home=schrodinger_home,
        jobserver_dir=jobserver_dir,
        schrodinger_tmpdir=schrodinger_tmpdir,
    )

    status = subprocess.run(
        [str(jsc), "local-server-status"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if status.returncode == 0:
        return

    requested_local_server_dir = env.get("SCHRODINGER_LOCAL_SERVER_DIR")
    if requested_local_server_dir:
        configured = subprocess.run(
            [str(jsc), "local-server-dir", "--set", requested_local_server_dir],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if configured.returncode != 0:
            stdout = configured.stdout.strip()
            stderr = configured.stderr.strip()
            message_parts = [
                "Failed to configure Schrodinger local job server directory "
                f"with: {jsc} local-server-dir --set {requested_local_server_dir}"
            ]
            if stdout:
                message_parts.append(f"stdout:\n{stdout}")
            if stderr:
                message_parts.append(f"stderr:\n{stderr}")
            raise base.PreprocessError("\n\n".join(message_parts))

    started = subprocess.run(
        [str(jsc), "local-server-start"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if started.returncode != 0:
        stdout = started.stdout.strip()
        stderr = started.stderr.strip()
        message_parts = [f"Failed to start Schrodinger local job server with: {jsc} local-server-start"]
        if stdout:
            message_parts.append(f"stdout:\n{stdout}")
        if stderr:
            message_parts.append(f"stderr:\n{stderr}")
        raise base.PreprocessError("\n\n".join(message_parts))


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    args = base.parse_args(argv)
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


def _resolve_schrodinger_root(
    *,
    schrodinger_root: str | Path | None,
    prepwizard_exe: str | Path | None,
    structconvert_exe: str | Path | None,
) -> Path:
    candidates: list[Path] = []

    if schrodinger_root is not None:
        candidates.append(Path(schrodinger_root).expanduser())

    env_root = os.environ.get("SCHRODINGER")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    for tool_path in (prepwizard_exe, structconvert_exe):
        if tool_path is None:
            continue
        tool = Path(tool_path).expanduser().resolve()
        candidates.append(tool.parent.parent)

    candidates.extend(_discover_schrodinger_roots())

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen or not normalized.is_dir():
            continue
        seen.add(normalized)
        if _find_utility(normalized, PREPWIZARD_NAMES) is not None:
            return normalized

    raise base.PreprocessError(
        "Could not find a Schrodinger installation. Set SCHRODINGER, pass "
        "--schrodinger-root, or provide --prepwizard-exe/--structconvert-exe."
    )


def _discover_schrodinger_roots() -> list[Path]:
    roots: list[Path] = []
    patterns = (
        str(Path.home() / "schrodinger*"),
        str(Path.home() / "Schrodinger*"),
        "/opt/schrodinger*",
        "/opt/Schrodinger*",
        "/usr/local/schrodinger*",
        "/usr/local/Schrodinger*",
    )
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match).expanduser()
            if path.is_dir():
                roots.append(path)

    roots.sort(key=_schrodinger_sort_key, reverse=True)
    return roots


def _schrodinger_sort_key(path: Path) -> tuple[int, int]:
    match = SCHRODINGER_VERSION_RE.fullmatch(path.name)
    if match is None:
        return (-1, -1)
    return (int(match.group("year")), int(match.group("release")))


def _resolve_utility(root: Path, names: tuple[str, ...], label: str) -> Path:
    utility = _find_utility(root, names)
    if utility is None:
        raise base.PreprocessError(
            f"Could not find Schrodinger {label} under {root / 'utilities'} "
            f"(looked for {', '.join(names)})."
        )
    return utility


def _find_utility(root: Path, names: tuple[str, ...]) -> Path | None:
    utilities_dir = root / "utilities"
    for name in names:
        candidate = utilities_dir / name
        if candidate.is_file():
            return candidate.resolve()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
