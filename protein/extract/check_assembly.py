# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/extract/check_assembly.py

Quick CLI that reports what biological assemblies an mmCIF declares and
whether running the pocket extractor with ``--biological-assembly`` would
change the result.

Usage::

    python -m protein.extract.check_assembly data/1stp/proteins/1stp.cif
    python -m protein.extract.check_assembly data/4hhb/proteins/4hhb.cif --json

Exit codes:
    0   No expansion needed (AU already equals bio-unit for every assembly,
        or no assemblies declared).
    1   At least one assembly differs from the AU - expansion would
        meaningfully change the extraction. Useful for shell-driven
        auto-selection, e.g.::

            if python -m protein.extract.check_assembly "$CIF"; then
                ./run_pipeline.sh "$PDB"
            else
                ./run_pipeline.sh "$PDB" --bio
            fi

    2   CLI usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._core import describe_biological_assemblies, PocketExtractionError


def _format_human(report: dict[str, object]) -> str:
    lines: list[str] = []
    au_chains = report["au_chains"]  # type: ignore[index]
    au_count = report["au_chain_count"]  # type: ignore[index]
    lines.append(f"Asymmetric unit: {au_count} chain(s) -> {list(au_chains)}")
    assemblies = report["assemblies"]  # type: ignore[index]
    if not assemblies:
        lines.append("No biological assemblies declared in this file.")
        return "\n".join(lines)
    lines.append(f"Biological assemblies declared: {len(assemblies)}")
    for a in assemblies:  # type: ignore[assignment]
        verdict = (
            "EXPAND (bio != AU)"
            if a["differs_from_au"]
            else "no-op (bio == AU, just renames chains)"
        )
        lines.append(
            f"  - id={a['id']!r} "
            f"({a['oligomeric_details']}): "
            f"{a['bio_chain_count']} chain(s) -> {verdict}"
        )
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report whether a given mmCIF's biological assembly differs "
            "from its asymmetric unit."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to an mmCIF (.cif) file.",
    )
    parser.add_argument(
        "--assembly-id",
        default="1",
        help=(
            "Assembly id to use for the exit-code decision (default: '1'). "
            "All declared assemblies are still listed in the report."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of human-readable text.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        report = describe_biological_assemblies(input_path)
    except PocketExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, default=list))
    else:
        print(_format_human(report))

    assemblies = report["assemblies"]  # type: ignore[index]
    target = next(
        (a for a in assemblies if a["id"] == args.assembly_id),  # type: ignore[union-attr]
        None,
    )
    if target is None:
        return 0  # chosen assembly not declared -> nothing to expand
    return 1 if target["differs_from_au"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
