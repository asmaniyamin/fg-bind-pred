#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Populate a dataset mirror root and optionally replace a repo-local dataset path
with a symlink to the mirrored location.

This is aimed at large PDBbind-style batches where storing prepared structures
directly in the repo tree becomes awkward.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing dataset tree into a mirror location and optionally "
            "swap in a symlink at the original repo path."
        )
    )
    parser.add_argument("--source-root", required=True, help="Existing dataset root to mirror.")
    parser.add_argument("--mirror-root", required=True, help="Destination mirror root.")
    parser.add_argument(
        "--link-path",
        default=None,
        help=(
            "Optional path that should become a symlink to mirror-root. "
            "If omitted, the script only populates the mirror."
        ),
    )
    parser.add_argument(
        "--copy-existing",
        action="store_true",
        help="Copy existing files from source-root into mirror-root before linking.",
    )
    parser.add_argument(
        "--replace-link",
        action="store_true",
        help=(
            "Allow replacing an existing non-symlink link-path by first moving it "
            "aside to a timestamped backup."
        ),
    )
    parser.add_argument(
        "--adopt-source",
        action="store_true",
        help=(
            "After the mirror has been populated, delete source-root and replace it "
            "with a symlink to mirror-root. This is destructive by design."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without modifying the filesystem.",
    )
    return parser.parse_args(argv)


def copy_tree(source_root: Path, mirror_root: Path, dry_run: bool) -> tuple[int, int]:
    copied_files = 0
    copied_dirs = 0
    for source_path in sorted(source_root.rglob("*")):
        relative = source_path.relative_to(source_root)
        target_path = mirror_root / relative
        if source_path.is_dir():
            copied_dirs += 1
            if not dry_run:
                target_path.mkdir(parents=True, exist_ok=True)
            continue
        copied_files += 1
        if dry_run:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    return copied_dirs, copied_files


def ensure_symlink(link_path: Path, mirror_root: Path, replace_link: bool, dry_run: bool) -> str:
    if link_path.is_symlink():
        target = link_path.resolve()
        if target == mirror_root.resolve():
            return f"{link_path} already points to {mirror_root}"
        raise RuntimeError(f"{link_path} is already a symlink to {target}, not {mirror_root}")

    if link_path.exists():
        if not replace_link:
            raise RuntimeError(
                f"{link_path} already exists and is not a symlink; rerun with --replace-link "
                "if you want it moved aside."
            )
        backup = link_path.with_name(f"{link_path.name}.bak.{time.strftime('%Y%m%d-%H%M%S')}")
        if dry_run:
            return f"Would move {link_path} -> {backup} and create symlink to {mirror_root}"
        link_path.rename(backup)
        link_path.symlink_to(mirror_root, target_is_directory=True)
        return f"Moved {link_path} -> {backup} and created symlink to {mirror_root}"

    if dry_run:
        return f"Would create symlink {link_path} -> {mirror_root}"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(mirror_root, target_is_directory=True)
    return f"Created symlink {link_path} -> {mirror_root}"


def adopt_source_root(source_root: Path, mirror_root: Path, dry_run: bool) -> str:
    if source_root.is_symlink():
        target = source_root.resolve()
        if target == mirror_root.resolve():
            return f"{source_root} already points to {mirror_root}"
        raise RuntimeError(f"{source_root} is already a symlink to {target}, not {mirror_root}")
    if dry_run:
        return f"Would delete {source_root} and replace it with a symlink to {mirror_root}"
    if not mirror_root.exists():
        raise RuntimeError(f"Mirror root does not exist yet: {mirror_root}")
    shutil.rmtree(source_root)
    source_root.symlink_to(mirror_root, target_is_directory=True)
    return f"Deleted {source_root} and replaced it with a symlink to {mirror_root}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = Path(args.source_root).expanduser().resolve()
    mirror_root = Path(args.mirror_root).expanduser().resolve()
    link_path = Path(args.link_path).expanduser().resolve() if args.link_path else None

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    if args.copy_existing:
        if args.dry_run:
            print(f"Would copy {source_root} -> {mirror_root}")
        else:
            mirror_root.mkdir(parents=True, exist_ok=True)
        copied_dirs, copied_files = copy_tree(source_root, mirror_root, args.dry_run)
        print(
            f"{'Would copy' if args.dry_run else 'Copied'} "
            f"{copied_dirs} directories and {copied_files} files into {mirror_root}"
        )
    elif not mirror_root.exists() and not args.dry_run:
        mirror_root.mkdir(parents=True, exist_ok=True)

    if link_path is not None:
        message = ensure_symlink(
            link_path=link_path,
            mirror_root=mirror_root,
            replace_link=args.replace_link,
            dry_run=args.dry_run,
        )
        print(message)
    if args.adopt_source:
        print(adopt_source_root(source_root, mirror_root, args.dry_run))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
