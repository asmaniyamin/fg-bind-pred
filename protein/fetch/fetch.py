# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/fetch/fetch.py

Download PDBx/mmCIF coordinate files from RCSB PDB and verify that the
atom_site table still contains per-atom B-factor values.

The official RCSB file download service exposes uncompressed mmCIF files at:
    https://files.rcsb.org/download/<pdb_id>.cif

Example:
    python protein/fetch/fetch.py 4hhb --output-dir data/proteins
"""

from __future__ import annotations

import argparse
import gzip
import re
import shlex
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable


DEFAULT_DOWNLOAD_ROOT = "https://files.rcsb.org/download"
DEFAULT_TIMEOUT_SECONDS = 30.0
MISSING_MMCIF_VALUES = frozenset({"?", "."})
FLOAT_WITH_OPTIONAL_ESD = re.compile(
    r"^(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\(\d+\))?$"
)

# Canonical experimental-method tokens that appear in _exptl.method.
# Matching is substring-based on the upper-cased value.
_X_RAY_TOKENS = ("X-RAY DIFFRACTION", "X-RAY POWDER DIFFRACTION", "NEUTRON DIFFRACTION")
_EM_TOKENS = ("ELECTRON MICROSCOPY", "ELECTRON CRYSTALLOGRAPHY", "CRYO-EM")
_NMR_TOKENS = ("NMR",)  # covers "SOLUTION NMR", "SOLID-STATE NMR"
_THEORETICAL_TOKENS = ("THEORETICAL MODEL", "PREDICTED")


class FetchError(RuntimeError):
    """Raised when an mmCIF file cannot be downloaded from RCSB."""


class BFactorError(FetchError):
    """Raised when the downloaded mmCIF file does not contain usable B-factors."""


class MetadataError(FetchError):
    """Raised when required structure metadata is missing or unparseable."""


class ResolutionError(FetchError):
    """Raised when the structure's resolution is worse than the configured cutoff."""


class CompletenessError(FetchError):
    """Raised when reflection completeness is below the configured cutoff."""


class ExperimentalMethodError(FetchError):
    """Raised when the structure's experimental method is not in the allowed set."""


class NonXrayMethodWarning(UserWarning):
    """Emitted when a fetched structure is not determined by X-ray diffraction.

    Cryo-EM, NMR, and theoretical models do not carry X-ray-style B-factors;
    downstream pocket extraction and B-factor filtering should adapt.
    """


class LowResolutionWarning(UserWarning):
    """Emitted when a structure's resolution is borderline but not rejected."""


@dataclass(frozen=True, slots=True)
class BFactorSummary:
    """Summary statistics for the atom_site B-factor column."""

    atom_rows: int
    numeric_values: int
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    column_name: str = "_atom_site.B_iso_or_equiv"

    @property
    def has_numeric_values(self) -> bool:
        """Return True when at least one atom row has a numeric B-factor."""
        return self.numeric_values > 0


@dataclass(frozen=True, slots=True)
class StructureMetadata:
    """Experimental metadata extracted from an mmCIF header.

    All fields are optional because many mmCIF entries omit one or more of
    them (NMR entries often lack resolution and completeness; theoretical
    models typically lack all three).
    """

    method: str | None = None
    resolution_angstroms: float | None = None
    completeness_percent: float | None = None

    @property
    def normalized_method(self) -> str:
        return (self.method or "").strip().upper()

    @property
    def is_x_ray(self) -> bool:
        return any(token in self.normalized_method for token in _X_RAY_TOKENS)

    @property
    def is_electron_microscopy(self) -> bool:
        return any(token in self.normalized_method for token in _EM_TOKENS)

    @property
    def is_nmr(self) -> bool:
        return any(token in self.normalized_method for token in _NMR_TOKENS)

    @property
    def is_theoretical(self) -> bool:
        return any(token in self.normalized_method for token in _THEORETICAL_TOKENS)

    @property
    def has_reliable_b_factors(self) -> bool:
        """True when B-factors are expected to carry experimental meaning.

        X-ray refinement B-factors are directly comparable. Cryo-EM maps use
        B-factor sharpening, NMR ensembles rarely carry meaningful B-factors,
        and theoretical models have none at all.
        """
        return self.is_x_ray


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Result object returned after a successful download."""

    pdb_id: str
    output_path: Path
    source_url: str
    b_factors: BFactorSummary | None = None
    metadata: StructureMetadata | None = None


def normalize_pdb_id(pdb_id: str) -> str:
    """
    Normalize a user-supplied PDB identifier or file stem.

    Supports classic 4-character PDB IDs like ``4hhb`` and explicit file stems
    like ``pdb_00004hhb``.
    """
    token = pdb_id.strip()
    if not token:
        raise ValueError("PDB ID cannot be empty.")

    lower_token = token.lower()
    for suffix in (".cif.gz", ".cif"):
        if lower_token.endswith(suffix):
            token = token[: -len(suffix)]
            lower_token = token.lower()
            break

    if re.fullmatch(r"[a-z0-9]{4}", lower_token):
        return lower_token

    if re.fullmatch(r"pdb_[a-z0-9]{8,12}", lower_token):
        return lower_token

    raise ValueError(
        "Unsupported PDB identifier. Use a classic 4-character ID like "
        "'4hhb' or an explicit file stem like 'pdb_00004hhb'."
    )


def default_output_path(pdb_id: str, output_dir: str | Path | None = None) -> Path:
    """Build the default output path for a downloaded mmCIF file."""
    normalized = normalize_pdb_id(pdb_id)
    directory = Path.cwd() if output_dir is None else Path(output_dir)
    return directory / f"{normalized}.cif"


def build_download_urls(pdb_id: str) -> list[tuple[str, bool]]:
    """
    Build candidate RCSB download URLs.

    The short ``<pdb_id>.cif`` URL is tried first, followed by the explicit
    ``pdb_0000<pdb_id>.cif`` style that RCSB also documents for classic IDs.
    Gzipped variants are used as fallbacks and transparently decompressed.
    """
    normalized = normalize_pdb_id(pdb_id)
    stems = [normalized]

    if len(normalized) == 4:
        stems.append(f"pdb_0000{normalized}")

    urls: list[tuple[str, bool]] = []
    for stem in stems:
        urls.append((f"{DEFAULT_DOWNLOAD_ROOT}/{stem}.cif", False))
        urls.append((f"{DEFAULT_DOWNLOAD_ROOT}/{stem}.cif.gz", True))
    return urls


def fetch_mmcif(
    pdb_id: str,
    output_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    overwrite: bool = False,
    ensure_b_factors: bool = True,
    ensure_metadata: bool = True,
    min_resolution: float | None = None,
    min_completeness: float | None = None,
    require_xray: bool = False,
    on_warning: Callable[[str], None] | None = None,
) -> FetchResult:
    """
    Download an RCSB PDBx/mmCIF file and optionally verify metadata + B-factors.

    Args:
        pdb_id: Classic 4-character PDB ID or explicit RCSB file stem.
        output_path: Full path to the output ``.cif`` file.
        output_dir: Directory used when ``output_path`` is not provided.
        timeout: Network timeout in seconds.
        overwrite: Whether to replace an existing file.
        ensure_b_factors: If True, verify ``_atom_site.B_iso_or_equiv`` and
            collect summary statistics. For non-X-ray entries (cryo-EM, NMR,
            theoretical models) a missing or empty B-factor column is
            downgraded to a warning rather than an error.
        ensure_metadata: If True, parse ``_exptl.method``, resolution, and
            completeness from the mmCIF header and run the validation gates.
        min_resolution: Maximum acceptable d-spacing in Å. Values *worse than*
            this (numerically larger) cause :class:`ResolutionError`.
        min_completeness: Minimum acceptable reflection completeness as a
            percentage. Values below this cause :class:`CompletenessError`.
        require_xray: If True, reject any non-X-ray experimental method with
            :class:`ExperimentalMethodError` instead of just warning.
        on_warning: Optional callable invoked with each warning string. Useful
            for surfacing warnings in batch drivers without relying on the
            ``warnings`` module.

    Returns:
        FetchResult describing the downloaded file.
    """
    normalized = normalize_pdb_id(pdb_id)
    target_path = (
        Path(output_path)
        if output_path is not None
        else default_output_path(normalized, output_dir=output_dir)
    )

    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without overwrite=True: {target_path}"
        )

    payload, source_url = _download_mmcif_bytes(normalized, timeout=timeout)
    cif_text = payload.decode("utf-8")

    metadata = parse_structure_metadata(cif_text) if ensure_metadata else None

    if metadata is not None:
        validate_structure_metadata(
            metadata,
            min_resolution=min_resolution,
            min_completeness=min_completeness,
            require_xray=require_xray,
            on_warning=on_warning,
        )

    b_factor_summary: BFactorSummary | None = None
    if ensure_b_factors:
        try:
            b_factor_summary = summarize_b_factors(cif_text)
        except BFactorError as exc:
            # X-ray entries must have B-factors; for other methods, demote
            # to a warning so the pipeline can still consume the file.
            if metadata is not None and not metadata.has_reliable_b_factors:
                _emit_warning(
                    f"{normalized}: missing or empty B-factors "
                    f"(method={metadata.method or 'unknown'}); continuing without "
                    "B-factor summary.",
                    NonXrayMethodWarning,
                    on_warning,
                )
            else:
                raise

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(payload)

    return FetchResult(
        pdb_id=normalized,
        output_path=target_path,
        source_url=source_url,
        b_factors=b_factor_summary,
        metadata=metadata,
    )


# Candidate mmCIF keys per metadata field, in priority order.
# The first key present in the file wins.
_METHOD_KEYS: tuple[str, ...] = ("_exptl.method",)
_RESOLUTION_KEYS: tuple[str, ...] = (
    "_refine.ls_d_res_high",
    "_reflns.d_resolution_high",
    "_em_3d_reconstruction.resolution",
)
_COMPLETENESS_KEYS: tuple[str, ...] = (
    "_refine.ls_percent_reflns_obs",
    "_reflns.percent_possible_obs",
)


def parse_structure_metadata(cif_text: str) -> StructureMetadata:
    """Parse experimental method, resolution, and completeness in a single pass.

    The mmCIF text is walked exactly once. Every candidate key for every
    field is collected on that walk; the scanner short-circuits as soon as
    all requested keys are resolved. Priority between fallback keys is
    expressed declaratively by the tuple order in :data:`_RESOLUTION_KEYS`
    and :data:`_COMPLETENESS_KEYS`.

    All fields are optional; missing values come back as ``None`` rather
    than raising. Use :func:`validate_structure_metadata` to enforce gates.
    """
    found = _scan_cif_values(cif_text, _METHOD_KEYS + _RESOLUTION_KEYS + _COMPLETENESS_KEYS)

    def _first(candidates: tuple[str, ...]) -> str | None:
        return next((found[k] for k in candidates if k in found), None)

    def _first_float(candidates: tuple[str, ...]) -> float | None:
        raw = _first(candidates)
        return _parse_cif_float(raw.strip()) if raw is not None else None

    return StructureMetadata(
        method=_first(_METHOD_KEYS),
        resolution_angstroms=_first_float(_RESOLUTION_KEYS),
        completeness_percent=_first_float(_COMPLETENESS_KEYS),
    )


def validate_structure_metadata(
    metadata: StructureMetadata,
    *,
    min_resolution: float | None = None,
    min_completeness: float | None = None,
    require_xray: bool = False,
    on_warning: Callable[[str], None] | None = None,
) -> None:
    """Enforce optional gates on parsed structure metadata.

    Raises:
        ExperimentalMethodError: If ``require_xray`` is True and the method
            is not X-ray (or is missing).
        ResolutionError: If a numeric resolution is present and worse than
            ``min_resolution``.
        CompletenessError: If a numeric completeness is present and below
            ``min_completeness``.

    Also emits :class:`NonXrayMethodWarning` for cryo-EM / NMR / theoretical
    structures so downstream B-factor consumers can adapt.
    """
    method_label = metadata.method or "unknown"

    if require_xray and not metadata.is_x_ray:
        raise ExperimentalMethodError(
            f"Experimental method '{method_label}' is not X-ray diffraction; "
            "pass require_xray=False to accept."
        )

    if not metadata.is_x_ray:
        if metadata.is_electron_microscopy:
            kind = "cryo-EM / electron microscopy"
        elif metadata.is_nmr:
            kind = "NMR"
        elif metadata.is_theoretical:
            kind = "theoretical / predicted model"
        else:
            kind = f"non-X-ray ({method_label})"
        _emit_warning(
            f"Structure was determined by {kind}; B-factors may be missing, "
            "uniform, or sharpening-derived rather than experimental.",
            NonXrayMethodWarning,
            on_warning,
        )

    if (
        min_resolution is not None
        and metadata.resolution_angstroms is not None
        and metadata.resolution_angstroms > min_resolution
    ):
        raise ResolutionError(
            f"Resolution {metadata.resolution_angstroms:.2f} Å is worse than "
            f"the configured cutoff of {min_resolution:.2f} Å."
        )

    if (
        min_completeness is not None
        and metadata.completeness_percent is not None
        and metadata.completeness_percent < min_completeness
    ):
        raise CompletenessError(
            f"Completeness {metadata.completeness_percent:.1f}% is below the "
            f"configured cutoff of {min_completeness:.1f}%."
        )


def _emit_warning(
    message: str,
    category: type[Warning],
    on_warning: Callable[[str], None] | None,
) -> None:
    """Route a warning through both ``warnings.warn`` and an optional sink."""
    warnings.warn(message, category, stacklevel=2)
    if on_warning is not None:
        on_warning(message)


def summarize_b_factors(cif_text: str) -> BFactorSummary:
    """
    Summarize ``_atom_site.B_iso_or_equiv`` values from an mmCIF file.

    Raises:
        BFactorError: If the atom_site loop or B-factor column is missing, or if
            the column exists but contains no numeric values.
    """
    lines = cif_text.splitlines()
    line_count = len(lines)
    index = 0

    while index < line_count:
        stripped = lines[index].strip()
        if stripped != "loop_":
            index += 1
            continue

        index += 1
        columns: list[str] = []

        while index < line_count:
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped.startswith("_"):
                columns.append(stripped.split()[0])
                index += 1
                continue
            break

        if not columns or not any(column.startswith("_atom_site.") for column in columns):
            continue

        if "_atom_site.B_iso_or_equiv" not in columns:
            raise BFactorError(
                "Downloaded mmCIF contains an atom_site loop but no "
                "'_atom_site.B_iso_or_equiv' column."
            )

        column_count = len(columns)
        b_factor_index = columns.index("_atom_site.B_iso_or_equiv")
        atom_rows = 0
        values: list[float] = []
        token_buffer: list[str] = []

        while index < line_count:
            raw_line = lines[index]
            stripped = raw_line.strip()

            if not stripped:
                index += 1
                continue

            if stripped == "#":
                break

            if stripped == "loop_" or stripped.startswith("data_"):
                break

            if stripped.startswith("_") and not token_buffer:
                break

            if raw_line.startswith(";"):
                raise BFactorError(
                    "Encountered unexpected multiline text while parsing atom_site."
                )

            token_buffer.extend(_tokenize_mmcif_line(raw_line))

            while len(token_buffer) >= column_count:
                row = token_buffer[:column_count]
                del token_buffer[:column_count]
                atom_rows += 1
                parsed_value = _parse_cif_float(row[b_factor_index])
                if parsed_value is not None:
                    values.append(parsed_value)

            index += 1

        if token_buffer:
            raise BFactorError("atom_site loop appears truncated or malformed.")

        if not values:
            raise BFactorError(
                "Downloaded mmCIF has a B-factor column, but no numeric "
                "B-factor values were found."
            )

        return BFactorSummary(
            atom_rows=atom_rows,
            numeric_values=len(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=mean(values),
        )

    raise BFactorError("No atom_site loop was found in the downloaded mmCIF file.")


def _download_mmcif_bytes(pdb_id: str, *, timeout: float) -> tuple[bytes, str]:
    """Try documented RCSB download URLs until one succeeds."""
    errors: list[str] = []

    for url, is_gzipped in build_download_urls(pdb_id):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "small-molecule-mod-toolkit-cs273b/1.0 "
                    "(protein fetcher; contact local user)"
                )
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
            continue
        except urllib.error.URLError as exc:
            errors.append(f"{url} -> {exc.reason}")
            continue

        if is_gzipped:
            try:
                payload = gzip.decompress(payload)
            except OSError as exc:
                raise FetchError(f"Failed to decompress {url}: {exc}") from exc

        return payload, url

    raise FetchError(
        "Unable to download an mmCIF file from RCSB for "
        f"{pdb_id}. Attempted URLs: {', '.join(errors)}"
    )


def _tokenize_mmcif_line(line: str) -> list[str]:
    """Split one mmCIF data line into tokens while respecting quotes."""
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _parse_cif_float(token: str) -> float | None:
    """Parse a numeric mmCIF token, including values with uncertainty suffixes."""
    if token in MISSING_MMCIF_VALUES:
        return None

    match = FLOAT_WITH_OPTIONAL_ESD.fullmatch(token)
    if not match:
        return None

    return float(match.group("value"))


# ---------------------------------------------------------------------------
# Single-pass mmCIF scanner
# ---------------------------------------------------------------------------
#
# The scanner walks the file exactly once, looking for any of the requested
# keys. A key can appear in one of three syntactic forms (see PDBx/mmCIF spec):
#
#   (a) inline scalar      `_key value`
#   (b) next-line scalar   `_key\n value`   or   `_key\n;multi-line\n;`
#   (c) loop_ column       first data row of a `loop_` block
#
# The main loop dispatches on the current line: a `loop_` sentinel hands off
# to `_read_loop_first_row`, a line that starts with one of the wanted keys
# hands off to `_value_from_scalar`, and anything else is noise to skip over.
# The scan short-circuits as soon as every wanted key has been resolved.


def _scan_cif_values(
    cif_text: str, keys: Iterable[str]
) -> dict[str, str]:
    """Return the first non-missing value for each key, in one pass.

    Missing-value sentinels (``?`` / ``.``) are treated as "key not found"
    rather than returned verbatim. For a key that appears multiple times,
    the first occurrence wins. For loop columns, only the first data row
    is consulted - all metadata we care about is single-valued per entry.
    """
    wanted: set[str] = set(keys)
    found: dict[str, str] = {}
    lines = cif_text.splitlines()
    n = len(lines)
    i = 0

    while i < n and wanted:
        raw = lines[i]
        stripped = raw.strip()

        if stripped == "loop_":
            hits, i = _read_loop_first_row(lines, i + 1, n, wanted)
            for key, value in hits.items():
                if value not in MISSING_MMCIF_VALUES:
                    found[key] = value
                wanted.discard(key)
            continue

        matched = _match_scalar_key(stripped, wanted)
        if matched is not None:
            value, i = _value_from_scalar(lines, i, n, matched)
            if value is not None and value not in MISSING_MMCIF_VALUES:
                found[matched] = value
            wanted.discard(matched)
            continue

        i += 1

    return found


def _extract_cif_value(cif_text: str, key: str) -> str | None:
    """Single-key convenience wrapper around :func:`_scan_cif_values`."""
    return _scan_cif_values(cif_text, (key,)).get(key)


def _extract_cif_float(cif_text: str, key: str) -> float | None:
    """Extract ``key`` and parse it as a float; ``None`` if absent or non-numeric."""
    raw = _extract_cif_value(cif_text, key)
    return _parse_cif_float(raw.strip()) if raw is not None else None


def _match_scalar_key(stripped_line: str, wanted: set[str]) -> str | None:
    """Return the key in ``wanted`` (if any) that begins this line.

    The match requires a word boundary after the key so that
    ``_refine.ls_d_res_high`` does not incidentally match
    ``_refine.ls_d_res_high_esd``.
    """
    for key in wanted:
        if _line_starts_with_key(stripped_line, key):
            return key
    return None


def _line_starts_with_key(stripped_line: str, key: str) -> bool:
    """True iff a stripped line begins with ``key`` followed by EOL or whitespace."""
    if not stripped_line.startswith(key):
        return False
    if len(stripped_line) == len(key):
        return True
    return stripped_line[len(key)].isspace()


def _value_from_scalar(
    lines: list[str], start: int, n: int, key: str
) -> tuple[str | None, int]:
    """Extract a scalar value for ``key`` starting at ``lines[start]``.

    Handles both forms (a) and (b) above. Returns ``(value, next_index)``.
    """
    stripped = lines[start].strip()
    rest = stripped[len(key):].strip()
    if rest:
        tokens = _tokenize_mmcif_line(rest)
        return (tokens[0] if tokens else None), start + 1

    # Value lives on a subsequent line, possibly as a `;...;` block.
    j = start + 1
    while j < n:
        line = lines[j]
        stripped2 = line.strip()
        if not stripped2:
            j += 1
            continue
        if line.startswith(";"):
            block: list[str] = [line[1:]]
            j += 1
            while j < n and not lines[j].startswith(";"):
                block.append(lines[j])
                j += 1
            return "\n".join(block).strip(), j + 1  # past closing `;`
        if stripped2.startswith("_") or stripped2 == "loop_" or stripped2.startswith("data_"):
            return None, j
        tokens = _tokenize_mmcif_line(line)
        return (tokens[0] if tokens else None), j + 1
    return None, j


def _read_loop_first_row(
    lines: list[str], start: int, n: int, wanted: set[str]
) -> tuple[dict[str, str], int]:
    """Parse a ``loop_`` block starting at ``lines[start]``.

    Returns ``(hits, next_index)`` where ``hits`` maps each wanted column
    (intersected with the loop's columns) to its first-row value. When the
    loop contains none of the wanted keys, ``hits`` is empty and the index
    is advanced past the whole loop body.
    """
    # Phase 1: column headers.
    columns: list[str] = []
    i = start
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("_"):
            columns.append(s.split()[0])
            i += 1
            continue
        break

    wanted_cols = [(idx, col) for idx, col in enumerate(columns) if col in wanted]

    # Phase 2: first data row (only if we actually want something from this loop).
    hits: dict[str, str] = {}
    if wanted_cols:
        tokens: list[str] = []
        col_count = len(columns)
        while i < n and len(tokens) < col_count:
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            if stripped == "#" or stripped == "loop_" or stripped.startswith("data_"):
                break
            if stripped.startswith("_") and not tokens:
                break
            if line.startswith(";"):
                block = [line[1:]]
                i += 1
                while i < n and not lines[i].startswith(";"):
                    block.append(lines[i])
                    i += 1
                tokens.append("\n".join(block).strip())
                i += 1
            else:
                tokens.extend(_tokenize_mmcif_line(line))
                i += 1
        if len(tokens) >= col_count:
            for idx, col in wanted_cols:
                hits[col] = tokens[idx]

    # Phase 3: fast-forward past the rest of the loop body so the main
    # scanner resumes at the next record, not in the middle of data rows.
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s == "#" or s == "loop_" or s.startswith("data_") or s.startswith("_"):
            break
        i += 1

    return hits, i


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the fetch CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Download a PDBx/mmCIF coordinate file from RCSB and verify that "
            "the file contains atom-level B-factors."
        )
    )
    parser.add_argument(
        "pdb_id",
        help="Classic 4-character PDB ID such as 4hhb.",
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Exact path for the downloaded .cif file.",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where <pdb_id>.cif will be written.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Network timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    parser.add_argument(
        "--skip-b-factor-check",
        action="store_true",
        help="Download the file without validating the atom_site B-factor column.",
    )
    parser.add_argument(
        "--skip-metadata-check",
        action="store_true",
        help=(
            "Skip parsing _exptl.method, resolution, and completeness from the "
            "mmCIF header. Metadata gating flags are ignored when this is set."
        ),
    )
    parser.add_argument(
        "--min-resolution",
        type=float,
        default=None,
        metavar="ANGSTROMS",
        help=(
            "Reject the structure if its refined resolution is *worse than* "
            "this value (numerically larger d-spacing in Å). Example: 3.0 "
            "rejects anything coarser than 3.0 Å."
        ),
    )
    parser.add_argument(
        "--min-completeness",
        type=float,
        default=None,
        metavar="PERCENT",
        help=(
            "Reject the structure if reflection completeness is below this "
            "percentage (0–100)."
        ),
    )
    parser.add_argument(
        "--require-xray",
        action="store_true",
        help=(
            "Reject any non-X-ray experimental method (cryo-EM, NMR, "
            "theoretical models). Default is to warn but accept."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    import sys

    args = parse_args(argv)

    def _stderr_warn(message: str) -> None:
        print(f"warning: {message}", file=sys.stderr)

    result = fetch_mmcif(
        args.pdb_id,
        output_path=args.output,
        output_dir=args.output_dir,
        timeout=args.timeout,
        overwrite=args.overwrite,
        ensure_b_factors=not args.skip_b_factor_check,
        ensure_metadata=not args.skip_metadata_check,
        min_resolution=args.min_resolution,
        min_completeness=args.min_completeness,
        require_xray=args.require_xray,
        on_warning=_stderr_warn,
    )

    print(f"Downloaded {result.pdb_id} from {result.source_url}")
    print(f"Saved mmCIF to {result.output_path.resolve()}")

    if result.metadata is not None:
        meta = result.metadata
        method = meta.method or "unknown"
        resolution = (
            f"{meta.resolution_angstroms:.2f} Å"
            if meta.resolution_angstroms is not None
            else "n/a"
        )
        completeness = (
            f"{meta.completeness_percent:.1f}%"
            if meta.completeness_percent is not None
            else "n/a"
        )
        print(
            "Metadata: "
            f"method={method}, "
            f"resolution={resolution}, "
            f"completeness={completeness}"
        )

    if result.b_factors is not None:
        summary = result.b_factors
        print(
            "B-factor summary: "
            f"rows={summary.atom_rows}, "
            f"numeric={summary.numeric_values}, "
            f"min={summary.min_value:.2f}, "
            f"mean={summary.mean_value:.2f}, "
            f"max={summary.max_value:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
