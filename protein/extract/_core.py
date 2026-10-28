# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/extract/_core.py

Low-level primitives shared between ``find_pocket_unprocessed.py`` and
``find_pocket_processed.py``:

    * registries and constants (water / solvent-ion exclusion, metals,
      cofactor families)
    * exception + dataclass types (ResidueKey, AtomRecord, TemplateBond,
      StructConnLink, LigandGroup, Simple* reduced-mmCIF adapters)
    * small parsing helpers (as_list, parse_int, normalize_*)
    * mmCIF loading and indexing
    * ``is_candidate_ligand_residue`` and group-classification helpers

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem

try:
    from openbabel import openbabel as ob
    from openbabel import pybel
except ImportError:  # pragma: no cover - environment-dependent fallback
    ob = None
    pybel = None

try:
    import gemmi  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-dependent fallback
    gemmi = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants and registries
# ---------------------------------------------------------------------------

DEFAULT_POCKET_RADIUS = 10.0
DEFAULT_MIN_HEAVY_ATOMS = 6
DEFAULT_OUTPUT_SUFFIX = "_pockets"

WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "H2O"})
COMMON_EXCLUDED_LIGANDS = frozenset(
    {
        "ACE", "ACT", "ACY", "BME", "BR", "BU1", "BU2", "BU3", "CA", "CD",
        "CL", "CO", "CU", "DMS", "DOD", "EDO", "EOH", "FE", "FMT", "GOL",
        "HG", "HOH", "IOD", "IPA", "K", "MES", "MG", "MN", "MPD", "NA",
        "NH4", "NI", "NO3", "PEG", "PGE", "PG4", "PO4", "SO4", "TRS", "UNX",
        "ZN",
    }
)
METAL_ELEMENTS = frozenset(
    {
        "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA",
        "ZN", "MN", "FE", "CO", "NI", "CU", "CD", "HG",
    }
)
STRUCT_CONN_GROUPING_TYPES = frozenset({"covale"})
BOND_LIKE_CONN_TYPES = frozenset({"covale", "disulf", "metalc"})
MISSING_CONN_VALUES = frozenset({"", ".", "?"})

# Known biochemical cofactor residue codes, grouped by family.
#
# These are PDB/CCD three-letter codes for molecules that are usually shared
# catalytic machinery rather than the compound of scientific interest. The
# default pipeline still extracts them as ligands so they remain analyzable,
# but they are tagged with a ``cofactor_family`` and ``categories=['cofactor']``
# in the manifest so downstream consumers can filter them in or out. The
# ``--exclude-cofactors`` CLI flag drops them entirely.
#
# Scope: high-confidence, commonly-encountered codes only. Chlorophylls and
# non-iron porphyrins are intentionally *not* grouped under ``heme`` to keep
# the family tags accurate.
COFACTOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "heme": ("HEM", "HEA", "HEB", "HEC", "HAS", "HDD", "HEO", "HEV", "VER", "SRM"),
    "flavin": ("FAD", "FDA", "FMN", "FMH", "RBF"),
    "nicotinamide": ("NAD", "NAI", "NAJ", "NAP", "NDP", "NHD", "NHO"),
    "coenzyme_a": ("COA", "ACO", "SCA", "MCA", "HMG", "HXC", "MYA"),
    "vitamin_b6": ("PLP", "PMP", "PDP"),
    "thiamine": ("TDP", "TPP", "TD6", "HTL"),
    "folate": ("THF", "FOL", "DHF", "MEF", "FON", "5FO", "MDH"),
    "adomet": ("SAM", "SAH", "SFG", "MTA"),
    "biotin": ("BTN",),
    "lipoate": ("LPA",),
    "quinone": ("UQ1", "UQ2", "UQ6", "UQ8", "UQ9", "UQ10", "MQ7", "MQ8", "MQ9", "MQE", "PQQ"),
    "cobalamin": ("B12", "B1Z", "CNC", "COB", "COY", "CBY"),
    "iron_sulfur": ("FES", "SF4", "F3S", "CLF", "FS4", "NFU", "SF3"),
}
COFACTOR_LOOKUP: dict[str, str] = {
    resname: family
    for family, resnames in COFACTOR_FAMILIES.items()
    for resname in resnames
}


# ---------------------------------------------------------------------------
# Exception and dataclass types
# ---------------------------------------------------------------------------

class PocketExtractionError(RuntimeError):
    """Raised when the mmCIF pocket extraction workflow fails."""


@dataclass(frozen=True, slots=True)
class ResidueKey:
    """Stable identifier for one residue instance."""

    chain_id: str
    resseq: int
    insertion_code: str
    resname: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_id": self.chain_id,
            "resseq": self.resseq,
            "insertion_code": self.insertion_code,
            "resname": self.resname,
        }

    def compact_label(self) -> str:
        suffix = self.insertion_code if self.insertion_code else ""
        return f"{self.chain_id}_{self.resname}_{self.resseq}{suffix}"


@dataclass(frozen=True, slots=True)
class AtomRecord:
    """Serializable atom record for ligands and pocket residues."""

    atom_index: int
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    residue_key: ResidueKey

    def to_dict(self) -> dict[str, object]:
        return {
            "atom_index": self.atom_index,
            "atom_name": self.atom_name,
            "element": self.element,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "residue": self.residue_key.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TemplateBond:
    """One mmCIF template bond from ``_chem_comp_bond``."""

    atom_name_1: str
    atom_name_2: str
    order: str
    aromatic: bool


@dataclass(frozen=True, slots=True)
class StructConnLink:
    """One residue-residue link from ``_struct_conn``.

    The first five fields come from the minimal unprocessed-path parse. The
    trailing three (``value_order``, ``details``, ``role``) carry processed
    mmCIF annotations that the unprocessed path simply leaves blank.
    """

    residue_key_1: ResidueKey
    atom_name_1: str
    residue_key_2: ResidueKey
    atom_name_2: str
    conn_type: str
    value_order: str = ""
    details: str = ""
    role: str = ""


@dataclass(frozen=True, slots=True)
class LigandGroup:
    """Connected ligand residue group to be extracted as one ligand."""

    ligand_id: str
    residue_keys: tuple[ResidueKey, ...]
    cofactor_family: str | None = None

    @property
    def categories(self) -> list[str]:
        """Tag labels for this ligand (currently just ``['cofactor']`` when applicable)."""
        return ["cofactor"] if self.cofactor_family else []


# ---------------------------------------------------------------------------
# Simple adapters used when Biopython can't parse a reduced mmCIF
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SimpleAtom:
    """Minimal atom adapter for processed mmCIF fallback parsing."""

    name: str
    element: str
    coord: np.ndarray

    def get_name(self) -> str:
        return self.name


@dataclass(slots=True)
class SimpleChain:
    """Minimal chain adapter for processed mmCIF fallback parsing."""

    id: str
    residues: list["SimpleResidue"] = field(default_factory=list)

    def __iter__(self):
        return iter(self.residues)


@dataclass(slots=True)
class SimpleResidue:
    """Minimal residue adapter for processed mmCIF fallback parsing."""

    resname: str
    id: tuple[str, int, str]
    parent: SimpleChain
    atoms: list[SimpleAtom] = field(default_factory=list)

    def __iter__(self):
        return iter(self.atoms)

    def get_atoms(self):
        return iter(self.atoms)


@dataclass(slots=True)
class SimpleModel:
    """Minimal model adapter for processed mmCIF fallback parsing."""

    index: int
    chains: list[SimpleChain]

    def __iter__(self):
        return iter(self.chains)


@dataclass(slots=True)
class SimpleStructure:
    """Minimal structure adapter for processed mmCIF fallback parsing."""

    models: list[SimpleModel]

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index: int) -> SimpleModel:
        return self.models[index]


# ---------------------------------------------------------------------------
# Scalar parsing helpers
# ---------------------------------------------------------------------------

def as_list(value: object | None) -> list[object]:
    """Normalize mmCIF dictionary values to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_int(value: object) -> int:
    """Parse integer-like mmCIF fields."""
    text = str(value).strip()
    if text in {"", ".", "?"}:
        raise ValueError("Missing integer value.")
    return int(float(text))


def normalize_insertion_code(value: object) -> str:
    """Normalize mmCIF insertion codes so missing values become empty strings."""
    text = str(value).strip()
    if text in {"", ".", "?", " "}:
        return ""
    return text


def normalize_element_symbol(raw: str) -> str:
    """Normalize an mmCIF element symbol to the case RDKit expects.

    RDKit's periodic table is case-sensitive: ``Chem.Atom("FE")`` raises
    ``RuntimeError: Element 'FE' not found`` because the lookup is keyed on
    ``"Fe"``. mmCIF's ``_atom_site.type_symbol`` column is all-caps, so every
    multi-letter element (Fe, Cl, Br, Mg, Ca, Zn, …) trips this unless the
    symbol is re-cased. Single-letter symbols round-trip cleanly through
    ``.capitalize()``.
    """
    return (raw or "").strip().capitalize()


# ---------------------------------------------------------------------------
# Cofactor classification
# ---------------------------------------------------------------------------

def classify_cofactor(resname: str) -> str | None:
    """Return the cofactor family for a residue name, or None if not a known cofactor."""
    return COFACTOR_LOOKUP.get(str(resname).strip().upper())


def classify_ligand_group(residue_keys: Sequence[ResidueKey]) -> str | None:
    """Return the cofactor family for a ligand group, or None if none apply.

    Iterates in caller-provided order (callers pass residues sorted by
    ``ligand_sort_key``) so the choice is deterministic when a group mixes
    a cofactor with covalent partners.
    """
    for key in residue_keys:
        family = classify_cofactor(key.resname)
        if family is not None:
            return family
    return None


# ---------------------------------------------------------------------------
# Residue / ligand identity helpers
# ---------------------------------------------------------------------------

def residue_key_from_biopython(chain_id: str, residue) -> ResidueKey:
    """Convert a Bio.PDB residue to a ResidueKey."""
    _, resseq, insertion_code = residue.id
    return ResidueKey(
        chain_id=str(chain_id).strip(),
        resseq=int(resseq),
        insertion_code=normalize_insertion_code(insertion_code),
        resname=str(residue.resname).strip().upper(),
    )


def ligand_sort_key(residue_key: ResidueKey) -> tuple[str, int, str, str]:
    """Stable sorting key for residues and ligands."""
    return (
        residue_key.chain_id,
        residue_key.resseq,
        residue_key.insertion_code,
        residue_key.resname,
    )


def make_ligand_id(index: int, residue_keys: Sequence[ResidueKey]) -> str:
    """Create a filesystem-safe ligand identifier."""
    import re
    residue_fragment = "__".join(key.compact_label() for key in residue_keys)
    safe_fragment = re.sub(r"[^A-Za-z0-9_.-]+", "_", residue_fragment)
    return f"ligand_{index:03d}_{safe_fragment}"


def collect_connected_component(
    start_key: ResidueKey,
    adjacency: dict[ResidueKey, set[ResidueKey]],
) -> set[ResidueKey]:
    """Collect one graph connected component of ligand residues."""
    stack = [start_key]
    component: set[ResidueKey] = set()
    while stack:
        current = stack.pop()
        if current in component:
            continue
        component.add(current)
        stack.extend(adjacency.get(current, ()))
    return component


def collect_residue_atoms(residues: Sequence[object]) -> list[AtomRecord]:
    """Collect atoms from one or more residues in a stable order."""
    atom_records: list[AtomRecord] = []
    atom_index = 0
    for residue in residues:
        residue_key = residue_key_from_biopython(residue.parent.id, residue)
        for atom in residue.get_atoms():
            x, y, z = atom.coord
            atom_records.append(
                AtomRecord(
                    atom_index=atom_index,
                    atom_name=atom.get_name().strip(),
                    element=atom.element,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    residue_key=residue_key,
                )
            )
            atom_index += 1
    return atom_records


# ---------------------------------------------------------------------------
# RDKit bond helpers
# ---------------------------------------------------------------------------

def sanitize_molecule_best_effort(mol: Chem.Mol) -> None:
    """Sanitize when possible but keep partially valid molecules writable."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)


# Human-readable names for Chem.SanitizeFlags so the audit can decompose the
# bitmask returned by SanitizeMol(catchErrors=True) into something the
# manifest can surface without shipping an integer.
_SANITIZE_FLAG_NAMES: tuple[tuple[str, int], ...] = tuple(
    (name, int(getattr(Chem.SanitizeFlags, name)))
    for name in (
        "SANITIZE_CLEANUP",
        "SANITIZE_PROPERTIES",
        "SANITIZE_SYMMRINGS",
        "SANITIZE_KEKULIZE",
        "SANITIZE_FINDRADICALS",
        "SANITIZE_SETAROMATICITY",
        "SANITIZE_SETCONJUGATION",
        "SANITIZE_SETHYBRIDIZATION",
        "SANITIZE_CLEANUPCHIRALITY",
        "SANITIZE_ADJUSTHS",
    )
    if hasattr(Chem.SanitizeFlags, name)
)


def validate_ligand_sanitization(mol: Chem.Mol) -> dict[str, object]:
    """Report whether ``mol`` fully passes RDKit sanitization, without mutating it.

    Runs ``Chem.SanitizeMol`` with ``catchErrors=True`` on a copy, so the
    caller's molecule (which will be written to SDF) is untouched. Returns
    a manifest-friendly dict:

        {
            "sanitized": bool,
            "failed_operations": [str, ...],   # empty when sanitized is True
            "error": str | None,               # set only when the call itself raised
        }

    Unlike :func:`sanitize_molecule_best_effort`, failure is *observable*
    here - the orchestrator decides the policy (record vs. raise).
    """
    try:
        probe = Chem.Mol(mol)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "sanitized": False,
            "failed_operations": [],
            "error": f"failed to copy mol for sanitization probe: {exc}",
        }

    try:
        failed_flag = Chem.SanitizeMol(
            probe,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL,
            catchErrors=True,
        )
    except Exception as exc:
        # catchErrors=True should return a bitmask rather than raise, but a
        # completely malformed mol can still blow up before it gets there.
        return {
            "sanitized": False,
            "failed_operations": [],
            "error": f"SanitizeMol raised despite catchErrors: {exc}",
        }

    failed_int = int(failed_flag)
    if failed_int == 0:
        return {"sanitized": True, "failed_operations": [], "error": None}

    failed_names = [name for name, flag in _SANITIZE_FLAG_NAMES if failed_int & flag]
    if not failed_names:
        failed_names = [f"raw_flag={failed_int}"]
    return {"sanitized": False, "failed_operations": failed_names, "error": None}


# Coordinate-sanity tunables. Picked to flag the obvious failure modes
# without false-positiving on legitimate edge cases:
#
#   * 0.7 Å lower bound is well below any chemically sensible
#     non-bonded contact (typical vdW contact ≈ 2.0-2.4 Å). A ligand
#     atom this close to a protein atom almost certainly indicates
#     a coordinate error or an alternate-conformation collision the
#     extractor failed to resolve.
#
#   * 8.0 Å upper bound on the closest ligand-protein contact catches
#     "the ligand is floating in space" cases. Pocket extraction's
#     whole point is *binding*: a ligand whose closest heavy atom is
#     more than 8 Å from any protein atom is not in a pocket.
#
# Both are intentionally generous - anything inside the band is
# left to downstream science to interpret.
DEFAULT_COORD_CLASH_FLOOR = 0.7
DEFAULT_COORD_ORPHAN_CEILING = 8.0


def audit_coordinate_sanity(
    *,
    ligand_atoms: Sequence["AtomRecord"],
    protein_atom_coords: "np.ndarray | None" = None,
    protein_kdtree: "object | None" = None,
    min_contact_floor: float = DEFAULT_COORD_CLASH_FLOOR,
    max_contact_ceiling: float = DEFAULT_COORD_ORPHAN_CEILING,
    report_cap: int = 5,
) -> dict[str, object]:
    """Report obvious coordinate-failure modes for the extracted ligand.

    Catches the two failure modes that are essentially always wrong:

        1. Non-finite coordinates. Any ligand atom with NaN or
           Inf in x/y/z is flagged. These come from corrupted input
           CIFs or arithmetic in upstream steps.
        2. Implausible ligand-protein geometry. Each ligand atom's
           closest protein atom is computed once via the prebuilt
           protein ``cKDTree``. A contact closer than
           ``min_contact_floor`` indicates a clash; the *minimum*
           closest contact across all ligand atoms exceeding
           ``max_contact_ceiling`` indicates an orphan ligand sitting
           outside any pocket.

    The ``protein_kdtree`` is the same one
    ``_search.build_pocket_search_index`` already builds for residue
    search, so this audit reuses it free of charge - no additional
    ``cKDTree`` build cost. ``protein_atom_coords`` is accepted only as
    a defensive fallback for callers that have coords but no tree
    (currently unused by the orchestrator).

    Returns a manifest-friendly dict; never mutates inputs. The
    orchestrator decides whether to escalate failure to an
    exception via ``--strict-validity`` (same gate as the other
    validity auditors).
    """
    issues: list[str] = []
    nan_examples: list[dict[str, object]] = []
    inf_examples: list[dict[str, object]] = []
    clash_examples: list[dict[str, object]] = []

    finite_coords: list[tuple[float, float, float]] = []
    finite_atom_records: list["AtomRecord"] = []
    nan_count = 0
    inf_count = 0

    for record in ligand_atoms:
        coords = (record.x, record.y, record.z)
        is_nan = any(c != c for c in coords)  # NaN-aware without numpy
        is_inf = any(c == float("inf") or c == float("-inf") for c in coords)
        if is_nan:
            nan_count += 1
            if len(nan_examples) < report_cap:
                nan_examples.append(_coord_atom_example(record))
            continue
        if is_inf:
            inf_count += 1
            if len(inf_examples) < report_cap:
                inf_examples.append(_coord_atom_example(record))
            continue
        finite_coords.append(coords)
        finite_atom_records.append(record)

    if nan_count:
        issues.append(f"{nan_count} ligand atoms with NaN coordinates")
    if inf_count:
        issues.append(f"{inf_count} ligand atoms with Inf coordinates")

    min_distance: float | None = None
    max_distance: float | None = None
    clash_count = 0
    orphan = False

    has_finite_ligand = bool(finite_coords)
    has_protein = protein_kdtree is not None or (
        protein_atom_coords is not None and len(protein_atom_coords) > 0
    )

    if has_finite_ligand and has_protein:
        ligand_array = np.asarray(finite_coords, dtype=np.float64)
        if protein_kdtree is not None:
            distances, _ = protein_kdtree.query(ligand_array, k=1)
            distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        else:
            # Defensive fallback: brute-force min only used when the
            # caller passes coords without a tree. Quadratic but safe.
            protein_array = np.asarray(protein_atom_coords, dtype=np.float64)
            deltas = ligand_array[:, None, :] - protein_array[None, :, :]
            distances = np.sqrt(np.sum(deltas * deltas, axis=2)).min(axis=1)

        min_distance = float(distances.min())
        max_distance = float(distances.max())

        clash_mask = distances < float(min_contact_floor)
        clash_count = int(clash_mask.sum())
        if clash_count:
            issues.append(
                f"{clash_count} ligand atoms within {min_contact_floor} Å of a "
                f"protein atom (closest = {min_distance:.3f} Å)"
            )
            for idx in np.flatnonzero(clash_mask)[:report_cap]:
                example = _coord_atom_example(finite_atom_records[int(idx)])
                example["distance_to_protein"] = float(distances[idx])
                clash_examples.append(example)

        if min_distance > float(max_contact_ceiling):
            orphan = True
            issues.append(
                f"closest ligand-protein contact {min_distance:.3f} Å exceeds "
                f"orphan ceiling {max_contact_ceiling} Å"
            )
    elif has_finite_ligand and not has_protein:
        # No protein to compare against (degenerate empty-protein case
        # the search index already short-circuits). Don't synthesize a
        # "pass": flag it so downstream code can decide.
        issues.append("no protein atoms available for distance check")

    coordinate_sanity_ok = not issues
    return {
        "coordinate_sanity_ok": coordinate_sanity_ok,
        "ligand_atom_count": len(ligand_atoms),
        "nan_atom_count": nan_count,
        "inf_atom_count": inf_count,
        "nan_atom_examples": nan_examples,
        "inf_atom_examples": inf_examples,
        "min_ligand_protein_distance": min_distance,
        "max_ligand_protein_distance": max_distance,
        "clash_atom_count": clash_count,
        "clash_atom_examples": clash_examples,
        "orphan_ligand": orphan,
        "min_contact_floor": float(min_contact_floor),
        "max_contact_ceiling": float(max_contact_ceiling),
        "issues": issues,
    }


def _coord_atom_example(record: "AtomRecord") -> dict[str, object]:
    """Render one offending atom as a JSON-clean dict for the audit report.

    NaN / Inf coordinates are coerced to ``None`` rather than emitted
    literally. Standard JSON has no representation for non-finite
    floats; Python's ``json`` module emits them as bareword ``NaN``/
    ``Infinity`` (non-standard, rejected by strict parsers) and
    round-tripping breaks equality because ``nan != nan``. Emitting
    ``None`` is the honest "this atom had no measurable position"
    answer, and stays parseable everywhere.
    """
    return {
        "residue": record.residue_key.to_dict(),
        "atom_name": record.atom_name,
        "x": _finite_or_none(record.x),
        "y": _finite_or_none(record.y),
        "z": _finite_or_none(record.z),
    }


def _finite_or_none(value: float) -> float | None:
    """Replace NaN / +-Inf with None so the audit example is strict-JSON-clean."""
    if value != value:  # NaN
        return None
    if value == float("inf") or value == float("-inf"):
        return None
    return float(value)


def audit_atom_completeness(
    mol: Chem.Mol,
    source_atom_records: Sequence["AtomRecord"],
    *,
    missing_report_cap: int = 20,
) -> dict[str, object]:
    """Check that every ``(residue_key, atom_name)`` from the builder survived.

    The builder tags each RDKit atom with ``atom_name``/``chain_id``/
    ``resseq``/``resname``/``insertion_code`` in
    ``_populate_rdkit_mol_from_residues``. If the post-cleanup mol is missing
    any of those identifiers, an atom was silently dropped between
    construction and SDF write. When atoms lack tags entirely (e.g. an
    upstream step rebuilt the mol without preserving props), we fall back to
    a pure count comparison and flag it in ``tagging_gap``.

    The report is capped at ``missing_report_cap`` missing atoms so a
    pathological case doesn't bloat the manifest; ``missing_count`` is the
    true total.
    """
    expected: set[tuple[ResidueKey, str]] = {
        (record.residue_key, record.atom_name) for record in source_atom_records
    }
    preserved: set[tuple[ResidueKey, str]] = set()
    untagged_atoms = 0

    for atom in mol.GetAtoms():
        if not atom.HasProp("atom_name") or not atom.HasProp("chain_id"):
            untagged_atoms += 1
            continue
        try:
            key = (
                ResidueKey(
                    chain_id=atom.GetProp("chain_id"),
                    resseq=atom.GetIntProp("resseq"),
                    insertion_code=atom.GetProp("insertion_code")
                    if atom.HasProp("insertion_code")
                    else "",
                    resname=atom.GetProp("resname"),
                ),
                atom.GetProp("atom_name"),
            )
        except Exception:
            untagged_atoms += 1
            continue
        preserved.add(key)

    missing = sorted(
        expected - preserved,
        key=lambda item: (item[0].chain_id, item[0].resseq, item[1]),
    )

    # When tags are unreliable, the count comparison is the only honest
    # signal. Treat "complete" as "no tagged atom is missing AND the total
    # mol atom count is at least the source atom count" - the cleaner is
    # allowed to add atoms (explicit hydrogens) but not remove them.
    count_sufficient = mol.GetNumAtoms() >= len(source_atom_records)
    complete = not missing and count_sufficient

    return {
        "complete": complete,
        "expected_atom_count": len(expected),
        "preserved_tagged_atom_count": len(preserved),
        "total_mol_atom_count": mol.GetNumAtoms(),
        "untagged_atom_count": untagged_atoms,
        "missing_count": len(missing),
        "missing_atoms": [
            {"residue": residue_key.to_dict(), "atom_name": atom_name}
            for residue_key, atom_name in missing[:missing_report_cap]
        ],
    }


def map_template_bond_order(order: str, *, aromatic: bool) -> Chem.BondType:
    """Map mmCIF bond order tokens to RDKit bond types."""
    normalized = str(order).strip().lower()
    if aromatic or normalized in {"arom", "delo"}:
        return Chem.BondType.AROMATIC
    return {
        "sing": Chem.BondType.SINGLE,
        "doub": Chem.BondType.DOUBLE,
        "trip": Chem.BondType.TRIPLE,
        "quad": Chem.BondType.QUADRUPLE,
    }.get(normalized, Chem.BondType.SINGLE)


def map_openbabel_bond_order(order: int, *, aromatic: bool) -> Chem.BondType:
    """Map OpenBabel bond orders to RDKit bond types."""
    if aromatic:
        return Chem.BondType.AROMATIC
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.QUADRUPLE,
    }.get(int(order), Chem.BondType.SINGLE)


def add_rdkit_bond(
    rw_mol: Chem.RWMol,
    atom_index_1: int,
    atom_index_2: int,
    *,
    order: str,
    aromatic: bool,
) -> None:
    """Add a bond if it does not already exist."""
    if atom_index_1 == atom_index_2:
        return
    if rw_mol.GetBondBetweenAtoms(atom_index_1, atom_index_2) is not None:
        return

    bond_type = map_template_bond_order(order, aromatic=aromatic)
    rw_mol.AddBond(atom_index_1, atom_index_2, bond_type)
    bond = rw_mol.GetBondBetweenAtoms(atom_index_1, atom_index_2)
    if bond is not None and (aromatic or bond_type == Chem.BondType.AROMATIC):
        bond.SetIsAromatic(True)
        rw_mol.GetAtomWithIdx(atom_index_1).SetIsAromatic(True)
        rw_mol.GetAtomWithIdx(atom_index_2).SetIsAromatic(True)


# ---------------------------------------------------------------------------
# mmCIF loading and indexing
# ---------------------------------------------------------------------------

def load_structure_and_cif(input_path: Path):
    """Load the structure tree and raw mmCIF dictionary."""
    cif_dict = MMCIF2Dict(str(input_path))
    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    try:
        structure = parser.get_structure(input_path.stem, str(input_path))
    except KeyError:
        structure = build_simple_structure_from_atom_site(cif_dict)
    return structure, cif_dict


def expand_to_biological_assembly(
    input_path: Path,
    assembly_id: str = "1",
    output_path: Path | None = None,
) -> Path:
    """Expand ``input_path`` to the named biological assembly and write mmCIF.

    Uses gemmi to apply the symmetry operators declared in
    ``_pdbx_struct_assembly_gen`` / ``_pdbx_struct_oper_list``, giving
    symmetry copies distinct chain IDs (``A`` → ``A1``, ``A2``, ...).

    Returns the path to the expanded mmCIF file, which is suitable input to
    :func:`load_structure_and_cif`.

    If ``output_path`` is None, the expanded file is written next to the
    input as ``{stem}_assembly_{assembly_id}.cif``.
    """
    if gemmi is None:
        raise PocketExtractionError(
            "gemmi is required for biological-assembly expansion, "
            "but it is not installed. Install with `pip install gemmi`."
        )

    structure = gemmi.read_structure(str(input_path))
    available = [a.name for a in structure.assemblies]
    if not available:
        raise PocketExtractionError(
            f"No biological assemblies are declared in {input_path.name}. "
            "Run without --biological-assembly to extract on the asymmetric unit."
        )
    if assembly_id not in available:
        raise PocketExtractionError(
            f"Assembly '{assembly_id}' not found in {input_path.name}. "
            f"Available assemblies: {', '.join(available)}"
        )

    structure.transform_to_assembly(
        assembly_id,
        gemmi.HowToNameCopiedChain.AddNumber,
    )

    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_assembly_{assembly_id}.cif"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expanded_doc = structure.make_mmcif_document()
    expanded_block = expanded_doc.sole_block()

    # gemmi.make_mmcif_document() drops component-level dictionaries
    # (_chem_comp_atom, _chem_comp_bond). They are residue-type keyed, so
    # AU and bio-unit share identical rows. Splice them from the source CIF.
    source_doc = gemmi.cif.read_file(str(input_path))
    source_block = source_doc.sole_block()
    for category in ("_chem_comp_atom.", "_chem_comp_bond."):
        rows = source_block.get_mmcif_category(category)
        if rows and any(rows.values()):
            expanded_block.set_mmcif_category(category, rows)

    expanded_doc.write_file(str(output_path))
    return output_path


def describe_biological_assemblies(input_path: Path) -> dict[str, object]:
    """Summarise what biological assemblies are declared in ``input_path``.

    Returns a dict of the shape::

        {
            "au_chains": ["A"],
            "au_chain_count": 1,
            "assemblies": [
                {
                    "id": "1",
                    "oligomeric_details": "tetrameric",
                    "bio_chains": ["A1", "A2", "A3", "A4"],
                    "bio_chain_count": 4,
                    "differs_from_au": True,
                },
                ...
            ],
        }

    The ``differs_from_au`` flag is a coarse heuristic: True when the
    bio-unit has a different chain count than the asymmetric unit.
    That is the common "1 → tetramer" signal used to decide whether
    running with ``--biological-assembly`` would change the extraction.
    Equal chain counts usually mean the AU already *is* the bio-unit and
    expansion would only rename chains.
    """
    if gemmi is None:
        raise PocketExtractionError(
            "gemmi is required for biological-assembly inspection, "
            "but it is not installed. Install with `pip install gemmi`."
        )

    au = gemmi.read_structure(str(input_path))
    au_chains = [c.name for c in au[0]]

    assembly_reports: list[dict[str, object]] = []
    for assembly in au.assemblies:
        expanded = gemmi.read_structure(str(input_path))
        expanded.transform_to_assembly(
            assembly.name, gemmi.HowToNameCopiedChain.AddNumber
        )
        bio_chains = [c.name for c in expanded[0]]
        assembly_reports.append(
            {
                "id": assembly.name,
                "oligomeric_details": assembly.oligomeric_details,
                "bio_chains": bio_chains,
                "bio_chain_count": len(bio_chains),
                "differs_from_au": len(bio_chains) != len(au_chains),
            }
        )

    return {
        "au_chains": au_chains,
        "au_chain_count": len(au_chains),
        "assemblies": assembly_reports,
    }


def should_expand_to_biological_assembly(
    input_path: Path, assembly_id: str = "1"
) -> bool:
    """Return True iff the named assembly's chain count differs from the AU.

    Shortcut over :func:`describe_biological_assemblies` for the common
    decision "should I pass ``--biological-assembly`` to the extractor?".
    Returns False if no assemblies are declared.
    """
    report = describe_biological_assemblies(input_path)
    for assembly in report["assemblies"]:  # type: ignore[union-attr]
        if assembly["id"] == assembly_id:
            return bool(assembly["differs_from_au"])
    return False


def index_model_residues(model) -> dict[ResidueKey, object]:
    """Create a residue lookup keyed by chain/resseq/insertion code/resname."""
    residue_lookup: dict[ResidueKey, object] = {}
    for chain in model:
        for residue in chain:
            key = residue_key_from_biopython(chain.id, residue)
            residue_lookup[key] = residue
    return residue_lookup


def build_simple_structure_from_atom_site(cif_dict: dict[str, object]) -> SimpleStructure:
    """Fallback structure builder for reduced mmCIF files lacking Biopython-required fields."""
    required_keys = [
        "_atom_site.label_atom_id",
        "_atom_site.type_symbol",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    missing_keys = [key for key in required_keys if key not in cif_dict]
    if missing_keys:
        raise PocketExtractionError(
            "The mmCIF file is missing required atom_site columns for fallback parsing: "
            + ", ".join(missing_keys)
        )

    atom_names = as_list(cif_dict.get("_atom_site.label_atom_id"))
    elements = as_list(cif_dict.get("_atom_site.type_symbol"))
    resnames = as_list(cif_dict.get("_atom_site.label_comp_id"))
    chain_ids = as_list(cif_dict.get("_atom_site.auth_asym_id"))
    insertion_codes = as_list(cif_dict.get("_atom_site.pdbx_PDB_ins_code"))
    xs = as_list(cif_dict.get("_atom_site.Cartn_x"))
    ys = as_list(cif_dict.get("_atom_site.Cartn_y"))
    zs = as_list(cif_dict.get("_atom_site.Cartn_z"))

    if "_atom_site.auth_seq_id" in cif_dict:
        seq_ids = as_list(cif_dict.get("_atom_site.auth_seq_id"))
    elif "_atom_site.label_seq_id" in cif_dict:
        seq_ids = as_list(cif_dict.get("_atom_site.label_seq_id"))
    else:
        raise PocketExtractionError(
            "The mmCIF file does not contain _atom_site.auth_seq_id or _atom_site.label_seq_id."
        )

    component_metadata = build_component_metadata(cif_dict)
    chain_map: dict[str, SimpleChain] = {}
    residue_map: dict[tuple[str, int, str, str], SimpleResidue] = {}

    for values in zip(atom_names, elements, resnames, chain_ids, seq_ids, insertion_codes, xs, ys, zs):
        atom_name, element, resname, chain_id, seq_id, ins_code, x, y, z = values
        try:
            resseq = parse_int(seq_id)
        except ValueError:
            continue
        normalized_chain = str(chain_id).strip()
        normalized_resname = str(resname).strip().upper()
        normalized_ins_code = normalize_insertion_code(ins_code)
        hetflag = infer_hetflag_from_component_type(normalized_resname, component_metadata)

        chain = chain_map.get(normalized_chain)
        if chain is None:
            chain = SimpleChain(id=normalized_chain)
            chain_map[normalized_chain] = chain

        residue_key = (normalized_chain, resseq, normalized_ins_code, normalized_resname)
        residue = residue_map.get(residue_key)
        if residue is None:
            residue = SimpleResidue(
                resname=normalized_resname,
                id=(hetflag, resseq, normalized_ins_code if normalized_ins_code else " "),
                parent=chain,
            )
            residue_map[residue_key] = residue
            chain.residues.append(residue)

        residue.atoms.append(
            SimpleAtom(
                name=str(atom_name).strip(),
                element=str(element).strip(),
                coord=np.asarray([float(x), float(y), float(z)], dtype=np.float32),
            )
        )

    for chain in chain_map.values():
        for residue in chain.residues:
            atom_name_set = {atom.get_name().strip().upper() for atom in residue.atoms}
            hetflag, resseq, ins_code = residue.id
            if looks_like_polymer_backbone(atom_name_set):
                residue.id = (" ", resseq, ins_code)

    model = SimpleModel(index=0, chains=list(chain_map.values()))
    return SimpleStructure(models=[model])


def infer_hetflag_from_component_type(
    resname: str,
    component_metadata: dict[str, dict[str, str]],
) -> str:
    """Infer a Bio.PDB-style residue hetflag from chem-comp metadata."""
    metadata = component_metadata.get(resname)
    if metadata is None:
        return f"H_{resname}"
    component_type = metadata.get("type", "").lower()
    if "linking" in component_type and "peptide-like" not in component_type:
        return " "
    return f"H_{resname}"


def looks_like_polymer_backbone(atom_names: set[str]) -> bool:
    """Heuristic to suppress polymer residues when chem_comp metadata is sparse."""
    protein_backbone = {"N", "CA", "C", "O"}
    dna_backbone = {"P", "O5'", "C5'", "C4'", "C3'"}
    rna_backbone = {"P", "O5'", "C5'", "C4'", "C3'", "O2'"}
    return (
        protein_backbone.issubset(atom_names)
        or dna_backbone.issubset(atom_names)
        or rna_backbone.issubset(atom_names)
    )


def build_component_metadata(cif_dict: dict[str, object]) -> dict[str, dict[str, str]]:
    """Build a lookup of ``_chem_comp`` metadata keyed by residue name."""
    component_metadata: dict[str, dict[str, str]] = {}
    ids = as_list(cif_dict.get("_chem_comp.id"))
    types = as_list(cif_dict.get("_chem_comp.type"))
    names = as_list(cif_dict.get("_chem_comp.name"))
    for comp_id, comp_type, comp_name in zip(ids, types, names):
        component_metadata[str(comp_id).upper()] = {
            "type": str(comp_type),
            "name": str(comp_name),
        }
    return component_metadata


def build_template_bonds(cif_dict: dict[str, object]) -> dict[str, list[TemplateBond]]:
    """Build intra-component bond templates from ``_chem_comp_bond``."""
    bonds_by_component: dict[str, list[TemplateBond]] = defaultdict(list)
    comp_ids = as_list(cif_dict.get("_chem_comp_bond.comp_id"))
    atom_1 = as_list(cif_dict.get("_chem_comp_bond.atom_id_1"))
    atom_2 = as_list(cif_dict.get("_chem_comp_bond.atom_id_2"))
    orders = as_list(cif_dict.get("_chem_comp_bond.value_order"))
    aromatics = as_list(cif_dict.get("_chem_comp_bond.pdbx_aromatic_flag"))

    for comp_id, name_1, name_2, order, aromatic in zip(comp_ids, atom_1, atom_2, orders, aromatics):
        bonds_by_component[str(comp_id).upper()].append(
            TemplateBond(
                atom_name_1=str(name_1).strip(),
                atom_name_2=str(name_2).strip(),
                order=str(order).strip(),
                aromatic=str(aromatic).strip().upper() == "Y",
            )
        )
    return dict(bonds_by_component)


def mmcif_column_or_default(
    cif_dict: dict[str, object],
    key: str,
    row_count: int,
    default: str,
) -> list[str]:
    """Return an mmCIF loop column as strings, or a same-length default-filled list."""
    values = as_list(cif_dict.get(key))
    if not values:
        return [default] * row_count
    return [str(value) for value in values]


def normalize_conn_value(value: object) -> str:
    """Normalize a ``_struct_conn`` scalar field for comparison."""
    return str(value).strip().lower()


def is_bond_like_connection(link: StructConnLink) -> bool:
    """Processed-path filter: True when a record carries a real bond order,
    or when its ``conn_type`` is one of covalent/disulfide/metal coordination.
    Works correctly on unprocessed links too (they have ``value_order=""``,
    which is in ``MISSING_CONN_VALUES``, so the decision falls to ``conn_type``).
    """
    if normalize_conn_value(link.value_order) not in MISSING_CONN_VALUES:
        return True
    return link.conn_type in BOND_LIKE_CONN_TYPES


def build_struct_conn_links(cif_dict: dict[str, object]) -> list[StructConnLink]:
    """Parse residue-residue links from ``_struct_conn`` using auth IDs.

    Populates the superset of fields shared between the processed and
    unprocessed paths. When the extra columns (``pdbx_value_order``,
    ``details``, ``pdbx_role``) are absent, the corresponding fields default
    to the empty string - which the ``is_bond_like_connection`` filter
    correctly treats as "no explicit bond order".
    """
    conn_types = as_list(cif_dict.get("_struct_conn.conn_type_id"))
    atom_1 = as_list(cif_dict.get("_struct_conn.ptnr1_label_atom_id"))
    atom_2 = as_list(cif_dict.get("_struct_conn.ptnr2_label_atom_id"))
    chain_1 = as_list(cif_dict.get("_struct_conn.ptnr1_auth_asym_id"))
    chain_2 = as_list(cif_dict.get("_struct_conn.ptnr2_auth_asym_id"))
    resname_1 = as_list(cif_dict.get("_struct_conn.ptnr1_auth_comp_id"))
    resname_2 = as_list(cif_dict.get("_struct_conn.ptnr2_auth_comp_id"))
    resseq_1 = as_list(cif_dict.get("_struct_conn.ptnr1_auth_seq_id"))
    resseq_2 = as_list(cif_dict.get("_struct_conn.ptnr2_auth_seq_id"))
    ins_1 = as_list(cif_dict.get("_struct_conn.pdbx_ptnr1_PDB_ins_code"))
    ins_2 = as_list(cif_dict.get("_struct_conn.pdbx_ptnr2_PDB_ins_code"))
    row_count = len(conn_types)
    value_orders = mmcif_column_or_default(cif_dict, "_struct_conn.pdbx_value_order", row_count, "")
    details_col = mmcif_column_or_default(cif_dict, "_struct_conn.details", row_count, "")
    roles = mmcif_column_or_default(cif_dict, "_struct_conn.pdbx_role", row_count, "")

    links: list[StructConnLink] = []
    for values in zip(
        conn_types,
        chain_1, resname_1, resseq_1, ins_1, atom_1,
        chain_2, resname_2, resseq_2, ins_2, atom_2,
        value_orders, details_col, roles,
    ):
        (
            conn_type,
            auth_chain_1, auth_resname_1, auth_resseq_1, auth_ins_1, auth_atom_1,
            auth_chain_2, auth_resname_2, auth_resseq_2, auth_ins_2, auth_atom_2,
            value_order, detail_text, role_text,
        ) = values
        try:
            key_1 = ResidueKey(
                chain_id=str(auth_chain_1).strip(),
                resseq=parse_int(auth_resseq_1),
                insertion_code=normalize_insertion_code(auth_ins_1),
                resname=str(auth_resname_1).strip().upper(),
            )
            key_2 = ResidueKey(
                chain_id=str(auth_chain_2).strip(),
                resseq=parse_int(auth_resseq_2),
                insertion_code=normalize_insertion_code(auth_ins_2),
                resname=str(auth_resname_2).strip().upper(),
            )
        except ValueError:
            continue

        links.append(
            StructConnLink(
                residue_key_1=key_1,
                atom_name_1=str(auth_atom_1).strip(),
                residue_key_2=key_2,
                atom_name_2=str(auth_atom_2).strip(),
                conn_type=str(conn_type).strip().lower(),
                value_order=normalize_conn_value(value_order),
                details=str(detail_text).strip(),
                role=str(role_text).strip(),
            )
        )
    return links


# ---------------------------------------------------------------------------
# Ligand group detection (shared between unprocessed and processed paths)
# ---------------------------------------------------------------------------

def _default_is_grouping_bond(link: StructConnLink) -> bool:
    """Unprocessed-path grouping strategy: only ``covale`` links group residues."""
    return link.conn_type in STRUCT_CONN_GROUPING_TYPES


def detect_ligand_groups(
    *,
    residue_lookup: dict[ResidueKey, object],
    component_metadata: dict[str, dict[str, str]],
    struct_conn_links: Sequence[StructConnLink],
    include_resnames: set[str],
    exclude_resnames: set[str],
    min_heavy_atoms: int,
    exclude_cofactors: bool = False,
    is_grouping_bond: Callable[[StructConnLink], bool] | None = None,
) -> list[LigandGroup]:
    """Detect ligand residue groups, excluding common solvent/ion artifacts.

    ``exclude_cofactors``: when True, skip residues that ``classify_cofactor``
    recognizes (HEM, FAD, NAD, FMN, etc.). Force-included residues from
    ``include_resnames`` are still kept even when they are cofactors.

    ``is_grouping_bond``: predicate selecting ``_struct_conn`` records that
    should fuse two candidate residues into one ligand group. Defaults to
    the unprocessed-path rule (covalent links only). The processed path
    passes :func:`is_bond_like_connection` so explicit bond orders and
    disulfide / metal-coordination links also group.
    """
    if is_grouping_bond is None:
        is_grouping_bond = _default_is_grouping_bond

    candidate_keys: list[ResidueKey] = []
    for key, residue in residue_lookup.items():
        if not is_candidate_ligand_residue(
            residue_key=key,
            residue=residue,
            component_metadata=component_metadata,
            include_resnames=include_resnames,
            exclude_resnames=exclude_resnames,
            min_heavy_atoms=min_heavy_atoms,
        ):
            continue
        if (
            exclude_cofactors
            and key.resname not in include_resnames
            and classify_cofactor(key.resname) is not None
        ):
            continue
        candidate_keys.append(key)

    candidate_set = set(candidate_keys)
    adjacency: dict[ResidueKey, set[ResidueKey]] = {key: set() for key in candidate_keys}
    for link in struct_conn_links:
        if not is_grouping_bond(link):
            continue
        if link.residue_key_1 in candidate_set and link.residue_key_2 in candidate_set:
            adjacency[link.residue_key_1].add(link.residue_key_2)
            adjacency[link.residue_key_2].add(link.residue_key_1)

    groups: list[LigandGroup] = []
    seen: set[ResidueKey] = set()
    for index, start_key in enumerate(sorted(candidate_keys, key=ligand_sort_key), start=1):
        if start_key in seen:
            continue
        component = collect_connected_component(start_key, adjacency)
        seen.update(component)
        ordered_keys = tuple(sorted(component, key=ligand_sort_key))
        ligand_id = make_ligand_id(index, ordered_keys)
        groups.append(
            LigandGroup(
                ligand_id=ligand_id,
                residue_keys=ordered_keys,
                cofactor_family=classify_ligand_group(ordered_keys),
            )
        )
    return groups


# ---------------------------------------------------------------------------
# Candidate ligand residue classifier
# ---------------------------------------------------------------------------

def is_candidate_ligand_residue(
    *,
    residue_key: ResidueKey,
    residue,
    component_metadata: dict[str, dict[str, str]],
    include_resnames: set[str],
    exclude_resnames: set[str],
    min_heavy_atoms: int,
) -> bool:
    """Heuristic ligand classifier tuned to avoid waters, ions, and additives."""
    resname = residue_key.resname
    if resname in include_resnames:
        return True
    if resname in WATER_RESNAMES:
        return False
    if resname in COMMON_EXCLUDED_LIGANDS or resname in exclude_resnames:
        return False

    metadata = component_metadata.get(resname)
    if metadata is not None:
        component_type = metadata.get("type", "").lower()
        if "linking" in component_type and "peptide-like" not in component_type:
            return False

    atoms = list(residue.get_atoms())
    if not atoms:
        return False

    heavy_atom_count = sum(1 for atom in atoms if atom.element.upper() != "H")
    if heavy_atom_count < min_heavy_atoms:
        return False

    heavy_elements = {atom.element.upper() for atom in atoms if atom.element.upper() != "H"}
    if heavy_elements and heavy_elements.issubset(METAL_ELEMENTS):
        return False

    atom_names = {atom.get_name().strip().upper() for atom in atoms}
    if looks_like_polymer_backbone(atom_names):
        return False

    hetflag = str(residue.id[0]).strip()
    if metadata is None:
        return bool(hetflag and not hetflag.startswith("W"))

    return True
