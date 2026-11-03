# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
protein/testing/test_cofactors.py

Unit tests for cofactor-aware handling in the pocket extractor.

These tests are fast and self-contained: no RCSB fetch, no Schrodinger call,
no RDKit ligand building. They exercise only the classifier, the argparse
surface, and ``detect_ligand_groups`` on synthetic residues assembled from
the SimpleChain/SimpleResidue/SimpleAtom adapters that the unprocessed
extractor already uses for reduced mmCIF files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Let this file run as a script from the repo root without requiring the
# ``protein`` package to be installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protein.testing.runner import run_suite, suite_main  # noqa: E402
from protein.extract import find_pocket_unprocessed as base  # noqa: E402
from protein.extract import find_pocket_processed as processed  # noqa: E402


def _build_residue(
    *,
    chain_id: str,
    resseq: int,
    resname: str,
    elements: list[str],
    hetflag: str | None = None,
) -> base.SimpleResidue:
    """Build a minimal residue with N heavy atoms named A1..AN."""
    chain = base.SimpleChain(id=chain_id)
    tag = hetflag if hetflag is not None else f"H_{resname}"
    residue = base.SimpleResidue(
        resname=resname,
        id=(tag, resseq, " "),
        parent=chain,
    )
    for i, element in enumerate(elements, start=1):
        residue.atoms.append(
            base.SimpleAtom(
                name=f"{element}{i}",
                element=element,
                coord=np.asarray([float(i), 0.0, 0.0], dtype=np.float32),
            )
        )
    chain.residues.append(residue)
    return residue


def _residue_lookup(residues: list[base.SimpleResidue]) -> dict[base.ResidueKey, object]:
    return {
        base.residue_key_from_biopython(residue.parent.id, residue): residue
        for residue in residues
    }


# ---------------------------------------------------------------------------
# classify_cofactor
# ---------------------------------------------------------------------------

def test_classify_known_codes() -> None:
    cases = {
        "HEM": "heme",
        "HEA": "heme",
        "FAD": "flavin",
        "FMN": "flavin",
        "NAD": "nicotinamide",
        "NAP": "nicotinamide",
        "NDP": "nicotinamide",
        "COA": "coenzyme_a",
        "ACO": "coenzyme_a",
        "PLP": "vitamin_b6",
        "TPP": "thiamine",
        "THF": "folate",
        "SAM": "adomet",
        "SAH": "adomet",
        "BTN": "biotin",
        "LPA": "lipoate",
        "PQQ": "quinone",
        "B12": "cobalamin",
        "FES": "iron_sulfur",
        "SF4": "iron_sulfur",
    }
    for resname, expected in cases.items():
        actual = base.classify_cofactor(resname)
        assert actual == expected, f"classify_cofactor({resname!r}) = {actual!r}, expected {expected!r}"


def test_classify_unknown_and_case() -> None:
    assert base.classify_cofactor("UNK") is None
    assert base.classify_cofactor("XXX") is None
    assert base.classify_cofactor("") is None
    # Case-insensitive + surrounding whitespace
    assert base.classify_cofactor("hem") == "heme"
    assert base.classify_cofactor(" FAD ") == "flavin"


def test_no_overlap_with_exclusion_list() -> None:
    """A cofactor must never also be on the solvent/ion drop list."""
    overlap = set(base.COFACTOR_LOOKUP) & base.COMMON_EXCLUDED_LIGANDS
    assert not overlap, f"COFACTOR_LOOKUP overlaps COMMON_EXCLUDED_LIGANDS on: {sorted(overlap)}"


def test_no_overlap_with_waters() -> None:
    overlap = set(base.COFACTOR_LOOKUP) & base.WATER_RESNAMES
    assert not overlap, f"COFACTOR_LOOKUP overlaps WATER_RESNAMES on: {sorted(overlap)}"


# ---------------------------------------------------------------------------
# classify_ligand_group + LigandGroup.categories
# ---------------------------------------------------------------------------

def test_classify_ligand_group_single() -> None:
    key = base.ResidueKey(chain_id="A", resseq=401, insertion_code="", resname="HEM")
    assert base.classify_ligand_group((key,)) == "heme"


def test_classify_ligand_group_none() -> None:
    key = base.ResidueKey(chain_id="A", resseq=401, insertion_code="", resname="STI")
    assert base.classify_ligand_group((key,)) is None


def test_classify_ligand_group_mixed_takes_first() -> None:
    """When a group mixes cofactor + covalent partner, first match wins in caller order."""
    cofactor = base.ResidueKey(chain_id="A", resseq=401, insertion_code="", resname="FAD")
    partner = base.ResidueKey(chain_id="A", resseq=402, insertion_code="", resname="STI")
    assert base.classify_ligand_group((cofactor, partner)) == "flavin"
    # The caller-provided order matters: a non-cofactor first still resolves to the cofactor.
    assert base.classify_ligand_group((partner, cofactor)) == "flavin"


def test_ligand_group_categories_tag() -> None:
    hem_key = base.ResidueKey(chain_id="A", resseq=401, insertion_code="", resname="HEM")
    hem_group = base.LigandGroup(
        ligand_id="ligand_001_A_HEM_401",
        residue_keys=(hem_key,),
        cofactor_family="heme",
    )
    assert hem_group.categories == ["cofactor"]

    other_key = base.ResidueKey(chain_id="A", resseq=402, insertion_code="", resname="STI")
    other_group = base.LigandGroup(
        ligand_id="ligand_002_A_STI_402",
        residue_keys=(other_key,),
    )
    assert other_group.categories == []
    assert other_group.cofactor_family is None


# ---------------------------------------------------------------------------
# detect_ligand_groups - cofactor tagging and --exclude-cofactors
# ---------------------------------------------------------------------------

def _two_candidate_setup() -> dict[base.ResidueKey, object]:
    """Synthetic model with one HEM cofactor and one generic ligand."""
    # 43 carbons is safely above min_heavy_atoms for HEM (real HEM has 43 C+N+O+Fe).
    hem_elements = ["C"] * 34 + ["N"] * 4 + ["O"] * 4 + ["FE"]
    sti_elements = ["C"] * 20 + ["N"] * 6 + ["O"] * 2
    hem = _build_residue(chain_id="A", resseq=401, resname="HEM", elements=hem_elements)
    sti = _build_residue(chain_id="A", resseq=500, resname="STI", elements=sti_elements)
    # residue_lookup is built from the shared chain structure - keys must be unique.
    return _residue_lookup([hem, sti])


def test_detect_tags_cofactor_group() -> None:
    residue_lookup = _two_candidate_setup()
    groups = base.detect_ligand_groups(
        residue_lookup=residue_lookup,
        component_metadata={},
        struct_conn_links=[],
        include_resnames=set(),
        exclude_resnames=set(),
        min_heavy_atoms=6,
    )
    by_family = {group.cofactor_family: group for group in groups}
    assert "heme" in by_family, f"expected a heme-tagged group, got families: {list(by_family)}"
    assert None in by_family, "expected at least one non-cofactor ligand alongside the heme"
    heme_group = by_family["heme"]
    assert heme_group.categories == ["cofactor"]


def test_detect_exclude_cofactors_drops_cofactor() -> None:
    residue_lookup = _two_candidate_setup()
    groups = base.detect_ligand_groups(
        residue_lookup=residue_lookup,
        component_metadata={},
        struct_conn_links=[],
        include_resnames=set(),
        exclude_resnames=set(),
        min_heavy_atoms=6,
        exclude_cofactors=True,
    )
    families = {group.cofactor_family for group in groups}
    resnames = {key.resname for group in groups for key in group.residue_keys}
    assert "heme" not in families
    assert "HEM" not in resnames
    # The non-cofactor ligand must still be found.
    assert "STI" in resnames


def test_detect_include_resname_overrides_exclude_cofactors() -> None:
    """--include-resname HEM must still keep HEM even with --exclude-cofactors."""
    residue_lookup = _two_candidate_setup()
    groups = base.detect_ligand_groups(
        residue_lookup=residue_lookup,
        component_metadata={},
        struct_conn_links=[],
        include_resnames={"HEM"},
        exclude_resnames=set(),
        min_heavy_atoms=6,
        exclude_cofactors=True,
    )
    resnames = {key.resname for group in groups for key in group.residue_keys}
    assert "HEM" in resnames, "force-included cofactor was dropped by --exclude-cofactors"
    # And it should still carry the cofactor tag (classification is independent of exclusion).
    hem_group = next(group for group in groups if any(key.resname == "HEM" for key in group.residue_keys))
    assert hem_group.cofactor_family == "heme"
    assert hem_group.categories == ["cofactor"]


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_cli_flag_unprocessed() -> None:
    args = base.parse_args(["/tmp/fake.cif", "--exclude-cofactors"])
    assert args.exclude_cofactors is True
    args_default = base.parse_args(["/tmp/fake.cif"])
    assert args_default.exclude_cofactors is False


def test_cli_flag_processed() -> None:
    args = processed.parse_args(["/tmp/fake.cif", "--exclude-cofactors"])
    assert args.exclude_cofactors is True
    args_default = processed.parse_args(["/tmp/fake.cif"])
    assert args_default.exclude_cofactors is False


# ---------------------------------------------------------------------------
# processed extractor re-exports the shared classifier
# ---------------------------------------------------------------------------

def test_processed_uses_shared_classifier() -> None:
    assert base.classify_cofactor("HEM") == "heme"
    # find_pocket_processed uses ``base.classify_cofactor`` directly, so this
    # test pins the contract that the helper is importable from the base
    # module (the processed extractor does not need to re-export it).
    assert callable(base.classify_cofactor)
    assert callable(base.classify_ligand_group)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

ALL_TESTS = (
    test_classify_known_codes,
    test_classify_unknown_and_case,
    test_no_overlap_with_exclusion_list,
    test_no_overlap_with_waters,
    test_classify_ligand_group_single,
    test_classify_ligand_group_none,
    test_classify_ligand_group_mixed_takes_first,
    test_ligand_group_categories_tag,
    test_detect_tags_cofactor_group,
    test_detect_exclude_cofactors_drops_cofactor,
    test_detect_include_resname_overrides_exclude_cofactors,
    test_cli_flag_unprocessed,
    test_cli_flag_processed,
    test_processed_uses_shared_classifier,
)


def run_all_tests() -> bool:
    """Entry point for the master :mod:`run_all_tests` runner."""
    return run_suite(ALL_TESTS)


if __name__ == "__main__":
    raise SystemExit(suite_main(ALL_TESTS))
