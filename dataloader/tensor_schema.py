# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


MMGBSA_ENTITY_ORDER = ["complex", "receptor", "ligand", "differences"]
MMGBSA_COMPONENT_ORDER = [
    "BOND",
    "ANGLE",
    "DIHED",
    "VDWAALS",
    "EEL",
    "1-4 VDW",
    "1-4 EEL",
    "EGB",
    "ESURF",
    "G gas",
    "G solv",
    "TOTAL",
    "DELTA G gas",
    "DELTA G solv",
    "DELTA TOTAL",
]
MMGBSA_RESIDUE_CHANNELS = [
    "internal",
    "vdw",
    "elec",
    "polar",
    "nonpolar",
    "total",
]
MASKED_MODE_ORDER = ["ghost", "charge_only", "lj_only"]
LIGAND_ATOM_FEATURE_NAMES = [
    "atomic_num",
    "formal_charge",
    "partial_charge",
    "is_aromatic",
    "hybridization",
    "chiral_tag",
    "explicit_h",
    "implicit_h",
    "isotope",
]
LIGAND_EDGE_FEATURE_NAMES = [
    "bond_type",
    "is_conjugated",
    "is_in_ring",
    "bond_stereo",
    "bond_dir",
]
FG_FEATURE_NAMES = [
    "num_atoms",
    "total_partial_charge",
    "max_radial_extent",
    "mean_partial_charge",
]
POCKET_ATOM_FEATURE_NAMES = [
    "atomic_num",
    "residue_type_id",
    "residue_base_type_id",
    "residue_variant_id",
    "atom_name_id",
    "is_backbone",
    "is_sidechain",
    "is_water",
    "nearest_ligand_dist",
    "nearest_fg_centroid_dist",
]
RESIDUE_FEATURE_NAMES = [
    "residue_type_id",
    "residue_base_type_id",
    "residue_variant_id",
    "atom_count",
    "min_ligand_dist",
]


STANDARD_RESIDUE_VOCAB = [
    "UNK",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ASX",
    "GLX",
    "SEC",
    "PYL",
    "HOH",
    "WAT",
    "ZN",
    "MG",
    "CA",
    "NA",
    "K",
    "CL",
    "MN",
    "FE",
    "CO",
    "CU",
]
STANDARD_RESIDUE_BASE_VOCAB = [
    "UNK",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ASX",
    "GLX",
    "SEC",
    "PYL",
    "HOH",
    "WAT",
    "ZN",
    "MG",
    "CA",
    "NA",
    "K",
    "CL",
    "MN",
    "FE",
    "CO",
    "CU",
]
STANDARD_RESIDUE_VARIANT_VOCAB = [
    "NONE",
    "HID",
    "HIE",
    "HIP",
    "ASH",
    "GLH",
    "LYN",
    "CYM",
    "CYX",
    "CYN",
    "MSE",
]

BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}


@dataclass(slots=True)
class TensorVocab:
    """Mutable vocabularies for categorical tensor encodings."""

    fg_type: dict[str, int] = field(default_factory=lambda: {"UNK": 0})
    fg_parent_type: dict[str, int] = field(default_factory=lambda: {"NONE": 0, "UNK": 1})
    residue_type: dict[str, int] = field(
        default_factory=lambda: {name: i for i, name in enumerate(STANDARD_RESIDUE_VOCAB)}
    )
    residue_base_type: dict[str, int] = field(
        default_factory=lambda: {name: i for i, name in enumerate(STANDARD_RESIDUE_BASE_VOCAB)}
    )
    residue_variant: dict[str, int] = field(
        default_factory=lambda: {name: i for i, name in enumerate(STANDARD_RESIDUE_VARIANT_VOCAB)}
    )
    pocket_atom_name: dict[str, int] = field(default_factory=lambda: {"UNK": 0})

    def id_for_fg_type(self, value: str | None) -> int:
        token = (value or "UNK").strip() or "UNK"
        return self._get_or_add(self.fg_type, token)

    def id_for_fg_parent_type(self, value: str | None) -> int:
        token = "NONE" if not value else value.strip() or "NONE"
        return self._get_or_add(self.fg_parent_type, token)

    def id_for_residue_type(self, value: str | None) -> int:
        token = (value or "UNK").strip().upper() or "UNK"
        return self._get_or_add(self.residue_type, token)

    def id_for_residue_base_type(self, value: str | None) -> int:
        token = (value or "UNK").strip().upper() or "UNK"
        return self._get_or_add(self.residue_base_type, token)

    def id_for_residue_variant(self, value: str | None) -> int:
        token = (value or "NONE").strip().upper() or "NONE"
        return self._get_or_add(self.residue_variant, token)

    def id_for_pocket_atom_name(self, value: str | None) -> int:
        token = (value or "UNK").strip().upper() or "UNK"
        return self._get_or_add(self.pocket_atom_name, token)

    @staticmethod
    def _get_or_add(vocab: dict[str, int], token: str) -> int:
        if token not in vocab:
            vocab[token] = len(vocab)
        return vocab[token]


DEFAULT_TENSOR_VOCAB = TensorVocab()


def _ordered_tokens(vocab: dict[str, int]) -> list[str]:
    return [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]


def build_feature_schema(vocab: TensorVocab | None = None) -> dict[str, object]:
    vocab = vocab or DEFAULT_TENSOR_VOCAB
    return {
        "ligand": {
            "atom_feature_names": LIGAND_ATOM_FEATURE_NAMES,
            "edge_feature_names": LIGAND_EDGE_FEATURE_NAMES,
        },
        "functional_groups": {
            "feature_names": FG_FEATURE_NAMES,
        },
        "pocket": {
            "atom_feature_names": POCKET_ATOM_FEATURE_NAMES,
            "residue_feature_names": RESIDUE_FEATURE_NAMES,
        },
        "mmgbsa": {
            "entity_order": MMGBSA_ENTITY_ORDER,
            "component_order": MMGBSA_COMPONENT_ORDER,
            "residue_channels": MMGBSA_RESIDUE_CHANNELS,
            "masked_mode_order": MASKED_MODE_ORDER,
        },
        "categorical_vocabularies": {
            "fg_type": _ordered_tokens(vocab.fg_type),
            "fg_parent_type": _ordered_tokens(vocab.fg_parent_type),
            "residue_type": _ordered_tokens(vocab.residue_type),
            "residue_base_type": _ordered_tokens(vocab.residue_base_type),
            "residue_variant": _ordered_tokens(vocab.residue_variant),
            "pocket_atom_name": _ordered_tokens(vocab.pocket_atom_name),
        },
    }
