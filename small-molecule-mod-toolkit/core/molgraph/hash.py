# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# core/molgraph/hash.py
from __future__ import annotations
import hashlib
import numpy as np
from core.structs import MolGraph

"""
RECOMMENDED USAGE:
mg = canonicalize(mg, stereo=False)[0] or stereo=True, depending on need
validate_molgraph(mg)
key = hash_molgraph(mg, mode="topology") # (or stereo) [NOTE 3]
"""

"""
WARNING
Stereo-canonical order is stereo-dependent, so you should not expect atom indices to line up across stereoisomers.
If you want to compare stereoisomers atom-by-atom you will need a mapping strategy (I recommend topology-based).
"""

def _sha(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def _i8(x):  return int(x).to_bytes(1, "little", signed=True)
def _u8(x):  return int(x).to_bytes(1, "little", signed=False)
def _i16(x): return int(x).to_bytes(2, "little", signed=True)
def _u16(x): return int(x).to_bytes(2, "little", signed=False)
def _i32(x): return int(x).to_bytes(4, "little", signed=True)
def _u32(x): return int(x).to_bytes(4, "little", signed=False)


def hash_topology(mg: MolGraph) -> str:
    a = mg.arrays
    n = int(a.atomic_num.shape[0])
    m = int(a.bonds.shape[0])

    parts = [b"TOPOv1"]

    # atoms (canonical order assumed)
    for i in range(n):
        parts.append(_i16(a.atomic_num[i]))
        parts.append(_i8(a.formal_charge[i]) if a.formal_charge is not None else _i8(0))
        parts.append(_u8(int(a.is_aromatic[i])) if a.is_aromatic is not None else _u8(0))
        parts.append(_i16(a.isotope[i]) if a.isotope is not None else _i16(0))
        parts.append(_i8(a.explicit_h[i]) if a.explicit_h is not None else _i8(0))
        parts.append(_i8(a.implicit_h[i]) if a.implicit_h is not None else _i8(0))

    # bonds (already normalized + sorted)
    for b in range(m):
        u, v = int(a.bonds[b, 0]), int(a.bonds[b, 1])
        parts.append(_u32(u)); parts.append(_u32(v))
        parts.append(_i8(a.bond_type[b]) if a.bond_type is not None else _i8(0))
        parts.append(_u8(int(a.is_conjugated[b])) if a.is_conjugated is not None else _u8(0))
        parts.append(_u8(int(a.is_in_ring[b])) if a.is_in_ring is not None else _u8(0))
        parts.append(_i8(a.bond_resonance_type[b]) if a.bond_resonance_type is not None else _i8(0))

    return _sha(b"".join(parts))


def hash_stereo(mg: MolGraph) -> str:
    a = mg.arrays
    n = int(a.atomic_num.shape[0])
    m = int(a.bonds.shape[0])

    parts = [b"STEREOv1"]

    for i in range(n):
        parts.append(_i16(a.atomic_num[i]))
        parts.append(_i8(a.formal_charge[i]) if a.formal_charge is not None else _i8(0))
        parts.append(_u8(int(a.is_aromatic[i])) if a.is_aromatic is not None else _u8(0))
        parts.append(_i16(a.isotope[i]) if a.isotope is not None else _i16(0))
        parts.append(_i8(a.explicit_h[i]) if a.explicit_h is not None else _i8(0))
        parts.append(_i8(a.implicit_h[i]) if a.implicit_h is not None else _i8(0))
        parts.append(_i8(a.chiral_tag[i]) if a.chiral_tag is not None else _i8(0))
        # cip_code is optional and not always stable; include only if you want:
        # parts.append(_i8(a.cip_code[i]) if a.cip_code is not None else _i8(0))

    for b in range(m):
        u, v = int(a.bonds[b, 0]), int(a.bonds[b, 1])
        parts.append(_u32(u)); parts.append(_u32(v))
        parts.append(_i8(a.bond_type[b]) if a.bond_type is not None else _i8(0))
        parts.append(_u8(int(a.is_conjugated[b])) if a.is_conjugated is not None else _u8(0))
        parts.append(_u8(int(a.is_in_ring[b])) if a.is_in_ring is not None else _u8(0))
        parts.append(_i8(a.bond_resonance_type[b]) if a.bond_resonance_type is not None else _i8(0))
        parts.append(_i8(a.bond_dir[b]) if a.bond_dir is not None else _i8(0))
        parts.append(_i8(a.bond_stereo[b]) if a.bond_stereo is not None else _i8(0))

    return _sha(b"".join(parts))


def hash_molgraph(mg: MolGraph, mode: str = "stereo") -> str:
    if mode == "topology":
        return hash_topology(mg)
    if mode == "stereo":
        return hash_stereo(mg)
    raise ValueError(f"unknown hash mode: {mode}")
