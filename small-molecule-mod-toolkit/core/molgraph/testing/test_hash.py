# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
test_hash.py

Tests for core/molgraph/hash.py (hash_topology/hash_stereo/hash_molgraph).
These tests assume hashes are computed on canonicalized graphs (recommended contract).

Run:
  python -m core.molgraph.testing.test_hash
(or call run_all_tests()).
"""

from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

import sys
import traceback
import numpy as np

from core.structs import MolGraph

"""
WARNING
Stereo-canonical order is stereo-dependent, so you should not expect atom indices to line up across stereoisomers.
If you want to compare stereoisomers atom-by-atom you will need a mapping strategy (I recommend topology-based).
"""


def debug_hash_inputs(mg: MolGraph, *, mode: str = "stereo") -> None:
    a = mg.arrays
    n = int(a.atomic_num.shape[0])
    m = int(a.bonds.shape[0])

    print("=" * 78)
    print(f"HASH DEBUG (mode={mode})")
    print("=" * 78)

    print("ATOMS (canonical order assumed):")
    for i in range(n):
        rec = {
            "idx": i,
            "Z": int(a.atomic_num[i]),
            "chg": int(a.formal_charge[i]) if a.formal_charge is not None else 0,
            "arom": int(a.is_aromatic[i]) if a.is_aromatic is not None else 0,
            "iso": int(a.isotope[i]) if a.isotope is not None else 0,
            "eH": int(a.explicit_h[i]) if a.explicit_h is not None else 0,
            "iH": int(a.implicit_h[i]) if a.implicit_h is not None else 0,
        }
        if mode == "stereo":
            rec["chir"] = int(a.chiral_tag[i]) if a.chiral_tag is not None else 0
            star = " *CHIRAL*" if rec["chir"] != 0 else ""
        else:
            star = ""
        print("  ", rec, star)
    
    # after printing atoms, if mode == "stereo"
    if mode == "stereo" and a.chiral_tag is not None:
        ch = [i for i in range(n) if int(a.chiral_tag[i]) != 0]
        print("CHIRAL ATOMS:", ch)

    print()
    print("BONDS (u < v, sorted):")
    for b in range(m):
        rec = {
            "idx": b,
            "u": int(a.bonds[b, 0]),
            "v": int(a.bonds[b, 1]),
            "type": int(a.bond_type[b]) if a.bond_type is not None else 0,
            "conj": int(a.is_conjugated[b]) if a.is_conjugated is not None else 0,
            "ring": int(a.is_in_ring[b]) if a.is_in_ring is not None else 0,
            "res": int(a.bond_resonance_type[b]) if a.bond_resonance_type is not None else 0,
        }
        if mode == "stereo":
            rec["dir"] = int(a.bond_dir[b]) if a.bond_dir is not None else 0
            rec["st"] = int(a.bond_stereo[b]) if a.bond_stereo is not None else 0
        print("  ", rec)   # <-- NO star here

    print()
    if mode == "topology":
        print("FINAL HASH:", hash_topology(mg))
    elif mode == "stereo":
        print("FINAL HASH:", hash_stereo(mg))
    else:
        raise ValueError(mode)

    print("=" * 78)


# --- builder (RDKit -> MolGraph) ---
try:
    from chem.build.create_molgraph import smiles_to_molgraph
except Exception as e:  # pragma: no cover
    smiles_to_molgraph = None  # type: ignore
    _IMPORT_BUILD_ERR = e
else:
    _IMPORT_BUILD_ERR = None

# --- canonicalize ---
_CANON_IMPORT_ERR = None
try:
    from core.molgraph.canonicalize import canonicalize
except Exception as e:  # pragma: no cover
    canonicalize = None  # type: ignore
    _CANON_IMPORT_ERR = e

# --- hash ---
_HASH_IMPORT_ERR = None
try:
    from core.molgraph.hash import hash_topology, hash_stereo, hash_molgraph
except Exception as e:  # pragma: no cover
    hash_topology = None  # type: ignore
    hash_stereo = None  # type: ignore
    hash_molgraph = None  # type: ignore
    _HASH_IMPORT_ERR = e

# --- permute_atoms (to simulate arbitrary atom ordering) ---
try:
    from chem.edit.atoms import permute_atoms
except Exception as e:  # pragma: no cover
    permute_atoms = None  # type: ignore
    _PERM_IMPORT_ERR = e
else:
    _PERM_IMPORT_ERR = None


def _build(smiles: str, *, coords=None):
    if smiles_to_molgraph is None:
        raise RuntimeError(f"smiles_to_molgraph import failed: {_IMPORT_BUILD_ERR}")
    mg = smiles_to_molgraph(smiles, compute_charges=True, keep_rdkit_mol=False, coords=coords)
    if mg is None:
        raise RuntimeError(f"Failed to build MolGraph from SMILES: {smiles}")
    return mg


def _canon(mg):
    if canonicalize is None:
        raise RuntimeError(f"canonicalize import failed: {_CANON_IMPORT_ERR}")
    # canonicalize() signature varies; we only need first return value
    out = canonicalize(mg)
    return out[0] if isinstance(out, tuple) else out


def test_hash_is_permutation_invariant_after_canonicalize():
    if hash_molgraph is None:
        raise RuntimeError(f"hash import failed: {_HASH_IMPORT_ERR}")
    if permute_atoms is None:
        raise RuntimeError(f"permute_atoms import failed: {_PERM_IMPORT_ERR}")

    mg = _build("CC(=O)O")  # acetic acid
    mg_c = _canon(mg)

    # Create an arbitrary permutation of atoms
    n = int(mg.arrays.atomic_num.shape[0])
    rng = np.random.default_rng(0)
    new_to_old = rng.permutation(n).astype(np.int64)

    mg_perm, _, _ = permute_atoms(mg, new_to_old)
    mg_perm_c = _canon(mg_perm)

    h1 = hash_molgraph(mg_c, mode="stereo")
    h2 = hash_molgraph(mg_perm_c, mode="stereo")
    assert h1 == h2, "Hash should be identical after canonicalize(), regardless of atom order"


def test_hash_ignores_coordinates():
    if hash_topology is None or hash_stereo is None:
        raise RuntimeError(f"hash import failed: {_HASH_IMPORT_ERR}")

    # Same molecule, one built without coords, one with coords
    mg0 = _canon(_build("Cc1ccccc1", coords=None))          # toluene, no coords
    mg1 = _canon(_build("Cc1ccccc1", coords="canonical"))   # toluene, canonical coords

    # By policy, topology/stereo hashes ignore pos/coord_*.
    assert hash_topology(mg0) == hash_topology(mg1), "Topology hash should ignore coordinates"
    assert hash_stereo(mg0) == hash_stereo(mg1), "Stereo hash should ignore coordinates"

    # Make sure coords actually differ in presence (sanity)
    assert (mg0.arrays.pos is None) != (mg1.arrays.pos is None), "Sanity: expected one graph to have coords"


def test_stereo_hash_distinguishes_enantiomers_but_topology_hash_does_not():
    if hash_topology is None or hash_stereo is None:
        raise RuntimeError(f"hash import failed: {_HASH_IMPORT_ERR}")

    # Enantiomer pair: alaninol-ish (simple chiral center)
    # Same connectivity, opposite handedness.
    a = canonicalize(_build("C[C@H](O)N"), stereo=False)[0]
    b = canonicalize(_build("C[C@@H](O)N"), stereo=False)[0]

    # Topology should match
    ht_a = hash_topology(a)
    ht_b = hash_topology(b)
    assert ht_a == ht_b, "Topology hash should ignore handedness and treat enantiomers as equivalent"
    
    a_s = canonicalize(_build("C[C@H](O)N"), stereo=True)[0]
    b_s = canonicalize(_build("C[C@@H](O)N"), stereo=True)[0]
    # Stereo should differ
    hs_a = hash_stereo(a_s)
    hs_b = hash_stereo(b_s)
    assert hs_a != hs_b, "Stereo hash should distinguish enantiomers"

    print("\n--- TOPOLOGY HASH INPUT A ---")
    debug_hash_inputs(a, mode="topology")
    print("\n--- TOPOLOGY HASH INPUT B ---")
    debug_hash_inputs(b, mode="topology")

    print("\n--- STEREO HASH INPUT A ---")
    debug_hash_inputs(a_s, mode="stereo")
    print("\n--- STEREO HASH INPUT B ---")
    debug_hash_inputs(b_s, mode="stereo")


def run_all_tests() -> bool:
    tests = [
        test_hash_is_permutation_invariant_after_canonicalize,
        test_hash_ignores_coordinates,
        test_stereo_hash_distinguishes_enantiomers_but_topology_hash_does_not,
    ]

    print("\n" + "#" * 60)
    print("# RUNNING ALL HASH TESTS")
    print("#" * 60)

    passed = 0
    for t in tests:
        name = t.__name__
        print("\n" + "=" * 60)
        print(f"TEST: {name}")
        print("=" * 60)
        try:
            t()
            print(f"✓ {name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            traceback.print_exc()

    print("\n" + "#" * 60)
    print(f"# RESULTS: {passed} passed, {len(tests) - passed} failed")
    print("#" * 60)
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
