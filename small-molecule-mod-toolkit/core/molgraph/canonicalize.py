# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# core/molgraph/canonicalize.py
from __future__ import annotations
import hashlib
import numpy as np

from core.structs import MolGraph
from chem.edit.atoms import permute_atoms

"""
WARNING
Stereo-canonical order is stereo-dependent, so you should not expect atom indices to line up across stereoisomers.
If you want to compare stereoisomers atom-by-atom you will need a mapping strategy (I recommend topology-based).
"""


def _h(x: bytes) -> bytes:
    return hashlib.sha256(x).digest()


def _build_adj(bonds: np.ndarray, n_atoms: int):
    adj = [[] for _ in range(n_atoms)]
    for bidx in range(bonds.shape[0]):
        u = int(bonds[bidx, 0]); v = int(bonds[bidx, 1])
        adj[u].append((v, bidx))
        adj[v].append((u, bidx))
    return adj


def _atom_seed(g: MolGraph, i: int, stereo: bool = True) -> bytes:
    a = g.arrays
    z = int(a.atomic_num[i])
    chg = int(a.formal_charge[i]) if a.formal_charge is not None else 0
    arom = int(bool(a.is_aromatic[i])) if a.is_aromatic is not None else 0
    iso = int(a.isotope[i]) if a.isotope is not None else 0
    chir = int(a.chiral_tag[i]) if (stereo and a.chiral_tag is not None) else 0
    eH = int(a.explicit_h[i]) if a.explicit_h is not None else 0
    iH = int(a.implicit_h[i]) if a.implicit_h is not None else 0
    return _h(f"Z={z};chg={chg};arom={arom};iso={iso};chir={chir};eH={eH};iH={iH}".encode())


def _bond_seed(g: MolGraph, bidx: int, stereo: bool = True) -> bytes:
    a = g.arrays
    bt = int(a.bond_type[bidx]) if a.bond_type is not None else 0
    conj = int(bool(a.is_conjugated[bidx])) if a.is_conjugated is not None else 0
    ring = int(bool(a.is_in_ring[bidx])) if a.is_in_ring is not None else 0
    bdir = int(a.bond_dir[bidx]) if (stereo and a.bond_dir is not None) else 0
    bst  = int(a.bond_stereo[bidx]) if (stereo and a.bond_stereo is not None) else 0
    res = int(a.bond_resonance_type[bidx]) if a.bond_resonance_type is not None else 0
    return _h(f"bt={bt};conj={conj};ring={ring};dir={bdir};st={bst};res={res}".encode())


def compute_canonical_order(g: MolGraph, *, iters: int = 4, stereo: bool = True) -> np.ndarray:
    """
    Return new_to_old permutation (length N) giving a deterministic atom order
    based on WL refinement over MolArrays.
    """
    a = g.arrays
    n = int(a.atomic_num.shape[0])
    bonds = a.bonds.astype(np.int64, copy=False) if a.bonds is not None else np.zeros((0, 2), dtype=np.int64)

    adj = _build_adj(bonds, n)
    atom_lbl = [_atom_seed(g, i, stereo) for i in range(n)]
    bond_lbl = [_bond_seed(g, b, stereo) for b in range(bonds.shape[0])] if bonds.shape[0] else []

    for _ in range(int(iters)):
        new = []
        for i in range(n):
            parts = []
            for j, bidx in adj[i]:
                parts.append(_h(atom_lbl[j] + bond_lbl[bidx]))
            parts.sort()
            new.append(_h(atom_lbl[i] + b"".join(parts)))
        atom_lbl = new

    def key(i: int):
        degree = len(adj[i])
        # deterministic tie-break: neighborhood signature + old index
        neigh = []
        for j, bidx in adj[i]:
            neigh.append(atom_lbl[j] + bond_lbl[bidx])
        neigh.sort()
        neigh_sig = _h(b"".join(neigh))
        return (atom_lbl[i], degree, neigh_sig, i)

    order = sorted(range(n), key=key)
    return np.array(order, dtype=np.int64)  # new_to_old


def canonicalize(g: MolGraph, *, iters: int = 4, stereo: bool = True):
    """
    Canonicalize atom/bond order. Delegates all remapping to permute_atoms().
    Returns: (g2, old_to_new, new_to_old, bond_perm)
    """
    new_to_old = compute_canonical_order(g, iters=iters, stereo=stereo)
    g2, old_to_new, bond_perm = permute_atoms(g, new_to_old)
    g2.meta = dict(getattr(g2, "meta", {}) or {})
    g2.meta["canonicalized"] = True
    return g2, old_to_new, new_to_old, bond_perm
