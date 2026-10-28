# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/ops/demo_ops_pipeline.py
from __future__ import annotations

if __package__ in (None, ""):
    import _path_bootstrap  # noqa: F401
else:
    from . import _path_bootstrap  # noqa: F401

from dataclasses import is_dataclass
from typing import Any, Optional, Tuple, Type

import numpy as np

from chem.build.create_molgraph import smiles_to_molgraph
from chem.build.molgraph_to_mol import molgraph_to_smiles

from chem.ops.base import apply_transform
from chem.ops.merge import MergeByLabels


# ============================================================
# Utility: find wrappers by name (so this file survives renames)
# ============================================================

def _pick_op_class(module, preferred_names: list[str]) -> Type[Any]:
    for name in preferred_names:
        if hasattr(module, name):
            cls = getattr(module, name)
            if isinstance(cls, type):
                return cls

    public = [n for n in dir(module) if not n.startswith("_")]
    print(f"\n[demo_ops_pipeline] Could not find any of: {preferred_names}")
    print(f"[demo_ops_pipeline] Available in {module.__name__}:")
    for n in public:
        obj = getattr(module, n)
        if isinstance(obj, type):
            tag = "class"
            if is_dataclass(obj):
                tag += " (dataclass)"
            print(f"  - {n}: {tag}")
    raise AttributeError(f"Missing expected op wrapper in {module.__name__}")


# ============================================================
# Printing
# ============================================================

def _smi(mg) -> str:
    s = molgraph_to_smiles(mg, canonical=True, isomeric=True)
    return str(s) if s is not None else "<SMILES:None>"


def _print_step(title: str, mg) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("SMILES:", _smi(mg))
    n = int(mg.arrays.atomic_num.shape[0])
    m = int(mg.arrays.bonds.shape[0])
    print(f"atoms={n} bonds={m}")
    if mg.arrays.pos is None:
        print("coords: None")
    else:
        print("coords:", mg.arrays.pos.shape, "frame=", mg.arrays.coord_frame)


# ============================================================
# Safety guards (conservative)
# ============================================================

_MAX_DEGREE = {
    6: 4,   # C
    7: 3,   # N (conservative)
    8: 2,   # O
    9: 1,   # F
    15: 5,  # P
    16: 6,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}


def _degrees(mg) -> np.ndarray:
    a = mg.arrays
    n = int(a.atomic_num.shape[0])
    deg = np.zeros(n, dtype=np.int64)
    for u, v in a.bonds.tolist():
        deg[int(u)] += 1
        deg[int(v)] += 1
    return deg


def _existing_edges(mg) -> set[Tuple[int, int]]:
    a = mg.arrays
    edges: set[Tuple[int, int]] = set()
    for u, v in a.bonds.tolist():
        x, y = int(u), int(v)
        if x > y:
            x, y = y, x
        edges.add((x, y))
    return edges


def _is_arom(mg, i: int) -> bool:
    a = mg.arrays
    return bool(a.is_aromatic[i]) if a.is_aromatic is not None else False


def _can_add_single_bond(mg, i: int) -> bool:
    a = mg.arrays
    Z = int(a.atomic_num[i])
    if Z == 0:
        return False
    cap = _MAX_DEGREE.get(Z, 4)
    deg = _degrees(mg)[i]
    # avoid aromatic endpoints for demo stability
    if _is_arom(mg, i):
        return False
    return int(deg) < int(cap)


def _pick_safe_delete_atom_idx(mg) -> int:
    """
    Prefer deleting: non-aromatic heavy atom with degree 1.
    """
    a = mg.arrays
    n = int(a.atomic_num.shape[0])
    deg = _degrees(mg)

    cands = [i for i in range(n)
             if int(a.atomic_num[i]) != 0 and not _is_arom(mg, i) and int(deg[i]) == 1]
    if cands:
        return int(cands[-1])

    cands = [i for i in range(n)
             if int(a.atomic_num[i]) != 0 and not _is_arom(mg, i)]
    if cands:
        return int(cands[-1])

    raise RuntimeError("No suitable atom found to delete (non-dummy heavy atom).")


def _pick_safe_addbond_pair(mg) -> Optional[Tuple[int, int]]:
    """
    Choose a pair for AddBond that respects conservative degree caps and avoids aromatic atoms.
    Further restrict to C/N only to avoid O valence blowups in a generic demo.
    """
    a = mg.arrays
    n = int(a.atomic_num.shape[0])
    edges = _existing_edges(mg)

    def ok_atom(i: int) -> bool:
        Z = int(a.atomic_num[i])
        if Z not in (6, 7):  # C/N only
            return False
        return _can_add_single_bond(mg, i)

    candidates = [i for i in range(n) if ok_atom(i)]
    for i in candidates:
        for j in candidates:
            if j <= i:
                continue
            if (i, j) in edges:
                continue
            return int(i), int(j)
    return None


def _find_dummy_with_label(mg, label: int) -> Optional[int]:
    a = mg.arrays
    if a.attachment_label is None:
        return None
    idxs = np.where((a.atomic_num == 0) & (a.attachment_label == label))[0]
    if idxs.size == 0:
        return None
    return int(idxs[0])


def _try_step(title: str, mg, op) -> Tuple[Any, bool]:
    """
    Apply a transform; if it makes SMILES conversion fail, rollback and return (mg, False).
    This lets the demo always reach merge.
    """
    mg0 = mg
    try:
        mg2, _info = apply_transform(mg, op)
    except Exception as e:
        print(f"[demo_ops_pipeline] Step failed ({title}): {e}")
        return mg0, False

    s = molgraph_to_smiles(mg2, canonical=True, isomeric=True)
    if s is None:
        print(f"[demo_ops_pipeline] Step produced unsanitizable graph; rolling back: {title}")
        return mg0, False

    _print_step(title, mg2)
    return mg2, True


# ============================================================
# Demo pipeline
# ============================================================

def main() -> None:
    # A: fused aromatic + chiral sidechain + alkene stereo + [*:1]
    smi_A = "c1ccc2cc([*:1])ccc2c1C[C@H](O)C/C=C\\C(=O)N"
    # B: heteroaromatic + sulfonamide + quaternary ammonium + [*:2]
    smi_B = "O=S(=O)(N)c1ncccc1C[N+](C)(C)C[*:2]"

    gA = smiles_to_molgraph(smi_A, coords="canonical", optimize_coords=False)
    gB = smiles_to_molgraph(smi_B, coords="canonical", optimize_coords=False)
    if gA is None or gB is None:
        raise RuntimeError("Failed to build one of the input MolGraphs from SMILES")

    _print_step("START A", gA)
    _print_step("START B", gB)

    # Import wrappers
    from chem.ops import atoms as ops_atoms
    from chem.ops import bonds as ops_bonds

    PermuteAtoms = _pick_op_class(ops_atoms, ["PermuteAtoms", "AtomPermute", "Permute"])
    DeleteAtom = _pick_op_class(ops_atoms, ["DeleteAtom", "RemoveAtom", "DelAtom"])

    AddBond = getattr(ops_bonds, "AddBond", None)
    DeleteBond = getattr(ops_bonds, "DeleteBond", None)
    # NOTE: we intentionally DO NOT do EditBond here because bond_type enums differ across projects.

    # 1) Permute atoms (stress mapping)
    nA = int(gA.arrays.atomic_num.shape[0])
    new_to_old = np.arange(nA, dtype=np.int64)[::-1]
    gA, _ = _try_step("A after PermuteAtoms(reverse)", gA, PermuteAtoms(new_to_old=new_to_old))
    print("op_log tail:", (gA.meta.get("op_log") or [])[-1])

    # 2) Delete a safe atom (avoid aromatic + avoid deleting merge dummy)
    del_idx = _pick_safe_delete_atom_idx(gA)
    merge_dummy = _find_dummy_with_label(gA, 1)
    if merge_dummy is not None and int(del_idx) == int(merge_dummy):
        # pick another non-aromatic heavy atom
        a = gA.arrays
        candidates = [i for i in range(int(a.atomic_num.shape[0]))
                      if i != int(merge_dummy) and int(a.atomic_num[i]) != 0 and not _is_arom(gA, i)]
        if not candidates:
            raise RuntimeError("Could not find a safe atom to delete without touching merge dummy.")
        del_idx = int(candidates[-1])

    gA, _ = _try_step(f"A after DeleteAtom(idx={del_idx})", gA, DeleteAtom(idx=del_idx))

    # 3) AddBond safely (C/N only), and IMPORTANT: use bond_type=0 (single bonds are encoded as 0)
    if AddBond is not None:
        pair = _pick_safe_addbond_pair(gA)
        if pair is not None:
            u, v = pair
            # Single bonds use type=0.
            gA, ok = _try_step(f"A after AddBond({u}-{v})", gA, AddBond(u=u, v=v, bond_type=0))
            if not ok:
                print("[demo_ops_pipeline] AddBond rolled back; continuing")
        else:
            print("[demo_ops_pipeline] No safe pair found for AddBond; skipping")

    # 4) DeleteBond safely: pick non-aromatic bond that won't disconnect (both ends degree>=2)
    if DeleteBond is not None:
        a = gA.arrays
        deg = _degrees(gA)
        m = int(a.bonds.shape[0])
        done = False
        for k in range(m):
            u = int(a.bonds[k, 0])
            v = int(a.bonds[k, 1])
            if int(a.atomic_num[u]) == 0 or int(a.atomic_num[v]) == 0:
                continue
            if _is_arom(gA, u) or _is_arom(gA, v):
                continue
            # Avoid disconnecting: skip leaf bonds
            if int(deg[u]) <= 1 or int(deg[v]) <= 1:
                continue

            gA, ok = _try_step(f"A after DeleteBond({u}-{v})", gA, DeleteBond(u=u, v=v))
            if ok:
                done = True
                break

        if not done:
            print("[demo_ops_pipeline] No safe non-disconnecting bond found for DeleteBond; skipping")

    # 5) Merge: require label 1 dummy still present
    if _find_dummy_with_label(gA, 1) is None:
        raise RuntimeError("Merge site label 1 was lost before merge (dummy missing).")

    merge_op = MergeByLabels(
        g_b=gB,
        label_a=1,
        label_b=2,
        validate_with_rdkit=True,
        coords="canonical",
        num_confs=1,
        optimize_coords=False,
        random_seed=123,
    )
    merged, _info = apply_transform(gA, merge_op)

    _print_step("MERGED result (A op-chain then merge with B)", merged)
    print("\nFinal SMILES:", _smi(merged))
    print("Final op_log tail:", (merged.meta.get("op_log") or [])[-3:])

    if np.any(merged.arrays.atomic_num == 0):
        print("[warn] merged still has dummy atoms; check merge_by_labels dummy removal behavior")

    print("\n✅ demo_ops_pipeline.py finished successfully")


if __name__ == "__main__":
    main()
