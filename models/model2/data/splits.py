# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from datetime import datetime


def _stable_score(token: str, seed: int) -> float:
    payload = f"{seed}:{token}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def pocket_hash_split(
    pdb_ids: list[str],
    pocket_sequences: dict[str, str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for pdb_id in sorted(pdb_ids):
        token = pocket_sequences.get(pdb_id) or pdb_id
        score = _stable_score(token, seed)
        if score < test_frac:
            test_ids.append(pdb_id)
        elif score < test_frac + val_frac:
            val_ids.append(pdb_id)
        else:
            train_ids.append(pdb_id)
    return train_ids, val_ids, test_ids


def time_split(
    pdb_ids: list[str],
    deposition_dates: dict[str, datetime],
    cutoff_date: str,
) -> tuple[list[str], list[str]]:
    cutoff = datetime.fromisoformat(cutoff_date)
    train_ids: list[str] = []
    test_ids: list[str] = []
    for pdb_id in sorted(pdb_ids):
        date = deposition_dates.get(pdb_id)
        if date is None or date <= cutoff:
            train_ids.append(pdb_id)
        else:
            test_ids.append(pdb_id)
    return train_ids, test_ids


def ligand_split(
    pdb_ids: list[str],
    ligand_smiles: dict[str, str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for pdb_id in sorted(pdb_ids):
        token = ligand_smiles.get(pdb_id) or pdb_id
        score = _stable_score(token, seed)
        if score < test_frac:
            test_ids.append(pdb_id)
        elif score < test_frac + val_frac:
            val_ids.append(pdb_id)
        else:
            train_ids.append(pdb_id)
    return train_ids, val_ids, test_ids
