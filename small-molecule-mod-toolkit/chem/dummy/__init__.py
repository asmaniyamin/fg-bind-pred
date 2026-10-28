# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

# chem/dummy/__init__.py
"""
Dummy (attachment-point) utilities.

Canonical rules for this project:
  - A dummy atom is any atom with arrays.atomic_num == 0.
  - A dummy's label (if any) lives ONLY in arrays.attachment_label[atom_idx].
  - Any other attachment representation (e.g. MolGraph.attachments, meta["label_to_index"])
    is derived / cached and should be rebuilt on demand.

Public API is intentionally small:
  - query: read-only helpers for locating dummies/labels/targets
  - edit: canonical mutations of attachment_label
  - derive: optional builders for derived attachment-point views
  - clean: optional validation/cleanup helpers
"""

from .query import (
    dummy_indices,
    dummy_labels,
    dummy_by_label,
    neighbors,
    dummy_target,
    is_insertion_dummy,
)

from .edit import (
    set_dummy_label,
    clear_dummy_label,
    relabel_dummy,
    enforce_label_invariant,
)

# derived view builder
try:
    from .derive import build_attachment_points  # noqa: F401
except Exception:  # pragma: no cover
    build_attachment_points = None  # type: ignore

# cleaning/validation helpers
try:
    from .clean import (
        validate_dummy_invariants,  # noqa: F401
        remove_orphan_dummies_mark_only,  # noqa: F401
    )
except Exception:  # pragma: no cover
    validate_dummy_invariants = None  # type: ignore
    remove_orphan_dummies_mark_only = None  # type: ignore


__all__ = [
    # query
    "dummy_indices",
    "dummy_labels",
    "dummy_by_label",
    "neighbors",
    "dummy_target",
    "is_insertion_dummy",
    # edit
    "set_dummy_label",
    "clear_dummy_label",
    "relabel_dummy",
    "enforce_label_invariant",
    # derive
    "build_attachment_points",
    # clean/validate
    "validate_dummy_invariants",
    "remove_orphan_dummies_mark_only",
]
