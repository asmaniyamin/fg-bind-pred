# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from .scienceplots.styles_discovery import read_styles_in_folders


PLOT_ROOT = Path(__file__).resolve().parent
SCIENCEPLOTS_ROOT = PLOT_ROOT / "scienceplots"
SCIENCEPLOTS_STYLES = SCIENCEPLOTS_ROOT / "styles"


def register_scienceplots() -> dict[str, object]:
    if not SCIENCEPLOTS_STYLES.is_dir():
        return {}
    stylesheets = read_styles_in_folders(str(SCIENCEPLOTS_STYLES))
    plt.style.core.update_nested_dict(plt.style.library, stylesheets)
    plt.style.core.available[:] = sorted(plt.style.library.keys())
    return stylesheets


def available_plot_styles() -> list[str]:
    register_scienceplots()
    return sorted(plt.style.available)


def parse_plot_style_spec(style_spec: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if style_spec is None:
        return ["science", "ieee"]
    if isinstance(style_spec, (list, tuple)):
        tokens = [str(item).strip() for item in style_spec if str(item).strip()]
    else:
        normalized = (
            str(style_spec)
            .replace("+", ",")
            .replace(";", ",")
            .replace("|", ",")
        )
        tokens = [token.strip() for token in normalized.split(",") if token.strip()]
    return tokens or ["science", "ieee"]


def _resolve_style_aliases(style_names: list[str]) -> list[str]:
    alias_map = {
        "science+ieee": ["science", "ieee"],
        "science+nature": ["science", "nature"],
        "science+scatter": ["science", "scatter"],
        "science+notebook": ["science", "notebook"],
    }
    resolved: list[str] = []
    for style_name in style_names:
        expanded = alias_map.get(style_name.lower(), [style_name])
        resolved.extend(expanded)
    return resolved


def _append_no_latex_if_needed(style_names: list[str]) -> list[str]:
    lower = {style.lower() for style in style_names}
    if "no-latex" in lower or "pgf" in lower or "latex-sans" in lower:
        return style_names
    if "science" in lower and shutil.which("latex") is None:
        return [*style_names, "no-latex"]
    return style_names


def apply_plot_styles(style_spec: str | list[str] | tuple[str, ...] | None) -> list[str]:
    register_scienceplots()
    styles = _resolve_style_aliases(parse_plot_style_spec(style_spec))
    styles = _append_no_latex_if_needed(styles)
    plt.style.use(styles)
    return styles
