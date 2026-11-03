#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot import apply_plot_styles


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL1_ROOT = REPO_ROOT / "models" / "model1"
PLOT_NEURAL_NET_ROOT = REPO_ROOT / "plot" / "PlotNeuralNet-main" / "PlotNeuralNet"
ARCH_SOURCES = [
    MODEL1_ROOT / "config.py",
    MODEL1_ROOT / "model" / "model.py",
    MODEL1_ROOT / "model" / "encoders.py",
    MODEL1_ROOT / "model" / "gnn.py",
    MODEL1_ROOT / "model" / "attention.py",
    MODEL1_ROOT / "model" / "pooling.py",
    MODEL1_ROOT / "model" / "heads.py",
    MODEL1_ROOT / "data" / "graph_builder.py",
    MODEL1_ROOT / "data" / "label_builder.py",
    MODEL1_ROOT / "training" / "loss.py",
    MODEL1_ROOT / "training" / "metrics.py",
]


@dataclass
class Box:
    name: str
    x: float
    y: float
    w: float
    h: float

    def anchor(self, side: str) -> tuple[float, float]:
        if side == "west":
            return (self.x, self.y + self.h / 2.0)
        if side == "east":
            return (self.x + self.w, self.y + self.h / 2.0)
        if side == "north":
            return (self.x + self.w / 2.0, self.y + self.h)
        if side == "south":
            return (self.x + self.w / 2.0, self.y)
        if side == "center":
            return (self.x + self.w / 2.0, self.y + self.h / 2.0)
        raise ValueError(side)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a parsed architecture diagram for models/model1.")
    parser.add_argument(
        "--report-dir",
        default="models/model1/reports/model1_architecture_20260521",
        help="Directory where architecture figures and summaries will be written.",
    )
    parser.add_argument(
        "--plot-style",
        default="science+ieee",
        help="Plot style spec passed through the local ./plot scienceplots loader.",
    )
    return parser.parse_args()


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def parse_model_config_defaults(config_path: Path) -> dict[str, Any]:
    module = ast.parse(config_path.read_text())
    defaults: dict[str, Any] = {}
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    defaults[stmt.target.id] = _literal(stmt.value)
    return defaults


def parse_forward_sequence(model_path: Path) -> list[str]:
    source = model_path.read_text()
    module = ast.parse(source)
    steps: list[str] = []
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != "PerFGModel":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "forward":
                continue
            for stmt in item.body:
                snippet = ast.get_source_segment(source, stmt) or ""
                first_line = snippet.splitlines()[0].strip()
                if first_line:
                    steps.append(first_line)
    return steps


def parse_init_modules(model_path: Path) -> list[str]:
    source = model_path.read_text()
    module = ast.parse(source)
    modules: list[str] = []
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != "PerFGModel":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            for stmt in item.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        modules.append(target.attr)
    return modules


def try_default_parameter_count() -> int | None:
    try:
        from models.model1.config import ModelConfig
        from models.model1.model.model import PerFGModel

        return int(PerFGModel(ModelConfig()).count_parameters())
    except Exception:
        return None


def build_architecture_summary() -> dict[str, Any]:
    config_defaults = parse_model_config_defaults(MODEL1_ROOT / "config.py")
    forward_sequence = parse_forward_sequence(MODEL1_ROOT / "model" / "model.py")
    init_modules = parse_init_modules(MODEL1_ROOT / "model" / "model.py")
    return {
        "title": "model1 pairwise FG-residue architecture",
        "files_parsed": [str(path.relative_to(REPO_ROOT)) for path in ARCH_SOURCES],
        "config_defaults": config_defaults,
        "default_parameter_count": try_default_parameter_count(),
        "init_modules": init_modules,
        "forward_sequence": forward_sequence,
        "module_notes": {
            "LigandAtomEncoder": {
                "source": "models/model1/model/encoders.py",
                "details": [
                    "Embeddings: atomic number 119->32, formal charge 9->4, hybridization 8->8, chiral tag 4->4",
                    "Scalars: partial charge, aromatic flag, explicit H, implicit H, isotope",
                    f"AEV branch: {config_defaults.get('aev_dim', 'NA')}-dim ligand-pocket radial environment",
                    f"Projection: 405 -> hidden_dim ({config_defaults.get('hidden_dim', 'NA')}) with LayerNorm",
                ],
            },
            "LigandBondEncoder": {
                "source": "models/model1/model/encoders.py",
                "details": [
                    "Embeddings: bond type 6->16, stereo 6->8, direction 6->8",
                    f"Projection: 34 -> ligand_edge_dim ({config_defaults.get('ligand_edge_dim', 'NA')})",
                ],
            },
            "FGNodeEncoder": {
                "source": "models/model1/model/encoders.py",
                "details": [
                    "Scalars: num_atoms, total_partial_charge, max_radial_extent, mean_partial_charge",
                    "Embeddings: fg_type 1024->16, fg_parent_type 256->8",
                    f"Projection: 28 -> hidden_dim ({config_defaults.get('hidden_dim', 'NA')})",
                ],
            },
            "PocketResidueEncoder": {
                "source": "models/model1/model/encoders.py",
                "details": [
                    "Embeddings: residue type 128->32, base type 128->16, variant 32->8",
                    f"Distance RBF expansion: {config_defaults.get('num_rbf_kernels', 'NA')} kernels over min ligand distance",
                    f"Projection: {32 + 16 + 8 + 1 + int(config_defaults.get('num_rbf_kernels', 16))} -> hidden_dim",
                ],
            },
            "LigandGNN": {
                "source": "models/model1/model/gnn.py",
                "details": [
                    f"{config_defaults.get('num_ligand_gnn_layers', 'NA')} x GATv2Layer over ligand bond graph",
                    f"Attention heads: {config_defaults.get('num_heads', 'NA')}, edge_dim: ligand_edge_dim",
                    "Residual + dropout + LayerNorm each layer",
                ],
            },
            "FGAttentionPool": {
                "source": "models/model1/model/pooling.py",
                "details": [
                    "Attention pooling of ligand atom embeddings into FG tokens",
                    "Uses fg_membership_index to aggregate atoms per FG",
                    "Merge MLP combines pooled atom context with h_fg_init",
                ],
            },
            "FGSelfAttentionGraph": {
                "source": "models/model1/model/attention.py",
                "details": [
                    f"Radius graph on FG centroids with cutoff {config_defaults.get('fg_edge_cutoff', 'NA')} A",
                    f"Distance edge RBF: {config_defaults.get('num_rbf_kernels', 'NA')} -> fg_edge_dim",
                    f"{config_defaults.get('num_fg_attn_layers', 'NA')} x GATv2Layer with {config_defaults.get('num_fg_graph_heads', 'NA')} heads",
                ],
            },
            "FGPocketCrossAttention": {
                "source": "models/model1/model/attention.py",
                "details": [
                    f"{config_defaults.get('num_cross_attn_layers', 'NA')} x CrossAttentionLayer",
                    f"MultiheadAttention heads: {config_defaults.get('num_cross_attn_heads', 'NA')}",
                    f"Mask built from FG-residue distances with cutoff {config_defaults.get('cross_attn_cutoff', 'NA')} A and nearest-residue fallback",
                ],
            },
            "FGMoleculeSelfAttention": {
                "source": "models/model1/model/attention.py",
                "details": [
                    f"{config_defaults.get('num_fg_global_attn_layers', 'NA')} x same-molecule FG self-attention after pocket conditioning",
                    "Fully connected within each ligand's FG set",
                ],
            },
            "FGResidueContactHead": {
                "source": "models/model1/model/heads.py",
                "details": [
                    "Scores FG-residue pairs rather than predicting contacts from a single FG token",
                    "Pair features: h_fg, h_res, elementwise product, abs diff, RBF(distance), raw distance",
                    "MLP -> 4 channel pair contribution + sigmoid cutoff gate -> index_add over residues per FG",
                    "Outputs per-FG [vdw, elec, polar, nonpolar], then contact_total = channel sum",
                ],
            },
            "PreorgHead": {
                "source": "models/model1/model/heads.py",
                "details": [
                    "MLP hidden_dim -> 64 -> 1 applied to h_fg_ctx",
                ],
            },
            "DeltaTotalHead": {
                "source": "models/model1/model/model.py",
                "details": [
                    "Complex-level head over pooled ligand, residue, and FG representations",
                    "Concatenates FG mean, FG sum, ligand mean, residue mean, log counts, sum(contact_total), sum(preorg)",
                    f"Input dim = 4 * hidden_dim + 5, then MLP -> hidden_dim -> 64 -> 1",
                ],
            },
            "Supervision": {
                "source": "models/model1/training/loss.py",
                "details": [
                    "Primary tasks: contact_total, 4 contact channels, delta_total",
                    "Optional tasks: preorg, aux_elec, aux_vdw, pairwise rank loss on contact_total",
                    "Metrics track Pearson/Spearman/MAE plus collapse diagnostics from pred/target std ratios",
                ],
            },
        },
    }


def add_box(ax, box: Box, title: str, body: str, facecolor: str, edgecolor: str, *, title_size: int = 11, body_size: int = 8) -> None:
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        box.x + 0.012,
        box.y + box.h - 0.022,
        title,
        fontsize=title_size,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.text(
        box.x + 0.012,
        box.y + box.h - 0.055,
        body,
        fontsize=body_size,
        va="top",
        ha="left",
        linespacing=1.2,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#4a4a4a",
    text: str | None = None,
    text_offset: tuple[float, float] = (0.0, 0.0),
    connectionstyle: str = "arc3,rad=0.0",
    linestyle: str = "-",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.35,
        color=color,
        connectionstyle=connectionstyle,
        linestyle=linestyle,
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2.0 + text_offset[0]
        my = (start[1] + end[1]) / 2.0 + text_offset[1]
        ax.text(mx, my, text, fontsize=8, color=color, ha="center", va="center")


def render_architecture_figure(summary: dict[str, Any], out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.0, 10.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    section_specs = [
        (0.015, 0.05, 0.185, 0.88, "#fbf6ef", "#e0c48a", "Inputs"),
        (0.215, 0.05, 0.205, 0.88, "#eef6fb", "#8db6d9", "Encoding"),
        (0.435, 0.05, 0.205, 0.88, "#f1f8f0", "#94c48a", "Context"),
        (0.655, 0.05, 0.145, 0.88, "#f7f1fb", "#b495d3", "Aggregation"),
        (0.815, 0.05, 0.17, 0.88, "#fff1f2", "#df8f97", "Heads and Outputs"),
    ]
    for x, y, w, h, fc, ec, label in section_specs:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=ec,
            facecolor=fc,
            alpha=0.22,
        )
        ax.add_patch(patch)
        ax.text(x + 0.008, y + h - 0.012, label, fontsize=12, fontweight="bold", va="top", ha="left", color=ec)

    ax.text(0.02, 0.975, "model1 architecture: pairwise FG-residue contact model", fontsize=18, fontweight="bold", va="top")
    config = summary["config_defaults"]
    param_count = summary.get("default_parameter_count")
    config_line = (
        f"default ModelConfig: hidden={config.get('hidden_dim')} | ligand_gnn={config.get('num_ligand_gnn_layers')} | "
        f"fg_graph={config.get('num_fg_attn_layers')} | cross_attn={config.get('num_cross_attn_layers')} | "
        f"fg_global={config.get('num_fg_global_attn_layers')} | rbf={config.get('num_rbf_kernels')} | "
        f"cross cutoff={config.get('cross_attn_cutoff')} A"
    )
    if param_count is not None:
        config_line += f" | default params={param_count / 1_000_000:.2f}M"
    ax.text(0.02, 0.946, config_line, fontsize=9, va="top")
    ax.text(
        0.02,
        0.925,
        "Topology reflects the current source tree; hyperparameter sweeps vary width/depth but keep the same dataflow.",
        fontsize=8.5,
        color="#444444",
        va="top",
    )

    boxes = {
        "ligand_input": Box("ligand_input", 0.03, 0.72, 0.15, 0.15),
        "fg_input": Box("fg_input", 0.03, 0.46, 0.15, 0.15),
        "pocket_input": Box("pocket_input", 0.03, 0.18, 0.15, 0.17),
        "aev_builder": Box("aev_builder", 0.08, 0.58, 0.10, 0.08),
        "ligand_enc": Box("ligand_enc", 0.24, 0.71, 0.14, 0.16),
        "fg_enc": Box("fg_enc", 0.24, 0.48, 0.14, 0.11),
        "res_enc": Box("res_enc", 0.24, 0.20, 0.14, 0.13),
        "ligand_gnn": Box("ligand_gnn", 0.46, 0.72, 0.15, 0.14),
        "fg_pool": Box("fg_pool", 0.46, 0.56, 0.15, 0.11),
        "fg_graph": Box("fg_graph", 0.46, 0.37, 0.15, 0.13),
        "cross_mask": Box("cross_mask", 0.67, 0.14, 0.10, 0.11),
        "cross_attn": Box("cross_attn", 0.67, 0.43, 0.11, 0.16),
        "fg_global": Box("fg_global", 0.67, 0.67, 0.11, 0.11),
        "pair_builder": Box("pair_builder", 0.67, 0.28, 0.11, 0.12),
        "complex_pool": Box("complex_pool", 0.67, 0.80, 0.11, 0.11),
        "preorg_head": Box("preorg_head", 0.84, 0.70, 0.12, 0.10),
        "contact_head": Box("contact_head", 0.84, 0.30, 0.12, 0.18),
        "delta_head": Box("delta_head", 0.84, 0.84, 0.12, 0.08),
        "output_contact": Box("output_contact", 0.85, 0.11, 0.11, 0.12),
    }

    add_box(
        ax,
        boxes["ligand_input"],
        "Ligand inputs",
        "Atom features:\nZ, formal charge,\npartial charge,\naromatic, hybrid,\nchirality, H counts,\nisotope\n\nBond graph:\nedge_index, edge_attr,\natom coordinates",
        "#fbe0c3",
        "#cb7f44",
    )
    add_box(
        ax,
        boxes["fg_input"],
        "FG inputs",
        "FG scalars:\nnum_atoms, charge,\nradial extent\n\nFG types:\nfg_type,\nfg_parent_type\n\nMembership and\nFG centroids",
        "#dff0d8",
        "#5f9f5f",
    )
    add_box(
        ax,
        boxes["pocket_input"],
        "Pocket inputs",
        "Residue features:\nresidue/base/variant ids,\natom count,\nmin ligand distance\n\nPocket atom types and\nresidue coordinates",
        "#d8e8f7",
        "#5a90c7",
    )
    add_box(
        ax,
        boxes["aev_builder"],
        "AEV builder",
        "Ligand coords +\npocket ECIF atom types\n-> 352-d radial\nenvironment vector",
        "#fdebd0",
        "#cb7f44",
        title_size=10,
        body_size=7.5,
    )
    add_box(
        ax,
        boxes["ligand_enc"],
        "Ligand encoders",
        "LigandAtomEncoder:\nembeddings + scalars + AEV\n-> h_atom0\n\nLigandBondEncoder:\nbond embeddings -> h_bond",
        "#f9d3a6",
        "#cb7f44",
    )
    add_box(
        ax,
        boxes["fg_enc"],
        "FGNodeEncoder",
        "FG scalars + type embeddings\n-> h_fg_init",
        "#d7efcf",
        "#5f9f5f",
    )
    add_box(
        ax,
        boxes["res_enc"],
        "PocketResidueEncoder",
        "Residue embeddings +\ndistance RBF\n-> h_res",
        "#d4e7fb",
        "#5a90c7",
    )
    add_box(
        ax,
        boxes["ligand_gnn"],
        "LigandGNN",
        f"{config.get('num_ligand_gnn_layers')} x GATv2Layer\nheads={config.get('num_heads')}, edge_dim={config.get('ligand_edge_dim')}\nresidual + LayerNorm",
        "#f7c58b",
        "#cb7f44",
    )
    add_box(
        ax,
        boxes["fg_pool"],
        "FGAttentionPool",
        "Attention pool atoms\nper functional group\nmerge pooled atom context\nwith h_fg_init",
        "#cfe8c8",
        "#5f9f5f",
    )
    add_box(
        ax,
        boxes["fg_graph"],
        "FGSelfAttentionGraph",
        f"FG centroid radius graph\ncutoff={config.get('fg_edge_cutoff')} A\nRBF edge projection\n{config.get('num_fg_attn_layers')} x GATv2Layer",
        "#c3e0b8",
        "#5f9f5f",
    )
    add_box(
        ax,
        boxes["cross_mask"],
        "FG-residue\nmasks",
        f"build_cross_attn_mask\nand batched masks\ncutoff={config.get('cross_attn_cutoff')} A\nnearest-residue fallback",
        "#e9dbf8",
        "#8f69b8",
        title_size=9,
        body_size=7.2,
    )
    add_box(
        ax,
        boxes["cross_attn"],
        "FGPocketCrossAttention",
        f"{config.get('num_cross_attn_layers')} x\nMultiheadAttention + FFN\nFG queries attend\nto residue tokens",
        "#e4d3f5",
        "#8f69b8",
    )
    add_box(
        ax,
        boxes["fg_global"],
        "FGMoleculeSelfAttention",
        f"{config.get('num_fg_global_attn_layers')} x same-molecule\nFG self-attention\nafter pocket conditioning",
        "#e8dcf8",
        "#8f69b8",
    )
    add_box(
        ax,
        boxes["pair_builder"],
        "FG-residue\npair builder",
        "Per-complex cdist on\nFG centroids vs residues\n+ shared masks +\ndistance RBF basis",
        "#efe5fb",
        "#8f69b8",
        title_size=9,
        body_size=7.2,
    )
    add_box(
        ax,
        boxes["complex_pool"],
        "Complex pooling",
        "FG mean + FG sum\nligand mean\nresidue mean\nlog counts\nsum(contact_total)\nsum(preorg)",
        "#eee6fb",
        "#8f69b8",
        title_size=10,
        body_size=7.2,
    )
    add_box(
        ax,
        boxes["preorg_head"],
        "PreorgHead",
        "MLP:\nh_fg_ctx -> 64 -> 1\nper FG",
        "#ffd9dc",
        "#c46d76",
    )
    add_box(
        ax,
        boxes["contact_head"],
        "FGResidueContactHead",
        "Pair features:\nh_fg, h_res,\nproduct, |diff|,\nRBF(d), d\n\nMLP + sigmoid cutoff gate\n-> 4 channel pair scores\n-> index_add over residues",
        "#ffd4d8",
        "#c46d76",
    )
    add_box(
        ax,
        boxes["delta_head"],
        "DeltaTotalHead",
        "MLP:\n(4h + 5) -> h -> 64 -> 1\nper complex",
        "#ffd7da",
        "#c46d76",
        title_size=10,
        body_size=7.6,
    )
    add_box(
        ax,
        boxes["output_contact"],
        "Outputs",
        "contact:\n[vdw, elec,\npolar, nonpolar]\n\ncontact_total =\nchannel sum\n\npreorg (FG)\ndelta_total\n(complex)",
        "#ffe7e9",
        "#c46d76",
    )

    add_arrow(ax, boxes["ligand_input"].anchor("east"), boxes["ligand_enc"].anchor("west"), color="#cb7f44")
    add_arrow(ax, boxes["pocket_input"].anchor("east"), boxes["res_enc"].anchor("west"), color="#5a90c7")
    add_arrow(ax, boxes["fg_input"].anchor("east"), boxes["fg_enc"].anchor("west"), color="#5f9f5f")
    add_arrow(ax, boxes["ligand_input"].anchor("south"), boxes["aev_builder"].anchor("west"), color="#cb7f44", connectionstyle="arc3,rad=-0.22")
    add_arrow(ax, boxes["pocket_input"].anchor("north"), boxes["aev_builder"].anchor("south"), color="#5a90c7", connectionstyle="arc3,rad=0.20")
    add_arrow(ax, boxes["aev_builder"].anchor("east"), (boxes["ligand_enc"].x + 0.02, boxes["ligand_enc"].y + 0.08), color="#cb7f44", text="AEV", text_offset=(0.0, 0.015))

    add_arrow(ax, boxes["ligand_enc"].anchor("east"), boxes["ligand_gnn"].anchor("west"), color="#cb7f44", text="h_atom0, h_bond", text_offset=(0.0, 0.015))
    add_arrow(ax, boxes["ligand_gnn"].anchor("south"), boxes["fg_pool"].anchor("north"), color="#cb7f44", connectionstyle="arc3,rad=0.05", text="h_atom", text_offset=(-0.01, 0.02))
    add_arrow(ax, boxes["fg_enc"].anchor("east"), boxes["fg_pool"].anchor("west"), color="#5f9f5f", text="h_fg_init", text_offset=(0.0, 0.015))
    add_arrow(ax, boxes["fg_input"].anchor("south"), boxes["fg_graph"].anchor("west"), color="#5f9f5f", connectionstyle="arc3,rad=-0.15", text="FG centroids", text_offset=(-0.02, -0.01))
    add_arrow(ax, boxes["fg_pool"].anchor("south"), boxes["fg_graph"].anchor("north"), color="#5f9f5f", text="h_fg", text_offset=(-0.01, 0.02))

    add_arrow(ax, boxes["fg_graph"].anchor("east"), boxes["cross_attn"].anchor("west"), color="#8f69b8", text="h_fg_ctx", text_offset=(0.0, 0.02))
    add_arrow(ax, boxes["res_enc"].anchor("east"), boxes["cross_attn"].anchor("west"), color="#5a90c7", connectionstyle="arc3,rad=-0.12", text="h_res", text_offset=(0.0, -0.03))
    add_arrow(ax, boxes["fg_input"].anchor("south"), boxes["cross_mask"].anchor("west"), color="#5f9f5f", connectionstyle="arc3,rad=-0.28")
    add_arrow(ax, boxes["pocket_input"].anchor("north"), boxes["cross_mask"].anchor("west"), color="#5a90c7", connectionstyle="arc3,rad=0.26")
    add_arrow(ax, boxes["cross_mask"].anchor("north"), boxes["cross_attn"].anchor("south"), color="#8f69b8", text="mask", text_offset=(0.04, 0.0))

    add_arrow(ax, boxes["cross_attn"].anchor("north"), boxes["fg_global"].anchor("south"), color="#8f69b8", text="h_fg_cross", text_offset=(-0.02, 0.015))
    add_arrow(ax, boxes["fg_global"].anchor("south"), boxes["pair_builder"].anchor("north"), color="#8f69b8", text="h_fg_final", text_offset=(-0.03, 0.015))
    add_arrow(ax, boxes["res_enc"].anchor("east"), boxes["pair_builder"].anchor("west"), color="#5a90c7", connectionstyle="arc3,rad=-0.08")
    add_arrow(ax, boxes["cross_mask"].anchor("north"), boxes["pair_builder"].anchor("south"), color="#8f69b8", text="shared masks", text_offset=(0.05, 0.0))
    add_arrow(ax, boxes["pair_builder"].anchor("east"), boxes["contact_head"].anchor("west"), color="#c46d76")

    add_arrow(ax, boxes["fg_graph"].anchor("east"), boxes["preorg_head"].anchor("west"), color="#c46d76", connectionstyle="arc3,rad=0.28", text="h_fg_ctx", text_offset=(-0.03, 0.02))
    add_arrow(ax, boxes["fg_global"].anchor("north"), boxes["complex_pool"].anchor("south"), color="#8f69b8", text="FG pooled", text_offset=(-0.03, 0.018))
    add_arrow(ax, boxes["ligand_gnn"].anchor("east"), boxes["complex_pool"].anchor("west"), color="#cb7f44", connectionstyle="arc3,rad=-0.15", text="ligand mean", text_offset=(0.0, 0.02))
    add_arrow(ax, boxes["res_enc"].anchor("east"), boxes["complex_pool"].anchor("west"), color="#5a90c7", connectionstyle="arc3,rad=0.15", text="residue mean", text_offset=(0.0, -0.02))
    add_arrow(ax, boxes["contact_head"].anchor("north"), boxes["complex_pool"].anchor("south"), color="#c46d76", connectionstyle="arc3,rad=0.10", text="sum contact_total", text_offset=(0.03, 0.01))
    add_arrow(ax, boxes["preorg_head"].anchor("south"), boxes["complex_pool"].anchor("east"), color="#c46d76", connectionstyle="arc3,rad=-0.12", text="sum preorg", text_offset=(0.04, 0.0))
    add_arrow(ax, boxes["complex_pool"].anchor("east"), boxes["delta_head"].anchor("west"), color="#c46d76")

    add_arrow(ax, boxes["contact_head"].anchor("south"), boxes["output_contact"].anchor("north"), color="#c46d76", text="4 channel contact", text_offset=(0.035, -0.01))
    add_arrow(ax, boxes["preorg_head"].anchor("south"), (boxes["output_contact"].x + 0.04, boxes["output_contact"].y + boxes["output_contact"].h), color="#c46d76", connectionstyle="arc3,rad=-0.22", text="preorg", text_offset=(-0.02, 0.005))
    add_arrow(ax, boxes["delta_head"].anchor("south"), (boxes["output_contact"].x + 0.08, boxes["output_contact"].y + boxes["output_contact"].h), color="#c46d76", connectionstyle="arc3,rad=0.18", text="delta_total", text_offset=(0.04, 0.015))

    footnote = (
        "Parsed sources: "
        + ", ".join(str(Path(path).relative_to(REPO_ROOT)) for path in ARCH_SOURCES[:6])
        + ", ... | Training adds MultiTaskUncertaintyLoss over contact_total, channel losses, delta_total, and optional rank loss."
    )
    ax.text(0.02, 0.025, textwrap.fill(footnote, width=150), fontsize=7.4, color="#555555", va="bottom")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_box(ax, box: Box, title: str, body: str, facecolor: str, edgecolor: str, *, title_size: int = 10, body_size: int = 8) -> None:
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle="square,pad=0.006",
        linewidth=1.35,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    title_artist = ax.text(
        box.x + 0.008,
        box.y + box.h - 0.014,
        title,
        fontsize=title_size,
        fontweight="bold",
        va="top",
        ha="left",
        color="#202020",
    )
    body_artist = ax.text(
        box.x + 0.008,
        box.y + box.h - 0.044,
        body,
        fontsize=body_size,
        va="top",
        ha="left",
        linespacing=1.12,
        color="#303030",
    )
    return {
        "box": box,
        "patch": patch,
        "title_text": title_artist,
        "body_text": body_artist,
    }


def add_path_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str = "#585858",
    text: str | None = None,
    text_xy: tuple[float, float] | None = None,
    linestyle: str = "-",
    linewidth: float = 1.25,
    registry: list[dict[str, Any]] | None = None,
    allow_boxes: set[str] | None = None,
) -> dict[str, Any]:
    if len(points) < 2:
        return {"points": points, "allow_boxes": set(allow_boxes or set())}
    for idx in range(len(points) - 2):
        (x0, y0), (x1, y1) = points[idx], points[idx + 1]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth, linestyle=linestyle, solid_capstyle="butt")
    arrow = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(arrow)
    if text:
        tx, ty = text_xy if text_xy is not None else points[len(points) // 2]
        ax.text(tx, ty, text, fontsize=7.5, color=color, ha="center", va="center")
    route = {"points": points, "allow_boxes": set(allow_boxes or set())}
    if registry is not None:
        registry.append(route)
    return route


def _bbox_contains(outer, inner, pad: float = 0.0) -> bool:
    return (
        inner.x0 >= outer.x0 + pad
        and inner.x1 <= outer.x1 - pad
        and inner.y0 >= outer.y0 + pad
        and inner.y1 <= outer.y1 - pad
    )


def _segment_hits_box(
    p0: tuple[float, float],
    p1: tuple[float, float],
    box: Box,
    *,
    eps: float = 1e-4,
) -> bool:
    x0, y0 = p0
    x1, y1 = p1
    left = box.x + eps
    right = box.x + box.w - eps
    bottom = box.y + eps
    top = box.y + box.h - eps
    if abs(y0 - y1) < eps:
        y = y0
        seg_left, seg_right = sorted((x0, x1))
        return bottom < y < top and seg_right > left and seg_left < right
    if abs(x0 - x1) < eps:
        x = x0
        seg_bottom, seg_top = sorted((y0, y1))
        return left < x < right and seg_top > bottom and seg_bottom < top
    return False


def check_layout(
    fig,
    artifacts: dict[str, dict[str, Any]],
    routes: list[dict[str, Any]],
    out_path: Path,
) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues: list[str] = []
    text_items: list[tuple[str, Any]] = []

    for name, artifact in artifacts.items():
        patch_bbox = artifact["patch"].get_window_extent(renderer)
        title_bbox = artifact["title_text"].get_window_extent(renderer)
        body_bbox = artifact["body_text"].get_window_extent(renderer)
        if not _bbox_contains(patch_bbox, title_bbox, pad=8.0):
            issues.append(f"{name}: title overflows box")
        if not _bbox_contains(patch_bbox, body_bbox, pad=8.0):
            issues.append(f"{name}: body overflows box")
        if title_bbox.overlaps(body_bbox):
            issues.append(f"{name}: title/body text overlap")
        text_items.append((f"{name}.title", title_bbox))
        text_items.append((f"{name}.body", body_bbox))

    for idx, (name_a, bbox_a) in enumerate(text_items):
        for name_b, bbox_b in text_items[idx + 1 :]:
            if name_a.split(".")[0] == name_b.split(".")[0]:
                continue
            if bbox_a.overlaps(bbox_b):
                issues.append(f"text overlap: {name_a} with {name_b}")

    for route_idx, route in enumerate(routes):
        points = route["points"]
        allow_boxes = route["allow_boxes"]
        for segment_idx in range(len(points) - 1):
            p0 = points[segment_idx]
            p1 = points[segment_idx + 1]
            for box_name, artifact in artifacts.items():
                if box_name in allow_boxes:
                    continue
                if _segment_hits_box(p0, p1, artifact["box"]):
                    issues.append(f"route {route_idx} segment {segment_idx} crosses {box_name}")

    out_path.write_text(json.dumps({"ok": not issues, "issues": issues}, indent=2))
    return issues


def render_architecture_figure(summary: dict[str, Any], out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    config = summary["config_defaults"]
    param_count = summary.get("default_parameter_count")
    ax.text(0.02, 0.97, "model1 architecture: pairwise FG-residue contact model", fontsize=18, fontweight="bold", va="top")
    config_line = (
        f"default config: hidden={config.get('hidden_dim')} | ligand_gnn={config.get('num_ligand_gnn_layers')} | "
        f"fg_graph={config.get('num_fg_attn_layers')} | cross_attn={config.get('num_cross_attn_layers')} | "
        f"fg_global={config.get('num_fg_global_attn_layers')} | rbf={config.get('num_rbf_kernels')} | "
        f"cross cutoff={config.get('cross_attn_cutoff')} A"
    )
    if param_count is not None:
        config_line += f" | default params={param_count / 1_000_000:.2f}M"
    ax.text(0.02, 0.935, config_line, fontsize=9, va="top")
    ax.text(
        0.02,
        0.915,
        "Primary FG-contact pathway is shown across the top. Metadata, pocket geometry, and whole-complex aggregation are routed below.",
        fontsize=8.4,
        color="#444444",
        va="top",
    )
    ax.plot([0.02, 0.98], [0.885, 0.885], color="#c7c7c7", linewidth=0.8)
    ax.text(0.02, 0.86, "Primary contact pathway", fontsize=11, fontweight="bold", color="#6b6b6b")
    ax.text(0.02, 0.52, "FG metadata branch", fontsize=10, fontweight="bold", color="#6b6b6b")
    ax.text(0.02, 0.27, "Pocket geometry branch", fontsize=10, fontweight="bold", color="#6b6b6b")
    ax.text(0.67, 0.27, "Whole-complex branch", fontsize=10, fontweight="bold", color="#6b6b6b")

    fill = "#fcfcfc"
    ligand_edge = "#786a59"
    fg_edge = "#647a67"
    pocket_edge = "#627588"
    interact_edge = "#78698a"
    head_edge = "#8a6c70"

    boxes = {
        "ligand_input": Box("ligand_input", 0.02, 0.64, 0.10, 0.11),
        "ligand_enc": Box("ligand_enc", 0.145, 0.64, 0.10, 0.11),
        "ligand_gnn": Box("ligand_gnn", 0.27, 0.64, 0.10, 0.11),
        "fg_pool": Box("fg_pool", 0.395, 0.64, 0.10, 0.11),
        "fg_graph": Box("fg_graph", 0.52, 0.64, 0.10, 0.11),
        "cross_attn": Box("cross_attn", 0.645, 0.64, 0.10, 0.11),
        "fg_global": Box("fg_global", 0.77, 0.64, 0.10, 0.11),
        "pair_head": Box("pair_head", 0.885, 0.62, 0.095, 0.15),
        "fg_input": Box("fg_input", 0.02, 0.39, 0.10, 0.09),
        "fg_enc": Box("fg_enc", 0.145, 0.39, 0.10, 0.09),
        "preorg_head": Box("preorg_head", 0.885, 0.39, 0.095, 0.09),
        "pocket_input": Box("pocket_input", 0.02, 0.14, 0.10, 0.09),
        "res_enc": Box("res_enc", 0.145, 0.14, 0.10, 0.09),
        "mask_builder": Box("mask_builder", 0.52, 0.14, 0.10, 0.09),
        "complex_pool": Box("complex_pool", 0.67, 0.14, 0.13, 0.09),
        "delta_head": Box("delta_head", 0.835, 0.14, 0.145, 0.09),
    }

    add_box(ax, boxes["ligand_input"], "Ligand inputs", "atom features\nbond graph + coords\nAEV branch", fill, ligand_edge)
    add_box(ax, boxes["ligand_enc"], "Ligand encoders", "atom/bond embeddings\n+ AEV projection", fill, ligand_edge)
    add_box(ax, boxes["ligand_gnn"], "LigandGNN", f"{config.get('num_ligand_gnn_layers')}x GATv2\nbond graph message passing", fill, ligand_edge)
    add_box(ax, boxes["fg_pool"], "FGAttentionPool", "attention pool\natoms -> FG tokens", fill, fg_edge)
    add_box(ax, boxes["fg_graph"], "FG graph context", f"FG centroid graph\n{config.get('num_fg_attn_layers')}x GATv2 + RBF edges", fill, fg_edge)
    add_box(ax, boxes["cross_attn"], "FG-pocket cross-attn", f"{config.get('num_cross_attn_layers')}x cross-attention\nFG queries, residue keys/values", fill, interact_edge)
    add_box(ax, boxes["fg_global"], "FG self-attn", f"{config.get('num_fg_global_attn_layers')}x same-molecule\nFG self-attention", fill, interact_edge)
    add_box(ax, boxes["pair_head"], "FG-residue pair head", "pair MLP(h_fg, h_res, d)\n4 channels / FG\ncontact_total = sum", fill, head_edge)
    add_box(ax, boxes["fg_input"], "FG inputs", "FG scalars + types\nmemberships + centroids", fill, fg_edge)
    add_box(ax, boxes["fg_enc"], "FGNodeEncoder", "metadata -> h_fg_init", fill, fg_edge)
    add_box(ax, boxes["preorg_head"], "Preorg head", "MLP on h_fg_ctx\npreorg / FG", fill, head_edge)
    add_box(ax, boxes["pocket_input"], "Pocket inputs", "residue features\natom coords/types", fill, pocket_edge)
    add_box(ax, boxes["res_enc"], "Pocket encoder", "residue embeddings\n+ distance RBF", fill, pocket_edge)
    add_box(ax, boxes["mask_builder"], "Mask builder", f"FG-residue masks\n{config.get('cross_attn_cutoff')} A cutoff + nearest fallback", fill, interact_edge)
    add_box(ax, boxes["complex_pool"], "Complex pooling", "FG mean/sum + ligand mean\n+ residue mean + summed heads", fill, interact_edge)
    add_box(ax, boxes["delta_head"], "DeltaTotalHead", "MLP on pooled complex\ndelta_total / complex", fill, head_edge)

    # Primary path
    add_path_arrow(ax, [boxes["ligand_input"].anchor("east"), boxes["ligand_enc"].anchor("west")])
    add_path_arrow(ax, [boxes["ligand_enc"].anchor("east"), boxes["ligand_gnn"].anchor("west")])
    add_path_arrow(ax, [boxes["ligand_gnn"].anchor("east"), boxes["fg_pool"].anchor("west")])
    add_path_arrow(ax, [boxes["fg_pool"].anchor("east"), boxes["fg_graph"].anchor("west")])
    add_path_arrow(ax, [boxes["fg_graph"].anchor("east"), boxes["cross_attn"].anchor("west")])
    add_path_arrow(ax, [boxes["cross_attn"].anchor("east"), boxes["fg_global"].anchor("west")])
    add_path_arrow(ax, [boxes["fg_global"].anchor("east"), boxes["pair_head"].anchor("west")])

    # FG metadata
    add_path_arrow(ax, [boxes["fg_input"].anchor("east"), boxes["fg_enc"].anchor("west")])
    add_path_arrow(
        ax,
        [boxes["fg_enc"].anchor("north"), (boxes["fg_enc"].anchor("north")[0], 0.59), (boxes["fg_pool"].anchor("south")[0], 0.59), boxes["fg_pool"].anchor("south")],
    )
    add_path_arrow(
        ax,
        [boxes["fg_enc"].anchor("east"), (0.41, boxes["fg_enc"].anchor("east")[1]), (0.41, 0.59), (boxes["fg_graph"].anchor("south")[0], 0.59), boxes["fg_graph"].anchor("south")],
    )

    # Pocket branch
    add_path_arrow(ax, [boxes["pocket_input"].anchor("east"), boxes["res_enc"].anchor("west")])
    add_path_arrow(
        ax,
        [boxes["res_enc"].anchor("east"), (0.60, boxes["res_enc"].anchor("east")[1]), (0.60, 0.59), (boxes["cross_attn"].anchor("south")[0], 0.59), boxes["cross_attn"].anchor("south")],
    )
    add_path_arrow(
        ax,
        [boxes["res_enc"].anchor("east"), (0.66, boxes["res_enc"].anchor("east")[1]), (0.66, 0.57), (boxes["pair_head"].anchor("south")[0], 0.57), boxes["pair_head"].anchor("south")],
    )
    add_path_arrow(
        ax,
        [boxes["fg_input"].anchor("south"), (boxes["fg_input"].anchor("south")[0], 0.11), (boxes["mask_builder"].x + 0.02, 0.11), (boxes["mask_builder"].x + 0.02, boxes["mask_builder"].y), boxes["mask_builder"].anchor("south")],
        linestyle="--",
        text="FG centroids",
        text_xy=(0.31, 0.102),
    )
    add_path_arrow(
        ax,
        [boxes["pocket_input"].anchor("east"), (0.38, boxes["pocket_input"].anchor("east")[1]), (0.38, 0.11), (boxes["mask_builder"].x + 0.05, 0.11), (boxes["mask_builder"].x + 0.05, boxes["mask_builder"].y), boxes["mask_builder"].anchor("south")],
        linestyle="--",
        text="residue geometry",
        text_xy=(0.47, 0.102),
    )
    add_path_arrow(
        ax,
        [boxes["mask_builder"].anchor("north"), (boxes["mask_builder"].anchor("north")[0], 0.58), (boxes["cross_attn"].anchor("south")[0] - 0.01, 0.58), boxes["cross_attn"].anchor("south")],
        linestyle="--",
    )
    add_path_arrow(
        ax,
        [boxes["mask_builder"].anchor("east"), (0.70, boxes["mask_builder"].anchor("east")[1]), (0.70, 0.55), (boxes["pair_head"].anchor("south")[0] - 0.01, 0.55), boxes["pair_head"].anchor("south")],
        linestyle="--",
    )

    # Auxiliary heads and whole-complex branch
    add_path_arrow(
        ax,
        [boxes["fg_graph"].anchor("south"), (boxes["fg_graph"].anchor("south")[0], 0.45), (boxes["preorg_head"].anchor("west")[0] - 0.01, 0.45), boxes["preorg_head"].anchor("west")],
    )
    add_path_arrow(
        ax,
        [boxes["ligand_gnn"].anchor("south"), (boxes["ligand_gnn"].anchor("south")[0], 0.25), (boxes["complex_pool"].anchor("west")[0], 0.25), boxes["complex_pool"].anchor("west")],
    )
    add_path_arrow(
        ax,
        [boxes["fg_global"].anchor("south"), (boxes["fg_global"].anchor("south")[0], 0.27), (boxes["complex_pool"].anchor("north")[0], 0.27), boxes["complex_pool"].anchor("north")],
    )
    add_path_arrow(
        ax,
        [boxes["pair_head"].anchor("south"), (boxes["pair_head"].anchor("south")[0], 0.31), (boxes["complex_pool"].anchor("east")[0], 0.31), boxes["complex_pool"].anchor("east")],
        text="sum(contact_total)",
        text_xy=(0.82, 0.323),
    )
    add_path_arrow(
        ax,
        [boxes["preorg_head"].anchor("south"), (boxes["preorg_head"].anchor("south")[0], 0.27), (boxes["complex_pool"].anchor("east")[0] - 0.01, 0.27), boxes["complex_pool"].anchor("east")],
        text="sum(preorg)",
        text_xy=(0.87, 0.283),
    )
    add_path_arrow(ax, [boxes["complex_pool"].anchor("east"), boxes["delta_head"].anchor("west")])

    ax.text(
        0.02,
        0.04,
        "Supervision: contact_total + [vdw, elec, polar, nonpolar] + delta_total; optional preorg and pairwise rank loss. "
        "The pairwise FG-residue head and the complex pooling branch are the architecture changes most relevant to the next sweep.",
        fontsize=7.8,
        color="#505050",
        va="bottom",
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_architecture_slide_figure(summary: dict[str, Any], out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    config = summary["config_defaults"]
    ax.text(0.03, 0.95, "model1 architecture: presentation view", fontsize=18, fontweight="bold", va="top")
    ax.text(
        0.03,
        0.91,
        f"h={config.get('hidden_dim')}, ligand_gnn={config.get('num_ligand_gnn_layers')}, fg_graph={config.get('num_fg_attn_layers')}, "
        f"cross_attn={config.get('num_cross_attn_layers')}, fg_global={config.get('num_fg_global_attn_layers')}, rbf={config.get('num_rbf_kernels')}",
        fontsize=8.8,
        color="#444444",
        va="top",
    )

    fill = "#fdfdfd"
    ligand_edge = "#6f665c"
    fg_edge = "#657467"
    pocket_edge = "#5f6f80"
    interact_edge = "#6d657f"
    head_edge = "#7f6669"
    arrow_color = "#5c6270"

    boxes = {
        "ligand_stream": Box("ligand_stream", 0.05, 0.60, 0.18, 0.15),
        "fg_stream": Box("fg_stream", 0.28, 0.60, 0.18, 0.15),
        "interaction": Box("interaction", 0.51, 0.60, 0.18, 0.15),
        "pair_head": Box("pair_head", 0.74, 0.60, 0.19, 0.15),
        "preorg_head": Box("preorg_head", 0.28, 0.24, 0.18, 0.14),
        "residue_stream": Box("residue_stream", 0.51, 0.24, 0.18, 0.14),
        "complex_head": Box("complex_head", 0.74, 0.22, 0.21, 0.18),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    routes: list[dict[str, Any]] = []

    artifacts["ligand_stream"] = add_box(
        ax,
        boxes["ligand_stream"],
        "Ligand stream",
        "atom/bond encoders\nAEV branch\nLigandGNN on bond graph",
        fill,
        ligand_edge,
        title_size=11,
        body_size=8.4,
    )
    artifacts["fg_stream"] = add_box(
        ax,
        boxes["fg_stream"],
        "FG stream",
        "FGNodeEncoder\nFGAttentionPool\nFG centroid graph context",
        fill,
        fg_edge,
        title_size=11,
        body_size=8.3,
    )
    artifacts["interaction"] = add_box(
        ax,
        boxes["interaction"],
        "Interaction backbone",
        "FG-pocket cross-attention\nsame-molecule FG self-attention",
        fill,
        interact_edge,
        title_size=10.8,
        body_size=8.15,
    )
    artifacts["pair_head"] = add_box(
        ax,
        boxes["pair_head"],
        "Pairwise contact head",
        "FGResidueContactHead\npair MLP(h_fg, h_res, d)\n4 channels / FG + contact_total",
        fill,
        head_edge,
        title_size=10.8,
        body_size=8.0,
    )
    artifacts["preorg_head"] = add_box(
        ax,
        boxes["preorg_head"],
        "Preorg head",
        "MLP on FG context\npreorg per FG",
        fill,
        head_edge,
        title_size=11,
        body_size=8.3,
    )
    artifacts["residue_stream"] = add_box(
        ax,
        boxes["residue_stream"],
        "Pocket geometry stream",
        "PocketResidueEncoder\ndistance masks\nnearest-residue fallback",
        fill,
        pocket_edge,
        title_size=10.6,
        body_size=8.15,
    )
    artifacts["complex_head"] = add_box(
        ax,
        boxes["complex_head"],
        "Whole-complex head",
        "pooled ligand + FG + residue summaries\n+ sum(contact_total, preorg)\nDeltaTotalHead -> delta_total",
        fill,
        head_edge,
        title_size=10.6,
        body_size=7.75,
    )

    add_path_arrow(ax, [boxes["ligand_stream"].anchor("east"), boxes["fg_stream"].anchor("west")], color=arrow_color, registry=routes, allow_boxes={"ligand_stream", "fg_stream"})
    add_path_arrow(ax, [boxes["fg_stream"].anchor("east"), boxes["interaction"].anchor("west")], color=arrow_color, registry=routes, allow_boxes={"fg_stream", "interaction"})
    add_path_arrow(ax, [boxes["interaction"].anchor("east"), boxes["pair_head"].anchor("west")], color=arrow_color, registry=routes, allow_boxes={"interaction", "pair_head"})
    add_path_arrow(
        ax,
        [boxes["residue_stream"].anchor("north"), (boxes["residue_stream"].anchor("north")[0], 0.53), boxes["interaction"].anchor("south")],
        color=arrow_color,
        registry=routes,
        allow_boxes={"residue_stream", "interaction"},
    )
    add_path_arrow(
        ax,
        [boxes["fg_stream"].anchor("south"), (boxes["fg_stream"].anchor("south")[0], 0.43), (boxes["preorg_head"].anchor("north")[0], 0.43), boxes["preorg_head"].anchor("north")],
        color=arrow_color,
        registry=routes,
        allow_boxes={"fg_stream", "preorg_head"},
    )
    add_path_arrow(
        ax,
        [boxes["pair_head"].anchor("south"), (boxes["pair_head"].anchor("south")[0], 0.43), boxes["complex_head"].anchor("north")],
        color=arrow_color,
        registry=routes,
        allow_boxes={"pair_head", "complex_head"},
    )
    add_path_arrow(
        ax,
        [
            boxes["preorg_head"].anchor("east"),
            (boxes["preorg_head"].anchor("east")[0] + 0.02, boxes["preorg_head"].anchor("east")[1]),
            (boxes["preorg_head"].anchor("east")[0] + 0.02, 0.19),
            (boxes["complex_head"].anchor("south")[0], 0.19),
            boxes["complex_head"].anchor("south"),
        ],
        color=arrow_color,
        registry=routes,
        allow_boxes={"preorg_head", "complex_head"},
    )

    ax.text(
        0.04,
        0.09,
        "Training targets: per-FG contact_total + [vdw, elec, polar, nonpolar], optional preorg, and per-complex delta_total.",
        fontsize=8.2,
        color="#4a4a4a",
    )
    ax.text(
        0.04,
        0.06,
        "Whole-complex head also consumes pooled ligand, FG, and residue context; omitted arrows are intentional to keep the slide legible.",
        fontsize=8.2,
        color="#4a4a4a",
    )

    issues = check_layout(fig, artifacts, routes, out_prefix.with_name(out_prefix.name + "_layout_check.json"))
    if issues:
        raise RuntimeError("layout check failed: " + "; ".join(issues))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _tikz_box(
    name: str,
    *,
    at: tuple[float, float],
    caption: str,
    xlabel: str = " ",
    zlabel: str = " ",
    fill: str = "\\ConvColor",
    width: float = 2.0,
    height: float = 10.0,
    depth: float = 10.0,
) -> str:
    caption = _tex_escape(caption)
    xlabel = _tex_escape(xlabel)
    zlabel = _tex_escape(zlabel)
    x, y = at
    return rf"""
\pic[shift={{({x},{y},0)}}] at (0,0,0) {{
    Box={{
        name={name},
        caption={{{caption}}},
        xlabel={{"{xlabel}",}},
        zlabel={{{zlabel}}},
        fill={fill},
        height={height},
        width={width},
        depth={depth}
    }}
}};
"""


def _tikz_connection_raw(src: str, dst: str) -> str:
    return rf"\draw [connection]  ({src}-east) -- node {{\midarrow}} ({dst}-west);"


def _tikz_draw_raw(src_anchor: str, dst_anchor: str, label: str | None = None) -> str:
    node = f" node {{ {_tex_escape(label)} }}" if label else ""
    return rf"\draw [connection] ({src_anchor}) --{node} ({dst_anchor});"


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def render_plotneuralnet_companion(summary: dict[str, Any], out_prefix: Path) -> dict[str, str | bool]:
    rel_layers_root = os.path.relpath(PLOT_NEURAL_NET_ROOT, out_prefix.parent).replace("\\", "/")
    tex_path = out_prefix.with_suffix(".tex")

    arch = [
        r"\input{" + rel_layers_root + r"/Layers/init.tex}",
    ]
    # Use the packaged header/box styles directly so the result stays compatible with PlotNeuralNet.
    arch = [
        rf"""
\documentclass[border=8pt, multi, tikz]{{standalone}}
\usepackage{{import}}
\subimport{{{rel_layers_root}/Layers/}}{{init}}
\usetikzlibrary{{positioning}}
\begin{{document}}
\begin{{tikzpicture}}
\tikzstyle{{connection}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw=\edgecolor,opacity=0.7]
\def\LigandColor{{rgb:orange,5;red,1;white,8}}
\def\FGColor{{rgb:green,4;black,0.8;white,8}}
\def\PocketColor{{rgb:blue,4;green,0.8;white,8}}
\def\AttentionColor{{rgb:violet,5;blue,1;white,8}}
\def\HeadColor{{rgb:red,4;black,0.8;white,9}}
"""
    ]

    cfg = summary["config_defaults"]
    arch.extend(
        [
            _tikz_box("ligin", at=(0.0, 3.0), caption="Ligand inputs", xlabel="atoms+bonds", zlabel="coords", fill="\\LigandColor", width=2.2, height=12, depth=12),
            _tikz_box("fgin", at=(0.0, 0.0), caption="FG inputs", xlabel="types+membership", zlabel="centroids", fill="\\FGColor", width=2.2, height=10, depth=10),
            _tikz_box("pocin", at=(0.0, -3.0), caption="Pocket inputs", xlabel="residue feats", zlabel="coords", fill="\\PocketColor", width=2.2, height=12, depth=12),
            _tikz_box("ligenc", at=(4.0, 3.0), caption="Ligand encoders", xlabel="atom+bond", zlabel=f"h={cfg.get('hidden_dim')}", fill="\\LigandColor", width=2.3, height=12, depth=12),
            _tikz_box("fgenc", at=(4.0, 0.0), caption="FGNodeEncoder", xlabel="fg init", zlabel=f"h={cfg.get('hidden_dim')}", fill="\\FGColor", width=2.3, height=10, depth=10),
            _tikz_box("pocenc", at=(4.0, -3.0), caption="PocketResidueEncoder", xlabel="res tokens", zlabel=f"h={cfg.get('hidden_dim')}", fill="\\PocketColor", width=2.6, height=12, depth=12),
            _tikz_box("liggnn", at=(8.4, 3.0), caption="LigandGNN", xlabel=f"{cfg.get('num_ligand_gnn_layers')}x GATv2", zlabel="h_atom", fill="\\LigandColor", width=2.3, height=12, depth=12),
            _tikz_box("fgpool", at=(8.4, 0.8), caption="FGAttentionPool", xlabel="atom->FG", zlabel="h_fg", fill="\\FGColor", width=2.4, height=10, depth=10),
            _tikz_box("fgctx", at=(12.8, 0.8), caption="FGSelfAttentionGraph", xlabel=f"{cfg.get('num_fg_attn_layers')}x graph attn", zlabel="h_fg_ctx", fill="\\FGColor", width=2.8, height=10, depth=10),
            _tikz_box("maskbox", at=(12.8, -2.3), caption="FG-res masks", xlabel=f"cutoff {cfg.get('cross_attn_cutoff')}A", zlabel="pair mask", fill="\\AttentionColor", width=2.2, height=7.5, depth=7.5),
            _tikz_box("crossattn", at=(17.4, 0.0), caption="FGPocketCrossAttention", xlabel=f"{cfg.get('num_cross_attn_layers')}x MHA", zlabel="h_fg_cross", fill="\\AttentionColor", width=2.8, height=11, depth=11),
            _tikz_box("fgglobal", at=(21.8, 1.8), caption="FGMoleculeSelfAttention", xlabel=f"{cfg.get('num_fg_global_attn_layers')}x", zlabel="h_fg_final", fill="\\AttentionColor", width=2.8, height=10, depth=10),
            _tikz_box("pairhead", at=(26.4, 0.4), caption="FGResidueContactHead", xlabel="pair MLP+gate", zlabel="4 ch / FG", fill="\\HeadColor", width=3.0, height=12, depth=12),
            _tikz_box("preorghead", at=(26.4, 3.8), caption="PreorgHead", xlabel="MLP", zlabel="1 / FG", fill="\\HeadColor", width=2.2, height=7.5, depth=7.5),
            _tikz_box("deltapool", at=(26.4, -3.2), caption="Complex pooling", xlabel="FG mean/sum + counts", zlabel="concat", fill="\\AttentionColor", width=3.0, height=8.5, depth=8.5),
            _tikz_box("deltahead", at=(31.2, -3.2), caption="DeltaTotalHead", xlabel="MLP", zlabel="1 / complex", fill="\\HeadColor", width=2.4, height=8.5, depth=8.5),
            _tikz_box("outbox", at=(31.2, 0.8), caption="Outputs", xlabel="contact, total, preorg", zlabel="delta", fill="\\HeadColor", width=2.8, height=12, depth=12),
            _tikz_connection_raw("ligin", "ligenc"),
            _tikz_connection_raw("fgin", "fgenc"),
            _tikz_connection_raw("pocin", "pocenc"),
            _tikz_connection_raw("ligenc", "liggnn"),
            _tikz_draw_raw("liggnn-south", "fgpool-north", "h_atom"),
            _tikz_connection_raw("fgenc", "fgpool"),
            _tikz_connection_raw("fgpool", "fgctx"),
            _tikz_connection_raw("pocenc", "maskbox"),
            _tikz_draw_raw("fgin-south", "maskbox-north", "centroids"),
            _tikz_connection_raw("fgctx", "crossattn"),
            _tikz_draw_raw("pocenc-east", "crossattn-south", "h_res"),
            _tikz_draw_raw("maskbox-north", "crossattn-south", "mask"),
            _tikz_draw_raw("crossattn-north", "fgglobal-south", "FG tokens"),
            _tikz_draw_raw("fgglobal-south", "pairhead-north", "h_fg_final"),
            _tikz_draw_raw("pocenc-east", "pairhead-south", "h_res"),
            _tikz_draw_raw("maskbox-east", "pairhead-south", "dists+mask"),
            _tikz_draw_raw("fgctx-north", "preorghead-south", "h_fg_ctx"),
            _tikz_draw_raw("liggnn-east", "deltapool-west", "lig mean"),
            _tikz_draw_raw("pocenc-east", "deltapool-south", "res mean"),
            _tikz_draw_raw("fgglobal-east", "deltapool-north", "FG mean/sum"),
            _tikz_draw_raw("pairhead-east", "deltapool-north", "sum total"),
            _tikz_draw_raw("preorghead-south", "deltapool-north", "sum preorg"),
            _tikz_connection_raw("deltapool", "deltahead"),
            _tikz_draw_raw("pairhead-east", "outbox-west", "contact"),
            _tikz_draw_raw("preorghead-east", "outbox-west", "preorg"),
            _tikz_draw_raw("deltahead-east", "outbox-south", "delta"),
            r"\end{tikzpicture}",
            r"\end{document}",
        ]
    )

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(arch))

    compile_ok = False
    compile_log = ""
    try:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        compile_ok = proc.returncode == 0 and tex_path.with_suffix(".pdf").is_file()
        compile_log = proc.stdout[-4000:] + proc.stderr[-4000:]
    except Exception as exc:
        compile_log = str(exc)

    if compile_log:
        tex_path.with_suffix(".log.txt").write_text(compile_log)
    return {
        "tex": str(tex_path),
        "pdf": str(tex_path.with_suffix(".pdf")),
        "compiled": compile_ok,
    }


def write_markdown_summary(summary: dict[str, Any], out_path: Path) -> None:
    cfg = summary["config_defaults"]
    lines = [
        "# model1 Architecture Summary",
        "",
        "Generated from the current source tree after parsing the architecture-related files in `models/model1`.",
        "",
        "## Defaults",
        "",
        f"- `hidden_dim={cfg.get('hidden_dim')}`",
        f"- `num_ligand_gnn_layers={cfg.get('num_ligand_gnn_layers')}`",
        f"- `num_fg_attn_layers={cfg.get('num_fg_attn_layers')}`",
        f"- `num_fg_global_attn_layers={cfg.get('num_fg_global_attn_layers')}`",
        f"- `num_cross_attn_layers={cfg.get('num_cross_attn_layers')}`",
        f"- `num_rbf_kernels={cfg.get('num_rbf_kernels')}`",
        f"- `fg_edge_cutoff={cfg.get('fg_edge_cutoff')}` A",
        f"- `cross_attn_cutoff={cfg.get('cross_attn_cutoff')}` A",
        "",
        "## Forward Sequence",
        "",
    ]
    for idx, step in enumerate(summary["forward_sequence"], start=1):
        lines.append(f"{idx}. `{step}`")
    lines.extend(
        [
            "",
            "## Parsed Files",
            "",
        ]
    )
    for path in summary["files_parsed"]:
        lines.append(f"- `{path}`")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    report_dir = REPO_ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_styles(args.plot_style)

    summary = build_architecture_summary()
    (report_dir / "model1_architecture_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_summary(summary, report_dir / "model1_architecture_summary.md")

    render_architecture_figure(summary, report_dir / "model1_architecture_detailed")
    render_architecture_slide_figure(summary, report_dir / "model1_architecture_full")
    render_architecture_slide_figure(summary, report_dir / "model1_architecture_slide")
    plotneuralnet_status = render_plotneuralnet_companion(summary, report_dir / "model1_architecture_plotneuralnet")
    (report_dir / "model1_architecture_plotneuralnet_status.json").write_text(json.dumps(plotneuralnet_status, indent=2))

    print(f"Wrote architecture report to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
