# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
from torch import Tensor


ANI2X_RS_VALUES = torch.linspace(0.80, 4.83, 16, dtype=torch.float32)


def get_rbf_centers(num_kernels: int, d_min: float, d_max: float) -> Tensor:
    return torch.linspace(float(d_min), float(d_max), int(num_kernels), dtype=torch.float32)


def get_rbf_widths(num_kernels: int, d_min: float, d_max: float) -> Tensor:
    if num_kernels <= 1:
        return torch.ones((1,), dtype=torch.float32)
    delta = (float(d_max) - float(d_min)) / float(num_kernels - 1)
    gamma = 1.0 / max(delta * delta, 1e-8)
    return torch.full((int(num_kernels),), gamma, dtype=torch.float32)


def rbf_expand(distances: Tensor, centers: Tensor, widths: Tensor) -> Tensor:
    distances = torch.nan_to_num(distances.float(), nan=0.0, posinf=0.0, neginf=0.0)
    centers = centers.to(device=distances.device, dtype=distances.dtype)
    widths = widths.to(device=distances.device, dtype=distances.dtype)
    while distances.dim() < 1:
        distances = distances.unsqueeze(0)
    diff = distances.unsqueeze(-1) - centers
    return torch.exp(-widths * diff.square())


def cosine_cutoff(distances: Tensor, rc: float) -> Tensor:
    rc = float(rc)
    scaled = torch.clamp(distances / max(rc, 1e-6), min=0.0, max=1.0)
    values = 0.5 * (torch.cos(scaled * math.pi) + 1.0)
    return torch.where(distances <= rc, values, torch.zeros_like(values))
