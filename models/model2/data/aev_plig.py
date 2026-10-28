# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
from torch import Tensor


def cutoff_cosine(distances: Tensor, cutoff: float) -> Tensor:
    return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5


def radial_terms(cutoff: float, eta_r: Tensor, shift_r: Tensor, distances: Tensor) -> Tensor:
    distances = distances.reshape(-1, 1, 1)
    fc = cutoff_cosine(distances, cutoff)
    ret = 0.25 * torch.exp(-eta_r * (distances - shift_r).square()) * fc
    return ret.flatten(start_dim=1)
