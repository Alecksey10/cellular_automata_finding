"""Симметричная заливка: левая половина зеркалится в правую."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...module import RandomizerModule
from ...space import Namespace
from .base import density_of


class SymmetricFill(RandomizerModule):
    id = "randomizer.symmetric"
    label = "симметричная заливка"

    def declare(self, ns: Namespace, view) -> None:
        ns.range("density", 0.1, 0.9, 0.1)

    def fill(
        self,
        data: np.ndarray,
        fixed: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        zero_state: int,
        states: int,
    ) -> None:
        density = density_of(self, fixed)
        height, width = data.shape
        data[:] = zero_state
        half = (width + 1) // 2
        left = np.zeros((height, half), dtype=np.int32)
        mask = rng.random((height, half)) < density
        if states <= 2:
            left[mask] = 1
        else:
            left[mask] = rng.integers(1, states, size=int(mask.sum()), dtype=np.int32)
        data[:, :half] = left
        data[:, width - half :] = left[:, ::-1]


__all__ = ["SymmetricFill"]
