"""Модули инициализации сетки.

Случайность берётся из ``numpy.random.Generator`` с seed из конфига, поэтому
один и тот же конфиг всегда даёт одно и то же начальное состояние.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..module import RandomizerModule
from ..space import Namespace


def _density(module: RandomizerModule, fixed: Mapping[str, Any]) -> float:
    return float(module.get(fixed, "density"))


class UniformFill(RandomizerModule):
    id = "randomizer.uniform"
    label = "равномерная случайная заливка"

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
        density = _density(self, fixed)
        height, width = data.shape
        data[:] = zero_state
        if states <= 2:
            data[rng.random((height, width)) < density] = 1
        else:
            mask = rng.random((height, width)) < density
            data[mask] = rng.integers(1, states, size=int(mask.sum()), dtype=np.int32)


class ClusteredFill(RandomizerModule):
    id = "randomizer.clustered"
    label = "случайные кластеры"

    def declare(self, ns: Namespace, view) -> None:
        ns.range("density", 0.1, 0.9, 0.1)
        ns.range("seeds", 2, 12, 2)

    def fill(
        self,
        data: np.ndarray,
        fixed: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        zero_state: int,
        states: int,
    ) -> None:
        density = _density(self, fixed)
        seeds = int(self.get(fixed, "seeds"))
        height, width = data.shape
        data[:] = zero_state
        for _ in range(seeds):
            cy = int(rng.integers(0, height))
            cx = int(rng.integers(0, width))
            radius = max(1, int(round(min(height, width) * density / 3)))
            y0 = max(0, cy - radius)
            y1 = min(height, cy + radius + 1)
            x0 = max(0, cx - radius)
            x1 = min(width, cx + radius + 1)
            patch = data[y0:y1, x0:x1]
            alive = rng.random(patch.shape) < density
            if states <= 2:
                patch[alive] = 1
            else:
                patch[alive] = rng.integers(1, states, size=int(alive.sum()), dtype=np.int32)


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
        density = _density(self, fixed)
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


__all__ = ["ClusteredFill", "SymmetricFill", "UniformFill"]
