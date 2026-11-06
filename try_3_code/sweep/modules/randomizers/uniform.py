"""Равномерная случайная заливка."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...module import RandomizerModule
from ...space import Namespace
from .base import density_of


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
        density = density_of(self, fixed)
        height, width = data.shape
        data[:] = zero_state
        if states <= 2:
            data[rng.random((height, width)) < density] = 1
        else:
            mask = rng.random((height, width)) < density
            data[mask] = rng.integers(1, states, size=int(mask.sum()), dtype=np.int32)


__all__ = ["UniformFill"]
