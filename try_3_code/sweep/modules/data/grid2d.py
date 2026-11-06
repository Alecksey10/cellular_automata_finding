"""Модуль данных: дискретная 2D-сетка.

Данные объявляют свои возможности (интерфейсы), поэтому правила и фильтры
проверяются на совместимость до исполнения. Сетка уже numba-совместима:
``int32``, C-order, без объектов.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...interfaces import Capability, CapabilityKind
from ...kernels import alive_count as _alive_count
from ...kernels import state_histogram
from ...module import DataModule, RandomizerModule
from ...space import Namespace


class Grid2D(DataModule):
    id = "grid2d"
    label = "дискретная 2D-сетка"

    def declare(self, ns: Namespace, view) -> None:
        ns.value("zero", 0)
        ns.range("states", 3, 15)
        ns.range("width", 64, 65, 32)
        ns.range("height", 64, 65, 32)

    def derive_states(self, fixed: Mapping[str, Any]) -> int:
        return int(self.get(fixed, "states"))

    def derive_zero(self, fixed: Mapping[str, Any]) -> int:
        return int(self.get(fixed, "zero"))

    def capabilities(self, fixed: Mapping[str, Any]) -> tuple[Capability, ...]:
        zero = self.derive_zero(fixed)
        states = self.derive_states(fixed)
        return (
            Capability(CapabilityKind.GRID_2D, {"zero": zero}, self.id),
            Capability(CapabilityKind.DIMENSION, {"value": 2}, self.id),
            Capability(CapabilityKind.MULTI_STATE, {"states": states, "zero": zero}, self.id),
            Capability(CapabilityKind.ZERO_STATE, {"zero": zero}, self.id),
            Capability(
                CapabilityKind.RESOLUTION,
                {"width": self.get(fixed, "width"), "height": self.get(fixed, "height")},
                self.id,
            ),
        )

    def create(
        self,
        fixed: Mapping[str, Any],
        rng: np.random.Generator,
        randomizer: RandomizerModule,
    ) -> np.ndarray:
        height = int(self.get(fixed, "height"))
        width = int(self.get(fixed, "width"))
        zero = self.derive_zero(fixed)
        states = self.derive_states(fixed)
        data = np.full((height, width), zero, dtype=np.int32)
        randomizer.fill(data, fixed, rng, zero_state=zero, states=states)
        return np.ascontiguousarray(data, dtype=np.int32)

    def metrics(self, data: np.ndarray, fixed: Mapping[str, Any]) -> dict[str, Any]:
        states = self.derive_states(fixed)
        histogram = state_histogram(data, states)
        total = float(data.size) or 1.0
        alive = int(data.size) - int(histogram[0])
        weighted = 0.0
        for state in range(1, states):
            weighted += float(state) * float(histogram[state])
        metrics: dict[str, Any] = {
            "zero_density": float(histogram[0]) / total,
            "alive_density": alive / total,
            "mean_state": weighted / alive if alive else 0.0,
            "max_state_share": (float(histogram.max()) / total) if states else 0.0,
            "occupied_states": float(np.count_nonzero(histogram)),
        }
        for state in range(states):
            metrics[f"share_{state}"] = float(histogram[state]) / total
        return metrics

    def alive_count(self, data: np.ndarray, fixed: Mapping[str, Any]) -> int:
        return int(_alive_count(data, self.derive_zero(fixed)))


__all__ = ["Grid2D"]
