"""Фильтр стабильности: запрещает вечную перестройку и период 2."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...kernels import as_mask, mask_difference
from ...module import BuildView, FilterModule, FilterRuntime, Observation, PASS, Verdict
from .base import REQUIRES


class StabilityRuntime(FilterRuntime):
    def __init__(self, max_change: float, forbid_period2: bool) -> None:
        self.max_change = float(max_change)
        self.forbid_period2 = bool(forbid_period2)
        self._previous: np.ndarray | None = None
        self._before_previous: np.ndarray | None = None

    def check(self, observation: Observation) -> Verdict:
        zero = int(observation.metrics.get("zero", 0))
        mask = as_mask(observation.data, zero)
        if self._previous is None:
            self._previous = mask
            return PASS
        difference = float(mask_difference(self._previous, mask))
        period2 = self._before_previous is not None and np.array_equal(self._before_previous, mask)
        self._before_previous = self._previous
        self._previous = mask
        if period2 and self.forbid_period2:
            return Verdict(False, "period2", {"difference": difference})
        if difference > self.max_change:
            return Verdict(False, "unstable", {"difference": difference})
        return PASS


class StabilityFilter(FilterModule):
    id = "filter.stability"
    label = "стабилизация"
    requires = REQUIRES

    def __init__(self, max_change: float = 0.3, forbid_period2: bool = True) -> None:
        self.max_change = float(max_change)
        self.forbid_period2 = bool(forbid_period2)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return StabilityRuntime(self.max_change, self.forbid_period2)


__all__ = ["StabilityFilter", "StabilityRuntime"]
