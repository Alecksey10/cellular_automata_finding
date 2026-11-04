"""Модули фильтрации: решают, продолжать ли выполнение конфига.

Фильтр не знает о стадиях и итерациях: он получает
:class:`~sweep.module.Observation` и возвращает :class:`~sweep.module.Verdict`.
Когда именно вызывать ``check`` — решает движок по ``Phase.frequency``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..interfaces import CapabilityKind, Requirement
from ..kernels import as_mask, cluster_stats, mask_difference, mask_iou, state_histogram
from ..module import (
    BuildView,
    FilterModule,
    FilterRuntime,
    Observation,
    PASS,
    Role,
    Verdict,
)

_REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
)


class DensityRuntime(FilterRuntime):
    def __init__(self, min_fill: float, max_fill: float) -> None:
        self.min_fill = float(min_fill)
        self.max_fill = float(max_fill)

    def check(self, observation: Observation) -> Verdict:
        density = float(observation.metrics.get("alive_density", 0.0))
        if density < self.min_fill:
            return Verdict(False, "too_sparse", {"alive_density": density})
        if density > self.max_fill:
            return Verdict(False, "too_dense", {"alive_density": density})
        return PASS


class DensityFilter(FilterModule):
    id = "filter.density"
    label = "допустимое заполнение"
    requires = _REQUIRES

    def __init__(self, min_fill: float = 0.02, max_fill: float = 0.95) -> None:
        self.min_fill = float(min_fill)
        self.max_fill = float(max_fill)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return DensityRuntime(self.min_fill, self.max_fill)


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
    requires = _REQUIRES

    def __init__(self, max_change: float = 0.3, forbid_period2: bool = True) -> None:
        self.max_change = float(max_change)
        self.forbid_period2 = bool(forbid_period2)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return StabilityRuntime(self.max_change, self.forbid_period2)


class ClustersRuntime(FilterRuntime):
    def __init__(self, min_components: int, min_mean_size: float, max_largest_share: float) -> None:
        self.min_components = int(min_components)
        self.min_mean_size = float(min_mean_size)
        self.max_largest_share = float(max_largest_share)

    def check(self, observation: Observation) -> Verdict:
        zero = int(observation.metrics.get("zero", 0))
        mask = as_mask(observation.data, zero)
        components, mean_size, largest_share = cluster_stats(mask)
        details = {
            "components": int(components),
            "mean_size": float(mean_size),
            "largest_share": float(largest_share),
        }
        if components < self.min_components:
            return Verdict(False, "too_few_components", details)
        if mean_size < self.min_mean_size:
            return Verdict(False, "small_clusters", details)
        if largest_share > self.max_largest_share:
            return Verdict(False, "one_huge_cluster", details)
        return Verdict(True, "", details)


class ClustersFilter(FilterModule):
    id = "filter.clusters"
    label = "структура кластеров"
    requires = _REQUIRES

    def __init__(
        self,
        min_components: int = 1,
        min_mean_size: float = 3.0,
        max_largest_share: float = 0.95,
    ) -> None:
        self.min_components = int(min_components)
        self.min_mean_size = float(min_mean_size)
        self.max_largest_share = float(max_largest_share)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return ClustersRuntime(self.min_components, self.min_mean_size, self.max_largest_share)


class PhasesRuntime(FilterRuntime):
    def __init__(self, min_share: float, states: int) -> None:
        self.min_share = float(min_share)
        self.states = int(states)

    def check(self, observation: Observation) -> Verdict:
        histogram = state_histogram(observation.data, self.states)
        total = float(observation.data.size) or 1.0
        shares = [float(histogram[state]) / total for state in range(self.states)]
        if any(share < self.min_share for share in shares):
            return Verdict(False, "missing_phase", {"shares": shares})
        return Verdict(True, "", {"shares": shares})


class PhasesFilter(FilterModule):
    id = "filter.phases"
    label = "все фазы представлены"
    requires = _REQUIRES + (Requirement(CapabilityKind.MULTI_STATE),)

    def __init__(self, min_share: float = 0.02) -> None:
        self.min_share = float(min_share)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return PhasesRuntime(self.min_share, int(fixed["grid2d.states"]))


class SymmetryRuntime(FilterRuntime):
    def __init__(self, zero: int, min_iou: float) -> None:
        self.zero = int(zero)
        self.min_iou = float(min_iou)

    def check(self, observation: Observation) -> Verdict:
        mask = as_mask(observation.data, self.zero)
        best = 1.0
        candidates = [mask[:, ::-1], mask[::-1, :], mask[::-1, ::-1]]
        for turns in (1, 2, 3):
            rotated = np.ascontiguousarray(np.rot90(mask, turns), dtype=np.uint8)
            if rotated.shape != mask.shape:
                continue  # неквадратную сетку можно сравнивать только с отражениями
            candidates.append(rotated)
            candidates.append(np.ascontiguousarray(np.fliplr(rotated), dtype=np.uint8))
        for candidate in candidates:
            best = max(best, float(mask_iou(mask, np.ascontiguousarray(candidate, dtype=np.uint8))))
        if best < self.min_iou:
            return Verdict(False, "asymmetric", {"iou": best})
        return PASS


class SymmetryFilter(FilterModule):
    id = "filter.symmetry"
    label = "симметрия"
    requires = _REQUIRES

    def __init__(self, min_iou: float = 0.9) -> None:
        self.min_iou = float(min_iou)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return SymmetryRuntime(int(fixed["grid2d.zero"]), self.min_iou)


ALL_FILTERS = [
    DensityFilter,
    StabilityFilter,
    ClustersFilter,
    PhasesFilter,
    SymmetryFilter,
]

__all__ = [
    "ALL_FILTERS",
    "ClustersFilter",
    "ClustersRuntime",
    "DensityFilter",
    "DensityRuntime",
    "PhasesFilter",
    "PhasesRuntime",
    "StabilityFilter",
    "StabilityRuntime",
    "SymmetryFilter",
    "SymmetryRuntime",
]
