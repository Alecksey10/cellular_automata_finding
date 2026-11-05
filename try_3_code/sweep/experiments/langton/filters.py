"""Фильтры отбора для муравья Ленгтона.

Проверки идут каждые ``every`` итераций (см. ``phase(..., every=100)``), но
решение принимается в конце: муравью нужно много шагов, чтобы «расцвести»,
поэтому ранние проверки только накапливают историю. Итоговая проверка — за
последние ``grace`` шагов фазы.

* :class:`SettleFilter` — «след перестал расти»: муравей зациклился в своём
  следе, а не уходит бесконечной магистралью.
* :class:`PatternFilter` — «паттерн симметричен»: на финальном кадре след
  совпадает с собственным отражением или поворотом.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...interfaces import CapabilityKind, Requirement
from ...module import (
    BuildView,
    FilterModule,
    FilterRuntime,
    Observation,
    PASS,
    Verdict,
)
from ...modules.filters import SymmetryRuntime

_REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
)

# сколько последних шагов фазы считаются решающими
GRACE = 300


def deadline_of(fixed: Mapping[str, Any], grace: int = GRACE) -> int:
    """Итерация, с которой проверка становится решающей."""

    total = int(fixed.get("phase.warmup", 0)) + int(fixed.get("phase.observe", 0))
    return max(0, total - int(grace))


class SettleRuntime(FilterRuntime):
    def __init__(
        self,
        max_growth: float,
        min_cells: float,
        window: int,
        deadline: int,
        max_density: float,
    ) -> None:
        self.max_growth = float(max_growth)
        self.min_cells = float(min_cells)
        self.window = max(1, int(window))
        self.deadline = int(deadline)
        self.max_density = float(max_density)
        self._previous: float | None = None

    def check(self, observation: Observation) -> Verdict:
        cells = float(observation.metrics.get("trail_cells", 0.0))
        iteration = int(observation.metrics.get("iteration", 0))
        growth = 0.0
        if self._previous is not None:
            growth = (cells - self._previous) / float(self.window)
        self._previous = cells
        if iteration < self.deadline:
            return PASS
        density = float(observation.metrics.get("alive_density", 0.0))
        details = {"growth": growth, "trail_cells": cells, "density": density}
        if cells < self.min_cells:
            return Verdict(False, "tiny_trail", details)
        # закрашенное целиком поле — не паттерн: у него и рост, и симметрия
        # выполняются тривиально
        if density > self.max_density:
            return Verdict(False, "saturated", details)
        if growth > self.max_growth:
            return Verdict(False, "still_growing", details)
        return Verdict(True, "", details)


class SettleFilter(FilterModule):
    id = "filter.ant_settle"
    label = "след стабилизировался"
    requires = _REQUIRES

    def __init__(
        self,
        max_growth: float = 0.2,
        min_cells: float = 150.0,
        window: int = 100,
        max_density: float = 0.7,
    ) -> None:
        self.max_growth = float(max_growth)
        self.min_cells = float(min_cells)
        self.window = int(window)
        self.max_density = float(max_density)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> SettleRuntime:
        return SettleRuntime(
            self.max_growth,
            self.min_cells,
            self.window,
            deadline_of(fixed),
            self.max_density,
        )


class PatternRuntime(SymmetryRuntime):
    """Симметрия проверяется только на решающих шагах."""

    def __init__(self, zero: int, min_iou: float, deadline: int) -> None:
        super().__init__(zero, min_iou)
        self.deadline = int(deadline)

    def check(self, observation: Observation) -> Verdict:
        if int(observation.metrics.get("iteration", 0)) < self.deadline:
            return PASS
        return super().check(observation)


class PatternFilter(FilterModule):
    id = "filter.ant_pattern"
    label = "симметричный паттерн"
    requires = _REQUIRES

    def __init__(self, min_iou: float = 0.6) -> None:
        self.min_iou = float(min_iou)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> PatternRuntime:
        return PatternRuntime(int(fixed["grid2d.zero"]), self.min_iou, deadline_of(fixed))


ALL_FILTERS = [SettleFilter, PatternFilter]

__all__ = [
    "ALL_FILTERS",
    "GRACE",
    "PatternFilter",
    "PatternRuntime",
    "SettleFilter",
    "SettleRuntime",
    "deadline_of",
]
