"""Фильтр плотности: доля живых клеток должна лежать в коридоре."""

from __future__ import annotations

from typing import Any, Mapping

from ...module import BuildView, FilterModule, FilterRuntime, Observation, PASS, Verdict
from .base import REQUIRES


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
    requires = REQUIRES

    def __init__(
        self,
        min_fill: float = 0.02,
        max_fill: float = 0.95,
        prefix: str | None = None,
    ) -> None:
        self.min_fill = float(min_fill)
        self.max_fill = float(max_fill)
        self.prefix = prefix

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        if self.prefix:
            min_fill = float(fixed[f"{self.prefix}.density_min"])
            max_fill = float(fixed[f"{self.prefix}.density_max"])
        else:
            min_fill = self.min_fill
            max_fill = self.max_fill
        return DensityRuntime(min_fill, max_fill)


__all__ = ["DensityFilter", "DensityRuntime"]
