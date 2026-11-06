"""Радиальный диск: евклидова, манхэттенская или чебышёвская метрика."""

from __future__ import annotations

from typing import Any, Mapping

from ..module import NeighborhoodSpec
from ..space import Namespace
from .base import BaseNeighborhood
from .geometry import METRICS


class RadiusDisc(BaseNeighborhood):
    id = "neighborhood.radius"
    label = "радиальный диск"

    def declare(self, ns: Namespace, view) -> None:
        ns.range("radius", 1, 4)
        ns.choice("metric", list(METRICS))
        ns.choice("wrap", [True, False])

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        radius = int(self.get(fixed, "radius"))
        metric = str(self.get(fixed, "metric"))
        wrap = bool(self.get(fixed, "wrap"))
        offsets = self.lattice.offsets(radius, metric)
        return self.make_spec(f"radius-{metric}-{radius}", offsets, offsets, wrap)


__all__ = ["RadiusDisc"]
