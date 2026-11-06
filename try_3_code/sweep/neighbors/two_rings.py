"""Два радиальных кольца: внутренний диск и внешнее кольцо."""

from __future__ import annotations

from typing import Any, Mapping

from ..module import NeighborhoodSpec
from ..space import Namespace
from .base import BaseNeighborhood


class TwoRings(BaseNeighborhood):
    id = "neighborhood.two_rings"
    label = "два радиальных кольца"

    def declare(self, ns: Namespace, view) -> None:
        ns.range("inner_radius", 1, 3)
        ns.range("outer_radius", 3, 6)
        ns.choice("metric", ["euclidean", "chebyshev"])
        ns.choice("wrap", [True, False])

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        inner_radius = int(self.get(fixed, "inner_radius"))
        outer_radius = int(self.get(fixed, "outer_radius"))
        metric = str(self.get(fixed, "metric"))
        wrap = bool(self.get(fixed, "wrap"))
        inner = self.lattice.offsets(inner_radius, metric)
        outer = self.lattice.ring(inner_radius, outer_radius, metric)
        return self.make_spec(
            f"two-rings-{metric}-{inner_radius}-{outer_radius}",
            outer,
            inner,
            wrap,
        )


__all__ = ["TwoRings"]
