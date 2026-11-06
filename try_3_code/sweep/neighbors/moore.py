"""Окрестность Мура: квадрат ``(2r+1)×(2r+1)`` без центра."""

from __future__ import annotations

from typing import Any, Mapping

from ..module import NeighborhoodSpec
from ..space import Namespace
from .base import BaseNeighborhood


class Moore(BaseNeighborhood):
    id = "neighborhood.moore"
    label = "квадрат (Мура)"

    def declare(self, ns: Namespace, view) -> None:
        ns.range("radius", 1, 4)
        ns.choice("wrap", [True, False])

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        radius = int(self.get(fixed, "radius"))
        wrap = bool(self.get(fixed, "wrap"))
        offsets = self.lattice.offsets(radius, "chebyshev")
        return self.make_spec(f"moore{radius}", offsets, offsets, wrap)


__all__ = ["Moore"]
