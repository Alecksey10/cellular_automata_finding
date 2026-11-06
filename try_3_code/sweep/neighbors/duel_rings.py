"""Кольца дуэли: внутреннее 3×3 и внешнее 5×5 для двух групп клеток."""

from __future__ import annotations

from typing import Any, Mapping

from ..module import NeighborhoodSpec
from ..space import Namespace
from .base import BaseNeighborhood


class DuelRings(BaseNeighborhood):
    """Два кольца 3×3 / 5×5 для дуэли двух групп."""

    id = "neighborhood.duel_rings"
    label = "два кольца 3×3 / 5×5"

    def declare(self, ns: Namespace, view) -> None:
        ns.choice("wrap", [True, False])

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        wrap = bool(self.get(fixed, "wrap"))
        inner = self.lattice.offsets(1, "chebyshev")
        outer = self.lattice.offsets(2, "chebyshev")
        return self.make_spec("duel-rings-3x5", outer, inner, wrap)


__all__ = ["DuelRings"]
