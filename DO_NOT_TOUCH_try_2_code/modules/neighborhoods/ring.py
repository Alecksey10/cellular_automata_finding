"""Окрестность «кольцо»: ``r1 < distance <= r2``."""

from __future__ import annotations

from casweep.fixed import Fixed
from casweep.space import DeclareView, Namespace

from .base import NeighborhoodModule, offsets_from_predicate


class RingNeighborhood(NeighborhoodModule):
    id = "neighborhood.ring"

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        super().declare(namespace, view)
        inner = namespace.range("r1", 1, 5, 1)
        namespace.range("r2", inner + 1, 8, 1)

    def offsets(self, fixed: Fixed) -> tuple[tuple[int, int], ...]:
        r1 = int(fixed["neighborhood.r1"])
        r2 = int(fixed["neighborhood.r2"])
        return offsets_from_predicate(r2, "ring", r1, r2)
