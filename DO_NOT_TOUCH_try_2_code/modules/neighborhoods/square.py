"""Окрестность «квадрат» (метрика Чебышёва)."""

from __future__ import annotations

from casweep.fixed import Fixed
from casweep.space import DeclareView, Namespace

from .base import NeighborhoodModule, offsets_from_predicate


class SquareNeighborhood(NeighborhoodModule):
    id = "neighborhood.square"

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        super().declare(namespace, view)
        namespace.range("size", 1, 5, 1)

    def offsets(self, fixed: Fixed) -> tuple[tuple[int, int], ...]:
        size = int(fixed["neighborhood.size"])
        return offsets_from_predicate(size, "square", size, size)
