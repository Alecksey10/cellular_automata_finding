"""Окрестность «круг» (евклидова метрика)."""

from __future__ import annotations

from casweep.fixed import Fixed
from casweep.space import DeclareView, Namespace

from .base import NeighborhoodModule, offsets_from_predicate


class RadiusNeighborhood(NeighborhoodModule):
    id = "neighborhood.radius"

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        super().declare(namespace, view)
        namespace.range("radius", 1, 6, 1)

    def offsets(self, fixed: Fixed) -> tuple[tuple[int, int], ...]:
        radius = int(fixed["neighborhood.radius"])
        return offsets_from_predicate(radius, "radius", radius, radius)
