"""Окрестности Мура фиксированного размера: 3×3 (8 соседей) и 5×5 (24 соседа).

Размер не перебирается: это отдельные модули, чтобы эксперимент явно называл
сетку соседства. Граница (``wrap``/``clip``) по-прежнему ось перебора.
"""

from __future__ import annotations

from casweep.fixed import Fixed

from .base import NeighborhoodModule, offsets_from_predicate


class Moore3Neighborhood(NeighborhoodModule):
    """Кольцо 3×3 вокруг клетки: 8 соседей."""

    id = "neighborhood.moore3"

    def offsets(self, fixed: Fixed) -> tuple[tuple[int, int], ...]:
        return offsets_from_predicate(1, "square", 1, 1)


class Moore5Neighborhood(NeighborhoodModule):
    """Квадрат 5×5 вокруг клетки: 24 соседа."""

    id = "neighborhood.moore5"

    def offsets(self, fixed: Fixed) -> tuple[tuple[int, int], ...]:
        return offsets_from_predicate(2, "square", 2, 2)
