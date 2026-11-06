"""Окрестность 9×9 без центральной клетки: окно видимости радиус 4.

80 соседей, внутреннее кольцо 3×3 (8 клеток) и внешнее 9×9. На таком окне
работает эксперимент ``maze9``: механизм выбирает подмаску внутри него.
"""

from __future__ import annotations

from .base import SquareWindow


class Square9x9(SquareWindow):
    id = "neighborhood.square9"
    label = "окно 9×9 (80 соседей)"
    radius = 4


__all__ = ["Square9x9"]
