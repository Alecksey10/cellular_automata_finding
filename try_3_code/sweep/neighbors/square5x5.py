"""Окрестность 5×5 без центральной клетки.

Это окно видимости «радиус 2» по Чебышёву: 24 соседа, внутреннее кольцо
3×3 (8 клеток) и внешнее кольцо 5×5 (16 клеток). Именно на таком окне
работает эксперимент ``omega5``.
"""

from __future__ import annotations

from .base import SquareWindow


class Square5x5(SquareWindow):
    id = "neighborhood.square5"
    label = "окно 5×5 (24 соседа)"
    radius = 2


__all__ = ["Square5x5"]
