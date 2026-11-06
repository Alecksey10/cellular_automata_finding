"""Та же сетка, но с интерфейсом симметричной инициализации."""

from __future__ import annotations

from .grid2d import Grid2D


class SymmetricGrid2D(Grid2D):
    id = "grid2d.symmetric"
    label = "2D-сетка с симметричной инициализацией"


__all__ = ["SymmetricGrid2D"]
