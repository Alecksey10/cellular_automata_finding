"""Сетка эксперимента ``omega5``: ровно 8 состояний."""

from __future__ import annotations

from ...modules.data import Grid2D
from ...space import Namespace


class OmegaGrid(Grid2D):
    """Та же int32-сетка, но с фиксированными восемью состояниями."""

    id = "grid2d"
    label = "2D-сетка, 8 состояний"

    def declare(self, ns: Namespace, view) -> None:
        ns.value("zero", 0)
        ns.value("states", 8)
        ns.range("width", 48, 65, 16)
        ns.range("height", 48, 65, 16)


__all__ = ["OmegaGrid"]
