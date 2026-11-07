"""Сетка симбиоза: ровно семь состояний."""

from __future__ import annotations

from ...modules.data import Grid2D
from ...space import Namespace


class SymbiosisGrid(Grid2D):
    """Та же int32-сетка, но с фиксированными семью состояниями."""

    id = "grid2d"
    label = "2D-сетка, 7 состояний"

    def declare(self, ns: Namespace, view) -> None:
        ns.value("zero", 0)
        ns.value("states", 7)
        ns.range("width", 64, 65, 16)
        ns.range("height", 64, 65, 16)


__all__ = ["SymbiosisGrid"]
