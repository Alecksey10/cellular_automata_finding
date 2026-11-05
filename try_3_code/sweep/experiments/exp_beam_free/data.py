"""Сетка эксперимента ``expBeamFree``: 9..20 состояний."""

from __future__ import annotations

from typing import Any, Mapping

from ...modules.data import Grid2D
from ...space import Namespace


class DuelGrid(Grid2D):
    id = "grid2d"
    label = "2D-сетка, 9..20 состояний"

    def declare(self, ns: Namespace, view) -> None:
        ns.value("zero", 0)
        ns.range("states", 9, 21)
        ns.range("width", 48, 65, 16)
        ns.range("height", 48, 65, 16)

    def derived_value(self, name: str, fixed: Mapping[str, Any]) -> Any:
        if name == "states":
            return int(self.get(fixed, "states"))
        return super().derived_value(name, fixed)


__all__ = ["DuelGrid"]
