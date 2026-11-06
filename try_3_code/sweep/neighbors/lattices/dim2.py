"""Двухмерная решётка."""

from __future__ import annotations

from ..geometry import disc_offsets, ring_offsets
from ..interfaces import Offsets


class Dim2:
    """Смещения в плоскости: ``dy`` и ``dx``."""

    dimensions = 2

    def offsets(self, radius: int, metric: str = "chebyshev") -> Offsets:
        return disc_offsets(radius, metric, self.dimensions)

    def ring(self, inner: int, outer: int, metric: str = "chebyshev") -> Offsets:
        return ring_offsets(inner, outer, metric, self.dimensions)


__all__ = ["Dim2"]
