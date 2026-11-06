"""Трёхмерная решётка: та же геометрия, другая размерность."""

from __future__ import annotations

from .dim2 import Dim2


class Dim3(Dim2):
    """Смещения в объёме: ``dz``, ``dy`` и ``dx``."""

    dimensions = 3


__all__ = ["Dim3"]
