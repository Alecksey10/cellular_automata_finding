"""Общая часть фильтров: требования к данным.

Фильтры работают с ``Observation`` (стадия, итерация, сетка, метрики) и
возвращают вердикт. Требования к данным объявлены здесь, чтобы одинаковые
наборы не копировались по файлам.
"""

from __future__ import annotations

from ...interfaces import CapabilityKind, Requirement

#: Минимум, нужный любому фильтру по клеточной сетке.
REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
)

#: То же плюс требование нескольких состояний.
REQUIRES_MULTI = REQUIRES + (Requirement(CapabilityKind.MULTI_STATE),)

__all__ = ["REQUIRES", "REQUIRES_MULTI"]
