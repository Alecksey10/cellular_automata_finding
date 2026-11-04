"""Предметные модули: данные, инициализация, соседство, фильтры.

Логики итерации здесь нет: каждое правило живёт в своём эксперименте
(`sweep/experiments/<name>/`), потому что это часть конкретной модели.
"""

from .data import Grid2D, SymmetricGrid2D
from .filters import (
    ALL_FILTERS,
    ClustersFilter,
    DensityFilter,
    PhasesFilter,
    StabilityFilter,
    SymmetryFilter,
)
from .neighborhoods import ALL_NEIGHBORHOODS, Moore, RadiusDisc, TwoRings
from .randomizers import ClusteredFill, SymmetricFill, UniformFill

__all__ = [
    "ALL_FILTERS",
    "ALL_NEIGHBORHOODS",
    "ClusteredFill",
    "ClustersFilter",
    "DensityFilter",
    "Grid2D",
    "Moore",
    "PhasesFilter",
    "RadiusDisc",
    "StabilityFilter",
    "SymmetricFill",
    "SymmetricGrid2D",
    "SymmetryFilter",
    "TwoRings",
    "UniformFill",
]
