"""Предметные модули: данные, инициализация, соседство, фильтры.

Логики итерации здесь нет: каждое правило живёт в своём эксперименте
(`sweep/experiments/<name>/`), потому что это часть конкретной модели.
"""

from .data import Grid2D, SymmetricGrid2D
from .filters import (
    ALL_FILTERS,
    ClustersFilter,
    DensityFilter,
    DynamicsFilter,
    GroupPresenceFilter,
    PhasesFilter,
    PresenceFilter,
    StabilityFilter,
    StateSharesFilter,
    SymmetryFilter,
)
from .neighborhoods import ALL_NEIGHBORHOODS, DuelRings, Moore, RadiusDisc, Square5x5, TwoRings
from .randomizers import ClusteredFill, SymmetricFill, UniformFill

__all__ = [
    "ALL_FILTERS",
    "ALL_NEIGHBORHOODS",
    "ClusteredFill",
    "ClustersFilter",
    "DensityFilter",
    "DynamicsFilter",
    "DuelRings",
    "GroupPresenceFilter",
    "Grid2D",
    "Moore",
    "PhasesFilter",
    "PresenceFilter",
    "RadiusDisc",
    "StabilityFilter",
    "SymmetricFill",
    "SymmetricGrid2D",
    "SymmetryFilter",
    "Square5x5",
    "StateSharesFilter",
    "TwoRings",
    "UniformFill",
]
