"""Предметные модули: данные, инициализация, фильтры.

Логики итерации здесь нет: каждое правило живёт в своём эксперименте
(``sweep/experiments/<name>/``), потому что это часть конкретной модели.
Соседство вынесено в отдельный пакет :mod:`sweep.neighbors`.

Каждая категория — пакет с файлом на модуль: ``data/``, ``randomizers/``,
``filters/``. Новая сетка, заливка или фильтр добавляется файлом рядом и
реэкспортом из ``__init__``; движок и эксперименты менять не нужно.
"""

from __future__ import annotations

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
from .randomizers import ClusteredFill, SymmetricFill, UniformFill

__all__ = [
    "ALL_FILTERS",
    "ClusteredFill",
    "ClustersFilter",
    "DensityFilter",
    "DynamicsFilter",
    "Grid2D",
    "GroupPresenceFilter",
    "PhasesFilter",
    "PresenceFilter",
    "StabilityFilter",
    "StateSharesFilter",
    "SymmetricFill",
    "SymmetricGrid2D",
    "SymmetryFilter",
    "UniformFill",
]
