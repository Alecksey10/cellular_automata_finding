"""Модули фильтрации: решают, продолжать ли выполнение конфига.

Фильтр не знает о стадиях и итерациях: он получает
:class:`~sweep.module.Observation` и возвращает :class:`~sweep.module.Verdict`.
Когда именно вызывать ``check`` — решает движок по ``Phase.frequency``.

Один фильтр — один файл; общие требования к данным лежат в ``base.py``.
"""

from __future__ import annotations

from .base import REQUIRES, REQUIRES_MULTI
from .clusters import ClustersFilter, ClustersRuntime
from .density import DensityFilter, DensityRuntime
from .dynamics import DynamicsFilter, DynamicsRuntime
from .group_presence import GroupPresenceFilter, GroupPresenceRuntime
from .phases import PhasesFilter, PhasesRuntime
from .presence import PresenceFilter, PresenceRuntime
from .stability import StabilityFilter, StabilityRuntime
from .state_shares import StateSharesFilter, StateSharesRuntime
from .symmetry import SymmetryFilter, SymmetryRuntime

ALL_FILTERS = [
    DensityFilter,
    StabilityFilter,
    ClustersFilter,
    PhasesFilter,
    SymmetryFilter,
    StateSharesFilter,
    DynamicsFilter,
    PresenceFilter,
]

__all__ = [
    "ALL_FILTERS",
    "ClustersFilter",
    "ClustersRuntime",
    "DensityFilter",
    "DensityRuntime",
    "DynamicsFilter",
    "DynamicsRuntime",
    "GroupPresenceFilter",
    "GroupPresenceRuntime",
    "PhasesFilter",
    "PhasesRuntime",
    "PresenceFilter",
    "PresenceRuntime",
    "REQUIRES",
    "REQUIRES_MULTI",
    "StabilityFilter",
    "StabilityRuntime",
    "StateSharesFilter",
    "StateSharesRuntime",
    "SymmetryFilter",
    "SymmetryRuntime",
]
