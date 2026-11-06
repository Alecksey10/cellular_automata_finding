"""База модулей: роли, требования, объявление осей, сборка.

Модуль — самостоятельный плагин. Он объявляет свои оси в
:class:`~sweep.space.Namespace` (только описание, без вычислений) и умеет
построить рабочий объект по зафиксированному конфигу. Ядро не знает, что
такое клетка, сосед или возраст: обо всём этом знают модули.

Пакет разложен по файлам: ``roles``, ``spec``, ``contracts``, ``base``,
``views``. Импортировать по-прежнему можно из ``sweep.module`` — здесь
только реэкспорт.
"""

from __future__ import annotations

from .base import (
    DataModule,
    FilterModule,
    FilterRuntime,
    Module,
    NeighborhoodModule,
    RandomizerModule,
    RuleModule,
    RuleRuntime,
)
from .contracts import Observation, PASS, Verdict
from .roles import ROLE_ORDER, Role
from .spec import NeighborhoodSpec
from .views import BuildView, DeclareView, PeerProxy

__all__ = [
    "BuildView",
    "DataModule",
    "DeclareView",
    "FilterModule",
    "FilterRuntime",
    "Module",
    "NeighborhoodModule",
    "NeighborhoodSpec",
    "Observation",
    "PASS",
    "PeerProxy",
    "ROLE_ORDER",
    "RandomizerModule",
    "Role",
    "RuleModule",
    "RuleRuntime",
    "Verdict",
]
