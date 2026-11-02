"""Базовый протокол модулей: роли, требования, объявление параметров, сборка."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping

from .fixed import Fixed
from .interfaces import Capability, Requirement
from .space import DeclareView, Namespace, Selection


class Role(Enum):
    """Роль модуля в сборке автомата."""

    DATA = "data"
    RANDOMIZER = "randomizer"
    NEIGHBORHOOD = "neighborhood"
    RULE = "rule"
    FILTER = "filter"


#: Порядок сборки: данные -> инициализация -> окрестность -> правило.
ROLE_ORDER: tuple[Role, ...] = (Role.DATA, Role.RANDOMIZER, Role.NEIGHBORHOOD, Role.RULE)


class Module:
    """Модуль эксперимента.

    Модуль не хранит состояние теста: свои оси он объявляет в :meth:`declare`,
    а рабочий объект собирает в :meth:`build` уже по зафиксированному конфигу.
    Экземпляр может нести неизменяемые настройки (например, критерии фильтра).
    """

    id: ClassVar[str] = ""
    role: ClassVar[Role]
    #: Возможности, которые модуль добавляет сборке (не зависят от конфига).
    provides: ClassVar[tuple[Capability, ...]] = ()
    #: Требования к уже собранному окружению.
    requires: ClassVar[tuple[Requirement, ...]] = ()
    #: Имена вычисляемых значений, доступных другим модулям через Selection.derived.
    derived: ClassVar[tuple[str, ...]] = ()
    #: Необязательный визуализатор (см. ``visualize.protocol.Visualizer``).
    visualizer: ClassVar[Any] = None

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        """Объявить оси параметров модуля (может быть пусто)."""

    def derive(self, fixed: Fixed) -> Mapping[str, Any]:
        """Вычислить значения, объявленные в :attr:`derived`."""

        return {}

    def capabilities(self, fixed: Fixed) -> tuple[Capability, ...]:
        """Возможности при конкретном конфиге (динамические, например число состояний)."""

        return self.provides

    def build(self, fixed: Fixed, view: "BuildView") -> Any:
        """Собрать рабочий объект по зафиксированному конфигу."""

        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.id} ({self.role.value})"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id}>"


@dataclass(frozen=True)
class FilterBinding:
    """Подключённый фильтр: модуль + префикс его параметров."""

    module: Module
    name: str

    @property
    def prefix(self) -> str:
        return f"filter.{self.name}"

    def key(self, param: str) -> str:
        return f"{self.prefix}.{param}"

    def __repr__(self) -> str:
        return f"FilterBinding({self.name!r}, {self.module.id!r})"


class BuildView:
    """Доступ модуля к зафиксированным значениям и уже собранным объектам."""

    def __init__(
        self,
        fixed: Fixed,
        prefix: str,
        built: dict[Role, Any],
        modules: Mapping[Role, Module],
    ) -> None:
        self.fixed = fixed
        self.prefix = prefix
        self._built = built
        self._modules = modules

    # свои параметры
    def key(self, name: str) -> str:
        return f"{self.prefix}.{name}"

    def value(self, name: str) -> Any:
        return self.fixed[self.key(name)]

    # окружение
    def object(self, role: Role) -> Any:
        """Рабочий объект уже собранного модуля роли."""

        return self._built[role]

    def module(self, role: Role) -> Module:
        return self._modules[role]

    def peer_value(self, role: Role, name: str) -> Any:
        return self.fixed[f"{role.value}.{name}"]

    def peer_derived(self, role: Role, name: str) -> Any:
        return self.fixed[f"{role.value}#{name}"]
