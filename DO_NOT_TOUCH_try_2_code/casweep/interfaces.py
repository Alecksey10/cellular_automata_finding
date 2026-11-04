"""Явные интерфейсы (возможности) автомата и требования модулей.

Вместо строковых проверок вида ``if "2d" in features`` модуль объявляет
типизированное :class:`Requirement`, а данные/другие модули — типизированную
:class:`Capability`. Проверка совместимости выполняется до прогона и попадает в
отчёт.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CapabilityKind(Enum):
    """Виды возможностей, которые предоставляют данные и модули."""

    TWO_D = "topology.2d"
    ONE_D = "topology.1d"
    THREE_D = "topology.3d"
    DISCRETE = "states.discrete"
    MULTI_STATE = "states.multi"
    ZERO_STATE = "states.zero"
    NEIGHBORHOOD_COUNT = "neighborhood.count"
    NEIGHBORHOOD_RINGS = "neighborhood.rings"
    CHRONOLOGICAL = "logic.chronological"


@dataclass(frozen=True)
class Capability:
    """Возможность с параметрами, например ``2d(width=64, height=64)``."""

    kind: CapabilityKind
    attrs: tuple[tuple[str, Any], ...] = ()

    def get(self, name: str, default: Any = None) -> Any:
        for key, value in self.attrs:
            if key == name:
                return value
        return default

    def __str__(self) -> str:
        if not self.attrs:
            return self.kind.value
        body = ", ".join(f"{key}={value!r}" for key, value in self.attrs)
        return f"{self.kind.value}({body})"


def capability(kind: CapabilityKind, **attrs: Any) -> Capability:
    """Собрать возможность; порядок атрибутов нормализуется."""

    return Capability(kind, tuple(sorted(attrs.items())))


class Comparison(Enum):
    EQ = "=="
    GE = ">="
    LE = "<="


@dataclass(frozen=True)
class Check:
    attr: str
    op: Comparison
    value: Any

    def test(self, candidate: Any) -> bool:
        if self.op is Comparison.EQ:
            return candidate == self.value
        if self.op is Comparison.GE:
            return candidate >= self.value
        if self.op is Comparison.LE:
            return candidate <= self.value
        raise AssertionError(f"Неизвестная операция сравнения: {self.op!r}")

    def describe(self) -> str:
        return f"{self.attr} {self.op.value} {self.value!r}"


def eq(attr: str, value: Any) -> Check:
    return Check(attr, Comparison.EQ, value)


def at_least(attr: str, value: Any) -> Check:
    return Check(attr, Comparison.GE, value)


def at_most(attr: str, value: Any) -> Check:
    return Check(attr, Comparison.LE, value)


_MISSING = object()


@dataclass(frozen=True)
class Requirement:
    """Требование модуля к окружению."""

    kind: CapabilityKind
    checks: tuple[Check, ...] = ()
    note: str = ""

    def matches(self, candidate: Capability) -> bool:
        if candidate.kind is not self.kind:
            return False
        for check in self.checks:
            value = candidate.get(check.attr, _MISSING)
            if value is _MISSING:
                return False
            if not check.test(value):
                return False
        return True

    def describe(self) -> str:
        if not self.checks:
            text = self.kind.value
        else:
            text = self.kind.value + "[" + ", ".join(c.describe() for c in self.checks) + "]"
        return text + (f" ({self.note})" if self.note else "")


def first_match(requirement: Requirement, available: Iterable[Capability]) -> Capability | None:
    for candidate in available:
        if requirement.matches(candidate):
            return candidate
    return None
