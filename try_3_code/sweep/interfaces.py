"""Возможности автомата и требования модулей.

Интерфейсы описаны явно: модуль данных объявляет набор ``Capability``,
модуль логики/фильтра — набор ``Requirement``. Проверка совместимости идёт до
исполнения теста, поэтому несовместимые пары «данные × правило» не тратят
процессор и не содержат строковых сравнений в горячем коде.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


class CapabilityKind(enum.Enum):
    GRID_2D = "grid_2d"
    DIMENSION = "dimension"
    MULTI_STATE = "multi_state"
    ZERO_STATE = "zero_state"
    NEIGHBORHOOD = "neighborhood"
    RESOLUTION = "resolution"


@dataclass(frozen=True)
class Capability:
    kind: CapabilityKind
    params: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""

    def get(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    def describe(self) -> str:
        body = ", ".join(f"{key}={value!r}" for key, value in self.params.items())
        prefix = f"{self.kind.value}"
        if body:
            prefix += f"({body})"
        return f"{prefix} <- {self.source}" if self.source else prefix


# Предикаты требований. Функции получают Capability и возвращают bool, а
# ``text`` нужен только для понятного отчёта о несовместимости.
Predicate = Callable[[Capability], bool]


def _predicate(text: str, fn: Predicate) -> Predicate:
    fn.text = text  # type: ignore[attr-defined]
    return fn


def ge(field: str, value: int) -> Predicate:
    return _predicate(f"states >= {value}" if field == "states" else f"{field} >= {value}",
                      lambda cap: cap.get(field) is not None and cap.get(field) >= value)


def eq(field: str, value: Any) -> Predicate:
    return _predicate(f"{field} == {value!r}", lambda cap: cap.get(field) == value)


def le(field: str, value: Any) -> Predicate:
    return _predicate(
        f"{field} <= {value}",
        lambda cap: cap.get(field) is not None and cap.get(field) <= value,
    )


def between(field: str, low: int, high: int) -> Predicate:
    return _predicate(
        f"{low} <= {field} <= {high}",
        lambda cap: cap.get(field) is not None and low <= cap.get(field) <= high,
    )


def always(_: Capability) -> bool:
    return True


always.text = "всегда"  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Requirement:
    kind: CapabilityKind
    checks: tuple[Predicate, ...] = ()
    text: str = ""

    def matches(self, capabilities: Iterable[Capability]) -> Capability | None:
        for cap in capabilities:
            if cap.kind is not self.kind:
                continue
            if all(check(cap) for check in self.checks):
                return cap
        return None

    def describe(self) -> str:
        if self.text:
            return self.text
        parts = [getattr(check, "text", "?") for check in self.checks]
        return f"{self.kind.value}[{', '.join(parts)}]" if parts else self.kind.value


def check_requirements(
    capabilities: Iterable[Capability],
    requirements: Iterable[Requirement],
) -> list[str]:
    """Вернуть список невыполненных требований (пусто — совместимо)."""
    caps = tuple(capabilities)
    missing: list[str] = []
    for requirement in requirements:
        if requirement.matches(caps) is None:
            missing.append(requirement.describe())
    return missing


def describe_capabilities(capabilities: Iterable[Capability]) -> str:
    return ", ".join(cap.describe() for cap in capabilities)
