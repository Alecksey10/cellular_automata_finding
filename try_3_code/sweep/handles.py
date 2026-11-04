"""Ленивые ссылки на параметры.

Описание эксперимента не исполняет код и не вычисляет значения: оси и границы
диапазонов ссылаются на параметры через :class:`Handle`. Значение появляется
только когда движок привязывает ось (:meth:`Handle.resolve`), поэтому один и
тот же объект описания переиспользуется для каждого конфига.

``Handle.get()`` возвращает сам handle: примеры из постановки пишут
``width.get()``, и такой код должен работать, оставаясь ленивым.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .errors import UnboundParameter

_BIN_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
    "min": min,
    "max": max,
}


class Handle:
    """Ссылка на значение параметра, известное только на этапе исполнения."""

    __slots__ = ()

    def resolve(self, values: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    # ``Generator.get()`` из постановки: остаёмся ленивыми.
    def get(self) -> "Handle":
        return self

    def keys(self) -> frozenset[str]:
        """Ключи, от которых зависит значение (для диагностики)."""
        return frozenset()

    def __add__(self, other: Any) -> "Handle":
        return Binary("+", self, as_handle(other))

    def __radd__(self, other: Any) -> "Handle":
        return Binary("+", as_handle(other), self)

    def __sub__(self, other: Any) -> "Handle":
        return Binary("-", self, as_handle(other))

    def __rsub__(self, other: Any) -> "Handle":
        return Binary("-", as_handle(other), self)

    def __mul__(self, other: Any) -> "Handle":
        return Binary("*", self, as_handle(other))

    def __rmul__(self, other: Any) -> "Handle":
        return Binary("*", as_handle(other), self)

    def __floordiv__(self, other: Any) -> "Handle":
        return Binary("//", self, as_handle(other))

    def __rfloordiv__(self, other: Any) -> "Handle":
        return Binary("//", as_handle(other), self)

    def __mod__(self, other: Any) -> "Handle":
        return Binary("%", self, as_handle(other))

    def __rmod__(self, other: Any) -> "Handle":
        return Binary("%", as_handle(other), self)

    def __repr__(self) -> str:  # pragma: no cover - отладочный вывод
        return f"<{type(self).__name__} {self}>"


@dataclass(frozen=True)
class Const(Handle):
    value: Any

    def resolve(self, values: Mapping[str, Any]) -> Any:
        return self.value

    def __str__(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class Ref(Handle):
    key: str

    def resolve(self, values: Mapping[str, Any]) -> Any:
        try:
            return values[self.key]
        except KeyError:
            raise UnboundParameter(self.key) from None

    def keys(self) -> frozenset[str]:
        return frozenset((self.key,))

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True)
class Derived(Handle):
    """Значение, вычисляемое из уже привязанных параметров.

    Например, ``max_count`` выбранной окрестности или ``states`` выбранного
    модуля данных. Функция обязана быть чистой: она вызывается многократно.
    """

    label: str
    fn: Callable[[Mapping[str, Any]], Any]

    def resolve(self, values: Mapping[str, Any]) -> Any:
        return self.fn(values)

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True)
class Binary(Handle):
    op: str
    left: Handle
    right: Handle

    def resolve(self, values: Mapping[str, Any]) -> Any:
        return _BIN_OPS[self.op](self.left.resolve(values), self.right.resolve(values))

    def keys(self) -> frozenset[str]:
        return self.left.keys() | self.right.keys()

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


def as_handle(value: Any) -> Handle:
    if isinstance(value, Handle):
        return value
    return Const(value)


def resolve(value: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(value, Handle):
        return value.resolve(values)
    return value


def resolve_int(value: Any, values: Mapping[str, Any], *, what: str = "значение") -> int:
    raw = resolve(value, values)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{what}: ожидалось целое, получено {raw!r}")
    if int(raw) != raw:
        raise TypeError(f"{what}: ожидалось целое, получено {raw!r}")
    return int(raw)


__all__ = [
    "Binary",
    "Const",
    "Derived",
    "Handle",
    "Ref",
    "as_handle",
    "resolve",
    "resolve_int",
]
