"""Ленивые ссылки на значения из пространства параметров.

В плане (planning-фазе) программа не исполняется: она объявляет оси перебора и
получает ``Handle``-ы вместо конкретных значений. Во время прогона теста все
значения фиксированы и берутся из :class:`casweep.fixed.Fixed`.
"""

from __future__ import annotations

from typing import Any, Mapping


class Handle:
    """Ссылка на значение по ключу пространства параметров."""

    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        if not key:
            raise ValueError("У ключа параметра не может быть пустого имени")
        self.key = key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Handle) and other.key == self.key

    def __hash__(self) -> int:
        return hash(("casweep.Handle", self.key))

    def __repr__(self) -> str:
        return f"Handle({self.key!r})"

    # Арифметика нужна для зависимых границ: range(born + 1, max_count + 1).
    def __add__(self, delta: int) -> "Shift":
        return Shift(self, delta)

    def __sub__(self, delta: int) -> "Shift":
        return Shift(self, -delta)


class Shift:
    """Значение параметра, сдвинутое на константу."""

    __slots__ = ("handle", "delta")

    def __init__(self, handle: Handle, delta: int) -> None:
        self.handle = handle
        self.delta = delta

    def __repr__(self) -> str:
        sign = "+" if self.delta >= 0 else "-"
        return f"{self.handle!r} {sign} {abs(self.delta)}"


def resolve(value: Any, binding: Mapping[str, Any]) -> Any:
    """Развернуть значение оси: число, ``Handle`` или ``Shift``."""

    if isinstance(value, Handle):
        return binding[value.key]
    if isinstance(value, Shift):
        return resolve(value.handle, binding) + value.delta
    return value


def referenced_keys(value: Any) -> tuple[str, ...]:
    """Ключи, от которых зависит выражение (для отладки/описаний)."""

    if isinstance(value, Handle):
        return (value.key,)
    if isinstance(value, Shift):
        return (value.handle.key,)
    return ()
