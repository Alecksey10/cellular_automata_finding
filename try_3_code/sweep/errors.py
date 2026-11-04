"""Ошибки движка перебора.

Все исключения уровня движка наследуются от :class:`SweepError`, чтобы CLI и
раннер могли отличать ошибку описания эксперимента от ошибки платформы.
"""

from __future__ import annotations


class SweepError(Exception):
    """Базовая ошибка движка."""


class UnboundParameter(SweepError, KeyError):
    """Ось ещё не получила значения при разрешении зависимости."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key

    def __str__(self) -> str:  # KeyError добавляет кавычки, они здесь лишние
        return f"параметр {self.key!r} не привязан"


class Incompatible(SweepError):
    """Конфигурация несовместима с интерфейсами выбранных модулей."""


class Cancelled(SweepError):
    """Прогон прерван пользователем."""
