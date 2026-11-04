"""Домены осей: какие значения ось может принимать.

Домен не хранит значения, а порождает их из уже привязанных параметров.
Границы и шаг ``RangeDomain`` — :class:`~sweep.handles.Handle`, поэтому
``height = grid.range("height", width, 128, 32)`` даёт для каждой ширины свой
набор высот.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .handles import Handle, as_handle, resolve


class Domain:
    """Интерфейс домена."""

    def values(self, bound: Mapping[str, Any]) -> list[Any]:
        raise NotImplementedError

    def size(self, bound: Mapping[str, Any]) -> int:
        return len(self.values(bound))

    def describe(self, bound: Mapping[str, Any] | None = None) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class RangeDomain(Domain):
    """Полуоткрытый диапазон, как :func:`range`.

    Целые границы дают целые значения, вещественные — значения с округлением
    до 10 знаков (плотности, вероятности).
    """

    lo: Handle
    hi: Handle
    step: Handle

    def values(self, bound: Mapping[str, Any]) -> list[Any]:
        lo = resolve(self.lo, bound)
        hi = resolve(self.hi, bound)
        step = resolve(self.step, bound)
        if step == 0:
            raise ValueError(f"шаг не может быть нулём (lo={lo!r}, hi={hi!r})")
        if isinstance(lo, float) or isinstance(hi, float) or isinstance(step, float):
            lo, hi, step = float(lo), float(hi), float(step)
            if step < 0:
                raise ValueError("шаг должен быть положительным")
            count = int(math.floor((hi - lo) / step + 1e-9))
            return [round(lo + i * step, 10) for i in range(max(0, count))]
        lo, hi, step = int(lo), int(hi), int(step)
        return list(range(lo, hi, step))

    def describe(self, bound: Mapping[str, Any] | None = None) -> str:
        if bound is None:
            return f"range({self.lo}, {self.hi}, {self.step})"
        values = self.values(bound)
        if not values:
            return "[]"
        return f"[{values[0]}..{values[-1]}] / {len(values)}"


@dataclass(frozen=True)
class ChoiceDomain(Domain):
    """Явный список вариантов (уже известных при описании)."""

    options: tuple[Any, ...]

    def values(self, bound: Mapping[str, Any]) -> list[Any]:
        return list(self.options)

    def describe(self, bound: Mapping[str, Any] | None = None) -> str:
        shown = ", ".join(_short(option) for option in self.options[:6])
        tail = ", ..." if len(self.options) > 6 else ""
        return f"{{{shown}{tail}}} / {len(self.options)}"


def _short(value: Any) -> str:
    text = getattr(value, "id", None) or repr(value)
    return str(text)


@dataclass(frozen=True)
class OptionalDomain(Domain):
    """Домен, который может не иметь значений при данной привязке.

    Используется зависимыми осями: пока не известен выбор, домен пуст, и
    дерево просто не даёт листьев.
    """

    inner: Domain
    condition: Handle

    def values(self, bound: Mapping[str, Any]) -> list[Any]:
        if not resolve(self.condition, bound):
            return []
        return self.inner.values(bound)

    def describe(self, bound: Mapping[str, Any] | None = None) -> str:
        return f"if {self.condition}: {self.inner.describe(bound)}"


def range_domain(lo: Any, hi: Any, step: Any = 1) -> RangeDomain:
    return RangeDomain(as_handle(lo), as_handle(hi), as_handle(step))


def choice_domain(options: Iterable[Any]) -> ChoiceDomain:
    if isinstance(options, (str, bytes)):
        raise TypeError("строка — не список вариантов")
    return ChoiceDomain(tuple(options))


def values_of(domain: Domain, bound: Mapping[str, Any]) -> Sequence[Any]:
    return domain.values(bound)


__all__ = [
    "ChoiceDomain",
    "Domain",
    "OptionalDomain",
    "RangeDomain",
    "choice_domain",
    "range_domain",
    "values_of",
]
