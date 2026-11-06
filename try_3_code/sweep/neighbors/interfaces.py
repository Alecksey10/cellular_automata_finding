"""Интерфейсы вычисления соседства.

Слой ничего не знает про numba и про конкретные правила. Он умеет две вещи:

* перечислить соседей точки в целочисленной решётке (2D или 3D) —
  :class:`Lattice`;
* отдать готовое описание окрестности плагину — :class:`Neighborhood`.

Смещения — кортежи длины ``dims``: ``(-1, 0)`` для 2D и ``(-1, 0, 1)`` для 3D.
Такой формат не зависит от размерности, поэтому геометрия 3D добавляется
новым классом ``Lattice``, а не правкой каждого правила.

Плагин-обёртка живёт в :mod:`sweep.module` (``NeighborhoodModule``): здесь
только протоколы, чтобы этот пакет можно было использовать отдельно от
экспериментов.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from ..module import NeighborhoodSpec

#: Набор смещений: ``dims`` целых чисел на соседа.
Offsets = tuple[tuple[int, ...], ...]


@runtime_checkable
class Lattice(Protocol):
    """Целочисленная решётка фиксированной размерности."""

    dimensions: int

    def offsets(self, radius: int, metric: str = "chebyshev") -> Offsets:
        """Все точки в радиусе ``radius`` без центра."""

    def ring(self, inner: int, outer: int, metric: str = "chebyshev") -> Offsets:
        """Точки между радиусами ``inner`` и ``outer``."""


@runtime_checkable
class Neighborhood(Protocol):
    """Окрестность, которую видит эксперимент.

    Совпадает с ``NeighborhoodModule`` по атрибутам, но не требует наследования
    от ``Module``: тесты и внешний код могут подставить свою реализацию.
    """

    id: str
    label: str
    dimensions: int

    def declare(self, ns: Any, view: Any) -> None:
        """Объявить оси окрестности."""

    def spec(self, fixed: Mapping[str, Any]) -> "NeighborhoodSpec":
        """Готовое описание смещений для зафиксированного конфига."""


__all__ = ["Lattice", "Neighborhood", "Offsets"]
