"""Геометрия соседства: смещения в решётке произвольной размерности.

Все функции возвращают кортежи смещений, отсортированные так же, как их ждёт
numba-ядро: первая ось меняется медленнее всех, последняя — быстрее всех.
Для 2D это привычный порядок ``dy`` снаружи, ``dx`` внутри, поэтому браузерный
прогон и Python-прогон обходят соседей в одном порядке.
"""

from __future__ import annotations

import itertools

import numpy as np

from .interfaces import Offsets

METRICS = ("euclidean", "manhattan", "chebyshev")


def _inside(point: tuple[int, ...], radius: int, metric: str) -> bool:
    if metric == "chebyshev":
        return max(abs(value) for value in point) <= radius
    if metric == "manhattan":
        return sum(abs(value) for value in point) <= radius
    return sum(value * value for value in point) <= radius * radius


def disc_offsets(radius: int, metric: str = "chebyshev", dims: int = 2) -> Offsets:
    """Все смещения в радиусе ``radius`` без центральной точки.

    ``dims=3`` даёт куб/шар/октаэдр вокруг клетки — та же функция, без
    отдельной реализации на размерность.
    """

    if dims < 1:
        raise ValueError("размерность должна быть положительной")
    if radius < 1:
        return ()
    spans = [range(-radius, radius + 1)] * dims
    return tuple(
        point
        for point in itertools.product(*spans)
        if any(point) and _inside(point, radius, metric)
    )


def ring_offsets(
    inner: int,
    outer: int,
    metric: str = "chebyshev",
    dims: int = 2,
) -> Offsets:
    """Смещения между двумя радиусами: ``disc(outer)`` минус ``disc(inner)``."""

    if outer < inner:
        raise ValueError("внешний радиус должен быть не меньше внутреннего")
    inner_set = set(disc_offsets(inner, metric, dims))
    return tuple(
        offset for offset in disc_offsets(outer, metric, dims) if offset not in inner_set
    )


def offset_arrays(offsets: Offsets, dims: int = 2) -> tuple[np.ndarray, ...]:
    """Кортежи смещений -> массивы ``int32`` по осям (для numba)."""

    if not offsets:
        return tuple(np.empty(0, dtype=np.int32) for _ in range(dims))
    array = np.asarray(offsets, dtype=np.int32)
    if array.ndim == 1:  # одна координата на смещение
        array = array.reshape(-1, 1)
    if array.shape[1] < dims:
        padding = np.zeros((array.shape[0], dims - array.shape[1]), dtype=np.int32)
        array = np.concatenate([array, padding], axis=1)
    return tuple(np.ascontiguousarray(array[:, axis]) for axis in range(dims))


__all__ = [
    "METRICS",
    "disc_offsets",
    "offset_arrays",
    "ring_offsets",
]
