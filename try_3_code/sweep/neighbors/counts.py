"""Счётчики соседей: одни и те же циклы для всех правил.

Каждая функция — обычное numba-ядро без Python-объектов, поэтому правило может
вызвать её из своего ``@njit``-шага: numba инлайнит вызов, лишних накладных
расходов нет. Это заменяет копию цикла «пройти по смещениям с wrap/границами»,
которая раньше была в каждом эксперименте.

Функции с суффиксом ``3d`` работают с трёхосной сеткой, остальные — с
двухмерной. Сигнатуры совпадают, поэтому правило может выбирать счётчик по
``spec.dims``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ..kernels import JIT_KWARGS


@njit(**JIT_KWARGS)
def count_nonzero(src, dy, dx, wrap, zero):
    """Число соседей, отличных от ``zero``, для каждой клетки."""

    height, width = src.shape
    total = dy.size
    out = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            count = 0
            for k in range(total):
                yy = y + dy[k]
                xx = x + dx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if src[yy, xx] != zero:
                    count += 1
            out[y, x] = count
    return out


@njit(**JIT_KWARGS)
def count_state(src, dy, dx, wrap, state):
    """Число соседей ровно в состоянии ``state``."""

    height, width = src.shape
    total = dy.size
    out = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            count = 0
            for k in range(total):
                yy = y + dy[k]
                xx = x + dx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if src[yy, xx] == state:
                    count += 1
            out[y, x] = count
    return out


@njit(**JIT_KWARGS)
def count_matching(src, dy, dx, wrap, target):
    """Число соседей, равных ``target[y, x]``.

    Нужно правилам, где сравниваемое состояние зависит от клетки: например,
    циклические фазы считают соседей *следующей* фазы.
    """

    height, width = src.shape
    total = dy.size
    out = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            want = target[y, x]
            count = 0
            for k in range(total):
                yy = y + dy[k]
                xx = x + dx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if src[yy, xx] == want:
                    count += 1
            out[y, x] = count
    return out


@njit(**JIT_KWARGS)
def count_nonzero3d(src, dy, dx, dz, wrap, zero):
    """То же, что :func:`count_nonzero`, но для сетки ``(depth, height, width)``."""

    depth, height, width = src.shape
    total = dy.size
    out = np.zeros((depth, height, width), dtype=np.int32)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                count = 0
                for k in range(total):
                    zz = z + dz[k]
                    yy = y + dy[k]
                    xx = x + dx[k]
                    if wrap:
                        zz = zz % depth
                        yy = yy % height
                        xx = xx % width
                    elif zz < 0 or zz >= depth or yy < 0 or yy >= height or xx < 0 or xx >= width:
                        continue
                    if src[zz, yy, xx] != zero:
                        count += 1
                out[z, y, x] = count
    return out


__all__ = [
    "count_matching",
    "count_nonzero",
    "count_nonzero3d",
    "count_state",
]
