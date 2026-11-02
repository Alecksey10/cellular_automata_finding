"""Numba-ядра фильтров: связные компоненты (облака)."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def component_stats(live, wrap):
    """Число и размер крупнейшей 8-связной компоненты живых клеток."""

    height, width = live.shape
    labels = np.zeros((height, width), np.int32)
    stack = np.empty(height * width, np.int64)
    components = 0
    largest = 0

    for i in range(height):
        for j in range(width):
            if live[i, j] == 0 or labels[i, j] != 0:
                continue
            components += 1
            labels[i, j] = components
            top = 0
            stack[0] = i * width + j
            size = 0
            while top >= 0:
                index = stack[top]
                top -= 1
                cell_i = index // width
                cell_j = index % width
                size += 1
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni = cell_i + di
                        nj = cell_j + dj
                        if wrap:
                            ni = ni % height
                            nj = nj % width
                        else:
                            if ni < 0 or ni >= height or nj < 0 or nj >= width:
                                continue
                        if live[ni, nj] != 0 and labels[ni, nj] == 0:
                            labels[ni, nj] = components
                            top += 1
                            stack[top] = ni * width + nj
            if size > largest:
                largest = size

    return components, largest
