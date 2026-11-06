"""Гистограммы и подсчёт живых клеток."""

from __future__ import annotations

import numpy as np
from numba import njit

from .jit import JIT_KWARGS


@njit(**JIT_KWARGS)
def state_histogram(data, states):
    """Гистограмма состояний ``[0, states)``."""

    histogram = np.zeros(states, dtype=np.int64)
    height, width = data.shape
    for y in range(height):
        for x in range(width):
            value = data[y, x]
            if value >= 0 and value < states:
                histogram[value] += 1
    return histogram


@njit(**JIT_KWARGS)
def alive_count(data, zero):
    """Число клеток, отличных от ``zero``."""

    height, width = data.shape
    total = 0
    for y in range(height):
        for x in range(width):
            if data[y, x] != zero:
                total += 1
    return total


__all__ = ["alive_count", "state_histogram"]
