"""Сравнение бинарных масок."""

from __future__ import annotations

import numpy as np
from numba import njit

from .jit import JIT_KWARGS


@njit(**JIT_KWARGS)
def mask_iou(left, right):
    """Intersection-over-union двух булевых масок."""

    intersection = 0
    union = 0
    height, width = left.shape
    for y in range(height):
        for x in range(width):
            a = left[y, x] != 0
            b = right[y, x] != 0
            if a and b:
                intersection += 1
            if a or b:
                union += 1
    if union == 0:
        return 1.0
    return intersection / union


@njit(**JIT_KWARGS)
def mask_difference(left, right):
    """Доля различающихся клеток."""

    height, width = left.shape
    total = height * width
    if total == 0:
        return 0.0
    changed = 0
    for y in range(height):
        for x in range(width):
            if left[y, x] != right[y, x]:
                changed += 1
    return changed / total


def as_mask(data: np.ndarray, zero: int) -> np.ndarray:
    return np.ascontiguousarray(data != zero, dtype=np.uint8)


__all__ = ["as_mask", "mask_difference", "mask_iou"]
