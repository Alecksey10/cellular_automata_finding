"""numba-ядра общего назначения: гистограммы, маски, кластеры, IoU."""

from __future__ import annotations

import numpy as np
from numba import njit

# cache=True нельзя: ядра принадлежат динамически импортируемым экспериментам.
JIT_KWARGS = {
    "cache": False,
    "nogil": True,
    "fastmath": True,
}


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


@njit(**JIT_KWARGS)
def cluster_stats(mask):
    """Связные компоненты (4-соседство): число, средний размер, доля максимума."""

    height, width = mask.shape
    seen = np.zeros((height, width), dtype=np.uint8)
    stack = np.empty(height * width, dtype=np.int32)
    sizes = np.empty(height * width, dtype=np.int32)
    components = 0
    total = 0
    largest = 0
    for sy in range(height):
        for sx in range(width):
            if mask[sy, sx] == 0 or seen[sy, sx]:
                continue
            top = 0
            stack[0] = sy * width + sx
            seen[sy, sx] = 1
            size = 0
            while top >= 0:
                index = stack[top]
                top -= 1
                y = index // width
                x = index - y * width
                size += 1
                if y > 0 and mask[y - 1, x] and not seen[y - 1, x]:
                    top += 1
                    stack[top] = (y - 1) * width + x
                    seen[y - 1, x] = 1
                if y + 1 < height and mask[y + 1, x] and not seen[y + 1, x]:
                    top += 1
                    stack[top] = (y + 1) * width + x
                    seen[y + 1, x] = 1
                if x > 0 and mask[y, x - 1] and not seen[y, x - 1]:
                    top += 1
                    stack[top] = y * width + (x - 1)
                    seen[y, x - 1] = 1
                if x + 1 < width and mask[y, x + 1] and not seen[y, x + 1]:
                    top += 1
                    stack[top] = y * width + (x + 1)
                    seen[y, x + 1] = 1
            sizes[components] = size
            components += 1
            total += size
            if size > largest:
                largest = size
    if components == 0:
        return 0, 0.0, 0.0
    mean = total / components
    share = largest / total if total else 0.0
    return components, mean, share


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


__all__ = [
    "JIT_KWARGS",
    "alive_count",
    "as_mask",
    "cluster_stats",
    "mask_difference",
    "mask_iou",
    "state_histogram",
]
