"""Связные компоненты бинарной маски."""

from __future__ import annotations

import numpy as np
from numba import njit

from .jit import JIT_KWARGS


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


__all__ = ["cluster_stats"]
