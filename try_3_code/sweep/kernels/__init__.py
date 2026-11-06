"""numba-ядра общего назначения: гистограммы, маски, кластеры, IoU.

Раньше это был один файл; теперь каждая тема в своём: ``jit`` (параметры
компиляции), ``histogram``, ``clusters``, ``masks``. Импорт не изменился —
``from sweep.kernels import JIT_KWARGS`` и т.д. работают как раньше.
"""

from __future__ import annotations

from .clusters import cluster_stats
from .histogram import alive_count, state_histogram
from .jit import JIT_KWARGS
from .masks import as_mask, mask_difference, mask_iou

__all__ = [
    "JIT_KWARGS",
    "alive_count",
    "as_mask",
    "cluster_stats",
    "mask_difference",
    "mask_iou",
    "state_histogram",
]
