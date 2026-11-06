"""Параметры компиляции общих numba-ядер."""

from __future__ import annotations

# cache=True нельзя: ядра принадлежат динамически импортируемым экспериментам.
JIT_KWARGS = {
    "cache": False,
    "nogil": True,
    "fastmath": True,
}

__all__ = ["JIT_KWARGS"]
