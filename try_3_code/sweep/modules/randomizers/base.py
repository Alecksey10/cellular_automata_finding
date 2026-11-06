"""Общая часть инициализации сетки."""

from __future__ import annotations

from typing import Any, Mapping

from ...module import RandomizerModule


def density_of(module: RandomizerModule, fixed: Mapping[str, Any]) -> float:
    """Плотность из осей модуля (``<id>.density``)."""

    return float(module.get(fixed, "density"))


__all__ = ["density_of"]
