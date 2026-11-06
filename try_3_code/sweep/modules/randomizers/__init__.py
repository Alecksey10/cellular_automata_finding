"""Модули инициализации сетки.

Случайность берётся из ``numpy.random.Generator`` с seed из конфига, поэтому
один и тот же конфиг всегда даёт одно и то же начальное состояние.
"""

from __future__ import annotations

from .base import density_of
from .clustered import ClusteredFill
from .symmetric import SymmetricFill
from .uniform import UniformFill

__all__ = ["ClusteredFill", "SymmetricFill", "UniformFill", "density_of"]
