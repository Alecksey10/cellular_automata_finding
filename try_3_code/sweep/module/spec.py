"""Описание готовой окрестности для numba-ядра.

``dims`` говорит правилу, сколько осей обходить: для 2D это ``dy``/``dx``,
для 3D добавляются ``dz``/``inner_dz``. Для 2D новые поля остаются пустыми,
поэтому старые правила и их ``bind`` не меняются.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NeighborhoodSpec:
    """Готовое описание соседства: только данные, совместимые с numba."""

    name: str
    dy: np.ndarray
    dx: np.ndarray
    inner_dy: np.ndarray
    inner_dx: np.ndarray
    wrap: bool
    max_count: int
    inner_max_count: int = 0
    dims: int = 2
    dz: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    inner_dz: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))

    @property
    def wrap_flag(self) -> int:
        return 1 if self.wrap else 0

    @property
    def is_3d(self) -> bool:
        return self.dims >= 3


__all__ = ["NeighborhoodSpec"]
