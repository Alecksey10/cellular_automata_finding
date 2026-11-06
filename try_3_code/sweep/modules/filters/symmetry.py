"""Фильтр симметрии: маска должна совпадать с отражениями и поворотами."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...kernels import as_mask, mask_iou
from ...module import BuildView, FilterModule, FilterRuntime, Observation, PASS, Verdict
from .base import REQUIRES


class SymmetryRuntime(FilterRuntime):
    def __init__(self, zero: int, min_iou: float) -> None:
        self.zero = int(zero)
        self.min_iou = float(min_iou)

    def check(self, observation: Observation) -> Verdict:
        mask = as_mask(observation.data, self.zero)
        # 0.0, а не 1.0: иначе max() всегда даёт >= 1 и фильтр не может отказать
        best = 0.0
        candidates = [mask[:, ::-1], mask[::-1, :], mask[::-1, ::-1]]
        for turns in (1, 2, 3):
            rotated = np.ascontiguousarray(np.rot90(mask, turns), dtype=np.uint8)
            if rotated.shape != mask.shape:
                continue  # неквадратную сетку можно сравнивать только с отражениями
            candidates.append(rotated)
            candidates.append(np.ascontiguousarray(np.fliplr(rotated), dtype=np.uint8))
        for candidate in candidates:
            best = max(best, float(mask_iou(mask, np.ascontiguousarray(candidate, dtype=np.uint8))))
        if best < self.min_iou:
            return Verdict(False, "asymmetric", {"iou": best})
        return PASS


class SymmetryFilter(FilterModule):
    id = "filter.symmetry"
    label = "симметрия"
    requires = REQUIRES

    def __init__(self, min_iou: float = 0.9) -> None:
        self.min_iou = float(min_iou)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return SymmetryRuntime(int(fixed["grid2d.zero"]), self.min_iou)


__all__ = ["SymmetryFilter", "SymmetryRuntime"]
