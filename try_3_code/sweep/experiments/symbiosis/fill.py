"""Заливка «колоний»: опора и её зависимые появляются вместе.

Случайная смесь всех состояний не даёт симбиозу шанса: опора рождается только
рядом с зависимыми, а зависимые — только рядом с опорой. Поэтому стартовая
сетка собирается из нескольких колоний: диск опоры, внутри — зависимые, по
краю — споры. Дальше правило само решает, выживет колония или распадётся.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...module import RandomizerModule
from ...space import Namespace
from .rule import ANCHOR, GRAZER, MATRIX, SPORE, WEAVER


class SymbiosisFill(RandomizerModule):
    """Несколько колоний: опора, зависимые и споры в одном пятне."""

    id = "randomizer.symbiosis"
    label = "колонии опоры и зависимых"

    def declare(self, ns: Namespace, view) -> None:
        ns.range("seeds", 2, 9, 2)
        ns.range("radius", 8, 15, 2)
        ns.range("spacing", 0.8, 0.96, 0.04)
        ns.range("mix", 0.2, 0.8, 0.2)

    def fill(
        self,
        data: np.ndarray,
        fixed: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        zero_state: int,
        states: int,
    ) -> None:
        seeds = int(self.get(fixed, "seeds"))
        radius = int(self.get(fixed, "radius"))
        spacing = float(self.get(fixed, "spacing"))
        mix = float(self.get(fixed, "mix"))
        height, width = data.shape
        data[:] = zero_state
        for seed in range(seeds):
            cy = int(rng.integers(0, height))
            cx = int(rng.integers(0, width))
            support = ANCHOR if seed % 2 == 0 else MATRIX
            dependent = GRAZER if support == ANCHOR else WEAVER
            y0 = max(0, cy - radius)
            y1 = min(height, cy + radius + 1)
            x0 = max(0, cx - radius)
            x1 = min(width, cx + radius + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            distance = np.hypot(yy - cy, xx - cx)
            patch = data[y0:y1, x0:x1]
            inside = distance <= radius
            # Разреженная опора: не сплошная плита, иначе правило переполнения
            # убивает её на первом же шаге.
            patch[inside] = zero_state
            keep = inside & (rng.random(inside.shape) >= spacing)
            patch[keep] = support
            dependents = inside & (rng.random(inside.shape) < mix)
            patch[dependents] = dependent
            rim = (distance > radius) & (distance <= radius + 2.0)
            spores = rim & (rng.random(rim.shape) < 0.15)
            patch[spores] = SPORE


__all__ = ["SymbiosisFill"]
