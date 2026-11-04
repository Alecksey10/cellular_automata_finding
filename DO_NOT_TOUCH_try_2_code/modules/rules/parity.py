"""Чётность: живая клетка, если живых соседей нечётное число.

Предельно симметричное (зависит только от суммы) и почти всегда нестабильное
правило. Нужно как контрольная точка: если фильтр «не мелькает» пропускает
чётность, значит он слишком мягкий.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least, eq
from casweep.module import BuildView, Module, Role
from casweep.trial import StepOutcome

from .kernels import parity_step


class ParityRule(Module):
    id = "rule.parity"
    role = Role.RULE
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.DISCRETE),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2), eq("states", 2))),
        Requirement(CapabilityKind.ZERO_STATE, (eq("state", 0),)),
        Requirement(CapabilityKind.NEIGHBORHOOD_COUNT, (at_least("max_count", 1),)),
    )

    def build(self, fixed: Fixed, view: BuildView) -> "ParityRuntime":
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        return ParityRuntime(
            di=neighborhood.di,
            dj=neighborhood.dj,
            wrap=neighborhood.wrap,
            zero_state=int(data.zero_state),
            birth_state=1,
        )


class ParityRuntime:
    __slots__ = ("di", "dj", "wrap", "zero_state", "birth_state")

    def __init__(
        self,
        di: np.ndarray,
        dj: np.ndarray,
        wrap: bool,
        zero_state: int,
        birth_state: int,
    ) -> None:
        self.di = di
        self.dj = dj
        self.wrap = wrap
        self.zero_state = zero_state
        self.birth_state = birth_state

    def step(self, src, dst) -> StepOutcome | None:
        if not np.any(src.array != self.zero_state):
            return StepOutcome(stop="вырождение: не осталось живых клеток")
        parity_step(
            src.array, dst.array, self.di, self.dj, self.zero_state, self.birth_state, self.wrap
        )
        return None

    def precompile(self) -> None:
        src = np.zeros((4, 4), dtype=np.int32)
        dst = np.empty((4, 4), dtype=np.int32)
        parity_step(src, dst, self.di, self.dj, self.zero_state, self.birth_state, self.wrap)
