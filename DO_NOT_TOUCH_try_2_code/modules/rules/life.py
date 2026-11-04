"""Игра «Жизнь» как отдельный модуль правила (B3/S23)."""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least, eq
from casweep.module import BuildView, Module, Role
from casweep.trial import StepOutcome

from .kernels import life_step


class LifeRule(Module):
    """Требует ровно два состояния и нулевое состояние 0.

    Это хороший пример динамической совместимости: при переборе ``states=2..9``
    правило доступно только для ``states=2``, остальные конфиги отсекаются.
    """

    id = "rule.life"
    role = Role.RULE
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.MULTI_STATE, (eq("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE, (eq("state", 0),)),
        Requirement(CapabilityKind.NEIGHBORHOOD_COUNT, (at_least("max_count", 1),)),
    )

    def build(self, fixed: Fixed, view: BuildView) -> "LifeRuntime":
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        return LifeRuntime(
            di=neighborhood.di,
            dj=neighborhood.dj,
            wrap=neighborhood.wrap,
            zero_state=int(data.zero_state),
            birth_state=1,
        )


class LifeRuntime:
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
        life_step(
            src.array,
            dst.array,
            self.di,
            self.dj,
            self.zero_state,
            self.birth_state,
            self.wrap,
        )
        return None

    def precompile(self) -> None:
        src = np.zeros((4, 4), dtype=np.int32)
        dst = np.empty((4, 4), dtype=np.int32)
        life_step(src, dst, self.di, self.dj, self.zero_state, self.birth_state, self.wrap)
