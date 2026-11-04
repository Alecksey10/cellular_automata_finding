"""Тоталистическое правило B/S: «сколько соседей» вместо «какие соседи».

Правило зависит только от числа живых соседей, поэтому симметрично относительно
любой перестановки клеток окрестности (а значит, и поворотов/отражений, если
симметрична сама окрестность). Это основное правило для сетки 3×3 с восемью
соседями: перебираются полосы рождения и выживания.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least, eq
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import StepOutcome

from .kernels import totalistic_step

#: Верхняя граница перебора порогов: больше 8 соседей в полосе не осмысленно.
COUNT_CAP = 8


class TotalisticRule(Module):
    """Правило «рождение в полосе B, выживание в полосе S» для двух состояний."""

    id = "rule.totalistic"
    role = Role.RULE
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.DISCRETE),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2), eq("states", 2))),
        Requirement(CapabilityKind.ZERO_STATE, (eq("state", 0),)),
        Requirement(CapabilityKind.NEIGHBORHOOD_COUNT, (at_least("max_count", 4),)),
    )

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        neighborhood = view.peer(Role.NEIGHBORHOOD)
        max_count = neighborhood.derived("max_count")
        cap = namespace.derive(
            "cap", lambda values: min(int(values[max_count.key]), COUNT_CAP)
        )
        born_lo = namespace.range("born_lo", 1, cap + 1, 1)
        namespace.range("born_hi", born_lo, cap + 1, 1)
        survive_lo = namespace.range("survive_lo", 0, cap, 1)
        namespace.range("survive_hi", survive_lo + 1, cap + 1, 1)

    def build(self, fixed: Fixed, view: BuildView) -> "TotalisticRuntime":
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        if int(data.states) != 2:
            raise ValueError("Тоталистическому правилу нужно ровно два состояния")
        return TotalisticRuntime(
            di=neighborhood.di,
            dj=neighborhood.dj,
            wrap=neighborhood.wrap,
            zero_state=int(data.zero_state),
            birth_state=1,
            born_lo=int(view.value("born_lo")),
            born_hi=int(view.value("born_hi")),
            survive_lo=int(view.value("survive_lo")),
            survive_hi=int(view.value("survive_hi")),
        )


class TotalisticRuntime:
    __slots__ = (
        "di",
        "dj",
        "wrap",
        "zero_state",
        "birth_state",
        "born_lo",
        "born_hi",
        "survive_lo",
        "survive_hi",
    )

    def __init__(
        self,
        di: np.ndarray,
        dj: np.ndarray,
        wrap: bool,
        zero_state: int,
        birth_state: int,
        born_lo: int,
        born_hi: int,
        survive_lo: int,
        survive_hi: int,
    ) -> None:
        self.di = di
        self.dj = dj
        self.wrap = wrap
        self.zero_state = zero_state
        self.birth_state = birth_state
        self.born_lo = born_lo
        self.born_hi = born_hi
        self.survive_lo = survive_lo
        self.survive_hi = survive_hi

    def step(self, src, dst) -> StepOutcome | None:
        if not np.any(src.array != self.zero_state):
            return StepOutcome(stop="вырождение: не осталось живых клеток")
        totalistic_step(
            src.array,
            dst.array,
            self.di,
            self.dj,
            self.born_lo,
            self.born_hi,
            self.survive_lo,
            self.survive_hi,
            self.zero_state,
            self.birth_state,
            self.wrap,
        )
        return None

    def precompile(self) -> None:
        src = np.zeros((4, 4), dtype=np.int32)
        dst = np.empty((4, 4), dtype=np.int32)
        totalistic_step(
            src,
            dst,
            self.di,
            self.dj,
            self.born_lo,
            self.born_hi,
            self.survive_lo,
            self.survive_hi,
            self.zero_state,
            self.birth_state,
            self.wrap,
        )
