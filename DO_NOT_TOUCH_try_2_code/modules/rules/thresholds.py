"""Правило с порогами рождения/смерти (аналог кода из постановки задачи)."""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import (
    CapabilityKind,
    Requirement,
    at_least,
    eq,
)
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import StepOutcome

from .kernels import threshold_step


class ThresholdRule(Module):
    """Клетка умирает/рождается по числу соседей, иначе стареет.

    Пороги объявляются здесь же и зависят от ``max_count`` окрестности:
    ``born_above`` в ``[1, max_count)``, ``death_below`` в ``[1, born_above)``,
    ``death_above`` в ``(born_above, max_count]``.
    """

    id = "rule.thresholds"
    role = Role.RULE
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE, (eq("state", 0),)),
        Requirement(CapabilityKind.NEIGHBORHOOD_COUNT, (at_least("max_count", 1),)),
    )

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        max_count = view.peer(Role.NEIGHBORHOOD).derived("max_count")
        born = namespace.range("born_above", 1, max_count)
        namespace.range("death_below", 1, born)
        namespace.range("death_above", born + 1, max_count + 1)

    def build(self, fixed: Fixed, view: BuildView) -> "ThresholdRuntime":
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        if data.zero_state != 0:
            raise ValueError("Пороговое правило ожидает нулевое состояние 0")
        return ThresholdRuntime(
            di=neighborhood.di,
            dj=neighborhood.dj,
            wrap=neighborhood.wrap,
            born_above=int(view.value("born_above")),
            death_below=int(view.value("death_below")),
            death_above=int(view.value("death_above")),
            zero_state=int(data.zero_state),
            birth_state=1,
            last_state=int(data.states) - 1,
        )


class ThresholdRuntime:
    __slots__ = (
        "di",
        "dj",
        "wrap",
        "born_above",
        "death_below",
        "death_above",
        "zero_state",
        "birth_state",
        "last_state",
    )

    def __init__(
        self,
        di: np.ndarray,
        dj: np.ndarray,
        wrap: bool,
        born_above: int,
        death_below: int,
        death_above: int,
        zero_state: int,
        birth_state: int,
        last_state: int,
    ) -> None:
        self.di = di
        self.dj = dj
        self.wrap = wrap
        self.born_above = born_above
        self.death_below = death_below
        self.death_above = death_above
        self.zero_state = zero_state
        self.birth_state = birth_state
        self.last_state = last_state

    def step(self, src, dst) -> StepOutcome | None:
        if not np.any(src.array != self.zero_state):
            return StepOutcome(stop="вырождение: не осталось живых клеток")
        threshold_step(
            src.array,
            dst.array,
            self.di,
            self.dj,
            self.born_above,
            self.death_below,
            self.death_above,
            self.zero_state,
            self.birth_state,
            self.last_state,
            self.wrap,
        )
        return None

    def precompile(self) -> None:
        src = np.zeros((4, 4), dtype=np.int32)
        dst = np.empty((4, 4), dtype=np.int32)
        threshold_step(
            src,
            dst,
            self.di,
            self.dj,
            self.born_above,
            self.death_below,
            self.death_above,
            self.zero_state,
            self.birth_state,
            self.last_state,
            self.wrap,
        )
