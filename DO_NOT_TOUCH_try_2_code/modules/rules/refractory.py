"""Возбудимая среда с рефрактерностью: возбуждение, отвердение, смерть.

Состояния: ``0`` — возбудимая клетка, ``1`` — возбуждённая, ``2..S-1`` —
рефрактерные фазы (клетка «отвердела» и не может возбудиться заново). Каждая
итерация повышает рефрактерную фазу, с последней клетка возвращается в ``0``.

Правило даёт бегущие волны и спирали: возбуждение распространяется по
невозбуждённым клеткам, за фронтом остаётся рефрактерный след. Порог
``threshold`` — сколько возбуждённых соседей нужно, чтобы зажечь клетку.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least, eq
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import StepOutcome
from modules.visualizers.refractory_phases import visualizer as refractory_phases_visualizer

from .kernels import refractory_step

COUNT_CAP = 8


class RefractoryRule(Module):
    id = "rule.refractory"
    role = Role.RULE
    visualizer = refractory_phases_visualizer
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.DISCRETE),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 4),)),
        Requirement(CapabilityKind.ZERO_STATE, (eq("state", 0),)),
        Requirement(CapabilityKind.NEIGHBORHOOD_COUNT, (at_least("max_count", 1),)),
    )

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        neighborhood = view.peer(Role.NEIGHBORHOOD)
        max_count = neighborhood.derived("max_count")
        cap = namespace.derive(
            "cap", lambda values: min(int(values[max_count.key]), COUNT_CAP)
        )
        namespace.range("threshold", 1, cap + 1, 1)

    def build(self, fixed: Fixed, view: BuildView) -> "RefractoryRuntime":
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        states = int(data.states)
        if states < 4:
            raise ValueError("Рефрактерному правилу нужно минимум четыре состояния")
        return RefractoryRuntime(
            di=neighborhood.di,
            dj=neighborhood.dj,
            wrap=neighborhood.wrap,
            zero_state=int(data.zero_state),
            last_state=states - 1,
            threshold=int(view.value("threshold")),
        )


class RefractoryRuntime:
    __slots__ = ("di", "dj", "wrap", "zero_state", "last_state", "threshold")

    def __init__(
        self,
        di: np.ndarray,
        dj: np.ndarray,
        wrap: bool,
        zero_state: int,
        last_state: int,
        threshold: int,
    ) -> None:
        self.di = di
        self.dj = dj
        self.wrap = wrap
        self.zero_state = zero_state
        self.last_state = last_state
        self.threshold = threshold

    def step(self, src, dst) -> StepOutcome | None:
        refractory_step(
            src.array,
            dst.array,
            self.di,
            self.dj,
            self.threshold,
            self.last_state,
            self.zero_state,
            self.wrap,
        )
        return None

    def precompile(self) -> None:
        src = np.zeros((4, 4), dtype=np.int32)
        dst = np.empty((4, 4), dtype=np.int32)
        refractory_step(
            src, dst, self.di, self.dj, self.threshold, self.last_state, self.zero_state, self.wrap
        )
