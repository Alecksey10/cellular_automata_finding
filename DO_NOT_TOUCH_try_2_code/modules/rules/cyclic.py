"""Циклическое правило: клетка «твердеет», затем умирает.

Состояния ``0..S-1`` — фазы цикла: ``0`` — пустая/самая уязвимая фаза, далее
клетка отвердевает с каждой фазой, а самая твёрдая фаза ``S-1`` сменяется
смертью (``0``). Переход в следующую фазу происходит, когда рядом не меньше
``threshold`` клеток уже в этой фазе — так по сетке бежит фронт фазы.

Два режима смерти (ось ``death``):

* ``next`` — фаза ``S-1`` умирает, когда рядом достаточно клеток фазы ``0``
  (классический циклический автомат с модульной арифметикой);
* ``always`` — фаза ``S-1`` умирает безусловно, поэтому волны затухают.

Правило тоталистическое: учитывается только число соседей нужной фазы, поэтому
оно симметрично относительно поворотов и отражений.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least, eq
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import StepOutcome
from modules.visualizers.cyclic_phases import visualizer as cyclic_phases_visualizer

from .kernels import cyclic_step

COUNT_CAP = 8


class CyclicRule(Module):
    id = "rule.cyclic"
    role = Role.RULE
    visualizer = cyclic_phases_visualizer
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.DISCRETE),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 3),)),
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
        namespace.choose("death", ("next", "always"))

    def build(self, fixed: Fixed, view: BuildView) -> "CyclicRuntime":
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        states = int(data.states)
        if states < 3:
            raise ValueError("Циклическому правилу нужно минимум три состояния")
        return CyclicRuntime(
            di=neighborhood.di,
            dj=neighborhood.dj,
            wrap=neighborhood.wrap,
            zero_state=int(data.zero_state),
            states=states,
            threshold=int(view.value("threshold")),
            always_die=view.value("death") == "always",
        )


class CyclicRuntime:
    __slots__ = ("di", "dj", "wrap", "zero_state", "states", "threshold", "always_die")

    def __init__(
        self,
        di: np.ndarray,
        dj: np.ndarray,
        wrap: bool,
        zero_state: int,
        states: int,
        threshold: int,
        always_die: bool,
    ) -> None:
        self.di = di
        self.dj = dj
        self.wrap = wrap
        self.zero_state = zero_state
        self.states = states
        self.threshold = threshold
        self.always_die = always_die

    def step(self, src, dst) -> StepOutcome | None:
        cyclic_step(
            src.array,
            dst.array,
            self.di,
            self.dj,
            self.threshold,
            self.states,
            self.zero_state,
            self.always_die,
            self.wrap,
        )
        return None

    def precompile(self) -> None:
        src = np.zeros((4, 4), dtype=np.int32)
        dst = np.empty((4, 4), dtype=np.int32)
        cyclic_step(
            src,
            dst,
            self.di,
            self.dj,
            self.threshold,
            self.states,
            self.zero_state,
            self.always_die,
            self.wrap,
        )
