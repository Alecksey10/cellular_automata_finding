"""Симметричное голосование: клетка принимает самое частое состояние соседей.

Правило не зависит от порядка соседей, поэтому инвариантно относительно всех
поворотов и отражений окрестности. Работает с любым числом состояний: при двух
состояниях это классическое мажоритарное правило, при большем — «стягивание»
в домены из нескольких фаз.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least, eq
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import StepOutcome

from .kernels import vote_step

COUNT_CAP = 8


class SymmetricVoteRule(Module):
    """Голосование большинством; ``quorum`` — минимум голосов за победителя."""

    id = "rule.vote"
    role = Role.RULE
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.DISCRETE),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE, (eq("state", 0),)),
        Requirement(CapabilityKind.NEIGHBORHOOD_COUNT, (at_least("max_count", 1),)),
    )

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        neighborhood = view.peer(Role.NEIGHBORHOOD)
        max_count = neighborhood.derived("max_count")
        cap = namespace.derive(
            "cap", lambda values: min(int(values[max_count.key]), COUNT_CAP)
        )
        namespace.range("quorum", 1, cap + 1, 1)

    def build(self, fixed: Fixed, view: BuildView) -> "VoteRuntime":
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        return VoteRuntime(
            di=neighborhood.di,
            dj=neighborhood.dj,
            wrap=neighborhood.wrap,
            zero_state=int(data.zero_state),
            states=int(data.states),
            quorum=int(view.value("quorum")),
        )


class VoteRuntime:
    __slots__ = ("di", "dj", "wrap", "zero_state", "states", "quorum")

    def __init__(
        self,
        di: np.ndarray,
        dj: np.ndarray,
        wrap: bool,
        zero_state: int,
        states: int,
        quorum: int,
    ) -> None:
        self.di = di
        self.dj = dj
        self.wrap = wrap
        self.zero_state = zero_state
        self.states = states
        self.quorum = quorum

    def step(self, src, dst) -> StepOutcome | None:
        if not np.any(src.array != self.zero_state):
            return StepOutcome(stop="вырождение: не осталось живых клеток")
        vote_step(
            src.array,
            dst.array,
            self.di,
            self.dj,
            self.quorum,
            self.states,
            self.zero_state,
            self.wrap,
        )
        return None

    def precompile(self) -> None:
        src = np.zeros((4, 4), dtype=np.int32)
        dst = np.empty((4, 4), dtype=np.int32)
        vote_step(
            src, dst, self.di, self.dj, self.quorum, self.states, self.zero_state, self.wrap
        )
