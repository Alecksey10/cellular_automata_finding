"""Правило «чем дольше клетка живёт, тем крепче».

Состояние живой клетки — её возраст ``1..max_age``. Чем больше возраст, тем
шире допустимая полоса соседей:

* рождение мёртвой клетки: ``inner >= born_inner`` и ``total >= born_total``;
* выживание: ``total`` попадает в ``[born_total - fortify*(age-1),
  crowd_limit + fortify*(age-1)]``;
* выжившая клетка взрослеет до ``max_age`` (число состояний минус один);
* не выжившая теряет один уровень возраста и умирает только с последнего.

Так старые клетки переносят и больше голода, и больше плотности, и дольше
сопротивляются гибели — облака «закаляются» и перестают мелькать.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least, eq
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import StepOutcome
from modules.visualizers.aging_fortress import visualizer as aging_fortress_visualizer

from .kernels import aging_step


class AgingFortressRule(Module):
    id = "rule.aging_fortress"
    role = Role.RULE
    visualizer = aging_fortress_visualizer
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 3),)),
        Requirement(CapabilityKind.ZERO_STATE, (eq("state", 0),)),
        Requirement(CapabilityKind.NEIGHBORHOOD_RINGS),
        Requirement(CapabilityKind.NEIGHBORHOOD_COUNT, (at_least("max_count", 1),)),
    )

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        neighborhood = view.peer(Role.NEIGHBORHOOD)
        max_inner = neighborhood.derived("max_inner")
        max_total = neighborhood.derived("max_count")

        # Ограничиваем перебор осмысленными «малыми» порогами рождения,
        # иначе произведение осей становится совсем необозримым.
        inner_cap = namespace.derive("inner_cap", lambda values: min(int(values[max_inner.key]), 5))
        total_cap = namespace.derive("total_cap", lambda values: min(int(values[max_total.key]), 12))

        namespace.range("born_inner", 1, inner_cap + 1, 1)
        namespace.range("born_total", 1, total_cap + 1, 1)

        # Переполнение должно убивать молодые клетки, иначе автомат просто
        # заливает всю сетку. Старые клетки переносят больше (fortify).
        fraction = namespace.range("crowd_fraction", 0.15, 0.66, 0.1)
        namespace.derive(
            "crowd_limit",
            lambda values: int(round(values[max_total.key] * values[fraction.key])),
        )
        namespace.range("fortify", 1, 4, 1)

        namespace.constrain(
            lambda born_total, crowd_limit: born_total <= crowd_limit,
            "born_total",
            "crowd_limit",
            name="rule.aging.band",
        )

    def build(self, fixed: Fixed, view: BuildView) -> "AgingRuntime":
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        rings = getattr(neighborhood, "rings", {})
        if "inner" not in rings or "outer" not in rings:
            raise ValueError("Правило требует окрестность с кольцами inner/outer")
        if int(data.states) < 3:
            raise ValueError("Правилу старения нужно минимум 3 состояния")
        return AgingRuntime(
            inner_di=rings["inner"][0],
            inner_dj=rings["inner"][1],
            outer_di=rings["outer"][0],
            outer_dj=rings["outer"][1],
            wrap=neighborhood.wrap,
            zero_state=int(data.zero_state),
            born_inner=int(view.value("born_inner")),
            born_total=int(view.value("born_total")),
            crowd_limit=int(view.value("crowd_limit")),
            fortify=int(view.value("fortify")),
            max_age=int(data.states) - 1,
        )


class AgingRuntime:
    __slots__ = (
        "inner_di",
        "inner_dj",
        "outer_di",
        "outer_dj",
        "wrap",
        "zero_state",
        "born_inner",
        "born_total",
        "crowd_limit",
        "fortify",
        "max_age",
    )

    def __init__(
        self,
        inner_di: np.ndarray,
        inner_dj: np.ndarray,
        outer_di: np.ndarray,
        outer_dj: np.ndarray,
        wrap: bool,
        zero_state: int,
        born_inner: int,
        born_total: int,
        crowd_limit: int,
        fortify: int,
        max_age: int,
    ) -> None:
        self.inner_di = inner_di
        self.inner_dj = inner_dj
        self.outer_di = outer_di
        self.outer_dj = outer_dj
        self.wrap = wrap
        self.zero_state = zero_state
        self.born_inner = born_inner
        self.born_total = born_total
        self.crowd_limit = crowd_limit
        self.fortify = fortify
        self.max_age = max_age

    def step(self, src, dst) -> StepOutcome | None:
        if not np.any(src.array != self.zero_state):
            return StepOutcome(stop="вырождение: не осталось живых клеток")
        aging_step(
            src.array,
            dst.array,
            self.inner_di,
            self.inner_dj,
            self.outer_di,
            self.outer_dj,
            self.born_inner,
            self.born_total,
            self.crowd_limit,
            self.fortify,
            self.max_age,
            self.zero_state,
            self.wrap,
        )
        return None

    def precompile(self) -> None:
        src = np.zeros((4, 4), dtype=np.int32)
        dst = np.empty((4, 4), dtype=np.int32)
        aging_step(
            src,
            dst,
            self.inner_di,
            self.inner_dj,
            self.outer_di,
            self.outer_dj,
            self.born_inner,
            self.born_total,
            self.crowd_limit,
            self.fortify,
            self.max_age,
            self.zero_state,
            self.wrap,
        )
