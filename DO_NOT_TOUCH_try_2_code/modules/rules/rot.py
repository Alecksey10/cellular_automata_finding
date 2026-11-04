"""Правило «старение с частичным гниением».

Состояние живой клетки — её прочность: ``1..states-1``. Клетка взрослеет,
пока число соседей (по двум кольцам) попадает в полосу выживания, и теряет
один уровень, когда не попадает. Клетка на последнем уровне не исчезает
сразу: с вероятностью ``rot`` она превращается в ``zero_state``. Поэтому
старое облако не мелькает, а постепенно гниёт по клеткам.

Рождение, смерть и выживание задаются порогами по счётчикам колец:

* ``born_inner`` — минимум соседей во внутреннем кольце для рождения;
* ``born_total`` — минимум соседей по сумме колец для рождения;
* ``survive_lo``, ``survive_hi`` — полоса суммы колец, в которой клетка
  выживает и взрослеет; вне полосы она теряет уровень.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least, eq
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import StepOutcome
from modules.visualizers.rot_aging import visualizer as rot_aging_visualizer

from .kernels import rot_aging_step

#: Доли старых клеток, которые гниют за кадр.
ROT_CHOICES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35)


class RotAgingRule(Module):
    """Старение + вероятностное гниение последнего состояния."""

    id = "rule.rot_aging"
    role = Role.RULE
    visualizer = rot_aging_visualizer
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

        # Пороги перебираются по всему диапазону колец: у r1=5 внутреннее
        # кольцо содержит 80 клеток, и обрезка порогов до 12 отбрасывала бы
        # все осмысленные режимы больших радиусов. Для больших пространств
        # используйте --random (см. README).
        namespace.range("born_inner", 1, max_inner + 1, 1)
        namespace.range("born_total", 1, max_total + 1, 1)
        namespace.range("survive_lo", 1, max_total + 1, 1)
        namespace.range("survive_hi", 1, max_total + 1, 1)
        namespace.choose("rot", ROT_CHOICES)

        # Полоса выживания должна содержать порог рождения: иначе клетка
        # рождается сразу вне полосы и не делает ни одного шага.
        namespace.constrain(
            lambda born_total, survive_lo: born_total <= survive_lo,
            "born_total",
            "survive_lo",
            name="rule.rot.born_within_band",
        )
        namespace.constrain(
            lambda survive_lo, survive_hi: survive_lo <= survive_hi,
            "survive_lo",
            "survive_hi",
            name="rule.rot.band_order",
        )

    def build(self, fixed: Fixed, view: BuildView) -> "RotAgingRuntime":
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        rings = getattr(neighborhood, "rings", {})
        if "inner" not in rings or "outer" not in rings:
            raise ValueError("Правило требует окрестность с кольцами inner/outer")
        if int(data.states) < 3:
            raise ValueError("Правилу старения нужно минимум 3 состояния")
        return RotAgingRuntime(
            inner_di=rings["inner"][0],
            inner_dj=rings["inner"][1],
            outer_di=rings["outer"][0],
            outer_dj=rings["outer"][1],
            wrap=neighborhood.wrap,
            zero_state=int(data.zero_state),
            born_inner=int(view.value("born_inner")),
            born_total=int(view.value("born_total")),
            survive_lo=int(view.value("survive_lo")),
            survive_hi=int(view.value("survive_hi")),
            last_state=int(data.states) - 1,
            rot=float(view.value("rot")),
        )


class RotAgingRuntime:
    """Рабочий объект одного теста; хранит RNG и буфер шума."""

    __slots__ = (
        "inner_di",
        "inner_dj",
        "outer_di",
        "outer_dj",
        "wrap",
        "zero_state",
        "born_inner",
        "born_total",
        "survive_lo",
        "survive_hi",
        "last_state",
        "rot",
        "rng",
        "noise",
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
        survive_lo: int,
        survive_hi: int,
        last_state: int,
        rot: float,
    ) -> None:
        self.inner_di = inner_di
        self.inner_dj = inner_dj
        self.outer_di = outer_di
        self.outer_dj = outer_dj
        self.wrap = wrap
        self.zero_state = zero_state
        self.born_inner = born_inner
        self.born_total = born_total
        self.survive_lo = survive_lo
        self.survive_hi = survive_hi
        self.last_state = last_state
        self.rot = rot
        self.rng: np.random.Generator | None = None
        self.noise: np.ndarray | None = None

    # --- RNG -------------------------------------------------------------
    def bind_rng(self, rng: np.random.Generator) -> None:
        """Получить генератор теста (вызывает ``run_trial`` до первой итерации)."""

        self.rng = rng

    def _noise_for(self, src) -> np.ndarray:
        if self.noise is None or self.noise.shape != src.array.shape:
            self.noise = np.empty(src.array.shape, dtype=np.float32)
        if self.rot > 0:
            rng = self.rng
            if rng is None:  # прямой вызов из теста/скрипта
                rng = np.random.default_rng(0)
                self.rng = rng
            self.noise[...] = rng.random(self.noise.shape, dtype=np.float32)
        else:
            self.noise[...] = 0.0
        return self.noise

    # --- итерация --------------------------------------------------------
    def step(self, src, dst) -> StepOutcome | None:
        if not np.any(src.array != self.zero_state):
            return StepOutcome(stop="вырождение: не осталось живых клеток")
        rot_aging_step(
            src.array,
            dst.array,
            self._noise_for(src),
            self.inner_di,
            self.inner_dj,
            self.outer_di,
            self.outer_dj,
            self.born_inner,
            self.born_total,
            self.survive_lo,
            self.survive_hi,
            self.last_state,
            self.rot,
            self.zero_state,
            self.wrap,
        )
        return None

    def precompile(self) -> None:
        src = np.zeros((4, 4), dtype=np.int32)
        dst = np.empty((4, 4), dtype=np.int32)
        noise = np.zeros((4, 4), dtype=np.float32)
        rot_aging_step(
            src,
            dst,
            noise,
            self.inner_di,
            self.inner_dj,
            self.outer_di,
            self.outer_dj,
            self.born_inner,
            self.born_total,
            self.survive_lo,
            self.survive_hi,
            self.last_state,
            self.rot,
            self.zero_state,
            self.wrap,
        )
