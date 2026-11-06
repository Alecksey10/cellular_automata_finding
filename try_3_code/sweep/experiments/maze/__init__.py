"""Эксперимент «лабиринт»: стохастический рост стен.

Сетка хранит два состояния: ``0`` — проход, ``1`` — стена. Из редкой случайной
затравки стены растут шаг за шагом. Клетка-проход становится стеной, если
соседей-стен в окне обзора ровно ``born_lo..born_hi`` (обычно 1–2: тогда
сеть растёт ветвями, а не пятнами), и только с вероятностью ``rate`` за шаг.
Клетка-стена выживает, если её соседей ``surv_lo..surv_hi``; иначе прорастает
проход.

Такой рост даёт именно лабиринт: стены — связная (диагонально) сетка, а
пустоты остаются единой связной системой проходов. Если разрешить рождение при
3–4 соседях, получаются замкнутые кольца и «каши»; если увеличить радиус окна,
стены и проходы становятся толще.

Оси перебора:

* ``grid2d.size`` — разрешение;
* ``maze.fill.density`` — плотность затравки;
* окрестность: квадрат Мура или диск с метрикой Чебышёва/Евклида/Манхэттена,
  радиус 1–2 — форма стен и толщина;
* ``born``/``surv`` — топология сети: дерево (1), кольца (2), пятна (3+);
* ``rate`` — скорость роста, она же плотность итоговой сети;
* окрестность ``wrap`` и ``border`` — тор или замкнутый рамкой лабиринт.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numba import njit

from ...experiment import Experiment
from ...interfaces import Capability, CapabilityKind, Requirement
from ...kernels import JIT_KWARGS
from ...module import (
    BuildView,
    NeighborhoodSpec,
    RandomizerModule,
    Role,
    RuleModule,
    RuleRuntime,
)
from ...modules.data import Grid2D
from ...modules.filters import DensityFilter
from ...neighbors import Moore, RadiusDisc
from .filters import StructureFilter
from .visualizer import MazeVisualizer

SIZES = (64, 96, 128)
FILL_DENSITIES = (0.03, 0.06, 0.12)
BORN_LO = (1, 2, 3)
BORN_HI = (1, 2, 3)
SURV_LO = (0, 1, 2)
SURV_HI = (1, 2, 3)
RATES = (0.35, 0.5, 0.75, 1.0)
BORDER_MODES = ("none", "wall")


@njit(**JIT_KWARGS)
def step_maze(src, dst, dy, dx, wrap, born_lo, born_hi, surv_lo, surv_hi, rate, border, noise):
    """Один шаг роста стен; ``noise`` — поле случайных чисел шага."""

    height, width = src.shape
    total = dy.size
    for y in range(height):
        for x in range(width):
            neighbors = 0
            for k in range(total):
                yy = y + dy[k]
                xx = x + dx[k]
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    if wrap:
                        yy = yy % height
                        xx = xx % width
                    else:
                        # за краем сетки — стена: лабиринт замкнут
                        neighbors += 1
                        continue
                if src[yy, xx] != 0:
                    neighbors += 1
            if src[y, x] != 0:
                dst[y, x] = 1 if surv_lo <= neighbors <= surv_hi else 0
            else:
                dst[y, x] = 1 if born_lo <= neighbors <= born_hi and noise[y, x] < rate else 0
            if border and (y == 0 or x == 0 or y == height - 1 or x == width - 1):
                dst[y, x] = 1


# ----------------------------------------------------------------------
# данные: квадратная сетка стен
# ----------------------------------------------------------------------
class MazeGrid(Grid2D):
    id = "grid2d"
    label = "сетка лабиринта"

    def declare(self, ns, view) -> None:
        ns.value("zero", 0)
        ns.choice("size", list(SIZES))

    def size(self, fixed: Mapping[str, Any]) -> int:
        return int(self.get(fixed, "size"))

    def derive_states(self, fixed: Mapping[str, Any]) -> int:
        return 2

    def derive_zero(self, fixed: Mapping[str, Any]) -> int:
        return int(self.get(fixed, "zero"))

    def capabilities(self, fixed: Mapping[str, Any]) -> tuple[Capability, ...]:
        size = self.size(fixed)
        zero = self.derive_zero(fixed)
        return (
            Capability(CapabilityKind.GRID_2D, {"zero": zero}, self.id),
            Capability(CapabilityKind.DIMENSION, {"value": 2}, self.id),
            Capability(
                CapabilityKind.MULTI_STATE,
                {"states": self.derive_states(fixed), "zero": zero},
                self.id,
            ),
            Capability(CapabilityKind.ZERO_STATE, {"zero": zero}, self.id),
            Capability(CapabilityKind.RESOLUTION, {"width": size, "height": size}, self.id),
        )

    def create(self, fixed, rng, randomizer) -> np.ndarray:
        size = self.size(fixed)
        data = np.zeros((size, size), dtype=np.int32)
        randomizer.fill(data, fixed, rng, zero_state=0, states=2)
        return np.ascontiguousarray(data, dtype=np.int32)


class NoiseFill(RandomizerModule):
    """Затравка лабиринта: редкие случайные стены."""

    id = "maze.fill"
    label = "затравка стен"

    def declare(self, ns, view) -> None:
        ns.choice("density", list(FILL_DENSITIES))

    def fill(self, data, fixed, rng, *, zero_state: int, states: int) -> None:
        share = float(self.get(fixed, "density"))
        mask = rng.random(data.shape) < share
        data[:] = zero_state
        if mask.any():
            data[mask] = 1


# ----------------------------------------------------------------------
# правило: рост стен с вероятностным рождением
# ----------------------------------------------------------------------
class MazeRuntime(RuleRuntime):
    def __init__(self, spec: NeighborhoodSpec, rule: tuple[int, int, int, int], rate: float, border: bool) -> None:
        self.dy = np.ascontiguousarray(spec.dy, dtype=np.int32)
        self.dx = np.ascontiguousarray(spec.dx, dtype=np.int32)
        self.wrap = bool(spec.wrap)
        self.born_lo, self.born_hi, self.surv_lo, self.surv_hi = rule
        self.rate = float(rate)
        self.border = bool(border)
        self._rng: np.random.Generator | None = None

    def step(self, src, dst, iteration: int, seed: int) -> None:
        if self._rng is None:
            self._rng = np.random.default_rng((int(seed) * 2654435761 + 7) % (2**32))
        noise = self._rng.random(src.shape, dtype=np.float32)
        step_maze(
            src,
            dst,
            self.dy,
            self.dx,
            self.wrap,
            self.born_lo,
            self.born_hi,
            self.surv_lo,
            self.surv_hi,
            self.rate,
            self.border,
            noise,
        )


class MazeRule(RuleModule):
    id = "rule.maze"
    label = "рост стен лабиринта"

    def declare(self, ns, view) -> None:
        ns.choice("born_lo", list(BORN_LO))
        ns.choice("born_hi", list(BORN_HI))
        ns.choice("surv_lo", list(SURV_LO))
        ns.choice("surv_hi", list(SURV_HI))
        ns.choice("rate", list(RATES))
        ns.choice("border", list(BORDER_MODES))

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> MazeRuntime:
        rule = (
            int(self.get(fixed, "born_lo")),
            int(self.get(fixed, "born_hi")),
            int(self.get(fixed, "surv_lo")),
            int(self.get(fixed, "surv_hi")),
        )
        return MazeRuntime(
            view.neighborhood(),
            rule,
            float(self.get(fixed, "rate")),
            str(self.get(fixed, "border")) == "wall",
        )


# ----------------------------------------------------------------------
# сборка эксперимента
# ----------------------------------------------------------------------
REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
    Requirement(CapabilityKind.NEIGHBORHOOD, (), "neighborhood"),
)


def build_maze() -> Experiment:
    experiment = Experiment(
        "maze",
        description="лабиринт из растущих стен: топология, окно обзора, толщина стен и проходов",
    )

    experiment.select(Role.DATA, MazeGrid())
    experiment.select(Role.RANDOMIZER, NoiseFill())
    experiment.select(Role.NEIGHBORHOOD, [Moore(), RadiusDisc()])
    experiment.select(Role.RULE, MazeRule())

    experiment.constrain(
        lambda values: values["rule.maze.born_hi"] >= values["rule.maze.born_lo"],
        ["rule.maze.born_hi", "rule.maze.born_lo"],
        name="born_order",
    )
    experiment.constrain(
        lambda values: values["rule.maze.surv_hi"] >= values["rule.maze.surv_lo"],
        ["rule.maze.surv_hi", "rule.maze.surv_lo"],
        name="survive_order",
    )

    # средняя плотность стен: 30–75%, то есть и стены, и проходы различимы
    density = DensityFilter(min_fill=0.3, max_fill=0.75)
    structure = StructureFilter()

    warmup = experiment.range("phase.warmup", 0, 1, 1)
    observe = experiment.range("phase.observe", 180, 301, 60)

    # сеть растёт всю фазу; решение принимается по последнему кадру
    experiment.phase("grow", warmup)
    experiment.phase("maze", observe, filters=[density, structure], frequency="end")

    experiment.visualizer = MazeVisualizer()
    return experiment


__all__ = [
    "MazeGrid",
    "MazeRule",
    "MazeRuntime",
    "NoiseFill",
    "StructureFilter",
    "build_maze",
    "step_maze",
]
