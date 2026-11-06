"""Эксперимент «лабиринт 9×9»: механизмы внутри окна радиуса 4.

Сетка — два состояния: ``0`` проход, ``1`` стена. Стены растут из редкой
затравки, как в :mod:`sweep.experiments.maze`, но решение принимается не по
всему окну 9×9, а по выбранной внутри него **механизму-подмаске**. Механизм —
это то, на что смотрит растущая стена:

* ``inner`` — кольцо 3×3: плотная застройка вплотную к клетке;
* ``middle`` — кольцо 5×5 без центрального 3×3;
* ``outer`` — внешнее кольцо 9×9 (радиус 3–4): стена «чувствует» далёкие стены;
* ``rim`` — самый край окна (радиус 4);
* ``cross`` — четыре луча по вертикали и горизонтали;
* ``diagonals`` — четыре диагонали;
* ``corners`` — четыре угловых блока 3×3;
* ``checker`` — шахматная подмаска;
* ``all`` — всё окно 9×9 без центра.

Пороги заданы в процентах от размера механизма, поэтому одно правило
переносится между механизмами разного размера. Разные механизмы дают разные
структурные элементы: ``inner`` выращивает ветвистое дерево, ``outer``
удерживает стены на расстоянии и делает широкие проходы, ``cross`` и
``diagonals`` вытягивают длинные прямые участки.

После прогрева карта должна меняться мало: фильтр динамики требует, чтобы за
шаг менялось не больше 10% клеток (``filter.dynamics``). Финальный кадр
проверяют плотность, структура проходов и графовый фильтр: стены и проходы —
крупные смежные компоненты, а не пыль из одиночных клеток.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...experiment import Experiment
from ...interfaces import CapabilityKind, Requirement
from ...module import BuildView, NeighborhoodSpec, RandomizerModule, Role, RuleModule, RuleRuntime
from ...modules.filters import DensityFilter, DynamicsFilter
from ...neighbors import Square9x9
from ..maze import MazeGrid, step_maze
from ..maze.filters import StructureFilter
from .filters import GraphFilter
from .visualizer import Maze9Visualizer

WINDOW = 4  # радиус окна 9×9
SIZES = (64, 96)
FILL_DENSITIES = (0.03, 0.06, 0.12)
MECHANISMS = (
    "inner",
    "middle",
    "outer",
    "rim",
    "cross",
    "diagonals",
    "corners",
    "checker",
    "all",
)
BORN_LO = (4, 8, 12, 20, 30)
BORN_HI = (8, 12, 20, 30, 40)
SURV_LO = (0, 4, 8, 12, 20)
SURV_HI = (20, 30, 40, 50, 60)
RATES = (0.35, 0.5, 0.75, 1.0)
BORDER_MODES = ("none", "wall")


def _ring(lo: int, hi: int) -> list[tuple[int, int]]:
    """Клетки окна, у которых расстояние Чебышёва в ``[lo, hi]``."""

    cells: list[tuple[int, int]] = []
    for dy in range(-WINDOW, WINDOW + 1):
        for dx in range(-WINDOW, WINDOW + 1):
            if lo <= max(abs(dy), abs(dx)) <= hi:
                cells.append((dy, dx))
    return cells


def mechanism_offsets(name: str) -> np.ndarray:
    """Смещения подмаски механизма внутри окна 9×9."""

    name = str(name)
    if name == "inner":
        cells = _ring(1, 1)
    elif name == "middle":
        cells = _ring(2, 2)
    elif name == "outer":
        cells = _ring(3, WINDOW)
    elif name == "rim":
        cells = _ring(WINDOW, WINDOW)
    elif name == "cross":
        cells = [(dy, 0) for dy in range(-WINDOW, WINDOW + 1) if dy]
        cells += [(0, dx) for dx in range(-WINDOW, WINDOW + 1) if dx]
    elif name == "diagonals":
        cells = [(k, k) for k in range(-WINDOW, WINDOW + 1) if k]
        cells += [(k, -k) for k in range(-WINDOW, WINDOW + 1) if k]
    elif name == "corners":
        cells = [
            (dy, dx)
            for dy in range(-WINDOW, WINDOW + 1)
            for dx in range(-WINDOW, WINDOW + 1)
            if abs(dy) >= 2 and abs(dx) >= 2
        ]
    elif name == "checker":
        cells = [
            (dy, dx)
            for dy in range(-WINDOW, WINDOW + 1)
            for dx in range(-WINDOW, WINDOW + 1)
            if (dy + dx) % 2 == 0 and (dy or dx)
        ]
    elif name == "all":
        cells = _ring(1, WINDOW)
    else:
        raise KeyError(f"неизвестный механизм {name!r}")
    offsets = np.asarray(sorted(cells), dtype=np.int32)
    return offsets


class Maze9Grid(MazeGrid):
    id = "grid2d"
    label = "сетка лабиринта 9×9"

    def declare(self, ns, view) -> None:
        ns.value("zero", 0)
        ns.choice("size", list(SIZES))


class SeedFill(RandomizerModule):
    """Затравка: редкие случайные стены."""

    id = "maze9.fill"
    label = "затравка стен"

    def declare(self, ns, view) -> None:
        ns.choice("density", list(FILL_DENSITIES))

    def fill(self, data, fixed, rng, *, zero_state: int, states: int) -> None:
        share = float(self.get(fixed, "density"))
        mask = rng.random(data.shape) < share
        data[:] = zero_state
        if mask.any():
            data[mask] = 1


class MechanismRuntime(RuleRuntime):
    def __init__(
        self,
        offsets: np.ndarray,
        spec: NeighborhoodSpec,
        rule: tuple[int, int, int, int],
        rate: float,
        border: bool,
    ) -> None:
        self.dy = np.ascontiguousarray(offsets[:, 0], dtype=np.int32)
        self.dx = np.ascontiguousarray(offsets[:, 1], dtype=np.int32)
        self.wrap = bool(spec.wrap)
        self.born_lo, self.born_hi, self.surv_lo, self.surv_hi = rule
        self.rate = float(rate)
        self.border = bool(border)
        self._rng: np.random.Generator | None = None

    def step(self, src, dst, iteration: int, seed: int) -> None:
        if self._rng is None:
            self._rng = np.random.default_rng((int(seed) * 2654435761 + 13) % (2**32))
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


def counts_for(size: int, percents: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Проценты от подмаски -> число клеток; рождение всегда требует соседа."""

    born_lo = max(1, int(round(percents[0] * size / 100.0)))
    born_hi = max(born_lo, int(round(percents[1] * size / 100.0)))
    surv_lo = max(0, int(round(percents[2] * size / 100.0)))
    surv_hi = max(surv_lo, int(round(percents[3] * size / 100.0)))
    return born_lo, born_hi, surv_lo, surv_hi


class MechanismRule(RuleModule):
    id = "rule.maze9"
    label = "механизм в окне 9×9"

    def declare(self, ns, view) -> None:
        ns.choice("mechanism", list(MECHANISMS))
        ns.choice("born_lo", list(BORN_LO))
        ns.choice("born_hi", list(BORN_HI))
        ns.choice("surv_lo", list(SURV_LO))
        ns.choice("surv_hi", list(SURV_HI))
        ns.choice("rate", list(RATES))
        ns.choice("border", list(BORDER_MODES))

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> MechanismRuntime:
        offsets = mechanism_offsets(str(self.get(fixed, "mechanism")))
        percents = (
            int(self.get(fixed, "born_lo")),
            int(self.get(fixed, "born_hi")),
            int(self.get(fixed, "surv_lo")),
            int(self.get(fixed, "surv_hi")),
        )
        return MechanismRuntime(
            offsets,
            view.neighborhood(),
            counts_for(len(offsets), percents),
            float(self.get(fixed, "rate")),
            str(self.get(fixed, "border")) == "wall",
        )


REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
)


def build_maze9() -> Experiment:
    experiment = Experiment(
        "maze9",
        description="лабиринт с окном 9×9: механизмы, малые изменения карты, графы стен и проходов",
    )

    experiment.select(Role.DATA, Maze9Grid())
    experiment.select(Role.RANDOMIZER, SeedFill())
    experiment.select(Role.NEIGHBORHOOD, Square9x9())
    experiment.select(Role.RULE, MechanismRule())

    experiment.constrain(
        lambda values: values["rule.maze9.born_hi"] >= values["rule.maze9.born_lo"],
        ["rule.maze9.born_hi", "rule.maze9.born_lo"],
        name="born_order",
    )
    experiment.constrain(
        lambda values: values["rule.maze9.surv_hi"] >= values["rule.maze9.surv_lo"],
        ["rule.maze9.surv_hi", "rule.maze9.surv_lo"],
        name="survive_order",
    )

    # после прогрева карта должна меняться не более чем на 10% за шаг
    dynamics = DynamicsFilter(min_change=0.0, max_change=0.1)
    density = DensityFilter(min_fill=0.3, max_fill=0.75)
    structure = StructureFilter()
    graph = GraphFilter()

    warmup = experiment.range("phase.warmup", 200, 401, 100)
    experiment.phase("grow", warmup)
    experiment.phase(
        "hold",
        40,
        filters=[dynamics, density, structure, graph],
        frequency="each",
    )

    experiment.visualizer = Maze9Visualizer()
    return experiment


__all__ = [
    "MECHANISMS",
    "Maze9Grid",
    "MechanismRule",
    "MechanismRuntime",
    "SeedFill",
    "build_maze9",
    "counts_for",
    "mechanism_offsets",
]
