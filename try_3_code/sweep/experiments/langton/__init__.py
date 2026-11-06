"""Эксперимент «муравей Ленгтона» на 2D-сетке.

Мир — это след муравьёв. В клетке хранится номер цвета (правила): муравей
читает цвет, поворачивается по таблице правил, сдвигает цвет на единицу и
делает шаг вперёд. Кодировка одной int32-сетки:

* ``0`` — клетка ещё не посещалась;
* ``1..colors`` — след цвета ``0..colors-1`` (``colors`` = число правил);
* ``colors+1..2*colors`` — тот же цвет плюс отметка «здесь стоит муравей».

Сами муравьи (координаты и курс) живут в рантайме правила: их не больше
восьми, в конфиг они не попадают. Конфиг задаёт начальные условия и ``seed``,
поэтому повторная сборка с тем же seed ставит муравьёв в те же клетки с теми
же курсами и даёт тот же след.

Курс — одно из восьми направлений, поэтому повороты берутся из множества
``±45°, ±90°, 180°``. Набор правил — строка символов:

* ``L`` / ``R`` — поворот на 90° влево / вправо;
* ``l`` / ``r`` — поворот на 45° влево / вправо;
* ``b`` — поворот на 180°;
* ``?`` — случайный поворот из этого набора (выбирается по seed).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numba import njit

from ...experiment import Experiment
from ...interfaces import Capability, CapabilityKind, Requirement, ge
from ...kernels import JIT_KWARGS
from ...module import (
    BuildView,
    DeclareView,
    RandomizerModule,
    Role,
    RuleModule,
    RuleRuntime,
)
from ...modules.data import Grid2D
from .filters import PatternFilter, SettleFilter
from .visualizer import LangtonVisualizer

# Направления по часовой стрелке, начиная с востока (ось Y вниз).
DY = (0, 1, 1, 1, 0, -1, -1, -1)
DX = (1, 1, 0, -1, -1, -1, 0, 1)

# Символ правила -> поворот в шагах по 45° (по часовой стрелке — плюс).
TURNS = {"L": -2, "R": 2, "l": -1, "r": 1, "b": 4}

# Наборы правил: 2, 4, 6 и 7 правил. Последний символ ``?`` разрешается seed'ом.
RULE_SETS = (
    # 2 правила: только ±90
    "LR",
    "RL",
    # 4 правила: ±45 и ±90 — циклические сдвиги и зеркала
    "lLrR",
    "LrRl",
    "rRlL",
    "RlLr",
    "lRrL",
    "RrLl",
    "rLlR",
    "LlRr",
    # 6 правил: ±45, ±90 и два поворота на 180
    "lLrRbb",
    "lLbRrb",
    "lLrbRb",
    "LlrRbb",
    "LlbrRb",
    "bLlRrb",
    # 7 правил: те же шесть и ещё один случайный поворот
    "lLrRb?",
    "lLrR?b",
    "lL?rRb",
    "?lLrRb",
    "lLrbR?",
    "b?lLrR",
)

INIT_MODES = ("empty", "random_light", "random_dense")
INIT_FILL = {"empty": 0.0, "random_light": 0.15, "random_dense": 0.3}


def parse_rules(spec: str, seed: int) -> tuple[int, ...]:
    """Строка правил -> повороты; ``?`` выбирается по seed."""

    rng = np.random.default_rng((int(seed) * 2654435761 + 17) % (2**32))
    turns: list[int] = []
    for symbol in spec:
        if symbol == "?":
            turns.append(int(rng.choice([1, -1, 2, -2, 4])))
        else:
            turns.append(TURNS[symbol])
    return tuple(turns)


# ----------------------------------------------------------------------
# данные: сетка следа
# ----------------------------------------------------------------------
@njit(**JIT_KWARGS)
def trail_stats(data, colors):
    """Счётчики следа: клетки по цветам, число отметок муравьёв, bbox."""

    height, width = data.shape
    counts = np.zeros(colors + 2, dtype=np.int64)
    visited = 0
    ants = 0
    min_y = height
    max_y = -1
    min_x = width
    max_x = -1
    for y in range(height):
        for x in range(width):
            value = data[y, x]
            if value > colors:
                ants += 1
                value -= colors
            if value > 0:
                visited += 1
                counts[value] += 1
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
    return counts, visited, ants, min_y, max_y, min_x, max_x


class LangtonGrid(Grid2D):
    """Сетка следа: ширина и высота задаются одним размером ``size``."""

    id = "grid2d"
    label = "сетка следа муравья"

    def declare(self, ns, view) -> None:
        ns.value("zero", 0)
        ns.choice("size", [96, 128])

    # ------------------------------------------------------------------
    def size(self, fixed: Mapping[str, Any]) -> int:
        return int(self.get(fixed, "size"))

    def colors(self, fixed: Mapping[str, Any]) -> int:
        """Число цветов = число правил выбранного набора."""

        return len(str(fixed["rule.langton.rules"]))

    def derive_states(self, fixed: Mapping[str, Any]) -> int:
        return 2 * self.colors(fixed) + 1

    def derive_zero(self, fixed: Mapping[str, Any]) -> int:
        return int(self.get(fixed, "zero"))

    def derived_value(self, name: str, fixed: Mapping[str, Any]) -> Any:
        if name == "states":
            return self.derive_states(fixed)
        return super().derived_value(name, fixed)

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
        randomizer.fill(data, fixed, rng, zero_state=0, states=self.colors(fixed))
        return np.ascontiguousarray(data, dtype=np.int32)

    def metrics(self, data: np.ndarray, fixed: Mapping[str, Any]) -> dict[str, Any]:
        colors = self.colors(fixed)
        counts, visited, ants, min_y, max_y, min_x, max_x = trail_stats(data, colors)
        total = float(data.size) or 1.0
        bbox = 0.0
        if visited:
            bbox = float((max_y - min_y + 1) * (max_x - min_x + 1))
        border = 0
        if visited:
            border = int(min_y == 0) + int(min_x == 0) + int(max_y == data.shape[0] - 1) + int(
                max_x == data.shape[1] - 1
            )
        metrics: dict[str, Any] = {
            "zero_density": 1.0 - visited / total,
            "alive_density": visited / total,
            "trail_density": visited / total,
            "trail_cells": float(visited),
            "ant_cells": float(ants),
            "occupied_states": float(np.count_nonzero(counts)),
            "max_state_share": float(counts.max()) / total if counts.size else 0.0,
            "mean_state": float(np.dot(np.arange(counts.size), counts)) / visited if visited else 0.0,
            "bbox_share": bbox / total,
            "bbox_fill": visited / bbox if bbox else 0.0,
            "border": float(border),
        }
        for color in range(1, colors + 1):
            metrics[f"share_{color}"] = float(counts[color]) / total
        return metrics

    def alive_count(self, data: np.ndarray, fixed: Mapping[str, Any]) -> int:
        return int(np.count_nonzero(data))


class TrailFill(RandomizerModule):
    """Начальный след: пусто или случайные цвета."""

    id = "trail"
    label = "начальный след"

    def declare(self, ns, view) -> None:
        ns.choice("init", list(INIT_MODES))

    def fill(self, data, fixed, rng, *, zero_state: int, states: int) -> None:
        share = INIT_FILL[str(self.get(fixed, "init"))]
        if share <= 0.0:
            data[:] = zero_state
            return
        mask = rng.random(data.shape) < share
        data[:] = zero_state
        if mask.any():
            data[mask] = rng.integers(1, states + 1, size=int(mask.sum()))


# ----------------------------------------------------------------------
# правило: муравьи
# ----------------------------------------------------------------------
class LangtonRuntime(RuleRuntime):
    """Муравьи как объекты рантайма: след живёт в сетке, курсы — здесь."""

    def __init__(self, turns: tuple[int, ...], count: int, place: str) -> None:
        self.turns = turns
        self.colors = len(turns)
        self.count = int(count)
        self.place = str(place)
        self.ants: list[tuple[int, int, int]] = []
        self.marked: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    def _reset(self, shape: tuple[int, int], seed: int) -> None:
        height, width = shape
        rng = np.random.default_rng((int(seed) * 2654435761 + 1) % (2**32))
        ants: list[tuple[int, int, int]] = []
        if self.place == "center":
            center_y, center_x = height // 2, width // 2
            for index in range(self.count):
                y = (center_y + index % 3 - 1) % height
                x = (center_x + (index // 3) % 3 - 1) % width
                heading = (index * 8) // max(1, self.count) % 8
                ants.append((y, x, heading))
        else:
            for _ in range(self.count):
                ants.append(
                    (
                        int(rng.integers(0, height)),
                        int(rng.integers(0, width)),
                        int(rng.integers(0, 8)),
                    )
                )
        self.ants = ants
        self.marked = []

    # ------------------------------------------------------------------
    def step(self, src, dst, iteration: int, seed: int) -> None:
        np.copyto(dst, src)
        if iteration == 0:
            self._reset(src.shape, seed)
        colors = self.colors
        height, width = dst.shape
        for y, x in self.marked:
            value = int(dst[y, x])
            if value > colors:
                dst[y, x] = value - colors
        self.marked = []
        for index in range(len(self.ants)):
            y, x, heading = self.ants[index]
            value = int(src[y, x])
            if value > colors:
                value -= colors
            color = value if value > 0 else 1
            heading = (heading + self.turns[color - 1]) % 8
            dst[y, x] = color % colors + 1
            next_y = (y + DY[heading]) % height
            next_x = (x + DX[heading]) % width
            target = int(src[next_y, next_x])
            if target > colors:
                target -= colors
            if target <= 0:
                target = 1
            dst[next_y, next_x] = target + colors
            self.marked.append((next_y, next_x))
            self.ants[index] = (next_y, next_x, heading)


class LangtonRule(RuleModule):
    id = "rule.langton"
    label = "муравей Ленгтона"

    def declare(self, ns, view: DeclareView) -> None:
        ns.choice("rules", list(RULE_SETS))
        ns.choice("ants", [1, 2, 4, 8])
        ns.choice("place", ["center", "random"])

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> LangtonRuntime:
        turns = parse_rules(str(self.get(fixed, "rules")), int(fixed.get("seed", 0)))
        return LangtonRuntime(turns, int(self.get(fixed, "ants")), str(self.get(fixed, "place")))


# ----------------------------------------------------------------------
# сборка эксперимента
# ----------------------------------------------------------------------
REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
    Requirement(CapabilityKind.MULTI_STATE, (ge("states", 5),), "multi_state (states >= 5)"),
)


def build_langton() -> Experiment:
    experiment = Experiment(
        "langton",
        description="муравей Ленгтона: след-сетка, наборы правил, отбор паттернов",
    )

    experiment.select(Role.DATA, LangtonGrid())
    experiment.select(Role.RANDOMIZER, TrailFill())
    # правило — модуль, а не inline-функция: у муравьёв есть состояние между
    # шагами (координаты и курсы), которого нет у чистой numba-итерации
    experiment.select(Role.RULE, LangtonRule())

    settle = SettleFilter(max_growth=0.2, min_cells=150.0)
    pattern = PatternFilter(min_iou=0.55)

    warmup = experiment.range("phase.warmup", 500, 2000, 500)
    observe = experiment.range("phase.observe", 3000, 6000, 1500)

    # муравей сначала свободно растёт, а отбор решает в конце: проверки идут
    # каждые 100 шагов, но отказ возможен только на последних 300
    experiment.phase("warmup", warmup)
    experiment.phase(
        "observe",
        observe,
        filters=[settle, pattern],
        frequency="each",
        every=100,
    )

    experiment.visualizer = LangtonVisualizer()
    return experiment



def web_card_details(values: Mapping[str, Any], seed: int = 0) -> dict[str, Any]:
    """Повороты и стартовые муравьи для браузерной карточки.

    Считает Python по тому же ``seed``, что и прогон, поэтому браузерный
    модуль не повторяет таблицу поворотов и расстановку муравьёв.
    """

    from . import LangtonRuntime, parse_rules

    turns = parse_rules(str(values["rule.langton.rules"]), seed)
    size = int(values.get("grid2d.size") or values["grid2d.width"])
    runtime = LangtonRuntime(turns, int(values["rule.langton.ants"]), str(values["rule.langton.place"]))
    runtime._reset((size, size), seed)
    return {
        "turns": [int(turn) for turn in turns],
        "ants": [[int(y), int(x), int(heading)] for y, x, heading in runtime.ants],
    }



__all__ = [
    "LangtonGrid",
    "LangtonRule",
    "LangtonRuntime",
    "PatternFilter",
    "RULE_SETS",
    "SettleFilter",
    "TrailFill",
    "build_langton",
    "web_card_details",
    "parse_rules",
    "trail_stats",
]
