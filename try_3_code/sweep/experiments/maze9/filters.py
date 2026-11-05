"""Графовый фильтр лабиринта.

Лабиринт — это два графа на одной сетке: граф стен и граф проходов.
Эксперимент ``maze9`` проверяет именно их, а не только плотность:

* проходы — один связный граф (иначе лабиринт распадается на изолированные
  карманы);
* средняя компонента стен не меньше нескольких клеток — стены это линии и
  развилки, а не пыль из одиночных клеток;
* почти нет «висящих» одиночных клеток ни у стен, ни у проходов: одиночная
  стена в проходе и одиночный проход в стене — это дырка, а не структура.

Компоненты считает :func:`sweep.kernels.cluster_stats` (обход в ширину по
4-соседству), степени вершин — сдвиги маски на одну клетку.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...interfaces import CapabilityKind, Requirement
from ...kernels import cluster_stats
from ...module import BuildView, FilterModule, FilterRuntime, Observation, PASS, Verdict

_REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
)


def degrees(mask: np.ndarray) -> np.ndarray:
    """Число 4-соседей той же маски для каждой клетки."""

    count = np.zeros(mask.shape, dtype=np.int16)
    count[1:] += mask[:-1]
    count[:-1] += mask[1:]
    count[:, 1:] += mask[:, :-1]
    count[:, :-1] += mask[:, 1:]
    return count


class GraphRuntime(FilterRuntime):
    def __init__(
        self,
        zero: int,
        min_free_largest: float,
        min_wall_mean: float,
        max_isolated_wall: float,
        max_isolated_free: float,
    ) -> None:
        self.zero = int(zero)
        self.min_free_largest = float(min_free_largest)
        self.min_wall_mean = float(min_wall_mean)
        self.max_isolated_wall = float(max_isolated_wall)
        self.max_isolated_free = float(max_isolated_free)

    def check(self, observation: Observation) -> Verdict:
        data = np.asarray(observation.data)
        wall = data != self.zero
        free = ~wall
        wall_cells = int(wall.sum())
        free_cells = int(free.sum())
        if wall_cells == 0 or free_cells == 0:
            return Verdict(False, "empty_phase", {"wall_cells": wall_cells, "free_cells": free_cells})

        free_components, _, free_largest = cluster_stats(free.view(np.uint8))
        wall_components, wall_mean, wall_largest = cluster_stats(wall.view(np.uint8))
        wall_degree = degrees(wall)
        free_degree = degrees(free)
        isolated_wall = float(((wall_degree == 0) & wall).sum()) / wall_cells
        isolated_free = float(((free_degree == 0) & free).sum()) / free_cells
        details: dict[str, Any] = {
            "free_components": int(free_components),
            "free_largest": float(free_largest),
            "wall_components": int(wall_components),
            "wall_mean": float(wall_mean),
            "wall_largest": float(wall_largest),
            "isolated_wall": isolated_wall,
            "isolated_free": isolated_free,
        }
        if free_largest < self.min_free_largest:
            return Verdict(False, "passages_split", details)
        if wall_mean < self.min_wall_mean:
            return Verdict(False, "wall_dust", details)
        if isolated_wall > self.max_isolated_wall:
            return Verdict(False, "hanging_walls", details)
        if isolated_free > self.max_isolated_free:
            return Verdict(False, "hanging_passages", details)
        return Verdict(True, "", details)


class GraphFilter(FilterModule):
    id = "filter.maze_graph"
    label = "графы стен и проходов"

    def __init__(
        self,
        min_free_largest: float = 0.85,
        min_wall_mean: float = 3.0,
        max_isolated_wall: float = 0.01,
        max_isolated_free: float = 0.01,
    ) -> None:
        self.min_free_largest = float(min_free_largest)
        self.min_wall_mean = float(min_wall_mean)
        self.max_isolated_wall = float(max_isolated_wall)
        self.max_isolated_free = float(max_isolated_free)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> GraphRuntime:
        return GraphRuntime(
            int(fixed["grid2d.zero"]),
            self.min_free_largest,
            self.min_wall_mean,
            self.max_isolated_wall,
            self.max_isolated_free,
        )


__all__ = ["GraphFilter", "GraphRuntime", "degrees"]
