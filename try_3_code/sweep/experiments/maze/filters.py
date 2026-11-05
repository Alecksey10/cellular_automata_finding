"""Структурный отбор лабиринта.

Лабиринт — не любая картинка из стен и проходов. Нужно, чтобы:

* и стен, и проходов было достаточно (средняя плотность в коридоре 30–80%);
* плотность была равномерной: в каждом квадрате ``tiles × tiles`` нет ни
  пустого угла, ни сплошной стены (``local_sparse`` / ``local_dense``);
* проходы были одной связной системой, а не набором изолированных карманов
  (``broken_passages``);
* стены были сеткой, а не пылью из одиночных клеток (``dusty_walls``).

Связность проходов считается по 4-соседству (через диагональ пройти нельзя),
поэтому сито из одиночных стен сразу даёт сотни изолированных карманов.
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


def interior_share(mask: np.ndarray) -> float:
    """Доля клеток маски, у которых все 4 соседа тоже в маске."""

    if mask.size == 0:
        return 0.0
    core = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return float(core.sum()) / total


def tile_density(mask: np.ndarray, tiles: int) -> tuple[float, float]:
    """Минимальная и максимальная плотность маски по плиткам ``tiles × tiles``."""

    height, width = mask.shape
    tiles = max(1, min(int(tiles), height, width))
    tile_h = max(1, height // tiles)
    tile_w = max(1, width // tiles)
    crop = mask[: tile_h * tiles, : tile_w * tiles]
    blocks = crop.reshape(tiles, tile_h, tiles, tile_w)
    density = blocks.mean(axis=(1, 3))
    return float(density.min()), float(density.max())


class StructureRuntime(FilterRuntime):
    def __init__(
        self,
        zero: int,
        local_lo: float,
        local_hi: float,
        tiles: int,
        min_free_share: float,
        min_free_component: float,
        min_wall_mean: float,
    ) -> None:
        self.zero = int(zero)
        self.local_lo = float(local_lo)
        self.local_hi = float(local_hi)
        self.tiles = int(tiles)
        self.min_free_share = float(min_free_share)
        self.min_free_component = float(min_free_component)
        self.min_wall_mean = float(min_wall_mean)

    def check(self, observation: Observation) -> Verdict:
        data = np.asarray(observation.data)
        total = float(data.size) or 1.0
        free = data == self.zero
        wall = ~free
        free_share = float(free.sum()) / total
        tile_min, tile_max = tile_density(free, self.tiles)
        free_components, _, free_largest = cluster_stats(free.view(np.uint8))
        wall_components, wall_mean, wall_largest = cluster_stats(wall.view(np.uint8))
        details: dict[str, Any] = {
            "free_share": free_share,
            "wall_share": 1.0 - free_share,
            "tile_min": tile_min,
            "tile_max": tile_max,
            "free_components": int(free_components),
            "free_largest": float(free_largest),
            "wall_components": int(wall_components),
            "wall_mean": float(wall_mean),
            "wall_largest": float(wall_largest),
            "interior_free": interior_share(free),
            "interior_wall": interior_share(wall),
        }
        if free_share < self.min_free_share:
            return Verdict(False, "few_passages", details)
        if tile_min < self.local_lo:
            return Verdict(False, "local_dense", details)
        if tile_max > self.local_hi:
            return Verdict(False, "local_sparse", details)
        if free_largest < self.min_free_component:
            return Verdict(False, "broken_passages", details)
        if wall_mean < self.min_wall_mean:
            return Verdict(False, "dusty_walls", details)
        return Verdict(True, "", details)


class StructureFilter(FilterModule):
    id = "filter.maze_structure"
    label = "структура лабиринта"
    requires = _REQUIRES

    def __init__(
        self,
        local_lo: float = 0.3,
        local_hi: float = 0.8,
        tiles: int = 8,
        min_free_share: float = 0.3,
        min_free_component: float = 0.85,
        min_wall_mean: float = 1.6,
    ) -> None:
        self.local_lo = float(local_lo)
        self.local_hi = float(local_hi)
        self.tiles = int(tiles)
        self.min_free_share = float(min_free_share)
        self.min_free_component = float(min_free_component)
        self.min_wall_mean = float(min_wall_mean)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> StructureRuntime:
        return StructureRuntime(
            int(fixed["grid2d.zero"]),
            self.local_lo,
            self.local_hi,
            self.tiles,
            self.min_free_share,
            self.min_free_component,
            self.min_wall_mean,
        )


__all__ = ["StructureFilter", "StructureRuntime", "interior_share", "tile_density"]
