"""Фильтр «автомат собирается в облака».

Облако — это 8-связная группа живых клеток с высокой локальной плотностью.
Критерии: заполнение в разумной полосе (не пусто и не залито целиком), средняя
компонента достаточно крупная (не пыль), большинство живых клеток имеет хотя бы
трёх живых соседей, средняя локальная плотность не ниже порога.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import FilterVerdict

from .kernels import component_stats


class CloudFilter(Module):
    """Критерий облака задаётся в эксперименте, а не перебирается.

    Порог — часть определения «желаемого» автомата, поэтому он фиксируется
    value-осями (они всё равно попадают в конфиг для воспроизводимости).
    """

    id = "filter.clouds"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(
        self,
        *,
        min_fill: float = 0.05,
        max_fill: float = 0.75,
        min_density: float = 0.45,
        min_clustered: float = 0.6,
        min_mean_component: float = 6.0,
    ) -> None:
        self.criteria = {
            "min_fill": min_fill,
            "max_fill": max_fill,
            "min_density": min_density,
            "min_clustered": min_clustered,
            "min_mean_component": min_mean_component,
        }

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        for name, value in self.criteria.items():
            namespace.value(name, value)

    def build(self, fixed: Fixed, view: BuildView) -> "CloudRuntime":
        return CloudRuntime(**{name: float(view.value(name)) for name in self.criteria})


def neighbour_counts(live: np.ndarray) -> np.ndarray:
    """Число живых 8-соседей для каждой клетки (края не заворачиваются)."""

    height, width = live.shape
    padded = np.pad(live.astype(np.uint8), 1)
    neighbours = np.zeros((height, width), dtype=np.int16)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbours += padded[1 + di : 1 + di + height, 1 + dj : 1 + dj + width]
    return neighbours


class CloudRuntime:
    __slots__ = ("min_fill", "max_fill", "min_density", "min_clustered", "min_mean_component")

    def __init__(
        self,
        min_fill: float,
        max_fill: float,
        min_density: float,
        min_clustered: float,
        min_mean_component: float,
    ) -> None:
        self.min_fill = min_fill
        self.max_fill = max_fill
        self.min_density = min_density
        self.min_clustered = min_clustered
        self.min_mean_component = min_mean_component

    def check(self, grid) -> FilterVerdict:
        live = grid.array != grid.zero_state
        count = int(np.count_nonzero(live))
        fill = float(count) / float(live.size)
        if count == 0:
            return FilterVerdict(False, "нет живых клеток", {"live": 0, "fill": 0.0})
        components, largest = component_stats(live, False)
        neighbours = neighbour_counts(live)
        mean_component = float(count) / float(components)
        density = float(neighbours[live].mean()) / 8.0
        clustered = float(np.mean(neighbours[live] >= 3))
        metrics = {
            "live": count,
            "fill": round(fill, 4),
            "components": int(components),
            "mean_component": round(mean_component, 2),
            "largest_share": round(float(largest) / float(count), 4),
            "density": round(density, 4),
            "clustered": round(clustered, 4),
        }
        if fill < self.min_fill:
            return FilterVerdict(
                False,
                f"слишком пусто: заполнение {fill:.2f} < {self.min_fill:.2f}",
                metrics,
            )
        if fill > self.max_fill:
            return FilterVerdict(
                False,
                f"залито целиком: заполнение {fill:.2f} > {self.max_fill:.2f}",
                metrics,
            )
        if mean_component < self.min_mean_component:
            return FilterVerdict(
                False,
                f"пыль вместо облаков: средняя компонента {mean_component:.1f} "
                f"< {self.min_mean_component:.1f}",
                metrics,
            )
        if density < self.min_density:
            return FilterVerdict(
                False,
                f"рыхлая структура: плотность {density:.2f} < {self.min_density:.2f}",
                metrics,
            )
        if clustered < self.min_clustered:
            return FilterVerdict(
                False,
                f"мало клеток в группах: {clustered:.2f} < {self.min_clustered:.2f}",
                metrics,
            )
        return FilterVerdict(True, metrics=metrics)
