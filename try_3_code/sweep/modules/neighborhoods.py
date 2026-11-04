"""Модули соседства.

Каждая реализация возвращает :class:`~sweep.module.NeighborhoodSpec` —
плоские массивы смещений ``(dy, dx)``, которые numba-ядро обходит без
Python-объектов. ``max_count`` — число соседей в полном кольце; правило
строит по нему свои пороги.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..interfaces import Capability, CapabilityKind
from ..module import NeighborhoodModule, NeighborhoodSpec
from ..space import Namespace


def _disc_offsets(radius: int, metric: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            if metric == "chebyshev":
                inside = max(abs(dy), abs(dx)) <= radius
            elif metric == "manhattan":
                inside = abs(dy) + abs(dx) <= radius
            else:  # euclidean
                inside = dy * dy + dx * dx <= radius * radius
            if inside:
                offsets.append((dy, dx))
    return offsets


def _arrays(offsets: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if not offsets:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty.copy()
    arr = np.asarray(offsets, dtype=np.int32)
    return arr[:, 0].copy(), arr[:, 1].copy()


class _BaseNeighborhood(NeighborhoodModule):
    def capabilities(self, fixed: Mapping[str, Any]) -> tuple[Capability, ...]:
        spec = self.spec(fixed)
        return (
            Capability(
                CapabilityKind.NEIGHBORHOOD,
                {
                    "max_count": spec.max_count,
                    "inner_max_count": spec.inner_max_count,
                    "wrap": spec.wrap,
                },
                self.id,
            ),
        )

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        raise NotImplementedError


class Moore(_BaseNeighborhood):
    id = "neighborhood.moore"
    label = "квадрат (Мура)"

    def declare(self, ns: Namespace, view) -> None:
        ns.range("radius", 1, 4)
        ns.choice("wrap", [True, False])

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        radius = int(self.get(fixed, "radius"))
        wrap = bool(self.get(fixed, "wrap"))
        offsets = _disc_offsets(radius, "chebyshev")
        dy, dx = _arrays(offsets)
        return NeighborhoodSpec(
            name=f"moore{radius}",
            dy=dy,
            dx=dx,
            inner_dy=dy,
            inner_dx=dx,
            wrap=wrap,
            max_count=len(offsets),
            inner_max_count=len(offsets),
        )


class RadiusDisc(_BaseNeighborhood):
    id = "neighborhood.radius"
    label = "радиальный диск"

    def declare(self, ns: Namespace, view) -> None:
        ns.range("radius", 1, 4)
        ns.choice("metric", ["euclidean", "manhattan", "chebyshev"])
        ns.choice("wrap", [True, False])

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        radius = int(self.get(fixed, "radius"))
        metric = str(self.get(fixed, "metric"))
        wrap = bool(self.get(fixed, "wrap"))
        offsets = _disc_offsets(radius, metric)
        dy, dx = _arrays(offsets)
        return NeighborhoodSpec(
            name=f"radius-{metric}-{radius}",
            dy=dy,
            dx=dx,
            inner_dy=dy,
            inner_dx=dx,
            wrap=wrap,
            max_count=len(offsets),
            inner_max_count=len(offsets),
        )


class TwoRings(_BaseNeighborhood):
    id = "neighborhood.two_rings"
    label = "два радиальных кольца"

    def declare(self, ns: Namespace, view) -> None:
        ns.range("inner_radius", 1, 3)
        ns.range("outer_radius", 3, 6)
        ns.choice("metric", ["euclidean", "chebyshev"])
        ns.choice("wrap", [True, False])

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        inner_radius = int(self.get(fixed, "inner_radius"))
        outer_radius = int(self.get(fixed, "outer_radius"))
        metric = str(self.get(fixed, "metric"))
        wrap = bool(self.get(fixed, "wrap"))
        inner_offsets = _disc_offsets(inner_radius, metric)
        outer_all = _disc_offsets(outer_radius, metric)
        inner_set = set(inner_offsets)
        outer_offsets = [offset for offset in outer_all if offset not in inner_set]
        inner_dy, inner_dx = _arrays(inner_offsets)
        outer_dy, outer_dx = _arrays(outer_offsets)
        return NeighborhoodSpec(
            name=f"two-rings-{metric}-{inner_radius}-{outer_radius}",
            dy=outer_dy,
            dx=outer_dx,
            inner_dy=inner_dy,
            inner_dx=inner_dx,
            wrap=wrap,
            max_count=len(outer_offsets),
            inner_max_count=len(inner_offsets),
        )


ALL_NEIGHBORHOODS = [Moore, RadiusDisc, TwoRings]

__all__ = ["ALL_NEIGHBORHOODS", "Moore", "RadiusDisc", "TwoRings"]
