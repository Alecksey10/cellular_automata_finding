"""Окрестность «два кольца»: внутренний диск ``r1`` и внешнее кольцо ``(r1, r2]``.

Перебор: ``r1`` от 1 до 5, ``r2`` от 1 до 7; ограничение ``r2 > r1``
отбрасывает пустое внешнее кольцо. Диапазоны полуоткрытые, как ``range`` в
Python. Кольца считаются отдельно, поэтому правило может требовать локальную
поддержку от внутреннего кольца и учитывать внешнее.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import Capability, CapabilityKind, capability
from casweep.module import BuildView
from casweep.space import DeclareView, Namespace

from .base import NeighborhoodModule, NeighborhoodRuntime, offsets_from_predicate


@lru_cache(maxsize=256)
def two_ring_offsets(
    r1: int, r2: int
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Вернуть (объединение, внутреннее кольцо, внешнее кольцо)."""

    inner = offsets_from_predicate(r1, "radius", r1, r1)
    outer = offsets_from_predicate(r2, "ring", r1, r2)
    return inner + outer, inner, outer


def _arrays(offsets: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray]:
    di = np.fromiter((cell[0] for cell in offsets), dtype=np.int64, count=len(offsets))
    dj = np.fromiter((cell[1] for cell in offsets), dtype=np.int64, count=len(offsets))
    return di, dj


class TwoRingNeighborhood(NeighborhoodModule):
    id = "neighborhood.two_rings"
    derived = ("max_count", "extent", "max_inner", "max_outer")

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        super().declare(namespace, view)  # boundary: wrap/clip
        namespace.range("r1", 1, 6, 1)  # 1..5
        namespace.range("r2", 1, 8, 1)  # 1..7, но должно быть больше r1
        namespace.constrain(
            lambda r1, r2: r2 > r1,
            "r1",
            "r2",
            name="neighborhood.two_rings.order",
        )

    def _rings(self, fixed: Fixed) -> tuple[tuple[tuple[int, int], ...], ...]:
        r1 = int(fixed["neighborhood.r1"])
        r2 = int(fixed["neighborhood.r2"])
        return two_ring_offsets(r1, r2)

    def offsets(self, fixed: Fixed) -> tuple[tuple[int, int], ...]:
        combined, _, _ = self._rings(fixed)
        return combined

    def derive(self, fixed: Fixed) -> dict[str, int]:
        combined, inner, outer = self._rings(fixed)
        return {
            "max_count": len(combined),
            "extent": self._extent(combined),
            "max_inner": len(inner),
            "max_outer": len(outer),
        }

    def capabilities(self, fixed: Fixed) -> tuple[Capability, ...]:
        derived = self.derive(fixed)
        return super().capabilities(fixed) + (
            capability(
                CapabilityKind.NEIGHBORHOOD_RINGS,
                rings=("inner", "outer"),
                max_inner=derived["max_inner"],
                max_outer=derived["max_outer"],
            ),
        )

    def build(self, fixed: Fixed, view: BuildView) -> NeighborhoodRuntime:
        combined, inner, outer = self._rings(fixed)
        di, dj = _arrays(combined)
        return NeighborhoodRuntime(
            module_id=self.id,
            di=di,
            dj=dj,
            wrap=view.value("boundary") == "wrap",
            rings={"inner": _arrays(inner), "outer": _arrays(outer)},
        )
