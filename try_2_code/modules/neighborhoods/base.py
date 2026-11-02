"""Общая часть модулей окрестности.

Окрестность описывается списком смещений ``(di, dj)``. Это данные, а не код:
благодаря этому одно numba-ядро правила работает с любой окрестностью,
перекомпиляция при смене радиуса/формы не нужна.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import (
    Capability,
    CapabilityKind,
    Requirement,
    at_least,
    capability,
)
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace

COMMON_REQUIRES: tuple[Requirement, ...] = (
    Requirement(CapabilityKind.TWO_D),
    Requirement(CapabilityKind.DISCRETE),
    Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
    Requirement(CapabilityKind.ZERO_STATE),
)


@lru_cache(maxsize=1024)
def offsets_from_predicate(rmax: int, kind: str, p1: int, p2: int) -> tuple[tuple[int, int], ...]:
    """Смещения окрестности, кэшируются по параметрам.

    ``kind``: ``square`` (Чебышёв), ``radius`` (Евклид), ``ring`` (кольцо).
    """

    cells: list[tuple[int, int]] = []
    for di in range(-rmax, rmax + 1):
        for dj in range(-rmax, rmax + 1):
            if di == 0 and dj == 0:
                continue
            if kind == "square":
                inside = max(abs(di), abs(dj)) <= p1
            elif kind == "radius":
                inside = di * di + dj * dj <= p1 * p1
            elif kind == "ring":
                distance_sq = di * di + dj * dj
                inside = p1 * p1 < distance_sq <= p2 * p2
            else:  # pragma: no cover - защита от опечатки
                raise ValueError(f"Неизвестный вид окрестности: {kind!r}")
            if inside:
                cells.append((di, dj))
    return tuple(cells)


class NeighborhoodRuntime:
    """Готовые смещения для правила итерации.

    ``rings`` — необязательные именованные кольца (см. ``neighborhood.two_rings``):
    правило может считать внутреннее и внешнее кольцо отдельно.
    """

    __slots__ = ("id", "di", "dj", "wrap", "rings")

    def __init__(
        self,
        module_id: str,
        di: np.ndarray,
        dj: np.ndarray,
        wrap: bool,
        rings: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.id = module_id
        self.di = di
        self.dj = dj
        self.wrap = wrap
        self.rings = dict(rings or {})

    @property
    def max_count(self) -> int:
        return int(self.di.size)

    def __repr__(self) -> str:
        boundary = "wrap" if self.wrap else "clip"
        rings = f", rings={list(self.rings)}" if self.rings else ""
        return (
            f"Neighborhood({self.id}, cells={self.max_count}, "
            f"boundary={boundary}{rings})"
        )


class NeighborhoodModule(Module):
    """Базовый класс окрестности: объявляет границу и производные значения."""

    role = Role.NEIGHBORHOOD
    requires = COMMON_REQUIRES
    derived = ("max_count", "extent")

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        namespace.choose("boundary", ("wrap", "clip"))

    # --- параметры окрестности ------------------------------------------
    def offsets(self, fixed: Fixed) -> tuple[tuple[int, int], ...]:
        raise NotImplementedError

    def _extent(self, offsets: tuple[tuple[int, int], ...]) -> int:
        return max((max(abs(di), abs(dj)) for di, dj in offsets), default=0)

    # --- протокол модуля -------------------------------------------------
    def derive(self, fixed: Fixed) -> dict[str, int]:
        offsets = self.offsets(fixed)
        return {"max_count": len(offsets), "extent": self._extent(offsets)}

    def capabilities(self, fixed: Fixed) -> tuple[Capability, ...]:
        offsets = self.offsets(fixed)
        return (
            capability(
                CapabilityKind.NEIGHBORHOOD_COUNT,
                max_count=len(offsets),
                extent=self._extent(offsets),
            ),
        )

    def build(self, fixed: Fixed, view: BuildView) -> NeighborhoodRuntime:
        offsets = self.offsets(fixed)
        di = np.fromiter((cell[0] for cell in offsets), dtype=np.int64, count=len(offsets))
        dj = np.fromiter((cell[1] for cell in offsets), dtype=np.int64, count=len(offsets))
        return NeighborhoodRuntime(
            module_id=self.id,
            di=di,
            dj=dj,
            wrap=view.value("boundary") == "wrap",
        )
