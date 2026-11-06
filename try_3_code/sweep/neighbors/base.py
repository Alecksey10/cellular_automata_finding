"""Общая обёртка окрестности: возможности и сборка ``NeighborhoodSpec``.

Конкретные окрестности отличаются только набором смещений и объявлением осей;
всё остальное (capabilities, массивы для numba, ``inner_*``) здесь.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..interfaces import Capability, CapabilityKind
from ..module import NeighborhoodModule, NeighborhoodSpec
from ..space import Namespace
from .geometry import offset_arrays
from .interfaces import Lattice, Offsets
from .lattices import Dim2


class BaseNeighborhood(NeighborhoodModule):
    """База окрестности: смещения -> ``NeighborhoodSpec``."""

    dimensions = 2
    lattice: Lattice = Dim2()

    def capabilities(self, fixed: Mapping[str, Any]) -> tuple[Capability, ...]:
        spec = self.spec(fixed)
        return (
            Capability(
                CapabilityKind.NEIGHBORHOOD,
                {
                    "max_count": spec.max_count,
                    "inner_max_count": spec.inner_max_count,
                    "wrap": spec.wrap,
                    "dims": spec.dims,
                },
                self.id,
            ),
        )

    def make_spec(
        self,
        name: str,
        offsets: Offsets,
        inner_offsets: Offsets,
        wrap: bool,
    ) -> NeighborhoodSpec:
        """Собрать spec из смещений, не думая про размерность."""

        dims = self.dimensions
        outer = offset_arrays(offsets, dims)
        inner = offset_arrays(inner_offsets, dims)
        extra: dict[str, Any] = {}
        if dims >= 3:
            extra["dz"] = outer[2]
            extra["inner_dz"] = inner[2]
        return NeighborhoodSpec(
            name=name,
            dy=outer[0],
            dx=outer[1],
            inner_dy=inner[0],
            inner_dx=inner[1],
            wrap=wrap,
            max_count=len(offsets),
            inner_max_count=len(inner_offsets),
            dims=dims,
            **extra,
        )


class SquareWindow(BaseNeighborhood):
    """Окно ``(2r+1)×(2r+1)`` с внутренним кольцом радиусом 1."""

    radius = 1

    def declare(self, ns: Namespace, view) -> None:
        ns.choice("wrap", [True, False])

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        wrap = bool(self.get(fixed, "wrap"))
        offsets = self.lattice.offsets(self.radius, "chebyshev")
        inner = self.lattice.offsets(1, "chebyshev")
        size = 2 * self.radius + 1
        return self.make_spec(f"square{size}x{size}", offsets, inner, wrap)


__all__ = ["BaseNeighborhood", "SquareWindow"]
