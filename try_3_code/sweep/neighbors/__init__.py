"""Соседство и подсчёт соседей.

Пакет разделён по файлам:

* ``interfaces`` — протоколы ``Lattice`` и ``Neighborhood``;
* ``geometry`` — смещения для 2D и 3D (диск, кольцо, окно);
* ``counts`` — numba-счётчики соседей (2D и 3D);
* ``lattices/`` — решётки ``Dim2`` и ``Dim3``;
* ``base`` — общая обёртка ``BaseNeighborhood`` и ``SquareWindow``;
* ``moore``, ``radius``, ``two_rings``, ``square5x5``, ``square9x9``,
  ``duel_rings`` — по файлу на окрестность.

Новая окрестность — это новый файл с объявлением осей и вызовом
``make_spec``; движок, сборка и эксперименты не меняются.

Пример 3D-окрестности (заготовка для будущего)::

    from .base import BaseNeighborhood
    from .lattices import Dim3

    class Cube(BaseNeighborhood):
        id = "neighborhood.cube"
        label = "куб 3×3×3"
        dimensions = 3
        lattice = Dim3()

        def declare(self, ns, view):
            ns.range("radius", 1, 2)
            ns.choice("wrap", [True, False])

        def spec(self, fixed):
            radius = int(self.get(fixed, "radius"))
            offsets = self.lattice.offsets(radius, "chebyshev")
            return self.make_spec(f"cube{radius}", offsets, offsets, bool(self.get(fixed, "wrap")))
"""

from __future__ import annotations

from . import counts, geometry, lattices
from .base import BaseNeighborhood, SquareWindow
from .duel_rings import DuelRings
from .interfaces import Lattice, Neighborhood, Offsets
from .lattices import Dim2, Dim3
from .moore import Moore
from .radius import RadiusDisc
from .square5x5 import Square5x5
from .square9x9 import Square9x9
from .two_rings import TwoRings

#: Все окрестности в порядке полного перебора.
ALL_NEIGHBORHOODS = [Moore, RadiusDisc, TwoRings, Square5x5, Square9x9, DuelRings]

__all__ = [
    "ALL_NEIGHBORHOODS",
    "BaseNeighborhood",
    "Dim2",
    "Dim3",
    "DuelRings",
    "Lattice",
    "Moore",
    "Neighborhood",
    "Offsets",
    "RadiusDisc",
    "Square5x5",
    "Square9x9",
    "SquareWindow",
    "TwoRings",
    "counts",
    "geometry",
    "lattices",
]
