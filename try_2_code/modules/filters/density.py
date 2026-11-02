"""Фильтр: доля живых клеток не ниже порога."""

from __future__ import annotations

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import FilterVerdict


class DensityFilter(Module):
    id = "filter.density"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        namespace.range("min_fill", 0.3, 0.9, 0.1)

    def build(self, fixed: Fixed, view: BuildView) -> "DensityRuntime":
        return DensityRuntime(min_fill=float(view.value("min_fill")))


class DensityRuntime:
    __slots__ = ("min_fill",)

    def __init__(self, min_fill: float) -> None:
        self.min_fill = min_fill

    def check(self, grid) -> FilterVerdict:
        fill = grid.live_fraction()
        metrics = {"fill": round(fill, 6)}
        if fill < self.min_fill:
            return FilterVerdict(False, f"заполнение {fill:.3f} < {self.min_fill:.2f}", metrics)
        return FilterVerdict(True, metrics=metrics)
