"""Фильтры «полоса»: заполнение и изменение картинки за итерацию.

Оба фильтра отсекают два тривиальных исхода: пустую/залитую сетку и полную
неподвижность или, наоборот, мелькание. Пороги задаёт эксперимент.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import FilterVerdict

from .metrics import live_mask


class FillBandFilter(Module):
    """Заполнение (доля живых клеток) должно лежать в полосе."""

    id = "filter.fill_band"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(self, *, min_fill: float = 0.05, max_fill: float = 0.85) -> None:
        self.criteria = {"min_fill": min_fill, "max_fill": max_fill}

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        for name, value in self.criteria.items():
            namespace.value(name, value)

    def build(self, fixed: Fixed, view: BuildView) -> "FillBandRuntime":
        return FillBandRuntime(**{name: float(view.value(name)) for name in self.criteria})


class FillBandRuntime:
    __slots__ = ("min_fill", "max_fill")

    def __init__(self, min_fill: float, max_fill: float) -> None:
        self.min_fill = min_fill
        self.max_fill = max_fill

    def check(self, grid) -> FilterVerdict:
        fill = float(grid.live_fraction())
        metrics = {"fill": round(fill, 6)}
        if fill < self.min_fill:
            return FilterVerdict(
                False, f"слишком пусто: заполнение {fill:.3f} < {self.min_fill:.2f}", metrics
            )
        if fill > self.max_fill:
            return FilterVerdict(
                False, f"залито: заполнение {fill:.3f} > {self.max_fill:.2f}", metrics
            )
        return FilterVerdict(True, metrics=metrics)


class ActivityFilter(Module):
    """Доля изменившихся клеток за итерацию должна лежать в полосе.

    Сравниваются маски «живая/мёртвая»: смена фазы у многосостоянийного
    автомата — это активность, а не смерть клетки.
    """

    id = "filter.activity"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(self, *, min_change: float = 0.0, max_change: float = 0.15) -> None:
        self.criteria = {"min_change": min_change, "max_change": max_change}

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        for name, value in self.criteria.items():
            namespace.value(name, value)

    def build(self, fixed: Fixed, view: BuildView) -> "ActivityRuntime":
        return ActivityRuntime(**{name: float(view.value(name)) for name in self.criteria})


class ActivityRuntime:
    __slots__ = ("min_change", "max_change", "previous")

    def __init__(self, min_change: float, max_change: float) -> None:
        self.min_change = min_change
        self.max_change = max_change
        self.previous: np.ndarray | None = None

    def check(self, grid) -> FilterVerdict:
        current = live_mask(grid)
        if self.previous is None:
            self.previous = current
            return FilterVerdict(True, metrics={"change": 0.0})
        change = float(np.mean(current != self.previous))
        self.previous = current
        metrics = {"change": round(change, 6)}
        if change > self.max_change:
            return FilterVerdict(
                False,
                f"мелькание: изменилось {change:.3f} > {self.max_change:.3f}",
                metrics,
            )
        if change < self.min_change:
            return FilterVerdict(
                False,
                f"застыло: изменилось {change:.4f} < {self.min_change:.4f}",
                metrics,
            )
        return FilterVerdict(True, metrics=metrics)
