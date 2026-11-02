"""Фильтр периода: автомат не должен колебаться с длинным циклом.

Каждая итерация запоминает маску «живая/мёртвая». Как только маска совпала с
одной из прошлых, известен период. Период больше ``max_period`` — отказ:
статичная картинка (период 1) и простые двухтактные структуры допустимы,
длинные циклы — это то самое мелькание.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import FilterVerdict

from .metrics import live_mask


class PeriodFilter(Module):
    id = "filter.period"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(self, *, max_period: int = 2, window: int = 16) -> None:
        self.criteria = {"max_period": int(max_period), "window": int(window)}

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        for name, value in self.criteria.items():
            namespace.value(name, value)

    def build(self, fixed: Fixed, view: BuildView) -> "PeriodRuntime":
        return PeriodRuntime(
            max_period=int(view.value("max_period")),
            window=int(view.value("window")),
        )


class PeriodRuntime:
    __slots__ = ("max_period", "window", "step", "seen", "order")

    def __init__(self, max_period: int, window: int) -> None:
        self.max_period = max_period
        self.window = max(2, window)
        self.step = 0
        self.seen: dict[bytes, int] = {}
        self.order: list[bytes] = []

    def check(self, grid) -> FilterVerdict:
        current = live_mask(grid)
        key = np.packbits(current).tobytes()
        step = self.step
        self.step += 1
        previous = self.seen.get(key)
        if previous is not None:
            period = step - previous
            metrics = {"period": float(period)}
            if period > self.max_period:
                return FilterVerdict(
                    False,
                    f"цикл длины {period} > {self.max_period}",
                    metrics,
                )
            return FilterVerdict(True, metrics=metrics)

        self.seen[key] = step
        self.order.append(key)
        if len(self.order) > self.window:
            self.seen.pop(self.order.pop(0), None)
        return FilterVerdict(True, metrics={"period": 0.0})
