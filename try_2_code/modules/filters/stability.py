"""Фильтр «автомат не мелькает».

Мелькание — это появление/исчезновение клеток, а не их старение: поэтому
сравниваются маски «живая/мёртвая», иначе медленно взрослеющее облако
выглядело бы мельканием. Ловится и период 2 (классический мигающий автомат).
Рантайм фильтра живёт один тест, поэтому хранит две предыдущие маски.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import FilterVerdict


class StabilityFilter(Module):
    """Порог мелькания фиксируется экспериментом (см. CloudFilter)."""

    id = "filter.stability"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(self, *, max_change: float = 0.2) -> None:
        self.max_change = max_change

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        namespace.value("max_change", self.max_change)

    def build(self, fixed: Fixed, view: BuildView) -> "StabilityRuntime":
        return StabilityRuntime(max_change=float(view.value("max_change")))


class StabilityRuntime:
    __slots__ = ("max_change", "previous", "before_previous")

    def __init__(self, max_change: float) -> None:
        self.max_change = max_change
        self.previous: np.ndarray | None = None
        self.before_previous: np.ndarray | None = None

    def check(self, grid) -> FilterVerdict:
        current = np.asarray(grid.array != grid.zero_state)
        if self.previous is None:
            self.previous = current
            return FilterVerdict(True, metrics={"change": 0.0})

        change = float(np.mean(current != self.previous))
        metrics: dict[str, float] = {"change": round(change, 4)}

        if change > self.max_change:
            self._remember(current)
            return FilterVerdict(
                False,
                f"мелькание: появилось/исчезло {change:.2f} клеток > {self.max_change:.2f}",
                metrics,
            )

        if self.before_previous is not None:
            period_two = float(np.mean(current == self.before_previous))
            metrics["period2"] = round(period_two, 4)
            if change > 0.02 and period_two > 0.95:
                self._remember(current)
                return FilterVerdict(
                    False,
                    f"период 2: картинка повторяется на {period_two:.2f}, "
                    f"изменение {change:.2f}",
                    metrics,
                )

        self._remember(current)
        return FilterVerdict(True, metrics=metrics)

    def _remember(self, current: np.ndarray) -> None:
        self.before_previous = self.previous
        self.previous = current
