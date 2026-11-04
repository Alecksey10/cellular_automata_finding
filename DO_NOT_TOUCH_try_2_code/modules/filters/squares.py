"""Фильтр: число однородных блоков 2x2 (аналог CountOfSquares из постановки)."""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import FilterVerdict


class MonochromeSquaresFilter(Module):
    id = "filter.squares"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
    )

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        namespace.range("max_blocks", 40, 141, 20)

    def build(self, fixed: Fixed, view: BuildView) -> "SquaresRuntime":
        return SquaresRuntime(max_blocks=int(view.value("max_blocks")))


class SquaresRuntime:
    __slots__ = ("max_blocks",)

    def __init__(self, max_blocks: int) -> None:
        self.max_blocks = max_blocks

    def check(self, grid) -> FilterVerdict:
        array = grid.array
        if array.shape[0] < 2 or array.shape[1] < 2:
            return FilterVerdict(True, metrics={"blocks": 0})
        blocks = (
            (array[:-1, :-1] == array[1:, :-1])
            & (array[:-1, :-1] == array[:-1, 1:])
            & (array[:-1, :-1] == array[1:, 1:])
        )
        count = int(np.count_nonzero(blocks))
        metrics = {"blocks": count}
        if count > self.max_blocks:
            return FilterVerdict(
                False, f"однородных блоков 2x2 {count} > {self.max_blocks}", metrics
            )
        return FilterVerdict(True, metrics=metrics)
