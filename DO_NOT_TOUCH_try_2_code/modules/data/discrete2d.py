"""Данные: двумерная дискретная сетка с конечным числом состояний."""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import Capability, CapabilityKind, capability
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace


class Grid2D:
    """Изменяемая сетка состояний.

    Модули работают с ``array`` напрямую (numba), поэтому обёртка тонкая.
    """

    __slots__ = ("array", "states", "zero_state")

    def __init__(self, array: np.ndarray, states: int, zero_state: int) -> None:
        self.array = array
        self.states = states
        self.zero_state = zero_state

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape  # type: ignore[return-value]

    def like(self) -> "Grid2D":
        return Grid2D(np.empty_like(self.array), self.states, self.zero_state)

    def live_fraction(self) -> float:
        return float(np.count_nonzero(self.array != self.zero_state)) / float(self.array.size)

    def count_of(self, state: int) -> int:
        return int(np.count_nonzero(self.array == state))

    def __repr__(self) -> str:
        return f"Grid2D(shape={self.shape}, states={self.states}, zero={self.zero_state})"


class DiscreteGrid2DFactory:
    """Создаёт сетки нужного размера и типа; живёт один тест."""

    def __init__(self, height: int, width: int, states: int, zero_state: int) -> None:
        self.height = int(height)
        self.width = int(width)
        self.states = int(states)
        self.zero_state = int(zero_state)

    def create(self) -> Grid2D:
        array = np.full((self.height, self.width), self.zero_state, dtype=np.int32)
        return Grid2D(array, self.states, self.zero_state)

    def like(self, grid: Grid2D) -> Grid2D:
        return Grid2D(np.empty_like(grid.array), grid.states, grid.zero_state)

    def __repr__(self) -> str:
        return (
            f"DiscreteGrid2DFactory({self.height}x{self.width}, "
            f"states={self.states}, zero={self.zero_state})"
        )


class DiscreteGrid2D(Module):
    """Двумерная дискретная сетка, параметры которой перебираются."""

    id = "data.discrete2d"
    role = Role.DATA

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        namespace.range("states", 2, 10, 1)
        width = namespace.range("width", 16, 128, 32)
        namespace.range("height", width, 129, 32)
        namespace.value("zero_state", 0)

    def capabilities(self, fixed: Fixed) -> tuple[Capability, ...]:
        return (
            capability(
                CapabilityKind.TWO_D,
                height=fixed["data.height"],
                width=fixed["data.width"],
            ),
            capability(CapabilityKind.DISCRETE),
            capability(CapabilityKind.MULTI_STATE, states=fixed["data.states"]),
            capability(CapabilityKind.ZERO_STATE, state=fixed["data.zero_state"]),
        )

    def build(self, fixed: Fixed, view: BuildView) -> DiscreteGrid2DFactory:
        return DiscreteGrid2DFactory(
            height=view.value("height"),
            width=view.value("width"),
            states=view.value("states"),
            zero_state=view.value("zero_state"),
        )
