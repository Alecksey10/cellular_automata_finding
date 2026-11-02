"""Модули инициализации сетки."""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace

_REQUIRES = (
    Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
    Requirement(CapabilityKind.DISCRETE),
    Requirement(CapabilityKind.ZERO_STATE),
)


class UniformRandomizer(Module):
    """Все состояния равновероятны."""

    id = "randomizer.uniform"
    role = Role.RANDOMIZER
    requires = _REQUIRES

    def build(self, fixed: Fixed, view: BuildView) -> "UniformRuntime":
        return UniformRuntime()


class UniformRuntime:
    def fill(self, grid, rng: np.random.Generator) -> None:
        values = rng.integers(0, grid.states, size=grid.array.shape, dtype=np.int32)
        grid.array[...] = values


class SparseRandomizer(Module):
    """Разреженная инициализация: живые клетки с заданной плотностью."""

    id = "randomizer.sparse"
    role = Role.RANDOMIZER
    requires = _REQUIRES

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        namespace.range("density", 0.05, 0.65, 0.1)
        namespace.value("live_state", 1)

    def build(self, fixed: Fixed, view: BuildView) -> "SparseRuntime":
        density = float(view.value("density"))
        live_state = int(view.value("live_state"))
        if density <= 0 or density >= 1:
            raise ValueError("Плотность должна быть в (0, 1)")
        return SparseRuntime(density=density, live_state=live_state)


class SparseRuntime:
    __slots__ = ("density", "live_state")

    def __init__(self, density: float, live_state: int) -> None:
        self.density = density
        self.live_state = live_state

    def fill(self, grid, rng: np.random.Generator) -> None:
        mask = rng.random(grid.array.shape) < self.density
        grid.array[...] = grid.zero_state
        grid.array[mask] = self.live_state
