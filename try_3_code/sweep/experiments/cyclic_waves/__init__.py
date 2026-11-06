"""Эксперимент «циклические волны».

Итерация ``step_cyclic`` лежит в этом же файле. Клетка переходит в следующую
фазу, если рядом достаточно соседей этой фазы; верхняя фаза сбрасывается в
ноль. Отбор требует, чтобы все фазы были представлены, а покрытие не
вырождалось в сплошную заливку.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ...experiment import Experiment
from ...interfaces import CapabilityKind, Requirement, ge
from ...kernels import JIT_KWARGS
from ...neighbors.counts import count_matching
from ...module import Role
from ...modules.data import Grid2D
from ...modules.filters import ClustersFilter, DensityFilter, DynamicsFilter, PhasesFilter
from ...neighbors import Moore, RadiusDisc
from ...modules.randomizers import SymmetricFill, UniformFill
from .visualizer import CyclicVisualizer


@njit(**JIT_KWARGS)
def step_cyclic(src, dst, dy, dx, wrap, zero, states, threshold, upper_resets):
    """Фаза растёт, если соседей следующей фазы не меньше порога.

    Соседи считаются общим счётчиком :func:`sweep.neighbors.counts.count_matching`:
    сравниваемое состояние — не константа, а «следующая фаза» каждой клетки.
    """

    height, width = src.shape
    last = states - 1
    upcoming = np.empty((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            nxt = src[y, x] + 1
            if nxt >= states:
                nxt = 0
            upcoming[y, x] = nxt
    counts = count_matching(src, dy, dx, wrap, upcoming)
    for y in range(height):
        for x in range(width):
            state = src[y, x]
            if state == last and upper_resets:
                dst[y, x] = zero
            elif counts[y, x] >= threshold:
                dst[y, x] = upcoming[y, x]
            else:
                dst[y, x] = state


def declare_rule(ns, view) -> None:
    max_count = view.peer(Role.NEIGHBORHOOD).derived("max_count")
    ns.range("threshold", 1, max_count + 1)
    ns.choice("upper_resets", [True, False])


def bind_rule(fixed, view) -> tuple:
    spec = view.neighborhood()
    return (
        spec.dy,
        spec.dx,
        spec.wrap_flag,
        int(fixed["grid2d.zero"]),
        int(fixed["grid2d.states"]),
        int(fixed["rule.cyclic.threshold"]),
        1 if bool(fixed["rule.cyclic.upper_resets"]) else 0,
    )


REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
    Requirement(CapabilityKind.MULTI_STATE, (ge("states", 3),), "multi_state (states >= 3)"),
    Requirement(CapabilityKind.NEIGHBORHOOD, (ge("max_count", 1),), "neighborhood.count"),
)


def build_cyclic_waves() -> Experiment:
    experiment = Experiment(
        "cyclic_waves",
        description="циклические фазы: k состояний и порог по следующей фазе",
        stop_on_extinct=True,
    )

    experiment.select(Role.DATA, Grid2D())
    experiment.select(Role.RANDOMIZER, [UniformFill(), SymmetricFill()])
    experiment.select(Role.NEIGHBORHOOD, [Moore(), RadiusDisc()])
    experiment.iterate(
        step_cyclic,
        module_id="rule.cyclic",
        label="циклические фазы (код в эксперименте)",
        requires=REQUIRES,
        declare=declare_rule,
        bind=bind_rule,
    )

    experiment.constrain(
        lambda values: values["grid2d.width"] * values["grid2d.height"] <= 96 * 96,
        ["grid2d.width", "grid2d.height"],
        name="cells_budget",
    )

    density = DensityFilter(min_fill=0.05, max_fill=0.95)
    dynamics = DynamicsFilter(min_change=0.001, max_change=0.90)
    phases = PhasesFilter(min_share=0.1)
    clusters = ClustersFilter(min_components=1, min_mean_size=2.0, max_largest_share=0.95)

    warmup = experiment.range("phase.warmup", 2, 6)
    observe = experiment.range("phase.observe", 8, 14)

    experiment.phase("warmup", warmup, filters=[density], frequency="end")
    experiment.phase("observe", observe, filters=[density, dynamics, phases, clusters], frequency="each")

    experiment.visualizer = CyclicVisualizer()
    return experiment


__all__ = ["bind_rule", "build_cyclic_waves", "declare_rule", "step_cyclic"]
