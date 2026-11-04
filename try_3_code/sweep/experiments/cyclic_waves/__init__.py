"""Эксперимент «циклические волны».

Итерация ``step_cyclic`` лежит в этом же файле. Клетка переходит в следующую
фазу, если рядом достаточно соседей этой фазы; верхняя фаза сбрасывается в
ноль. Отбор требует, чтобы все фазы были представлены, а покрытие не
вырождалось в сплошную заливку.
"""

from __future__ import annotations

from numba import njit

from ...experiment import Experiment
from ...interfaces import CapabilityKind, Requirement, ge
from ...kernels import JIT_KWARGS
from ...module import Role
from ...modules.data import Grid2D
from ...modules.filters import ClustersFilter, DensityFilter, PhasesFilter
from ...modules.neighborhoods import Moore, RadiusDisc
from ...modules.randomizers import SymmetricFill, UniformFill
from .visualizer import CyclicVisualizer


@njit(**JIT_KWARGS)
def step_cyclic(src, dst, dy, dx, wrap, zero, states, threshold, upper_resets):
    """Фаза растёт, если соседей следующей фазы не меньше порога."""

    height, width = src.shape
    total = dy.size
    last = states - 1
    for y in range(height):
        for x in range(width):
            state = src[y, x]
            nxt = state + 1
            if nxt >= states:
                nxt = 0
            count = 0
            for k in range(total):
                yy = y + dy[k]
                xx = x + dx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if src[yy, xx] == nxt:
                    count += 1
            if state == last and upper_resets:
                dst[y, x] = zero
            elif count >= threshold:
                dst[y, x] = nxt
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
    phases = PhasesFilter(min_share=0.02)
    clusters = ClustersFilter(min_components=1, min_mean_size=3.0, max_largest_share=0.95)

    warmup = experiment.range("phase.warmup", 2, 6)
    observe = experiment.range("phase.observe", 8, 14)

    experiment.phase("warmup", warmup, filters=[density], frequency="end")
    experiment.phase("observe", observe, filters=[density, phases, clusters], frequency="each")

    experiment.visualizer = CyclicVisualizer()
    return experiment


__all__ = ["bind_rule", "build_cyclic_waves", "declare_rule", "step_cyclic"]
