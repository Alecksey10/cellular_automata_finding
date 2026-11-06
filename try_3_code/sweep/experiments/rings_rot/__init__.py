"""Эксперимент «два кольца с гниением».

Итерация написана прямо здесь, включая детерминированную случайность гниения:
``step_rot`` получает ``iteration`` и ``seed`` и не зависит от глобального rng.
Правило различает внутреннее и внешнее кольцо: рождение смотрит на внутреннее,
выживание — на сумму. Живая клетка копит прочность, а на последнем уровне
медленно гниёт.
"""

from __future__ import annotations

from numba import njit

from ...experiment import Experiment
from ...interfaces import CapabilityKind, Requirement, ge
from ...kernels import JIT_KWARGS
from ...module import Role
from ...modules.data import Grid2D
from ...modules.filters import ClustersFilter, DensityFilter, StabilityFilter
from ...neighbors import Moore, RadiusDisc, TwoRings
from ...modules.randomizers import ClusteredFill, SymmetricFill, UniformFill
from .visualizer import RotAgingVisualizer


@njit(inline="always")
def _rand01(y, x, iteration, seed):
    """Детерминированное число [0, 1) из координат, итерации и seed."""

    mixed = (y * 73856093) ^ (x * 19349663) ^ (iteration * 83492791) ^ ((seed % 100000) * 2654435)
    mixed = (mixed * 1103515245 + 12345) & 0x7FFFFFFF
    mixed = (mixed * 1103515245 + 12345) & 0x7FFFFFFF
    return mixed / 2147483648.0


@njit(**JIT_KWARGS)
def step_rot(
    src,
    dst,
    iteration,
    seed,
    inner_dy,
    inner_dx,
    outer_dy,
    outer_dx,
    wrap,
    zero,
    max_state,
    born_inner,
    born_total,
    surv_lo,
    surv_hi,
    rot_prob,
):
    """Рождение по внутреннему кольцу, выживание по сумме, гниение сверху."""

    height, width = src.shape
    inner_total = inner_dy.size
    outer_total = outer_dy.size
    for y in range(height):
        for x in range(width):
            inner = 0
            outer = 0
            for k in range(outer_total):
                yy = y + outer_dy[k]
                xx = x + outer_dx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if src[yy, xx] != zero:
                    outer += 1
            for k in range(inner_total):
                yy = y + inner_dy[k]
                xx = x + inner_dx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if src[yy, xx] != zero:
                    inner += 1
            state = src[y, x]
            if state == zero:
                if inner >= born_inner and outer >= born_total:
                    dst[y, x] = 1
                else:
                    dst[y, x] = zero
                continue
            if surv_lo <= outer <= surv_hi:
                nxt = state + 1
                if nxt > max_state:
                    nxt = max_state
            else:
                nxt = state - 1
            if nxt < zero:
                nxt = zero
            if state == max_state and _rand01(y, x, iteration, seed) < rot_prob:
                nxt = zero
            dst[y, x] = nxt


def declare_rule(ns, view) -> None:
    max_count = view.peer(Role.NEIGHBORHOOD).derived("max_count")
    inner_max = view.peer(Role.NEIGHBORHOOD).derived("inner_max_count")
    ns.range("born_inner", 1, inner_max + 1)
    ns.range("born_total", 1, max_count + 1)
    ns.range("surv_lo", 1, max_count + 1)
    ns.range("surv_hi", ns.handle("surv_lo"), max_count + 1)
    ns.choice("rot", [0.02, 0.05, 0.1, 0.2])


def bind_rule(fixed, view) -> tuple:
    spec = view.neighborhood()
    return (
        spec.inner_dy,
        spec.inner_dx,
        spec.dy,
        spec.dx,
        spec.wrap_flag,
        int(fixed["grid2d.zero"]),
        int(fixed["grid2d.states"]) - 1,
        int(fixed["rule.rot_aging.born_inner"]),
        int(fixed["rule.rot_aging.born_total"]),
        int(fixed["rule.rot_aging.surv_lo"]),
        int(fixed["rule.rot_aging.surv_hi"]),
        float(fixed["rule.rot_aging.rot"]),
    )


REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
    Requirement(CapabilityKind.MULTI_STATE, (ge("states", 3),), "multi_state (states >= 3)"),
    Requirement(
        CapabilityKind.NEIGHBORHOOD,
        (ge("inner_max_count", 1),),
        "neighborhood.inner (inner_max_count >= 1)",
    ),
)


def build_rings_rot() -> Experiment:
    experiment = Experiment(
        "rings_rot",
        description="рождение по внутреннему кольцу, выживание по сумме, гниение",
        stop_on_extinct=True,
    )

    experiment.select(Role.DATA, Grid2D())
    experiment.select(Role.RANDOMIZER, [UniformFill(), ClusteredFill(), SymmetricFill()])
    experiment.select(Role.NEIGHBORHOOD, [TwoRings(), RadiusDisc(), Moore()])
    experiment.iterate(
        step_rot,
        module_id="rule.rot_aging",
        label="два кольца + гниение (код в эксперименте)",
        requires=REQUIRES,
        declare=declare_rule,
        bind=bind_rule,
        with_control=True,
    )

    experiment.constrain(
        lambda values: values["grid2d.width"] * values["grid2d.height"] <= 96 * 96,
        ["grid2d.width", "grid2d.height"],
        name="cells_budget",
    )

    density = DensityFilter(min_fill=0.03, max_fill=0.95)
    stability = StabilityFilter(max_change=0.28, forbid_period2=True)
    clusters = ClustersFilter(min_components=2, min_mean_size=5.0, max_largest_share=0.75)

    warmup = experiment.range("phase.warmup", 3, 7)
    observe = experiment.range("phase.observe", 8, 14)

    experiment.phase("warmup", warmup, filters=[density], frequency="end")
    experiment.phase("observe", observe, filters=[density, stability, clusters], frequency="each")

    experiment.visualizer = RotAgingVisualizer()
    return experiment


__all__ = ["bind_rule", "build_rings_rot", "declare_rule", "step_rot"]
