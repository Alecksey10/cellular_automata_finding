"""Эксперимент «жизнь»: B/S-пороги на 2D-сетке.

Итерация написана прямо здесь: ``step_life`` — обычная numba-функция, а
``experiment.iterate(...)`` связывает её с осями и рантаймом. Готовый
``RuleModule`` из общего каталога не нужен.

Перебираются окрестность, пороги рождения/выживания, плотность инициализации
и разрешение. Фильтры отбирают автоматы, которые не вымерли, не заполнили всё
и стабилизировались (статические фигуры и медленные осцилляции).
"""

from __future__ import annotations

from numba import njit

from ...experiment import Experiment
from ...interfaces import CapabilityKind, Requirement, eq, ge
from ...kernels import JIT_KWARGS
from ...module import Role
from ...modules.data import Grid2D
from ...modules.filters import ClustersFilter, DensityFilter, StabilityFilter
from ...modules.neighborhoods import Moore, RadiusDisc
from ...modules.randomizers import ClusteredFill, UniformFill
from .visualizer import LifeVisualizer


@njit(**JIT_KWARGS)
def step_life(src, dst, dy, dx, wrap, zero, born_lo, born_hi, surv_lo, surv_hi):
    """Одна итерация B/S: живые клетки — состояние 1, мёртвые — ``zero``."""

    height, width = src.shape
    total = dy.size
    for y in range(height):
        for x in range(width):
            neighbors = 0
            for k in range(total):
                yy = y + dy[k]
                xx = x + dx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if src[yy, xx] != zero:
                    neighbors += 1
            state = src[y, x]
            if state == zero:
                dst[y, x] = 1 if born_lo <= neighbors <= born_hi else zero
            else:
                dst[y, x] = 1 if surv_lo <= neighbors <= surv_hi else zero


def declare_rule(ns, view) -> None:
    """Оси порогов зависят от выбранной окрестности."""

    max_count = view.peer(Role.NEIGHBORHOOD).derived("max_count")
    ns.range("born_lo", 1, max_count + 1)
    ns.range("born_hi", ns.handle("born_lo"), max_count + 1)
    ns.range("surv_lo", 1, max_count + 1)
    ns.range("surv_hi", ns.handle("surv_lo"), max_count + 1)


def bind_rule(fixed, view) -> tuple:
    """Зафиксированные значения конфига -> аргументы numba-ядра."""

    spec = view.neighborhood()
    return (
        spec.dy,
        spec.dx,
        spec.wrap_flag,
        int(fixed["grid2d.zero"]),
        int(fixed["rule.life.born_lo"]),
        int(fixed["rule.life.born_hi"]),
        int(fixed["rule.life.surv_lo"]),
        int(fixed["rule.life.surv_hi"]),
    )


REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
    Requirement(CapabilityKind.MULTI_STATE, (eq("states", 2),), "bi_state (states == 2)"),
    Requirement(CapabilityKind.NEIGHBORHOOD, (ge("max_count", 1),), "neighborhood.count"),
)


def build_life() -> Experiment:
    experiment = Experiment(
        "life",
        description="B/S-пороги: рождение и выживание по числу соседей",
        stop_on_extinct=True,
    )

    experiment.select(Role.DATA, Grid2D())
    experiment.select(Role.RANDOMIZER, [UniformFill(), ClusteredFill()])
    experiment.select(Role.NEIGHBORHOOD, [Moore(), RadiusDisc()])
    experiment.iterate(
        step_life,
        module_id="rule.life",
        label="B/S-пороги (код в эксперименте)",
        requires=REQUIRES,
        declare=declare_rule,
        bind=bind_rule,
    )

    experiment.constrain(
        lambda values: values["grid2d.width"] * values["grid2d.height"] <= 96 * 96,
        ["grid2d.width", "grid2d.height"],
        name="cells_budget",
    )

    density = DensityFilter(min_fill=0.01, max_fill=0.9)
    stability = StabilityFilter(max_change=0.3, forbid_period2=True)
    clusters = ClustersFilter(min_components=1, min_mean_size=4.0, max_largest_share=0.85)

    warmup = experiment.range("phase.warmup", 0, 7, 3)
    observe = experiment.range("phase.observe", 10, 31, 10)

    experiment.phase("warmup", warmup, filters=[density], frequency="end")
    experiment.phase("observe", observe, filters=[density, stability, clusters], frequency="each")

    experiment.visualizer = LifeVisualizer()
    return experiment


__all__ = ["bind_rule", "build_life", "declare_rule", "step_life"]
