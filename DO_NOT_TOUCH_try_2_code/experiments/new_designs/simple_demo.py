"""Эксперимент «Простая сетка» — минимальный пример с experiment.space_range().

Показывает самый простой случай: задача диапазонов через experiment.space.range()
вместо определения их в module data.

Старый эксперимент two_rings по-прежнему работает через DiscreteGrid2D.
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules.data.simple_grid import GridDef


def build_experiment() -> Experiment:
    experiment = Experiment(
        "simple_grid_demo",
        "Минимальный эксперимент: диапазоны через experiment.space.range.",
    )

    # ============================================================
    # 1. ПАРАМЕТРЫ ПЕРЕБОРА — через experiment.space.range()
    # ============================================================
    experiment.space.range("data.width", 16, 130, 32)   # 16, 48, 80, 112
    experiment.space.range("data.height", 16, 130, 32)  # аналогично
    experiment.space.range("data.states", 2, 11, 1)     # 2..10

    # Constraint: width >= 32
    experiment.space.constrain(
        lambda values: values["data.width"] >= 32,
        ["data.width"],
        name="simple_demo.width_fits_rings",
    )

    # ============================================================
    # 2. РОЛИ И МОДУЛИ
    # ============================================================
    experiment.select(Role.RANDOMIZER, [UniformRandomizer()])
    # DATA role не нужен — grids создаются динамически из ranges
    from modules.neighborhoods import TwoRingNeighborhood
    experiment.select(Role.NEIGHBORHOOD, [TwoRingNeighborhood()])
    experiment.select(Role.RULE, [AgingFortressRule()])

    experiment.phase("default", 10)

    return experiment


def make_grids_from_ranges():
    """Собрать все GridDef на основе определенных ranges."""
    widths = [16, 48, 80, 112]
    heights = [16, 48, 80, 112]
    states = [2, 3, 4, 5, 6]

    result = []
    for w in widths:
        for h in heights:
            for s in states:
                result.append(
                    GridDef(width=w, height=h, states=s, zero_state=0)
                )
    return result