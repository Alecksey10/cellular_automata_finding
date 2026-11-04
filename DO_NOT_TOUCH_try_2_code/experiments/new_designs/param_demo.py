"""Эксперимент «Демонстрация explicit parameter sweeping».

Показывает, как задавать перебор размеров сетки явно в коде эксперимента,
используя experiment.space.range() вместо того, чтобы писать такие диапазоны
в module data (discrete2d.py и т.п.).

Архитектура:
- Диапазоны width, height, states задаются через experiment.space.range()
- GridDef из simple_grid используется для создания конкретных_grid
- Отсутствие соседних фильтров для упрощения демонстрации
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules.data.simple_grid import GridDef, create_grid
from modules.rules.aging import AgingFortressRule
from modules.neighborhoods.ring import RingNeighborhood
from modules.neighborhoods.ring import RingNeighborhood
from modules.neighborhoods.ring import RingNeighborhood
from modules import UniformRandomizer


def build_experiment() -> Experiment:
    experiment = Experiment(
        "param_demo",
        "Демонстрация explicit parameter sweeping через experiment.space.range.",
    )

    # ============================================================
    # 1. ПАРАМЕТРЫ ПЕРЕБОРА — через experiment.space.range()
    # ============================================================
    experiment.space.range("data.width", 16, 130, 32)   # 16, 48, 80, 112
    experiment.space.range("data.height", 16, 130, 32)  # 16, 48, 80, 112
    experiment.space.range("data.states", 2, 11, 1)     # 2..10

    # Constraints (ограничения) — тоже здесь, в коде эксперимента
    experiment.space.constrain(
        lambda values: values["data.width"] >= 32,
        ["data.width"],
        name="param_demo.width_fits_rings",
    )

    # ============================================================
    # 2. РОЛИ И МОДУЛИ
    # ============================================================
    experiment.select(Role.RANDOMIZER, [UniformRandomizer()])

    experiment.select(Role.NEIGHBORHOOD, [RingNeighborhood()])
    experiment.select(Role.RULE, [AgingFortressRule()])
    # DATA role не нужен — grids создаются динамически из ranges

    # Фаза по умолчанию
    # Хранятся значения phases в space
    warmup_val = experiment.space.value("stage.warmup") if "stage.warmup" in experiment.space else 10
    observe_val = experiment.space.value("stage.observe") if "stage.observe" in experiment.space else 10
    
    experiment.phase("default", warmup_val)

    return experiment


def make_grids_from_ranges():
    """Собрать GridDef на основе определенных ranges для перебора."""
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
