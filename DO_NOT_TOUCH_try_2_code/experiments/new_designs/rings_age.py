"""Эксперимент «Кольца с возрастом» — явный перебор через experiment.space.range().

Сравнение со старым two_rings:
- Параметры перебора (width, height, states) задаются через experiment.space.range()
- Вся ответственность за диапазоны liegt здесь, в коде эксперимента,
  а не в module data discrete2d.py
- GridDef из simple_grid используются для создания сеток, но диапазоны
  перебора задаются здесь.
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules.data.simple_grid import GridDef


def build_experiment() -> Experiment:
    experiment = Experiment(
        "rings_age",
        "Кольца с возрастом: диапазоны перебора через experiment.space.range.",
    )

    # ============================================================
    # 1. ПАРАМЕТРЫ ПЕРЕБОРА — через experiment.space.range()
    # ============================================================
    # width: 16, 48, 80, 112 (step 32)
    # height: от width до 129, step 32
    # states: 2..6
    # ============================================================
    experiment.space.range("data.width", 64, 64, 32)
    experiment.space.range("data.height", 64, 64, 32)
    experiment.space.range("data.states", 3, 7, 1)

    # Constraint: width >= 32 (как в старом two_rings, чтобы убирать малые сетки)
    experiment.space.constrain(
        lambda values: values["data.width"] >= 32,
        ["data.width"],
        name="rings_age.width_fits_rings",
    )

    # ============================================================
    # 2. РОЛИ И МОДУЛИ
    # ============================================================
    experiment.select(Role.RANDOMIZER, [UniformRandomizer()])
    # DATA role не нужен — grids создаются динамически
    from modules.neighborhoods import TwoRingNeighborhood
    experiment.select(Role.NEIGHBORHOOD, [TwoRingNeighborhood()])
    experiment.select(Role.RULE, [AgingFortressRule()])

    experiment.phase("default", 10)

    return experiment


# def make_grids_from_ranges():
#     """Собрать все GridDef на основе определенных ranges."""
#     widths = [16, 48, 80, 112]
#     heights_map = {
#         16: [16, 48, 80, 112],
#         48: [48, 80, 112],
#         80: [80, 112],
#         112: [112],
#     }
#     states = [2, 3, 4, 5, 6]

#     result = []
#     for w in widths:
#         for h in heights_map.get(w, [w]):
#             for s in states:
#                 result.append(
#                     GridDef(width=w, height=h, states=s, zero_state=0)
#                 )
#     return result