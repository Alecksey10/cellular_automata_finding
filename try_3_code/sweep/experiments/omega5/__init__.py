"""Эксперимент ``omega5``: 8 состояний, окно 5×5 и «хитрые» реакции.

Этот эксперимент намеренно отделяет три вещи друг от друга:

* **механизм счёта соседей** — как из 24 клеток окна 5×5 получить число:
  вся маска, внутреннее/внешнее кольцо, четверти, 8 направлений, крест,
  диагонали, строки/столбцы, шахматная маска, углы и т.п.;
* **категории** рождения, выживания и смерти — интервалы по этому числу;
* **особую реакцию** — одно из двадцати правил, где состояния ведут себя
  «вопреки» базовым порогам (например, 6 всегда стабильна, а 4 уничтожает
  все 3 в радиусе и сама переходит в 2).

Правило полностью описывается скалярными осями конфига. JSON-паспорт
(:func:`sweep.experiments.omega5.rule.rule_descriptor`) собирается из них и
показывается в галерее, поэтому автомат можно воспроизвести и нарисовать.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...experiment import Experiment
from ...interfaces import CapabilityKind, Requirement, eq, ge
from ...module import Role
from ...modules.filters import ClustersFilter, DensityFilter, PhasesFilter, StabilityFilter
from ...modules.neighborhoods import Square5x5
from ...modules.randomizers import ClusteredFill, SymmetricFill, UniformFill
from .data import OmegaGrid
from .kernel import step_omega
from .rule import (
    CATEGORIES,
    MECHANISMS,
    STATES,
    TRICKS,
    WINDOW_OFFSETS,
    category_interval,
    group_ids,
    mechanism_max,
    rule_descriptor,
    trick_code,
)
from .visualizer import OmegaVisualizer

PREFIX = "rule.omega5."


def declare_rule(ns, view) -> None:
    ns.choice("mechanism", list(MECHANISMS))
    ns.choice("birth", list(CATEGORIES))
    ns.choice("survival", list(CATEGORIES))
    ns.choice("death", list(CATEGORIES))
    ns.choice("birth_state", list(range(1, STATES)))
    ns.choice("trick", ["none", *TRICKS])


def _interval(fixed: Mapping[str, Any], kind: str, maximum: int) -> tuple[int, int]:
    return category_interval(str(fixed[PREFIX + kind]), maximum)


def bind_rule(fixed: Mapping[str, Any], view) -> tuple:
    spec = view.neighborhood()
    mechanism = str(fixed[PREFIX + "mechanism"])
    maximum = mechanism_max(mechanism)
    groups = group_ids(mechanism, zip(spec.dy.tolist(), spec.dx.tolist()))
    group = np.asarray(groups, dtype=np.int32)

    birth_lo, birth_hi = _interval(fixed, "birth", maximum)
    surv_lo, surv_hi = _interval(fixed, "survival", maximum)
    death_lo, death_hi = _interval(fixed, "death", maximum)

    return (
        spec.dy,
        spec.dx,
        group,
        spec.wrap_flag,
        int(fixed["grid2d.zero"]),
        int(fixed["grid2d.states"]),
        birth_lo,
        birth_hi,
        int(fixed[PREFIX + "birth_state"]),
        surv_lo,
        surv_hi,
        death_lo,
        death_hi,
        trick_code(str(fixed[PREFIX + "trick"])),
        MECHANISMS[mechanism].code,
    )


REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
    Requirement(CapabilityKind.MULTI_STATE, (ge("states", STATES),), f"states >= {STATES}"),
    Requirement(
        CapabilityKind.NEIGHBORHOOD,
        (eq("max_count", 24),),
        "5x5 neighborhood (max_count == 24)",
    ),
)


def build_omega5() -> Experiment:
    experiment = Experiment(
        "omega5",
        description="8 состояний, окно 5×5, категории B/S/D и 20 особых реакций",
        stop_on_extinct=True,
    )

    experiment.select(Role.DATA, OmegaGrid())
    experiment.select(Role.RANDOMIZER, [UniformFill(), ClusteredFill(), SymmetricFill()])
    experiment.select(Role.NEIGHBORHOOD, Square5x5())
    experiment.iterate(
        step_omega,
        module_id="rule.omega5",
        label="omega5: счёт 5×5 + B/S/D + трюки (код в эксперименте)",
        requires=REQUIRES,
        declare=declare_rule,
        bind=bind_rule,
    )

    experiment.constrain(
        lambda values: values["grid2d.width"] * values["grid2d.height"] <= 96 * 96,
        ["grid2d.width", "grid2d.height"],
        name="cells_budget",
    )

    density = DensityFilter(min_fill=0.02, max_fill=0.92)
    stability = StabilityFilter(max_change=0.35, forbid_period2=True)
    clusters = ClustersFilter(min_components=1, min_mean_size=2.0, max_largest_share=0.95)
    phases = PhasesFilter(min_share=0.002)

    warmup = experiment.range("phase.warmup", 0, 9, 3)
    observe = experiment.range("phase.observe", 8, 21, 4)

    experiment.phase("warmup", warmup, filters=[density], frequency="end")
    experiment.phase(
        "observe",
        observe,
        filters=[density, stability, clusters, phases],
        frequency="each",
    )

    experiment.visualizer = OmegaVisualizer()
    return experiment


def web_parameters(values: Mapping[str, Any]) -> dict[str, Any]:
    """Параметры, которые браузерный модуль не может вывести из одного конфига.

    ``build.py`` кладёт результат в ``card.params``. Группы соседей должны
    идти в том же порядке, что и ``neighborhoodOffsets`` в JS (окно 5×5 по
    строкам), а интервалы B/S/D уже посчитаны правилом.
    """

    info = rule_descriptor(values)
    mechanism = str(info["mechanism"]["id"])
    groups = group_ids(mechanism, WINDOW_OFFSETS)
    return {
        "groups": [int(group) for group in groups],
        "birth": list(info["categories"]["birth"]["interval"]),
        "survival": list(info["categories"]["survival"]["interval"]),
        "death": list(info["categories"]["death"]["interval"]),
        "birthState": int(info["categories"]["birth"].get("state", 1)),
        "trick": int(info["trick"]["code"]),
        "mechanism": int(info["mechanism"]["code"]),
    }


__all__ = [
    "bind_rule",
    "build_omega5",
    "declare_rule",
    "step_omega",
    "web_parameters",
]
