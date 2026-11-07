"""Эксперимент «симбиоз»: 7 состояний, крупная опора и секторы радиуса 5.

Два опорных состояния (радиусы 20 и 30) рождаются только там, где уже есть
достаточно подобных клеток **и** зависимых, которые без опоры не живут.
Зависимые состояния считают опору в круге радиуса 5 (80 клеток) и проверяют
её рисунок по секторам: половины, четверти, 6 или 8 долей. Секторы можно
считать по-разному — минимум занятых, все заняты, есть пустой, чётные против
нечётных, — поэтому одно и то же правило даёт разные формы колоний.

Споры — память колонии: зависимая клетка без опоры превращается в спору и
ждёт возвращения опоры. Импульсы — короткий сигнал, без которого зависимые
не размножаются; это не даёт колонии расти бесконечно.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...experiment import Experiment
from ...interfaces import CapabilityKind, Requirement, eq, ge
from ...module import Role
from ...modules.filters import ClustersFilter, DensityFilter, PhasesFilter, StabilityFilter
from ...neighbors.geometry import offset_arrays
from .data import SymbiosisGrid
from .fill import SymbiosisFill
from .kernel import step_symbiosis
from .rule import (
    METRICS,
    SECTOR_MODES,
    STATES,
    WINDOW_MODES,
    logic_offsets,
    sector_ids,
    spacing_offsets,
    support_masks,
    support_offsets,
)
from .visualizer import SymbiosisVisualizer

PREFIX = "rule.symbiosis."


def declare_rule(ns, view) -> None:
    """Оси правила: геометрия опоры, секторы и все пороги."""

    ns.choice("metric", list(METRICS))
    ns.choice("anchor_radius", [20, 24])
    ns.choice("matrix_radius", [26, 30])
    ns.choice("support_window", list(WINDOW_MODES))
    ns.choice("logic_window", list(WINDOW_MODES))
    ns.choice("sectors", [2, 4, 6, 8])
    ns.choice("sector_mode", list(SECTOR_MODES))
    ns.range("sector_need", 1, 4)

    ns.range("n_anchor_birth", 20, 85, 20)
    ns.range("m_anchor_birth", 1, 3)
    ns.range("n_anchor_surv", 2, 7)
    ns.range("m_anchor_surv", 0, 2)
    ns.range("n_anchor_max", 100, 501, 80)

    ns.range("n_matrix_birth", 30, 151, 20)
    ns.range("m_matrix_birth", 1, 3)
    ns.range("n_matrix_surv", 2, 7)
    ns.range("m_matrix_surv", 0, 2)
    ns.range("n_matrix_max", 180, 541, 120)

    ns.range("grazer_birth", 2, 7, 2)
    ns.range("grazer_surv", 1, 3)
    ns.range("grazer_max", 8, 25, 8)
    ns.range("weaver_birth", 2, 7, 2)
    ns.range("weaver_surv", 1, 3)
    ns.range("weaver_max", 8, 25, 8)

    ns.choice("wrap", [True, False])
    ns.choice("pulse_prob", [0.04, 0.08, 0.12])
    ns.choice("spore_decay", [0.01, 0.02, 0.04])


def bind_rule(fixed: Mapping[str, Any], view) -> tuple:
    """Собрать numba-совместимые массивы геометрии и пороги."""

    metric = str(fixed[PREFIX + "metric"])
    anchor_radius = int(fixed[PREFIX + "anchor_radius"])
    matrix_radius = int(fixed[PREFIX + "matrix_radius"])
    sectors = int(fixed[PREFIX + "sectors"])

    offsets = support_offsets(matrix_radius, metric)
    in_anchor, in_matrix = support_masks(
        offsets,
        anchor_radius,
        matrix_radius,
        str(fixed[PREFIX + "support_window"]),
        metric,
    )
    sdy, sdx = offset_arrays(offsets, 2)

    logic = logic_offsets(str(fixed[PREFIX + "logic_window"]), metric)
    ldy, ldx = offset_arrays(logic, 2)
    ndy, ndx = offset_arrays(spacing_offsets(), 2)
    sector = np.asarray(sector_ids(logic, sectors), dtype=np.int32)

    return (
        sdy,
        sdx,
        np.asarray(in_anchor, dtype=np.int32),
        np.asarray(in_matrix, dtype=np.int32),
        ldy,
        ldx,
        ndy,
        ndx,
        sector,
        int(bool(fixed[PREFIX + "wrap"])),
        int(fixed["grid2d.zero"]),
        int(fixed[PREFIX + "n_anchor_birth"]),
        int(fixed[PREFIX + "m_anchor_birth"]),
        int(fixed[PREFIX + "n_anchor_surv"]),
        int(fixed[PREFIX + "m_anchor_surv"]),
        int(fixed[PREFIX + "n_anchor_max"]),
        int(fixed[PREFIX + "n_matrix_birth"]),
        int(fixed[PREFIX + "m_matrix_birth"]),
        int(fixed[PREFIX + "n_matrix_surv"]),
        int(fixed[PREFIX + "m_matrix_surv"]),
        int(fixed[PREFIX + "n_matrix_max"]),
        int(fixed[PREFIX + "grazer_birth"]),
        int(fixed[PREFIX + "grazer_surv"]),
        int(fixed[PREFIX + "grazer_max"]),
        int(fixed[PREFIX + "weaver_birth"]),
        int(fixed[PREFIX + "weaver_surv"]),
        int(fixed[PREFIX + "weaver_max"]),
        sectors,
        SECTOR_MODES.index(str(fixed[PREFIX + "sector_mode"])),
        int(fixed[PREFIX + "sector_need"]),
        float(fixed[PREFIX + "pulse_prob"]),
        float(fixed[PREFIX + "spore_decay"]),
    )


REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
    Requirement(CapabilityKind.MULTI_STATE, (eq("states", STATES),), f"states == {STATES}"),
)


def build_symbiosis() -> Experiment:
    experiment = Experiment(
        "symbiosis",
        description="7 состояний: опора R20/R30, зависимые с секторами R5, споры и импульсы",
        stop_on_extinct=True,
    )

    experiment.select(Role.DATA, SymbiosisGrid())
    experiment.select(Role.RANDOMIZER, SymbiosisFill())
    experiment.iterate(
        step_symbiosis,
        module_id="rule.symbiosis",
        label="симбиоз: опора R20/R30 + секторы R5 (код в эксперименте)",
        requires=REQUIRES,
        declare=declare_rule,
        bind=bind_rule,
        with_control=True,
    )
    experiment.constrain(
        lambda values: values["grid2d.width"] * values["grid2d.height"] <= 64 * 64,
        ["grid2d.width", "grid2d.height"],
        name="cells_budget",
    )
    experiment.constrain(
        lambda values: values[PREFIX + "sector_need"] < values[PREFIX + "sectors"],
        [PREFIX + "sector_need", PREFIX + "sectors"],
        name="sector_need_fits",
    )
    experiment.constrain(
        lambda values: values[PREFIX + "grazer_surv"] <= values[PREFIX + "grazer_max"],
        [PREFIX + "grazer_surv", PREFIX + "grazer_max"],
        name="grazer_limits",
    )
    experiment.constrain(
        lambda values: values[PREFIX + "weaver_surv"] <= values[PREFIX + "weaver_max"],
        [PREFIX + "weaver_surv", PREFIX + "weaver_max"],
        name="weaver_limits",
    )

    density = DensityFilter(min_fill=0.02, max_fill=0.95)
    stability = StabilityFilter(max_change=0.35, forbid_period2=True)
    clusters = ClustersFilter(min_components=2, min_mean_size=3.0, max_largest_share=0.9)
    phases = PhasesFilter(min_share=0.001)

    warmup = experiment.range("phase.warmup", 0, 9, 3)
    observe = experiment.range("phase.observe", 12, 31, 6)

    experiment.phase("warmup", warmup, filters=[density], frequency="end")
    experiment.phase(
        "observe",
        observe,
        filters=[density, stability, clusters, phases],
        frequency="each",
    )

    experiment.visualizer = SymbiosisVisualizer()
    return experiment


def web_parameters(values: Mapping[str, Any]) -> dict[str, Any]:
    """Номера секторов круга R5 для браузерного модуля.

    ``atan2`` в Python и JavaScript может разойтись на границе сектора,
    поэтому геометрия считается один раз здесь и уезжает в карточку.
    """

    metric = str(values.get(PREFIX + "metric", "euclidean"))
    sectors = int(values.get(PREFIX + "sectors", 8))
    logic = logic_offsets(str(values.get(PREFIX + "logic_window", "disc")), metric)
    return {"sector": list(sector_ids(logic, sectors))}


__all__ = ["bind_rule", "build_symbiosis", "declare_rule", "step_symbiosis", "web_parameters"]
