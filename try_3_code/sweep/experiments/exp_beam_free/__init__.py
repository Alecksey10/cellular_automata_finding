"""Эксперимент ``expBeamFree``: дуэль двух групп клеток."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...experiment import Experiment
from ...interfaces import CapabilityKind, Requirement, eq, ge
from ...module import Role
from ...modules.filters import DensityFilter, DynamicsFilter, GroupPresenceFilter
from ...modules.neighborhoods import DuelRings
from ...modules.randomizers import ClusteredFill, SymmetricFill, UniformFill
from .data import DuelGrid
from .fill import BiasedDuelFill
from .rule import MECHANISMS, bind_rule, declare_rule, step_duel
from .visualizer import DuelVisualizer

PREFIX = "rule.duel."


def build_expBeamFree() -> Experiment:
    experiment = Experiment(
        "expBeamFree",
        description="дуэль двух групп: подавление, усиление и конверсия в двух кольцах",
        stop_on_extinct=True,
        web_dir="exp_beam_free",
    )

    experiment.select(Role.DATA, DuelGrid())
    experiment.select(
        Role.RANDOMIZER,
        [UniformFill(), ClusteredFill(), SymmetricFill(), BiasedDuelFill()],
    )
    experiment.select(Role.NEIGHBORHOOD, DuelRings())
    experiment.iterate(
        step_duel,
        module_id="rule.duel",
        label="duel: две группы в двух кольцах (код в эксперименте)",
        requires=(
            Requirement(CapabilityKind.GRID_2D),
            Requirement(CapabilityKind.ZERO_STATE),
            Requirement(CapabilityKind.MULTI_STATE, (ge("states", 9),), "states >= 9"),
            Requirement(
                CapabilityKind.NEIGHBORHOOD,
                (eq("max_count", 24),),
                "5x5 neighborhood (max_count == 24)",
            ),
        ),
        declare=declare_rule,
        bind=bind_rule,
    )

    experiment.constrain(
        lambda values: values["grid2d.width"] * values["grid2d.height"] <= 96 * 96,
        ["grid2d.width", "grid2d.height"],
        name="cells_budget",
    )

    density = DensityFilter(min_fill=0.05, max_fill=0.95)
    dynamics = DynamicsFilter(min_change=0.02, max_change=0.30, min_states=2, min_transitions=2)
    groups = GroupPresenceFilter(split_key=f"{PREFIX}split")

    warmup = experiment.range("phase.warmup", 3, 5)
    observe = experiment.range("phase.observe", 10, 11)

    experiment.phase("warmup", warmup, filters=[density], frequency="end")
    experiment.phase("observe", observe, filters=[density, dynamics, groups], frequency="each")

    experiment.visualizer = DuelVisualizer()
    return experiment


def web_parameters(values: Mapping[str, Any]) -> dict[str, Any]:
    """Код механизма для браузерного модуля (порядок ``MECHANISMS`` в Python)."""

    return {"mechanism": MECHANISMS.index(str(values.get("rule.duel.mechanism", MECHANISMS[0])))}


__all__ = ["build_expBeamFree", "web_parameters"]
