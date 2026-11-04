"""Эксперимент «3×3, много состояний»: голосование и циклические фазы.

Правила не зависят от порядка соседей, поэтому симметричны по построению, но
проверяется не форма, а динамика: волны фаз должны занимать заметную часть
сетки, иметь высокую энтропию распределения и не вырождаться в один домен.

Перебирается: число состояний (3..9), порог кворума голосования, порог
циклического правила, режим смерти, граница, размер сетки, инициализация.
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules import (
    ActivityFilter,
    CyclicRule,
    DiscreteGrid2D,
    EntropyFilter,
    FillBandFilter,
    Moore3Neighborhood,
    PeriodFilter,
    PhaseBalanceFilter,
    SparseRandomizer,
    SymmetricVoteRule,
    UniformRandomizer,
)


def build_experiment() -> Experiment:
    experiment = Experiment(
        "multi3",
        "3×3 (8 соседей), много состояний: голосование и циклические волны.",
    )

    experiment.select(Role.DATA, [DiscreteGrid2D()])
    experiment.select(Role.RANDOMIZER, [SparseRandomizer(), UniformRandomizer()])
    experiment.select(Role.NEIGHBORHOOD, [Moore3Neighborhood()])
    experiment.select(Role.RULE, [SymmetricVoteRule(), CyclicRule()])

    # Волнам фаз нужно минимум три состояния; двухсостоятельный режим —
    # это moore3.
    experiment.space.constrain(
        lambda values: values["data.states"] >= 3,
        ["data.states"],
        name="experiment.states_at_least_3",
    )

    fill = experiment.filter(FillBandFilter(min_fill=0.1, max_fill=0.9))
    activity = experiment.filter(ActivityFilter(min_change=0.01, max_change=0.35))
    phases = experiment.filter(PhaseBalanceFilter(min_share=0.02))
    entropy = experiment.filter(EntropyFilter(min_entropy=0.55, max_entropy=1.0))
    period = experiment.filter(PeriodFilter(max_period=3, window=24))

    warmup = experiment.space.range("stage.warmup", 5, 14, 4)
    observe = experiment.space.range("stage.observe", 10, 31, 10)

    experiment.phase("warmup", warmup, filters=[(fill, "end")])
    experiment.phase(
        "observe",
        observe,
        filters=[
            (fill, "each"),
            (activity, "each"),
            (phases, "each"),
            (entropy, "each"),
            (period, "each"),
        ],
    )
    return experiment
