"""Эксперимент «5×5»: 24 соседа, циклические и рефрактерные правила.

Правила описывают «твердение и смерть»: клетка проходит фазы, каждая следующая
твёрже, затем погибает. Отбор ищет волны: все фазы присутствуют, картинка
меняется, но не хаотично, цикл не длиннее трёх итераций.

Перебирается: число состояний (3..9; рефрактерному правилу нужно ≥ 4),
порог возбуждения, режим смерти циклического правила, граница, размер сетки,
инициализация.
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules import (
    ActivityFilter,
    ClusterFilter,
    CyclicRule,
    DiscreteGrid2D,
    EntropyFilter,
    FillBandFilter,
    Moore5Neighborhood,
    PeriodFilter,
    PhaseBalanceFilter,
    RefractoryRule,
    SparseRandomizer,
    UniformRandomizer,
)


def build_experiment() -> Experiment:
    experiment = Experiment(
        "cyclic5",
        "5×5 (24 соседа): циклические и рефрактерные правила «твердеет, затем умирает».",
    )

    experiment.select(Role.DATA, [DiscreteGrid2D()])
    experiment.select(Role.RANDOMIZER, [SparseRandomizer(), UniformRandomizer()])
    experiment.select(Role.NEIGHBORHOOD, [Moore5Neighborhood()])
    experiment.select(Role.RULE, [CyclicRule(), RefractoryRule()])

    # Циклу нужно минимум три состояния, рефрактерному правилу — четыре.
    # Ограничение зависит от выбранного правила и отсекает ветку заранее.
    experiment.space.constrain(
        lambda values: values["data.states"]
        >= (4 if values["rule.module"] == "rule.refractory" else 3),
        ["data.states", "rule.module"],
        name="experiment.states_for_rule",
    )

    fill = experiment.filter(FillBandFilter(min_fill=0.1, max_fill=0.9))
    activity = experiment.filter(ActivityFilter(min_change=0.01, max_change=0.4))
    phases = experiment.filter(PhaseBalanceFilter(min_share=0.02))
    entropy = experiment.filter(EntropyFilter(min_entropy=0.6, max_entropy=1.0))
    period = experiment.filter(PeriodFilter(max_period=3, window=24))
    clusters = experiment.filter(
        ClusterFilter(min_components=1, max_components=200, min_mean_component=4.0)
    )

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
            (clusters, "each"),
        ],
    )
    return experiment
