"""Эксперимент «3×3»: восемь соседей, симметричные правила, устойчивые структуры.

Перебирается:

* полосы рождения/выживания тоталистического правила (``B/S``);
* порог кворума голосования;
* граница ``wrap``/``clip``, число состояний (только 2 — так отсекает правило),
  размер сетки и плотность инициализации;
* число итераций прогрева и наблюдения.

Отбор: после прогрева заполнение в полосе, затем автомат обязан успокоиться —
цикл не длиннее 1, изменение маски не больше 2 % за итерацию, маска близка к
симметричной, кластеры не превращаются в пыль.
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules import (
    ActivityFilter,
    ClusterFilter,
    DiscreteGrid2D,
    FillBandFilter,
    LifeRule,
    Moore3Neighborhood,
    ParityRule,
    PeriodFilter,
    SparseRandomizer,
    SymmetricVoteRule,
    SymmetryFilter,
    TotalisticRule,
    UniformRandomizer,
)


def build_experiment() -> Experiment:
    experiment = Experiment(
        "moore3",
        "3×3 (8 соседей): тоталистические и симметричные правила.",
    )

    experiment.select(Role.DATA, [DiscreteGrid2D()])
    experiment.select(Role.RANDOMIZER, [SparseRandomizer(), UniformRandomizer()])
    experiment.select(Role.NEIGHBORHOOD, [Moore3Neighborhood()])
    experiment.select(
        Role.RULE,
        [TotalisticRule(), LifeRule(), SymmetricVoteRule(), ParityRule()],
    )

    # Сценарий «3×3» намеренно про два состояния: Life/Parity других не
    # допускают, а голосование при двух состояниях — это большинство.
    # Многосостоятельные волны живут в multi3/cyclic5.
    experiment.space.constrain(
        lambda values: values["data.states"] == 2,
        ["data.states"],
        name="experiment.two_states",
    )

    fill = experiment.filter(FillBandFilter(min_fill=0.15, max_fill=0.85))
    activity = experiment.filter(ActivityFilter(min_change=0.0, max_change=0.12))
    period = experiment.filter(PeriodFilter(max_period=1))
    symmetry = experiment.filter(SymmetryFilter(min_score=0.7))
    clusters = experiment.filter(
        ClusterFilter(min_components=1, max_components=80, min_mean_component=6.0)
    )

    warmup = experiment.space.range("stage.warmup", 4, 13, 4)
    observe = experiment.space.range("stage.observe", 10, 31, 10)

    # После прогрева проверяется только заполнение: структура ещё не сложилась.
    experiment.phase("warmup", warmup, filters=[(fill, "end")])
    experiment.phase(
        "observe",
        observe,
        filters=[
            (fill, "each"),
            (activity, "each"),
            (period, "each"),
            (symmetry, "each"),
            (clusters, "each"),
        ],
    )
    return experiment
