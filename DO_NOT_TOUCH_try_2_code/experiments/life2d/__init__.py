"""Эксперимент: 2D автоматы с пороговыми правилами и игрой «Жизнь».

Порядок объявления важен: каждая роль видит уже объявленные роли через
``DeclareView.peer`` (правило берёт ``max_count`` у окрестности).
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules import (
    DensityFilter,
    DiscreteGrid2D,
    LifeRule,
    MonochromeSquaresFilter,
    RadiusNeighborhood,
    RingNeighborhood,
    SparseRandomizer,
    SquareNeighborhood,
    ThresholdRule,
    UniformRandomizer,
)


def build_experiment() -> Experiment:
    experiment = Experiment(
        "life2d",
        "2D дискретные автоматы: пороговые правила и игра «Жизнь».",
    )

    experiment.select(Role.DATA, [DiscreteGrid2D()])
    experiment.select(Role.RANDOMIZER, [UniformRandomizer(), SparseRandomizer()])
    experiment.select(
        Role.NEIGHBORHOOD,
        [SquareNeighborhood(), RingNeighborhood(), RadiusNeighborhood()],
    )
    experiment.select(Role.RULE, [ThresholdRule(), LifeRule()])

    density = experiment.filter(DensityFilter())
    squares = experiment.filter(MonochromeSquaresFilter())

    warmup = experiment.space.range("stage.warmup", 0, 10, 2)
    observe = experiment.space.range("stage.observe", 10, 30, 10)

    # После прогрева фильтр проверяется один раз (как в примере), а на
    # наблюдении — каждую итерацию, чтобы рано отбрасывать скучные автоматы.
    experiment.phase("warmup", warmup, filters=[(density, "end")])
    experiment.phase("observe", observe, filters=[(density, "each"), (squares, "each")])
    return experiment
