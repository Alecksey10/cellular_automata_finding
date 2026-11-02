"""Эксперимент «два кольца»: старение, крепость и отбор облаков без мелькания.

Перебирается:

* ``r1`` от 1 до 5 и ``r2`` от ``r1 + 1`` до 8 — два кольца;
* пороги рождения по внутреннему кольцу и по сумме колец;
* доля максимальной плотности, при которой клетка ещё выживает;
* ``fortify`` — насколько старая клетка расширяет допустимую полосу;
* число состояний (3..6), размер сетки (ширина ≥ 32), инициализация;
* пороги фильтров «облака» и «стабильность».

После ровно 5 итераций проверяется, что автомат собрался в облака; затем
10–20 итераций он должен оставаться облачным и не мелькать.
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules import (
    AgingFortressRule,
    CloudFilter,
    DiscreteGrid2D,
    SparseRandomizer,
    StabilityFilter,
    TwoRingNeighborhood,
    UniformRandomizer,
)


def build_experiment() -> Experiment:
    experiment = Experiment(
        "two_rings",
        "Два кольца (r1/r2), старение и крепость; отбор облаков без мелькания.",
    )

    experiment.select(Role.DATA, [DiscreteGrid2D()])
    experiment.select(Role.RANDOMIZER, [SparseRandomizer(), UniformRandomizer()])
    experiment.select(Role.NEIGHBORHOOD, [TwoRingNeighborhood()])
    experiment.select(Role.RULE, [AgingFortressRule()])

    # Старение имеет смысл только при нескольких состояниях, а кольцо r2=7
    # требует сетки хотя бы 32 клетки по стороне. Эти ограничения не меняют
    # динамику, но убирают заведомо неинтересные конфиги из перебора.
    experiment.space.constrain(
        lambda values: 3 <= values["data.states"] <= 6,
        ["data.states"],
        name="experiment.states_3_6",
    )
    experiment.space.constrain(
        lambda values: values["data.width"] >= 32,
        ["data.width"],
        name="experiment.width_fits_rings",
    )

    clouds = experiment.filter(
        CloudFilter(
            min_fill=0.05,
            max_fill=0.85,
            min_density=0.4,
            min_clustered=0.6,
            min_mean_component=6.0,
        )
    )
    stability = experiment.filter(StabilityFilter(max_change=0.2))

    warmup = experiment.space.value("stage.warmup", 5)
    observe = experiment.space.range("stage.observe", 10, 30, 10)

    # Ровно 5 итераций прогрева: после них проверяем облака и запоминаем кадр
    # для фильтра стабильности (проверять мелькание пока не с чем).
    experiment.phase("warmup", warmup, filters=[(clouds, "end"), (stability, "end")])
    experiment.phase("observe", observe, filters=[(clouds, "each"), (stability, "each")])
    return experiment
