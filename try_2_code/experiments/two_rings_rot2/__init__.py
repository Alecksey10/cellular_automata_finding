"""Эксперимент «два кольца с частичным гниением».

Цель — стабильные автоматы, в которых старое ядро не мелькает, а медленно
гниёт: последнее состояние с вероятностью ``rot`` превращается в ``zero_state``.

Порядок прогона:

1. **Прогрев** 3–5 итераций: шум должен собраться в кластеры. На этом этапе
   ничего не проверяется — только запоминается последний кадр.
2. **Наблюдение** 10, 20 или 30 итераций, на каждом кадре:

   * ``filter.stability`` — доля появившихся/исчезнувших живых клеток не больше
     20 % (мелькание) и нет периода 2;
   * ``filter.zero_balance`` — доля ``zero_state`` меняется не больше чем на
     8 процентных пунктов за кадр;
   * ``filter.rot_structure`` — последнее состояние присутствует, но не
     захватывает сетку;
   * в конце — заполнение не пустое и не залитое.

Перебирается: ``r1`` 1..5, ``r2`` 1..7 (``r2 > r1``), пороги рождения по
внутреннему кольцу и по сумме, полоса выживания по сумме колец, вероятность
гниения, число состояний 3..6, размер сетки и инициализация.
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules import (
    DiscreteGrid2D,
    FillBandFilter,
    RotAgingRule,
    RotStructureFilter,
    SparseRandomizer,
    StabilityFilter,
    TwoRingNeighborhood,
    UniformRandomizer,
    ZeroShareFilter,
)


def build_experiment() -> Experiment:
    experiment = Experiment(
        "two_rings_rot2",
        "Два кольца: старение, вероятностное гниение последнего состояния, "
        "плавная доля мёртвых без мелькания.",
    )

    experiment.select(Role.DATA, [DiscreteGrid2D()])
    experiment.select(Role.RANDOMIZER, [SparseRandomizer(), UniformRandomizer()])
    experiment.select(Role.NEIGHBORHOOD, [TwoRingNeighborhood()])
    experiment.select(Role.RULE, [RotAgingRule()])

    # Число состояний — ровно то, что просили: 3..6.
    experiment.space.constrain(
        lambda values: 3 <= values["data.states"] <= 6,
        ["data.states"],
        name="experiment.states_3_6",
    )
    # Два кольца до r2=7 требуют сетки хотя бы 32 клетки по стороне.
    experiment.space.constrain(
        lambda values: values["data.width"] >= 64,
        ["data.width"],
        name="experiment.width_fits_rings",
    )
    # Цель — именно гниение: rot=0 даёт просто застывший автомат.
    experiment.space.constrain(
        lambda values: values["rule.rot"] > 0,
        ["rule.rot"],
        name="experiment.rot_positive",
    )

    stability = experiment.filter(StabilityFilter(max_change=0.2))
    zero_balance = experiment.filter(ZeroShareFilter(max_shift=0.08))
    rot_structure = experiment.filter(
        RotStructureFilter(min_last_share=0.01, max_last_share=0.5)
    )
    fill = experiment.filter(FillBandFilter(min_fill=0.05, max_fill=0.95))

    warmup = experiment.space.range("stage.warmup", 3, 6, 1)  # 3..5
    observe = experiment.space.range("stage.observe", 10, 31, 10)  # 10..30

    # Прогрев не проверяется: фильтры только запоминают последний кадр, чтобы
    # первое же наблюдение сравнивалось с концом прогрева.
    experiment.phase(
        "warmup",
        warmup,
        filters=[(stability, "end"), (zero_balance, "end")],
    )
    experiment.phase(
        "observe",
        observe,
        filters=[
            (stability, "each"),
            (zero_balance, "each"),
            (rot_structure, "each"),
            (fill, "end"),
        ],
    )
    return experiment
