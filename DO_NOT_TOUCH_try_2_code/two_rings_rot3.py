"""Эксперимент «два кольца с частичным гниением» — новый дизайн.

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

Архитектура:
* Параметры перебора (width, height, states, rule.rot, stage.warmup, stage.observe)
  задаются через experiment.space.range() в начале build_experiment().
* GridDef из simple_grid используется как контейнер данных, но диапазоны
  перебора задаются здесь, в коде эксперимента.
* Остальная логика (rule, neighborhood, randomizer, filters) сохранена из оригинала.
"""

from __future__ import annotations

from casweep.experiment import Experiment
from casweep.module import Role

from modules.data.simple_grid import GridDef
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
        "two_rings_rot3",
        "Два кольца: старение, вероятностное гниение последнего состояния, "
        "плавная доля мёртвых без мелькания.",
    )

    # ============================================================
    # 1. ПАРАМЕТРЫ ПЕРЕБОРА — через experiment.space.range()
    # ============================================================
    # data.width: 64, 96, 128 (step 32 от 64)
    # data.height: от width до 129, step 32
    # data.states: 3, 4, 5, 6
    # rule.rot: 1, 2, 3 (вероятность гниения последнего состояния)
    # stage.warmup: 3, 4, 5 итераций
    # stage.observe: 10, 20, 30 итераций

    experiment.space.range("data.width", 64, 96, 32)   # 64, 96, 128
    experiment.space.range("data.height", 64, 96, 32)  # от width до 129, step 32
    experiment.space.range("data.states", 3, 7, 1)      # 3, 4, 5, 6
    experiment.space.range("rule.rot", 1, 4, 1)         # 1, 2, 3

    # stage.warmup: 3..5 итераций
    experiment.space.range("stage.warmup", 3, 6, 1)     # 3, 4, 5

    # stage.observe: 10..30 итераций
    experiment.space.range("stage.observe", 10, 31, 10) # 10, 20, 30

    # Constraints (ограничения) — тоже здесь, в коде эксперимента
    experiment.space.constrain(
        lambda values: 3 <= values["data.states"] <= 6,
        ["data.states"],
        name="experiment.states_3_6",
    )

    experiment.space.constrain(
        lambda values: values["data.width"] >= 64,
        ["data.width"],
        name="experiment.width_fits_rings",
    )

    experiment.space.constrain(
        lambda values: values["rule.rot"] > 0,
        ["rule.rot"],
        name="experiment.rot_positive",
    )

    # ============================================================
    # 2. РОЛИ И МОДУЛИ
    # ============================================================
    experiment.select(Role.DATA, [DiscreteGrid2D()])
    experiment.select(Role.RANDOMIZER, [SparseRandomizer(), UniformRandomizer()])
    experiment.select(Role.NEIGHBORHOOD, [TwoRingNeighborhood()])
    experiment.select(Role.RULE, [RotAgingRule()])

    # ============================================================
    # 3. ФИЛЬТРЫ
    # ============================================================
    stability = experiment.filter(StabilityFilter(max_change=0.2))
    zero_balance = experiment.filter(ZeroShareFilter(max_shift=0.08))
    rot_structure = experiment.filter(
        RotStructureFilter(min_last_share=0.01, max_last_share=0.5)
    )
    fill = experiment.filter(FillBandFilter(min_fill=0.05, max_fill=0.95))

    # ============================================================
    # 4. ФАЗЫ
    # ============================================================
    # warmup phase: 3-5 итераций, фильтры только в конце ("end")
    experiment.phase(
        "warmup",
        "warmup",
        filters=[(stability, "end"), (zero_balance, "end")],
    )

    # observe phase: итерации из range stage.observe, фильтры на каждом шаге ("each")
    # и в конце ("end")
    experiment.phase(
        "observe",
        "observe",
        filters=[
            (stability, "each"),
            (zero_balance, "each"),
            (rot_structure, "each"),
            (fill, "end"),
        ],
    )

    return experiment


def make_grids_for_experiment():
    widths = [64, 96, 128]
    heights_map = {
        64: [64, 96, 128],
        96: [96, 128],
        128: [128],
    }
    states_list = [3, 4, 5, 6]
    rot_values = [1, 2, 3]

    result = []
    for w in widths:
        for h in heights_map.get(w, [w]):
            for s in states_list:
                for r in rot_values:
                    result.append(
                        {
                            "width": w,
                            "height": h,
                            "states": s,
                            "rot": r,
                            "zero_state": 0,
                        }
                    )
    return result
PYEOF
echo "two_rings_rot3.py rewritten"
