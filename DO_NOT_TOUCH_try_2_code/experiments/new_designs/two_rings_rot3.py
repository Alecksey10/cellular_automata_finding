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

# Parameter definitions — вынесены в отдельный модуль для ясности источников данных
from parameters.parameter_defs import (
    WIDTHS,
    HEIGHTS_MAP,
    STATES_LIST,
    ROT_VALUES,
    WARMUP_VALUES,
    OBSERVE_VALUES,
)


def build_experiment() -> Experiment:
    experiment = Experiment(
        "two_rings_rot3",
        "Два кольца: старение, вероятностное гниение последнего состояния, плавная доля мёртвых без мелькания.",
    )

    # ============================================================
    # 1. ПАРАМЕТРЫ ПЕРЕБОРА — range для регистрации + constrain для значений
    # ============================================================
    # Register data parameters in space using range (step 32 от 64)
    # Это необходимо, чтобы пространство значений знало о существовании этих ключей
    experiment.space.range("data.width", min(WIDTHS), max(WIDTHS) + 32, 32)   # 64, 96, 128
    experiment.space.range("data.height", min(WIDTHS), max(HEIGHTS_MAP.get(64, [64])) + 32, 32)  # от width до 129, step 32
    experiment.space.range("data.states", min(STATES_LIST), max(STATES_LIST) + 1, 1)      # 3, 4, 5, 6
    experiment.space.range("rule.rot", min(ROT_VALUES), max(ROT_VALUES) + 1, 1)         # 1, 2, 3

    # Stage warmup: 3..5 итераций (Handle из space)
    warmup = experiment.space.range("stage.warmup", 3, 6, 1)     # 3, 4, 5

    # Stage observe: 10..30 итераций (Handle из space)
    observe = experiment.space.range("stage.observe", 10, 31, 10) # 10, 20, 30

    # Now constrain data parameters to specific values from parameter_defs
    experiment.space.constrain(
        lambda values: values["data.width"] in WIDTHS,
        ["data.width"],
        name="experiment.width_from_defs",
    )

    experiment.space.constrain(
        lambda values: values["data.height"] in HEIGHTS_MAP.get(values["data.width"], [values["data.width"]]),
        ["data.height"],
        name="experiment.height_from_defs",
    )

    experiment.space.constrain(
        lambda values: values["data.states"] in STATES_LIST,
        ["data.states"],
        name="experiment.states_from_defs",
    )

    experiment.space.constrain(
        lambda values: values["rule.rot"] in ROT_VALUES,
        ["rule.rot"],
        name="experiment.rot_from_defs",
    )

    # Additional constraints
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
    # 4. ФАЗИ
    # ============================================================
    # warmup phase: 3-5 итераций, фильтры только в конце ("end")
    experiment.phase(
        "warmup",
        warmup,
        filters=[(stability, "end"), (zero_balance, "end")],
    )

    # observe phase: итерации из range stage.observe, фильтры на каждом шаге ("each")
    # и в конце ("end")
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
