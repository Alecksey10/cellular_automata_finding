"""Эксперимент ``omega5_dense``: плотные автоматы с выверенными долями.

Правило и окно те же, что у ``omega5`` (8 состояний, 5×5, 22 механизма,
20 реакций), но отбор другой:

* состояние 0 должно занимать не больше 5% клеток;
* доли остальных состояний задаются профилем до запуска;
* профиль проверяется на совместность: сумма минимумов ≤ 100% ≤ сумма
  максимумов;
* в конце прогона отдельная фаза ``verify`` сверяет доли всех состояний.

Профиль читается из ``omega5_dense/profile.json``, из файла в переменной
``SWEEP_OMEGA5_PROFILE`` или из ``--profile`` у команды ``run``. Значения
профиля попадают в конфиг обычными осями, поэтому запись прогона
воспроизводится даже после изменения файла.
"""

from __future__ import annotations

from ...experiment import Experiment
from ...module import Role
from ...modules.filters import DensityFilter, DynamicsFilter, PresenceFilter, StateSharesFilter
from ...neighbors import Square5x5
from ...modules.randomizers import ClusteredFill, SymmetricFill, UniformFill
from ..omega5 import REQUIRES, bind_rule, declare_rule, step_omega, web_parameters
from ..omega5.data import OmegaGrid
from .fill import ProfileFill
from .profile import load_density, load_profile
from .visualizer import DenseOmegaVisualizer

PREFIX = "dense"


def build_omega5_dense() -> Experiment:
    profile = load_profile()
    density_min, density_max = load_density()
    experiment = Experiment(
        "omega5_dense",
        description="плотные автоматы: 0 ≤ 5%, доли остальных состояний из профиля",
        stop_on_extinct=True,
    )

    for state, (lo, hi) in profile.items():
        experiment.value(f"{PREFIX}.share{state}_lo", lo)
        experiment.value(f"{PREFIX}.share{state}_hi", hi)

    experiment.value(f"{PREFIX}.density_min", density_min)
    experiment.value(f"{PREFIX}.density_max", density_max)

    # Динамика: доля реально сменивших состояние клеток за один шаг.
    experiment.choice(f"{PREFIX}.min_change", [0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    experiment.value(f"{PREFIX}.max_change", 1.0)
    # Разнообразие: сколько разных состояний должно присутствовать в кадре.
    experiment.choice(f"{PREFIX}.min_present_states", [4, 5, 6])

    experiment.select(Role.DATA, OmegaGrid())
    experiment.select(
        Role.RANDOMIZER,
        [ProfileFill(), UniformFill(), ClusteredFill(), SymmetricFill()],
    )
    experiment.select(Role.NEIGHBORHOOD, Square5x5())
    experiment.iterate(
        step_omega,
        module_id="rule.omega5",
        label="omega5: счёт 5×5 + B/S/D + трюки (код в эксперименте)",
        requires=REQUIRES,
        declare=declare_rule,
        bind=bind_rule,
    )

    experiment.constrain(
        lambda values: values["grid2d.width"] * values["grid2d.height"] <= 96 * 96,
        ["grid2d.width", "grid2d.height"],
        name="cells_budget",
    )

    density = DensityFilter(min_fill=0.2, max_fill=1.0)
    dynamics = DynamicsFilter(prefix=PREFIX)
    presence = PresenceFilter(prefix=PREFIX)
    shares = StateSharesFilter(prefix=PREFIX)

    warmup = experiment.range("phase.warmup", 4, 13, 4)
    observe = experiment.range("phase.observe", 8, 21, 6)

    experiment.phase("warmup", warmup, filters=[density], frequency="end")
    experiment.phase("observe", observe, filters=[dynamics], frequency="each")
    experiment.phase(
        "verify",
        experiment.value("phase.verify", 0),
        filters=[shares, presence],
        frequency="end",
    )

    experiment.visualizer = DenseOmegaVisualizer()
    return experiment


__all__ = ["build_omega5_dense", "web_parameters"]
