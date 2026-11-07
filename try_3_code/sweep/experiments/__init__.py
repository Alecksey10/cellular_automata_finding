"""Реестр экспериментов.

Каждый эксперимент живёт в своей директории ``<name>/``: ``__init__.py``
собирает описание, ``visualizer.py`` содержит его визуализатор. Эксперименты
собираются функциями: каждый прогон/работник получает свежий объект описания,
поэтому состояние фаз и фильтров не течёт между запусками.
"""

from __future__ import annotations

from typing import Callable

from ..experiment import Experiment
from .cyclic_waves import build_cyclic_waves
from .exp_beam_free import build_expBeamFree
from .langton import build_langton
from .life import build_life
from .maze import build_maze
from .maze9 import build_maze9
from .omega5 import build_omega5
from .omega5_dense import build_omega5_dense
from .rings_rot import build_rings_rot
from .symbiosis import build_symbiosis

BUILDERS: dict[str, Callable[[], Experiment]] = {
    "life": build_life,
    "rings_rot": build_rings_rot,
    "symbiosis": build_symbiosis,
    "cyclic_waves": build_cyclic_waves,
    "expBeamFree": build_expBeamFree,
    "langton": build_langton,
    "maze": build_maze,
    "maze9": build_maze9,
    "omega5": build_omega5,
    "omega5_dense": build_omega5_dense,
}

ALL_EXPERIMENTS = tuple(BUILDERS)


def get_experiment(name: str) -> Experiment:
    try:
        builder = BUILDERS[name]
    except KeyError:
        known = ", ".join(BUILDERS)
        raise KeyError(f"неизвестный эксперимент {name!r}; доступны: {known}") from None
    return builder()


def list_experiments() -> list[str]:
    return list(BUILDERS)


__all__ = ["ALL_EXPERIMENTS", "BUILDERS", "get_experiment", "list_experiments"]
