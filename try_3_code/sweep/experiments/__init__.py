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
from .life import build_life
from .rings_rot import build_rings_rot

BUILDERS: dict[str, Callable[[], Experiment]] = {
    "life": build_life,
    "rings_rot": build_rings_rot,
    "cyclic_waves": build_cyclic_waves,
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
