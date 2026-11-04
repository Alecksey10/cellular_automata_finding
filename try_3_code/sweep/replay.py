"""Повторная сборка автомата по сохранённому конфигу.

Прогон хранит только конфиг, статус и метрики. Кадров на диске нет: чтобы
показать автомат, его собирают заново из того же конфига. Поэтому запись в
`successes.jsonl` — это одновременно и результат отбора, и способ получить
автомат: параметры фиксированы, seed фиксирован, повтор даёт те же кадры.

Поверх конфига можно передать мета-параметры (:func:`meta_parameters`):
длительности фаз `phase.*` и `seed`. Так из карточки в браузере можно
пересобрать тот же автомат с большим числом итераций, не меняя прогон.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .assembly import assemble
from .config import Config
from .experiments import get_experiment
from .runtime import Trial, TrialResult
from .visualize.interface import Visualizer

DEFAULT_STATES = 2
META_PREFIX = "phase."
META_MAX_ITERATIONS = 2000
MAX_SEED = 2**31 - 1
# сколько кадров хранить в памяти при показе: длинный прогон прореживается
FRAME_BUDGET = 240


def experiment_for(name: str):
    """Свежий объект эксперимента по имени из записи прогона."""

    return get_experiment(name)


def visualizer_for(name: str) -> Visualizer:
    """Визуализатор эксперимента; базовый, если эксперимент неизвестен."""

    try:
        visualizer = experiment_for(name).visualizer
    except KeyError:
        return Visualizer()
    return visualizer if isinstance(visualizer, Visualizer) else Visualizer()


def states_of(record: Mapping[str, Any]) -> int:
    """Число состояний автомата, записанное в конфиге."""

    values = record.get("config", {}).get("values", {})
    try:
        return max(DEFAULT_STATES, int(values["grid2d.states"]))
    except (KeyError, TypeError, ValueError):
        return DEFAULT_STATES


# ----------------------------------------------------------------------
def meta_parameters(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Мета-параметры карточки: длительности фаз и seed."""

    config = record.get("config", {})
    values = config.get("values", {})
    meta: list[dict[str, Any]] = []
    for key, value in values.items():
        if key.startswith(META_PREFIX) and isinstance(value, int) and not isinstance(value, bool):
            meta.append(
                {
                    "key": key,
                    "label": key[len(META_PREFIX) :],
                    "value": int(value),
                    "min": 0,
                    "max": META_MAX_ITERATIONS,
                }
            )
    meta.append(
        {
            "key": "seed",
            "label": "seed",
            "value": int(config.get("seed", 0)),
            "min": 0,
            "max": MAX_SEED,
        }
    )
    return meta


def clean_overrides(record: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, int]:
    """Оставить только мета-параметры с числовыми значениями в допустимых границах."""

    if not overrides:
        return {}
    values = record.get("config", {}).get("values", {})
    allowed = {
        key
        for key, value in values.items()
        if key.startswith(META_PREFIX) and isinstance(value, int) and not isinstance(value, bool)
    }
    cleaned: dict[str, int] = {}
    for key, raw in overrides.items():
        if key != "seed" and key not in allowed:
            continue
        try:
            number = int(raw)
        except (TypeError, ValueError):
            continue
        limit = MAX_SEED if key == "seed" else META_MAX_ITERATIONS
        cleaned[str(key)] = max(0, min(number, limit))
    return cleaned


def config_for(record: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> Config:
    """Конфиг записи с применёнными мета-параметрами."""

    config = Config.from_record(record["config"])
    cleaned = clean_overrides(record, overrides)
    if not cleaned:
        return config
    values = dict(config.values)
    seed = config.seed
    for key, value in cleaned.items():
        if key == "seed":
            seed = value
        else:
            values[key] = value
    return Config(config.experiment, values, seed)


def rebuild(
    record: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    max_frames: int | None = FRAME_BUDGET,
) -> TrialResult:
    """Собрать автомат из записи прогона и исполнить его заново."""

    config = config_for(record, overrides)
    experiment = get_experiment(config.experiment)
    experiment.finalize()
    assembly = assemble(experiment, config)
    total = sum(assembly.iterations(phase) for phase in experiment.phases)
    if max_frames and total > max_frames:
        # прогон может оборваться фильтром намного раньше: сначала узнаём
        # фактическую длину без записи кадров, потом прореживаем ровно на неё
        probe = Trial(assembly, config.seed, record=False).run()
        stride = max(1, math.ceil(max(1, probe.iterations) / max_frames))
    else:
        stride = 1
    return Trial(assembly, config.seed, record=True, frame_stride=stride).run()


def frames_for(
    record: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
    limit: int | None = None,
    max_frames: int | None = FRAME_BUDGET,
) -> list[np.ndarray]:
    """Кадры конфига, полученные повторным исполнением."""

    result = rebuild(record, overrides=overrides, max_frames=max_frames)
    frames = [frame.data for frame in result.frames]
    return frames[:limit] if limit is not None else frames


__all__ = [
    "DEFAULT_STATES",
    "FRAME_BUDGET",
    "MAX_SEED",
    "META_MAX_ITERATIONS",
    "META_PREFIX",
    "clean_overrides",
    "config_for",
    "experiment_for",
    "frames_for",
    "meta_parameters",
    "rebuild",
    "states_of",
    "visualizer_for",
]
