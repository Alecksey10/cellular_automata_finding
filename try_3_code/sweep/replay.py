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
from .module import Role
from .runtime import Trial, TrialResult
from .visualize.interface import Visualizer

DEFAULT_STATES = 2
META_PREFIX = "phase."
# длинные прогоны (муравей Ленгтона) живут тысячами итераций
META_MAX_ITERATIONS = 20000
MAX_SEED = 2**31 - 1
# разрешение квадратной сетки можно менять прямо в галерее: виртуальная ось
# ``grid2d.size`` ставит и ``grid2d.size``, и пару ``grid2d.width/height``
SIZE_KEY = "grid2d.size"
RESOLUTION_MIN = 32
RESOLUTION_MAX = 192
RESOLUTION_STEP = 16
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
    """Сколько состояний у автомата из записи.

    Сетка может вычислять их из других параметров (у муравья Ленгтона число
    состояний зависит от набора правил), поэтому сначала спрашиваем модуль
    данных и только потом смотрим на сохранённую ось ``grid2d.states``.
    """

    config = record.get("config", {})
    values = dict(config.get("values", {}))
    try:
        experiment = get_experiment(str(config.get("experiment", "")))
        experiment.finalize()
        module = experiment.selected(Role.DATA, values)
        return max(DEFAULT_STATES, int(module.derived_value("states", values)))
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return max(DEFAULT_STATES, int(values["grid2d.states"]))
    except (KeyError, TypeError, ValueError):
        return DEFAULT_STATES


# ----------------------------------------------------------------------
def resolution_of(values: Mapping[str, Any]) -> int | None:
    """Сторона квадратной сетки, если её можно менять в галерее."""

    size = values.get(SIZE_KEY)
    if isinstance(size, int) and not isinstance(size, bool):
        return int(size)
    width = values.get("grid2d.width")
    height = values.get("grid2d.height")
    if isinstance(width, int) and isinstance(height, int) and not isinstance(width, bool) and width == height:
        return int(width)
    return None


def meta_parameters(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Мета-параметры карточки: длительности фаз, разрешение и seed."""

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
    size = resolution_of(values)
    if size is not None:
        meta.append(
            {
                "key": SIZE_KEY,
                "label": "размер",
                "value": size,
                "min": RESOLUTION_MIN,
                "max": RESOLUTION_MAX,
                "step": RESOLUTION_STEP,
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
    if resolution_of(values) is not None:
        allowed.add(SIZE_KEY)
    cleaned: dict[str, int] = {}
    for key, raw in overrides.items():
        if key != "seed" and key not in allowed:
            continue
        try:
            number = int(raw)
        except (TypeError, ValueError):
            continue
        if key == "seed":
            limit = MAX_SEED
        elif key == SIZE_KEY:
            limit = RESOLUTION_MAX
        else:
            limit = META_MAX_ITERATIONS
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
        elif key == SIZE_KEY:
            size = max(RESOLUTION_MIN, int(value))
            if SIZE_KEY in values:
                values[SIZE_KEY] = size
            else:
                values["grid2d.width"] = size
                values["grid2d.height"] = size
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
    "RESOLUTION_MAX",
    "RESOLUTION_MIN",
    "SIZE_KEY",
    "clean_overrides",
    "config_for",
    "experiment_for",
    "frames_for",
    "meta_parameters",
    "rebuild",
    "resolution_of",
    "states_of",
    "visualizer_for",
]
