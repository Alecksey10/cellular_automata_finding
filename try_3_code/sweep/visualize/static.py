"""Статический экспорт галереи: каталог, который можно отдать с GitHub Pages.

У GitHub Pages нет Python, поэтому пересобрать автомат по клику там нельзя.
Экспорт заранее исполняет каждый успешный конфиг, выбирает несколько кадров,
складывает их контактным листом в один PNG и встраивает все данные в
``index.html``. Страница получается самодостаточной: переключение
экспериментов и постраничная навигация работают, мета-параметры — нет.

Раскладка каталога::

    site/index.html
    site/.nojekyll
    site/runs/<run>/<digest>.png

Запуск: ``python -m sweep export --runs runs --out site``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..replay import meta_parameters, rebuild, states_of, visualizer_for
from ..storage import RunStore
from .engine import compose
from .page import PAGE
from .png import write_png

SHEET_COLUMNS = 4


def sample_frames(frames: Sequence[np.ndarray], limit: int) -> list[np.ndarray]:
    """Не больше ``limit`` кадров, равномерно по прогону, последний — всегда."""

    total = len(frames)
    if limit <= 0 or total <= limit:
        return list(frames)
    step = max(1, total // limit)
    chosen = list(frames[::step])[:limit]
    if chosen and chosen[-1] is not frames[-1]:
        chosen[-1] = frames[-1]
    return chosen


def _card(record: Mapping[str, Any], index: int, image: str, legend: list, rule: Any) -> dict[str, Any]:
    config = record.get("config", {})
    return {
        "index": index,
        "digest": record["digest"],
        "status": record.get("status"),
        "reason": record.get("reason"),
        "iterations": record.get("iterations", 0),
        "metrics": {
            key: value
            for key, value in record.get("metrics", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        "values": config.get("values", {}),
        "meta": meta_parameters(record),
        "frames": 1,
        "image": image,
        "legend": legend,
        "rule": rule,
    }


def export_site(
    stores: Mapping[str, RunStore],
    out_dir: str | Path,
    *,
    frames: int = 12,
    scale: int = 2,
    title: str = "sweep",
) -> dict[str, int]:
    """Записать статическую галерею и вернуть число прогонов и карточек."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs_payload: list[dict[str, Any]] = []
    cards_total = 0
    for name, store in stores.items():
        records = store.successes()
        frames_dir = out / "runs" / name
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        cards: list[dict[str, Any]] = []
        experiment = ""
        for index, record in enumerate(records):
            config = record.get("config", {})
            experiment = str(config.get("experiment") or experiment)
            result = rebuild(record)
            sample = sample_frames([frame.data for frame in result.frames], frames)
            if not sample:
                continue
            visualizer = visualizer_for(experiment)
            states = states_of(record)
            sheet = compose(sample, visualizer, states=states, cols=SHEET_COLUMNS, scale=max(1, scale))
            relative = f"runs/{name}/{record['digest']}.png"
            write_png(out / relative, sheet)
            legend = [[list(color), label] for color, label in visualizer.legend(states)]
            cards.append(_card(record, index, relative, legend, visualizer.rule_info(config.get("values", {}))))
            cards_total += 1
        runs_payload.append(
            {
                "name": name,
                "experiment": experiment,
                "directory": str(store.directory),
                "cards": cards,
            }
        )
    payload = {"build": "", "mode": "static", "title": title, "runs": runs_payload}
    page = PAGE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    (out / "index.html").write_text(page, encoding="utf-8")
    # GitHub Pages прогоняет Jekyll, а он игнорирует каталоги с подчёркиванием
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return {"runs": len(runs_payload), "cards": cards_total}


__all__ = ["SHEET_COLUMNS", "export_site", "sample_frames"]
