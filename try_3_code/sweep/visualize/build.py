"""Сборка статической галереи для GitHub Pages.

Раньше экспорт заранее рисовал PNG-кадры и складывал их рядом с ``index.html``.
Теперь эксперимент приносит браузерный модуль (``sweep/experiments/<имя>/web.js``),
а сборка склеивает ядро и модули в один скрипт ``assets/rules.js``. Страница
считает автоматы сама: ни Python, ни ``/api`` на GitHub Pages не нужны.

Раскладка каталога::

    site/index.html          данные карточек (сжатые сетки + конфиг)
    site/assets/rules.js     engine.js + runtime.js + модули экспериментов
    site/assets/gallery.js   карточки, страницы, анимация и контролы
    site/.nojekyll

Граница интерфейса описана в ``sweep/visualize/web/README.md``: эксперимент
приносит только ``create``/``step``, а размер, число итераций, скорость и
цикл берёт на себя браузерный ``Runner``.

Запуск: ``python -m sweep export --run-dir runs/maze9 --out site``.

Если у эксперимента нет ``web.js``, он в статическую сборку не попадает:
карточке нечем считать кадры в браузере.
"""

from __future__ import annotations

import base64
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..assembly import assemble
from ..config import Config
from ..experiments import get_experiment
from ..replay import meta_parameters, resolution_of, states_of, visualizer_for
from ..runtime import Trial

WEB_DIR = Path(__file__).with_name("web")
# порядок важен: helpers -> реестр/runner -> модули экспериментов
WEB_SCRIPTS = ("engine.js", "runtime.js")
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def experiment_directory(name: str) -> str:
    """Каталог пакета эксперимента (может отличаться от имени в реестре)."""

    try:
        from ..experiments import get_experiment

        experiment = get_experiment(name)
    except KeyError:
        return str(name)
    return str(getattr(experiment, "web_dir", None) or name)
CARD_METRICS = (
    "alive_density",
    "zero_density",
    "occupied_states",
    "max_state_share",
    "mean_state",
    "bbox_share",
    "trail_cells",
    "ant_cells",
)


def module_path(experiment: str) -> Path | None:
    """Браузерный модуль эксперимента, если он есть."""

    path = EXPERIMENTS_DIR / experiment_directory(experiment) / "web.js"
    return path if path.is_file() else None


def plane_bits(states: int) -> int:
    if states <= 2:
        return 1
    if states <= 16:
        return 4
    return 8


def pack_plane(data: np.ndarray, bits: int) -> str:
    """Сетка -> base64; пустая плоскость кодируется пустой строкой."""

    flat = np.asarray(data, dtype=np.uint8).reshape(-1)
    if not flat.any():
        return ""
    if bits >= 8:
        raw = flat.tobytes()
    elif bits == 4:
        if flat.size % 2:
            flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
        pairs = flat.reshape(-1, 2)
        raw = ((pairs[:, 0] << 4) | pairs[:, 1]).astype(np.uint8).tobytes()
    else:
        raw = np.packbits(flat).tobytes()
    return base64.b64encode(raw).decode("ascii")


def hex_color(rgb: Sequence[int]) -> str:
    return "#%02x%02x%02x" % tuple(int(value) for value in rgb)


def web_parameters_for(experiment: str, values: Mapping[str, Any]) -> dict[str, Any] | None:
    """Необязательный хук эксперимента для параметров браузерного модуля.

    Эксперимент может определить в своём пакете ``web_parameters(values)``;
    тогда результат попадёт в ``card.params``. Так сложные правила (например,
    ``omega5``) не дублируют таблицы Python в JavaScript.
    """

    directory = experiment_directory(experiment)
    try:
        module = importlib.import_module(f"{__package__.rsplit('.', 1)[0]}.experiments.{directory}")
    except ImportError:
        return None
    hook = getattr(module, "web_parameters", None)
    if not callable(hook):
        return None
    result = hook(values)
    return dict(result) if result else None


def web_card_details_for(experiment: str, values: Mapping[str, Any], seed: int = 0) -> dict[str, Any] | None:
    """Необязательный хук эксперимента для дополнительных полей карточки.

    Заменяет жёсткие ``if experiment == ...`` в сборке: эксперимент сам
    решает, какие данные нужны браузеру сверх сеток (например, муравьи
    Ленгтона). Результат сливается в карточку.
    """

    directory = experiment_directory(experiment)
    try:
        module = importlib.import_module(f"{__package__.rsplit('.', 1)[0]}.experiments.{directory}")
    except ImportError:
        return None
    hook = getattr(module, "web_card_details", None)
    if not callable(hook):
        return None
    try:
        result = hook(values, seed)
    except TypeError:
        result = hook(values)
    return dict(result) if result else None


def _run_trial(record: Mapping[str, Any]) -> Any:
    """Полный прогон конфига: первый кадр — затравка, последний — итог."""

    config = Config.from_record(record["config"])
    experiment = get_experiment(config.experiment)
    experiment.finalize()
    assembly = assemble(experiment, config)
    total = sum(assembly.iterations(phase) for phase in experiment.phases)
    return Trial(assembly, config.seed, record=True, frame_stride=max(1, total)).run()


def card_payload(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Карточка для браузера: конфиг, палитра и две сжатые сетки."""

    config = record["config"]
    all_values = dict(config.get("values", {}))
    values = {
        key: value for key, value in all_values.items() if not str(key).startswith("phase.")
    }
    phases = {
        key[len("phase.") :]: int(value)
        for key, value in all_values.items()
        if str(key).startswith("phase.") and isinstance(value, int) and not isinstance(value, bool)
    }
    experiment = str(config.get("experiment"))
    states = states_of(record)
    bits = plane_bits(states)
    result = _run_trial(record)
    frames = result.frames
    if not frames:
        raise ValueError(f"нет кадров: {record.get('digest')}")
    first = frames[0].data
    last = frames[-1].data
    height, width = first.shape
    visualizer = visualizer_for(experiment)
    colors = [hex_color(visualizer.state_color(state, states)) for state in range(states)]
    legend = [[hex_color(color), label] for color, label in visualizer.legend(states)]
    metrics = {
        key: round(float(record["metrics"][key]), 4)
        for key in CARD_METRICS
        if isinstance(record.get("metrics", {}).get(key), (int, float))
    }
    card: dict[str, Any] = {
        "index": index,
        "digest": record["digest"],
        "experiment": experiment,
        "status": record.get("status"),
        "iterations": int(result.iterations),
        "width": int(width),
        "height": int(height),
        "states": int(states),
        "bits": bits,
        "init": pack_plane(first, bits),
        "final": pack_plane(last, bits),
        "values": values,
        "phases": phases,
        "meta": meta_parameters(record),
        "nativeSize": resolution_of(all_values),
        "seed": int(config.get("seed", 0)),
        "metrics": metrics,
        "colors": colors,
        "legend": legend,
        "rule": visualizer.rule_info(values),
    }
    params = web_parameters_for(experiment, all_values)
    if params:
        card["params"] = params
    details = web_card_details_for(experiment, all_values, int(config.get("seed", 0)))
    if details:
        card.update(details)
    return card


def bundle_source(experiments: Sequence[str]) -> str:
    """Ядро + модули экспериментов одним скриптом (без сборщика и импортов)."""

    parts = [
        f"// --- {name} ---\n" + (WEB_DIR / name).read_text(encoding="utf-8")
        for name in WEB_SCRIPTS
    ]
    for experiment in experiments:
        path = module_path(experiment)
        if path is None:
            continue
        parts.append(f"// --- {experiment}/web.js ---\n" + path.read_text(encoding="utf-8"))
    parts.append(
        "globalThis.SweepRules = {\n"
        "  // реестр и жизненный цикл\n"
        "  experiments: EXPERIMENTS,\n"
        "  experimentIds: experimentIds,\n"
        "  createRunner: createRunner,\n"
        "  Runner: Runner,\n"
        "  // данные и отрисовка\n"
        "  gridFromBase64: gridFromBase64,\n"
        "  gridToBase64: gridToBase64,\n"
        "  resizeGrid: resizeGrid,\n"
        "  resizeCard: resizeCard,\n"
        "  makeRng: makeRng,\n"
        "  hashSeed: hashSeed,\n"
        "  paintGrid: paintGrid,\n"
        "  finalGrid: finalGrid,\n"
        "  // общие помощники правил\n"
        "  neighborhoodOffsets: neighborhoodOffsets,\n"
        "  innerNeighborhoodOffsets: innerNeighborhoodOffsets,\n"
        "  neighborhoodWrap: neighborhoodWrap,\n"
        "  mechanismOffsets: mechanismOffsets,\n"
        "  percentCounts: percentCounts,\n"
        "  makeMazeRule: makeMazeRule,\n"
        "  makeOmegaRule: makeOmegaRule,\n"
        "};\n"
    )
    return '(function () {\n"use strict";\n\n' + "\n\n".join(parts) + "\n})();\n"


def _check_bundle(path: Path) -> str | None:
    """Синтаксическая проверка собранного скрипта через node, если он есть."""

    node = shutil.which("node")
    if node is None:
        return None
    done = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise RuntimeError("node --check: " + (done.stderr.strip() or done.stdout.strip()))
    return "node --check"


def build_site(
    stores: Mapping[str, Any],
    out_dir: str | Path,
    *,
    title: str = "sweep",
    page_size: int = 24,
) -> dict[str, Any]:
    """Собрать статическую галерею; вернуть статистику сборки."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in (out / "runs", out / "assets"):
        if stale.exists():
            shutil.rmtree(stale)
    runs_payload: list[dict[str, Any]] = []
    skipped: list[str] = []
    used: list[str] = []
    cards_total = 0
    for name, store in stores.items():
        records = store.successes()
        if not records:
            continue
        experiment = str(records[0]["config"].get("experiment"))
        if module_path(experiment) is None:
            skipped.append(experiment)
            continue
        cards = [card_payload(record, index) for index, record in enumerate(records)]
        cards_total += len(cards)
        if experiment not in used:
            used.append(experiment)
        runs_payload.append(
            {
                "name": name,
                "experiment": experiment,
                "directory": str(store.directory),
                "cards": cards,
            }
        )
    if not runs_payload:
        raise RuntimeError("нечего собирать: ни у одного прогона нет web-модуля эксперимента")

    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rules_path = assets / "rules.js"
    rules_path.write_text(bundle_source(used), encoding="utf-8")
    shutil.copyfile(WEB_DIR / "gallery.js", assets / "gallery.js")
    checked = _check_bundle(rules_path)
    gallery_check = _check_bundle(assets / "gallery.js")
    if gallery_check:
        checked = ", ".join(part for part in (checked, gallery_check) if part)

    payload = {
        "title": title,
        "build": "",
        "mode": "web",
        "pageSize": int(page_size),
        "runs": runs_payload,
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (WEB_DIR / "page.html").read_text(encoding="utf-8")
    page = template.replace("__TITLE__", str(title)).replace("__DATA__", data)
    (out / "index.html").write_text(page, encoding="utf-8")
    # GitHub Pages прогоняет Jekyll, а он игнорирует каталоги с подчёркиванием
    (out / ".nojekyll").write_text("", encoding="utf-8")

    size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    return {
        "runs": len(runs_payload),
        "cards": cards_total,
        "experiments": used,
        "skipped": skipped,
        "bytes": size,
        "checked": checked,
    }


__all__ = [
    "build_site",
    "bundle_source",
    "card_payload",
    "hex_color",
    "module_path",
    "pack_plane",
    "plane_bits",
    "web_parameters_for",
]
