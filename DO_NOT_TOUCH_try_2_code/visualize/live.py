"""Живой просмотр автоматов в терминале — без PNG, HTML и прочих файлов.

Один стандарт для всех экспериментов: сетка с ``.array``, ``.states`` и
``.zero_state`` переводится в цветные полублоки (или ASCII, если цвет
отключён). Кадры перерисовываются на месте, поэтому ничего не сохраняется:
конфиг и seed — это всё, что нужно, чтобы повторить картинку.

Два режима:

* :func:`play` — один автомат, анимация в терминале;
* :func:`compare` — несколько автоматов рядом, покадрово.

Палитра та же, что у ``visualize.standard``: ``age`` для старения/прочности,
``hue`` для циклических фаз. При перенаправлении вывода в файл цвет и
перерисовка отключаются автоматически.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from casweep.fixed import Fixed
from casweep.storage import find_record, read_records
from casweep.trial import run_trial

from .standard import palette

_RAMP = " .:-=+*#%@"


# ---------------------------------------------------------------------------
# Стиль терминала
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TerminalStyle:
    scheme: str = "age"
    color: bool = True
    truecolor: bool = True
    clear: bool = True
    width: int = 0  # 0 — подогнать под терминал
    height: int = 0  # в символьных строках (одна строка = 2 ряда сетки)
    fps: float = 8.0


def terminal_color_default() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def truecolor_default() -> bool:
    return os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")


def terminal_size() -> tuple[int, int]:
    columns, rows = shutil.get_terminal_size((80, 24))
    return max(8, columns - 2), max(4, rows - 3)


# ---------------------------------------------------------------------------
# Кадр
# ---------------------------------------------------------------------------
def _sample_indices(size: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    if size <= 0:
        return np.zeros(count, dtype=np.int64)
    # linspace повторяет крайние индексы, если просим больше строк, чем есть:
    # маленькая сетка не должна ломать кадр при большом окне терминала.
    return np.linspace(0, size - 1, count).round().astype(np.int64)


def downsample(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Ближайшее прореживание 2D-массива до ``width × height``."""

    array = np.asarray(array)
    rows = _sample_indices(array.shape[0], height)
    columns = _sample_indices(array.shape[1], width)
    return array[np.ix_(rows, columns)]


def _plain_char(state: int, states: int, zero_state: int) -> str:
    if state == zero_state:
        return " "
    level = round((state / max(states - 1, 1)) * (len(_RAMP) - 1))
    return _RAMP[int(min(max(level, 1), len(_RAMP) - 1))]


def _sgr(foreground: np.ndarray, background: np.ndarray, truecolor: bool) -> str:
    if truecolor:
        fr, fg, fb = (int(value) for value in foreground)
        br, bg, bb = (int(value) for value in background)
        return f"\x1b[38;2;{fr};{fg};{fb}m\x1b[48;2;{br};{bg};{bb}m"
    fg = 16 + 36 * round(int(foreground[0]) / 51) + 6 * round(int(foreground[1]) / 51) + round(int(foreground[2]) / 51)
    bg = 16 + 36 * round(int(background[0]) / 51) + 6 * round(int(background[1]) / 51) + round(int(background[2]) / 51)
    return f"\x1b[38;5;{fg}m\x1b[48;5;{bg}m"


def frame_lines(
    grid,
    style: TerminalStyle | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    indent: str = "",
) -> list[str]:
    """Строки одного кадра.

    В цветном режиме одна строка терминала — два ряда сетки (полублок ``▀``);
    в ASCII-режиме — один ряд. ``height`` всегда задаёт число строк.
    """

    style = style or TerminalStyle()
    columns, rows = terminal_size()
    width = width or style.width or min(64, columns)
    height = height or style.height or max(6, min(24, rows))
    width = max(4, int(width))
    height = max(2, int(height))

    grid_rows = height * 2 if style.color else height
    sampled = downsample(np.asarray(grid.array), width, grid_rows)
    states = max(int(getattr(grid, "states", 2)), 2)
    zero_state = int(getattr(grid, "zero_state", 0))

    if not style.color:
        return [
            indent + "".join(_plain_char(int(state), states, zero_state) for state in row)
            for row in sampled
        ]

    colors = palette(states, zero_state, style.scheme)
    lines: list[str] = []
    for row in range(height):
        top = sampled[row * 2]
        bottom = sampled[row * 2 + 1]
        parts: list[str] = []
        for top_state, bottom_state in zip(top, bottom):
            parts.append(
                _sgr(
                    colors[int(top_state)],
                    colors[int(bottom_state)],
                    style.truecolor,
                )
                + "▀"
            )
        lines.append(indent + "".join(parts) + "\x1b[0m")
    return lines


def config_summary(fixed: Fixed, max_parts: int = 7) -> str:
    """Короткая подпись конфига для заголовка."""

    values = fixed.to_dict()
    parts: list[str] = []
    for key, label in (
        ("neighborhood.r1", "r1"),
        ("neighborhood.r2", "r2"),
        ("neighborhood.size", "size"),
        ("data.width", "w"),
        ("data.height", "h"),
        ("data.states", "S"),
        ("rule.rot", "rot"),
        ("rule.born_inner", "bi"),
        ("rule.born_total", "bt"),
        ("rule.survive_lo", "lo"),
        ("rule.survive_hi", "hi"),
        ("rule.birth_above", "born"),
        ("rule.threshold", "thr"),
    ):
        if key in values:
            parts.append(f"{label}={values[key]}")
        if len(parts) >= max_parts:
            break
    return " ".join(parts) or fixed.digest()


# ---------------------------------------------------------------------------
# Один автомат
# ---------------------------------------------------------------------------
def _header(experiment: Any, fixed: Fixed, seed: int) -> list[str]:
    return [
        f"{experiment.name}  конфиг {fixed.digest()}  seed={seed}",
        config_summary(fixed),
    ]


def _draw(
    lines: Sequence[str],
    header: Sequence[str],
    footer: str,
    style: TerminalStyle,
) -> None:
    if style.clear:
        sys.stdout.write("\x1b[H\x1b[J")
    for line in header:
        print(line)
    for line in lines:
        print(line)
    if footer:
        print(footer)
    sys.stdout.flush()


def play(
    experiment: Any,
    fixed: Fixed,
    *,
    seed: int = 0,
    steps: int | None = None,
    style: TerminalStyle | None = None,
) -> int:
    """Показать один автомат в терминале; файлов не создаёт."""

    style = style or TerminalStyle()
    delay = 1.0 / style.fps if style.fps > 0 else 0.0
    header = _header(experiment, fixed, seed)
    frame = {"number": 0}

    def on_frame(phase: str, step: int, grid) -> None:
        frame["number"] += 1
        lines = frame_lines(grid, style)
        footer = f"{phase:<10} итерация {frame['number']:>3}   [{fixed.digest()[:8]}]"
        _draw(lines, header, footer, style)
        if delay and style.clear:
            time.sleep(delay)

    result = run_trial(
        experiment,
        fixed,
        seed=seed,
        keep_state=False,
        on_frame=on_frame,
        max_iterations=steps,
    )
    footer = f"{result.status.value:<10} {result.reason}"
    # Последний кадр уже нарисован; после отказа/остановки показываем вердикт.
    print(footer)
    if result.error:
        print(result.error.splitlines()[-1])
    return 0 if result.status.value in ("passed", "rejected", "stopped") else 1


# ---------------------------------------------------------------------------
# Несколько автоматов рядом
# ---------------------------------------------------------------------------
class _Live:
    """Один автомат без фильтров: только правило и смена буферов."""

    __slots__ = ("fixed", "seed", "assembled", "grid", "scratch", "done", "stop")

    def __init__(self, experiment: Any, fixed: Fixed, seed: int) -> None:
        assembled = experiment.assemble(fixed)
        rng = np.random.default_rng(seed)
        binder = getattr(assembled.rule, "bind_rng", None)
        if callable(binder):
            binder(rng)
        grid = assembled.data_factory.create()
        assembled.randomizer.fill(grid, rng)
        self.fixed = fixed
        self.seed = seed
        self.assembled = assembled
        self.grid = grid
        self.scratch = assembled.data_factory.like(grid)
        self.done = False
        self.stop = ""

    def step(self) -> None:
        if self.done:
            return
        outcome = self.assembled.rule.step(self.grid, self.scratch)
        self.grid, self.scratch = self.scratch, self.grid
        if outcome is not None and outcome.stop:
            self.done = True
            self.stop = outcome.stop


def _tile_width(total: int, count: int) -> int:
    if count <= 0:
        return total
    gaps = 3 * (count - 1)
    return max(6, (total - gaps) // count)


def compare(
    experiment: Any,
    entries: Sequence[tuple[Fixed, int]],
    *,
    steps: int = 30,
    style: TerminalStyle | None = None,
) -> int:
    """Покадрово показать несколько автоматов рядом; файлов не создаёт."""

    if not entries:
        print("Нет автоматов для сравнения")
        return 1
    style = style or TerminalStyle()
    columns, rows = terminal_size()
    count = len(entries)
    tile_width = style.width or _tile_width(columns, count)
    tile_height = style.height or max(4, min(16, rows - 6))
    lives = [_Live(experiment, fixed, seed) for fixed, seed in entries]
    delay = 1.0 / style.fps if style.fps > 0 else 0.0

    def draw(number: int) -> None:
        if style.clear:
            sys.stdout.write("\x1b[H\x1b[J")
        headers = []
        for live in lives:
            values = live.fixed.to_dict()
            r1 = values.get("neighborhood.r1", "?")
            r2 = values.get("neighborhood.r2", "?")
            states = values.get("data.states", "?")
            header = f"{live.fixed.digest()[:8]} r={r1},{r2} S{states}"
            headers.append(header.ljust(tile_width)[:tile_width])
        print(f"{experiment.name}: {count} автоматов, кадр {number}")
        print(" │ ".join(headers))
        columns_lines = [
            frame_lines(live.grid, style, width=tile_width, height=tile_height)
            for live in lives
        ]
        for row in range(tile_height):
            print(" │ ".join(tile[row] for tile in columns_lines))
        sys.stdout.flush()

    for number in range(1, steps + 1):
        for live in lives:
            live.step()
        draw(number)
        if all(live.done for live in lives):
            break
        if delay and style.clear:
            time.sleep(delay)

    for live in lives:
        if live.done:
            print(f"{live.fixed.digest()[:8]}: {live.stop}")
    return 0


# ---------------------------------------------------------------------------
# Выбор конфига
# ---------------------------------------------------------------------------
def _parse_config(raw: str) -> dict[str, Any]:
    text = raw.strip()
    path = Path(raw)
    if not text.startswith("{") and path.exists():
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            text = [line for line in text.splitlines() if line.strip()][-1]
    data = json.loads(text)
    if isinstance(data, dict) and "config" in data:
        return data["config"]
    return data


def pick_feasible(experiment: Any, seed: int = 0, attempts: int = 512) -> Fixed:
    """Случайный, но совместимый по интерфейсам конфиг."""

    rng = random.Random(seed)
    for _ in range(attempts):
        try:
            fixed = Fixed.of(experiment.space.random_binding(rng))
        except Exception:  # noqa: BLE001 - редкий тупик ограничений
            continue
        if experiment.feasible(fixed) is None:
            return fixed
    raise SystemExit("Не удалось подобрать совместимый конфиг; укажите --config/--digest")


def pick_interesting(
    experiment: Any,
    seed: int = 0,
    *,
    candidates: int = 10,
    probe_steps: int = 6,
) -> Fixed:
    """Выбрать из нескольких случайных конфигов самый живой для показа.

    Случайный конфиг часто вымирает на первом шаге; короткий пробный прогон
    без сохранения чего-либо позволяет показать по умолчанию не пустую сетку.
    Критерий — заполнение около трети и заметное, но не полное изменение.
    """

    best: tuple[float, Fixed] | None = None
    for attempt in range(max(1, candidates)):
        fixed = pick_feasible(experiment, seed + attempt * 7919)
        try:
            live = _Live(experiment, fixed, seed + attempt * 7919)
        except Exception:  # noqa: BLE001 - несовместимый/сломанный конфиг
            continue
        previous = np.asarray(live.grid.array != live.grid.zero_state).copy()
        ease = 0.0
        for _ in range(max(1, probe_steps)):
            live.step()
            if live.done:
                break
            current = np.asarray(live.grid.array != live.grid.zero_state)
            ease = max(ease, float(np.mean(current != previous)))
            previous = current.copy()
        array = np.asarray(live.grid.array)
        fill = float(np.count_nonzero(array != live.grid.zero_state)) / float(array.size)
        # Интересно: не пусто, не залито, есть движение; середина шкалы в почёте.
        score = 1.0 - abs(fill - 0.35) - (1.0 if live.done else 0.0) + 0.5 * ease
        if best is None or score > best[0]:
            best = (score, fixed)
    if best is None:
        return pick_feasible(experiment, seed)
    return best[1]


def resolve_view_config(
    experiment: Any,
    *,
    config: str | None = None,
    digest: str | None = None,
    run_dir: str | None = None,
    index: int = 0,
    seed: int = 0,
    first: bool = False,
) -> tuple[Fixed, int]:
    """Вернуть конфиг и seed: явный, из прогона, первый или случайный."""

    if config:
        return Fixed.from_dict(_parse_config(config)), seed
    if digest or run_dir:
        if digest:
            if not run_dir:
                raise SystemExit("--digest требует --run-dir")
            record = find_record(run_dir, digest)
            if record is None:
                raise SystemExit(f"Конфиг {digest!r} не найден в {run_dir}")
            records = [record]
        else:
            records = read_records(run_dir, statuses=("passed",))
        if not records:
            raise SystemExit(f"В {run_dir} нет успешных записей")
        record = records[max(0, min(index, len(records) - 1))]
        return Fixed.from_dict(record["config"]), int(record.get("seed", seed))
    if first:
        fixed = next(experiment.iter_configs(feasible=True), None)
        if fixed is None:
            raise SystemExit("В эксперименте нет совместимых конфигов")
        return fixed, seed
    return pick_interesting(experiment, seed), seed


# ---------------------------------------------------------------------------
# Готовый вход для experiments/<name>/view.py
# ---------------------------------------------------------------------------
def main_for(experiment: Any, scheme: str = "age", argv: Sequence[str] | None = None) -> int:
    """Разобрать аргументы и показать один автомат (для ``view.py``)."""

    parser = argparse.ArgumentParser(
        description=f"Живой просмотр эксперимента {experiment.name} в терминале (без файлов)"
    )
    parser.add_argument("--config", default=None, help="JSON-конфиг строкой или файлом")
    parser.add_argument("--digest", default=None, help="дайджест конфига")
    parser.add_argument("--run-dir", default=None, help="каталог прогона с successes.jsonl")
    parser.add_argument("--index", type=int, default=0, help="какой успех взять (0 — первый)")
    parser.add_argument("--seed", type=int, default=0, help="seed инициализации")
    parser.add_argument(
        "--random", action="store_true", help="случайный живой конфиг (режим по умолчанию)"
    )
    parser.add_argument("--first", action="store_true", help="первый совместимый конфиг")
    parser.add_argument("--steps", type=int, default=None, help="бюджет итераций")
    parser.add_argument("--fps", type=float, default=8.0, help="кадров в секунду")
    parser.add_argument("--width", type=int, default=0, help="ширина в символах")
    parser.add_argument("--height", type=int, default=0, help="высота в строках")
    parser.add_argument("--no-color", action="store_true", help="ASCII вместо цветных блоков")
    parser.add_argument("--no-clear", action="store_true", help="не перерисовывать экран")
    args = parser.parse_args(argv)

    fixed, seed = resolve_view_config(
        experiment,
        config=args.config,
        digest=args.digest,
        run_dir=args.run_dir,
        index=args.index,
        seed=args.seed,
        first=args.first,
    )
    style = TerminalStyle(
        scheme=scheme,
        color=False if args.no_color else terminal_color_default(),
        truecolor=truecolor_default(),
        clear=not args.no_clear and terminal_color_default(),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    return play(experiment, fixed, seed=seed, steps=args.steps, style=style)
