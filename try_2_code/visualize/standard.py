"""Общий стандарт 2D-отрисовки клеточных автоматов.

Любая сетка, у которой есть ``.array``, ``.states`` и ``.zero_state``,
рисуется одинаково — независимо от того, какое это правило и какие у него
состояния. Стандарт задаёт:

* палитру (монотонная «возрастная» или фазовая по кругу тонов);
* один кадр в пикселях;
* горизонтальную легенду состояний;
* ленту кадров и контактный лист (несколько рядов).

HTML-вьюер и галерея в :mod:`visualize.viewer` используют те же кадры, поэтому
в браузере и в PNG картинка выглядит одинаково.
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .png import write_png

BACKGROUND = (12, 16, 22)

#: Монотонная шкала «молодая -> старая»: тёмно-красный, оранжевый, жёлтый, белый.
AGE_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (58, 18, 24)),
    (0.28, (150, 40, 34)),
    (0.55, (214, 104, 36)),
    (0.8, (238, 188, 70)),
    (1.0, (252, 250, 214)),
)

#: Минимально допустимый внешний радиус кольца: меньшие значения
#: не формируют насыщенный какую-либо форму и только замедляют прогон.
_MIN_RADIUS = 1
_MAX_RADIUS = 5

#: Порог «надёжности» выживания. Если доля клеток 1-го кольца в начале
#: итерации меньше этого значения, второе кольцо уже не успевает отработать
#: и автомат считается запущенным. Это константа фильтра stabilise.
_FIRSTRING_MIN_DENSITY = 0.3


def _interpolate_stops(t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    for index in range(len(AGE_STOPS) - 1):
        left_t, left_color = AGE_STOPS[index]
        right_t, right_color = AGE_STOPS[index + 1]
        if t <= right_t:
            span = right_t - left_t
            k = 0.0 if span <= 0 else (t - left_t) / span
            return tuple(
                int(round(left_color[channel] + (right_color[channel] - left_color[channel]) * k))
                for channel in range(3)
            )
    return AGE_STOPS[-1][1]


def palette(states: int, zero_state: int = 0, scheme: str = "age") -> np.ndarray:
    """Палитра ``(states, 3)`` uint8.

    ``scheme="age"`` — состояния растут от тёмного к светлому (возраст/прочность);
    ``scheme="hue"`` — фазы разнесены по кругу тонов (циклические автоматы).
    ``zero_state`` всегда тёмный фон.
    """

    states = max(int(states), 1)
    colors = np.zeros((states, 3), dtype=np.uint8)
    colors[:] = BACKGROUND
    if scheme not in ("age", "hue"):
        raise ValueError(f"Неизвестная схема палитры: {scheme!r}")
    if states <= 1:
        return colors

    if scheme == "age":
        span = states - 1
        for state in range(states):
            if state == zero_state:
                continue
            colors[state] = _interpolate_stops(state / span)
        if 0 <= zero_state < states:
            colors[zero_state] = BACKGROUND
        return colors

    for state in range(states):
        if state == zero_state:
            continue
        hue = ((state - 1) * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
        colors[state] = (int(red * 255), int(green * 255), int(blue * 255))
    if 0 <= zero_state < states:
        colors[zero_state] = BACKGROUND
    return colors


def render_frame(
    grid,
    scale: int = 8,
    *,
    scheme: str = "age",
    colors: np.ndarray | None = None,
    grid_lines: bool = False,
) -> np.ndarray:
    """Один кадр ``(height*scale, width*scale, 3)`` uint8."""

    array = np.asarray(grid.array)
    if colors is None:
        colors = palette(
            int(getattr(grid, "states", int(array.max()) + 1)),
            int(getattr(grid, "zero_state", 0)),
            scheme,
        )
    indexes = np.clip(array, 0, colors.shape[0] - 1)
    rgb = colors[indexes]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if grid_lines and scale >= 4:
        rgb[::scale, :, :] = (rgb[::scale, :, :].astype(np.uint16) * 4 // 5).astype(np.uint8)
        rgb[:, ::scale, :] = (rgb[:, ::scale, :].astype(np.uint16) * 4 // 5).astype(np.uint8)
    return rgb


def legend(
    states: int,
    zero_state: int = 0,
    *,
    scheme: str = "age",
    cell: int = 14,
    gap: int = 2,
    background: tuple[int, int, int] = BACKGROUND,
) -> np.ndarray:
    """Горизонтальная легенда: по квадрату на состояние."""

    states = max(int(states), 1)
    colors = palette(states, zero_state, scheme)
    width = states * cell + (states - 1) * gap
    canvas = np.zeros((cell, width, 3), dtype=np.uint8)
    canvas[:] = background
    for state in range(states):
        left = state * (cell + gap)
        canvas[:, left : left + cell, :] = colors[state]
    return canvas


def compose(
    frames: list[np.ndarray],
    *,
    columns: int | None = None,
    gap: int = 4,
    background: tuple[int, int, int] = BACKGROUND,
) -> np.ndarray:
    """Собрать кадры в один PNG: лента (``columns=None``) или лист."""

    if not frames:
        return np.zeros((max(1, gap), max(1, gap), 3), dtype=np.uint8) + np.array(background, dtype=np.uint8)
    height, width = frames[0].shape[:2]
    count = len(frames)
    columns = count if columns is None else max(1, int(columns))
    rows = (count + columns - 1) // columns
    canvas_width = columns * width + (columns - 1) * gap
    canvas_height = rows * height + (rows - 1) * gap
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = background
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        top = row * (height + gap)
        left = column * (width + gap)
        canvas[top : top + frame.shape[0], left : left + frame.shape[1]] = frame
    return canvas


def save_frame(grid, path: str | Path, *, scale: int = 8, scheme: str = "age", grid_lines: bool = False) -> Path:
    return write_png(path, render_frame(grid, scale, scheme=scheme, grid_lines=grid_lines))


def save_strip(frames: list[np.ndarray], path: str | Path, *, gap: int = 4) -> Path:
    return write_png(path, compose(frames, gap=gap))


def save_sheet(frames: list[np.ndarray], path: str | Path, *, columns: int = 5, gap: int = 4) -> Path:
    return write_png(path, compose(frames, columns=columns, gap=gap))


# ---------------------------------------------------------------------------
# Визуализатор модуля
# ---------------------------------------------------------------------------

_REQUIRED_WITHOUT_SKIP = re.compile(r"^\s*(def|class)\s+")


def can_render(module: object, scheme: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """Проверить, готов ли модуль визуализировать свой кадр.

    Возвращает ``(True, None)`` или ``(False, короткое_объяснение)``.
    Проверка не вызывает код модуля и не требует сторонних зависимостей:
    достаточно, чтобы в нём были имя ``render_frame`` или ``main_for``.

    .. versionadded:: 0.4.0
    """
    if module is None:
        return False, "модуль визуализации не передан"

    if hasattr(module, "render_frame") or hasattr(module, "main_for"):
        return True, None

    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return False, "не удалось получить исходный код модуля"

    lines = [line for line in source.splitlines() if _REQUIRED_WITHOUT_SKIP.match(line)]
    if not lines:
        return False, "в визуализаторе нет ни функции render_frame, ни функции main_for"

    name = getattr(module, "__name__", type(module).__name__)
    return False, f"лишь {name} содержит определения, но ни render_frame, ни main_for не найдено"


def default_view_config(
    states: int,
    *,
    scheme: str = "age",
    zero_state: int = 0,
    range_min: int = _MIN_RADIUS,
    range_max: int = _MAX_RADIUS,
) -> dict:
    """Вернуть конфигурацию визуализации по умолчанию."""
    return {"states": max(int(states), 2), "zero_state": int(zero_state), "scheme": scheme}


def load_view_config(path: str | Path, *, fallback: dict | None = None) -> dict:
    """Загрузить конфигурацию визуализации из JSON-файла с минимальной защитой."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("корень JSON-файла должен быть объектом")
        result = dict(fallback or {})
        result.update(loaded)
        return result
    except FileNotFoundError:
        return dict(fallback or {})
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"не удалось прочитать конфигурацию визуализации {path}: {exc}") from exc
