"""Совместимый фасад над общим 2D-стандартом (см. :mod:`visualize.standard`).

Здесь остались только имена, которые использовались раньше; вся отрисовка —
в ``standard``. Любой автомат с ``.array``/``.states``/``.zero_state`` рисуется
через один и тот же код.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .png import write_png
from .standard import (
    BACKGROUND,
    legend,
    palette as _palette,
    render_frame,
)

_ASCII_RAMP = " .:-=+*#%@"


def palette(states: int, zero_state: int = 0, scheme: str = "age") -> np.ndarray:
    return _palette(states, zero_state, scheme)


def render_pixels(
    grid,
    scale: int = 8,
    colors: np.ndarray | None = None,
    *,
    scheme: str = "age",
) -> np.ndarray:
    return render_frame(grid, scale, scheme=scheme, colors=colors)


def save_grid(
    grid,
    path: str | Path,
    scale: int = 8,
    colors: np.ndarray | None = None,
    *,
    scheme: str = "age",
    grid_lines: bool = False,
) -> Path:
    return write_png(
        path,
        render_frame(grid, scale, scheme=scheme, colors=colors, grid_lines=grid_lines),
    )


def ascii_grid(grid, max_width: int = 100) -> str:
    """Компактное текстовое представление для терминала."""

    array = np.asarray(grid.array)
    height, width = array.shape
    step = max(1, (width + max_width - 1) // max_width)
    states = max(int(grid.states), 2)
    lines = []
    for row in range(0, height, step):
        chars = []
        for column in range(0, width, step):
            level = int(array[row, column]) * (len(_ASCII_RAMP) - 1) // max(states - 1, 1)
            chars.append(_ASCII_RAMP[level])
        lines.append("".join(chars))
    return "\n".join(lines)


__all__ = [
    "BACKGROUND",
    "ascii_grid",
    "legend",
    "palette",
    "render_frame",
    "render_pixels",
    "save_grid",
]
