"""Терминальный просмотр кадра."""

from __future__ import annotations

import numpy as np

from .interface import Visualizer

RESET = "\x1b[0m"


def ansi(color: tuple[int, int, int]) -> str:
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m"


def render_terminal(data: np.ndarray, visualizer: Visualizer, *, states: int, scale: int = 1) -> str:
    height, width = data.shape
    lines: list[str] = []
    step = 2 if scale >= 2 else 1
    for y in range(0, height, step):
        chunks: list[str] = []
        for x in range(0, width, step):
            state = int(data[y, x])
            if state >= states:
                state = states - 1
            chunks.append(ansi(visualizer.state_color(state, states)) + "█")
        lines.append("".join(chunks) + RESET)
    return "\n".join(lines)


__all__ = ["ansi", "render_terminal"]
