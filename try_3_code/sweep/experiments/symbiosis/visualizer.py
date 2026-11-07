"""Визуализатор симбиоза: опора, зависимые, споры и импульсы."""

from __future__ import annotations

from ...visualize.interface import RGB, Visualizer

BACKGROUND: RGB = (10, 10, 16)
ANCHOR: RGB = (128, 92, 226)
MATRIX: RGB = (58, 132, 232)
GRAZER: RGB = (72, 200, 118)
WEAVER: RGB = (238, 158, 62)
SPORE: RGB = (222, 214, 142)
PULSE: RGB = (232, 72, 86)

COLORS = (BACKGROUND, ANCHOR, MATRIX, GRAZER, WEAVER, SPORE, PULSE)
LABELS = (
    "пустота",
    "опора R20",
    "матрица R30",
    "пасущийся (R5)",
    "ткач (R5)",
    "спора",
    "импульс",
)


class SymbiosisVisualizer(Visualizer):
    experiment = "symbiosis"

    def state_color(self, state: int, states: int) -> RGB:
        return COLORS[state] if 0 <= state < len(COLORS) else BACKGROUND

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        return [(COLORS[index], LABELS[index]) for index in range(len(COLORS))]


__all__ = ["SymbiosisVisualizer"]
