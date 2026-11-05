"""Визуализатор эксперимента «лабиринт»."""

from __future__ import annotations

from ...visualize.interface import RGB, Visualizer


class MazeVisualizer(Visualizer):
    experiment = "maze"

    def state_color(self, state: int, states: int) -> RGB:
        return (12, 14, 18) if state == 0 else (206, 212, 224)

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        return [((12, 14, 18), "проход"), ((206, 212, 224), "стена")]


__all__ = ["MazeVisualizer"]
