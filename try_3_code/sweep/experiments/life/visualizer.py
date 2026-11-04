"""Визуализатор эксперимента «жизнь»."""

from __future__ import annotations

from ...visualize.interface import RGB, Visualizer


class LifeVisualizer(Visualizer):
    experiment = "life"

    def state_color(self, state: int, states: int) -> RGB:
        return (16, 16, 20) if state == 0 else (240, 240, 240)

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        return [((16, 16, 20), "мёртвая"), ((240, 240, 240), "живая")]


__all__ = ["LifeVisualizer"]
