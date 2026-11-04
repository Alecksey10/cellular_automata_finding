"""Визуализатор «двух колец»: градиент прочности и гниения."""

from __future__ import annotations

from ...visualize.interface import RGB, Visualizer, ramp

YOUNG: RGB = (40, 170, 190)
MATURE: RGB = (245, 140, 40)
DEAD: RGB = (14, 16, 22)


class RotAgingVisualizer(Visualizer):
    experiment = "rings_rot"

    def state_color(self, state: int, states: int) -> RGB:
        if state <= 0:
            return DEAD
        return ramp(YOUNG, MATURE, (state - 1) / max(1, states - 2))

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        return [(DEAD, "мёртвая"), (YOUNG, "молодая"), (MATURE, "зрелая")]


__all__ = ["RotAgingVisualizer"]
