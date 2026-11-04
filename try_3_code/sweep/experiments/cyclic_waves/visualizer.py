"""Визуализатор «циклических волн»: круг тонов по фазам."""

from __future__ import annotations

from ...visualize.interface import RGB, Visualizer, wheel_color


class CyclicVisualizer(Visualizer):
    experiment = "cyclic_waves"

    def state_color(self, state: int, states: int) -> RGB:
        if state <= 0:
            return (10, 10, 14)
        return wheel_color(state, max(1, states))

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        return [
            (self.state_color(state, states), f"фаза {state}")
            for state in range(states)
        ]


__all__ = ["CyclicVisualizer"]
