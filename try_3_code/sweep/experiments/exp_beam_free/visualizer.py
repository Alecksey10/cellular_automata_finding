"""Визуализатор ``expBeamFree``."""

from __future__ import annotations

from ...visualize.interface import RGB, Visualizer, wheel_color


class DuelVisualizer(Visualizer):
    experiment = "expBeamFree"

    def state_color(self, state: int, states: int) -> RGB:
        if state <= 0:
            return (10, 11, 15)
        split = max(1, (states - 1) // 2)
        if state <= split:
            # группа A: сине-голубая гамма с уровнями
            level = (state - 1) / max(1, split - 1)
            return (
                int(40 + 40 * level),
                int(90 + 120 * level),
                int(180 + 60 * level),
            )
        # группа B: оранжево-красная гамма с уровнями
        level = (state - split - 1) / max(1, states - split - 2)
        return (
            int(180 + 60 * level),
            int(70 + 80 * level),
            int(40 + 40 * level),
        )

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        split = max(1, (states - 1) // 2)
        return [
            ((10, 11, 15), "0 — пустая"),
            (self.state_color(1, states), f"1..{split} — группа A"),
            (self.state_color(split + 1, states), f"{split + 1}..{states - 1} — группа B"),
        ]


__all__ = ["DuelVisualizer"]
