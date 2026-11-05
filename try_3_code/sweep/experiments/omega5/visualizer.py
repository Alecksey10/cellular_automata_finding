"""Визуализатор ``omega5``: восемь различимых состояний."""

from __future__ import annotations

from typing import Any, Mapping

from ...visualize.interface import RGB, Visualizer
from .rule import describe_rule, rule_descriptor

DEAD: RGB = (10, 11, 15)
PALETTE: tuple[RGB, ...] = (
    (225, 235, 255),  # 1 — светлая плазма
    (255, 196, 64),   # 2 — янтарь
    (240, 92, 92),    # 3 — красный
    (120, 230, 140),  # 4 — хищник
    (80, 180, 255),   # 5 — синий
    (200, 120, 255),  # 6 — фиолетовый
    (255, 120, 220),  # 7 — маджента
)


class OmegaVisualizer(Visualizer):
    experiment = "omega5"

    def state_color(self, state: int, states: int) -> RGB:
        if state <= 0:
            return DEAD
        return PALETTE[(state - 1) % len(PALETTE)]

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        return [(DEAD, "0 — мёртвая")] + [
            (self.state_color(state, states), f"{state}") for state in range(1, states)
        ]

    def rule_info(self, values: Mapping[str, Any]) -> dict[str, Any]:
        info = rule_descriptor(values)
        info["text"] = describe_rule(values)
        return info


__all__ = ["OmegaVisualizer", "PALETTE"]
