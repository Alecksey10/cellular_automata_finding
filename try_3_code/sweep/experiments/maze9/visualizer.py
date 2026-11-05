"""Визуализатор эксперимента «лабиринт 9×9»."""

from __future__ import annotations

from typing import Any, Mapping

from ...visualize.interface import RGB, Visualizer


class Maze9Visualizer(Visualizer):
    experiment = "maze9"

    def state_color(self, state: int, states: int) -> RGB:
        return (12, 14, 18) if state == 0 else (206, 212, 224)

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        return [((12, 14, 18), "проход"), ((206, 212, 224), "стена")]

    def rule_info(self, values: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return {
            "окно": "9×9",
            "механизм": values.get("rule.maze9.mechanism"),
            "рождение %": [values.get("rule.maze9.born_lo"), values.get("rule.maze9.born_hi")],
            "выживание %": [values.get("rule.maze9.surv_lo"), values.get("rule.maze9.surv_hi")],
            "rate": values.get("rule.maze9.rate"),
        }


__all__ = ["Maze9Visualizer"]
