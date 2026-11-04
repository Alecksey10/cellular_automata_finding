"""Общий интерфейс визуализаторов.

Визуализатор определяет только цвет состояния и подписи; раскладку кадров,
масштаб и запись PNG/HTML делает движок (:mod:`sweep.visualize.engine`).
Поэтому новый эксперимент добавляет один небольшой класс.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Rendered:
    png: Path
    html: Path | None = None


def clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


def ramp(start: RGB, end: RGB, t: float) -> RGB:
    t = max(0.0, min(1.0, float(t)))
    return tuple(clamp(a + (b - a) * t) for a, b in zip(start, end))  # type: ignore[return-value]


def wheel_color(state: int, states: int) -> RGB:
    if states <= 1:
        return (255, 255, 255)
    hue = (state % states) / states
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return clamp(r * 255), clamp(g * 255), clamp(b * 255)


class Visualizer:
    """Базовый визуализатор: белый на чёрном."""

    experiment: str = ""

    def state_color(self, state: int, states: int) -> RGB:
        if state == 0:
            return (16, 16, 20)
        return (235, 235, 235)

    def caption(self, iteration: int, phase: str, metrics: dict[str, Any] | None = None) -> str:
        return f"{phase} · итерация {iteration}"

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        return []

    def background(self) -> RGB:
        return (8, 8, 10)


__all__ = ["RGB", "Rendered", "Visualizer", "clamp", "ramp", "wheel_color"]
