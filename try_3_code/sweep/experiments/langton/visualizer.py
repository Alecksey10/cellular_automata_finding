"""Визуализатор эксперимента «муравей Ленгтона»."""

from __future__ import annotations

from ...visualize.interface import RGB, Visualizer, wheel_color


class LangtonVisualizer(Visualizer):
    experiment = "langton"

    def colors(self, states: int) -> int:
        # states = 2 * colors + 1
        return max(1, (int(states) - 1) // 2)

    def state_color(self, state: int, states: int) -> RGB:
        colors = self.colors(states)
        if state <= 0:
            return (10, 10, 12)
        ant = state > colors
        base = state - colors if ant else state
        red, green, blue = wheel_color(base - 1, colors)
        if ant:
            # муравей — тот же цвет, но ярче, чтобы его было видно на следе
            return (min(255, red + 80), min(255, green + 80), min(255, blue + 80))
        return (int(red * 0.55), int(green * 0.55), int(blue * 0.55))

    def legend(self, states: int) -> list[tuple[RGB, str]]:
        colors = self.colors(states)
        entries: list[tuple[RGB, str]] = [((10, 10, 12), "пусто")]
        entries.append((self.state_color(1, states), "след"))
        entries.append((self.state_color(colors + 1, states), "муравей"))
        return entries


__all__ = ["LangtonVisualizer"]
