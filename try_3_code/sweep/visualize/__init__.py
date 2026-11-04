"""Визуализация: общий интерфейс, PNG/HTML-движок, терминал."""

from .engine import compose, render, render_grid
from .interface import RGB, Rendered, Visualizer, ramp, wheel_color
from .png import write_png
from .terminal import render_terminal

__all__ = [
    "RGB",
    "Rendered",
    "Visualizer",
    "compose",
    "ramp",
    "render",
    "render_grid",
    "render_terminal",
    "wheel_color",
    "write_png",
]
