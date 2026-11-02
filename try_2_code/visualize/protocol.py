"""Протокол визуализатора и поиск реализации по модулям.

Модуль может объявить атрибут ``visualizer`` — экземпляр или класс
:class:`Visualizer`. Центральный просмотрщик сам проверяет модули и берёт
первую найденную реализацию; если её нет, остаётся общий 2D-стандарт.

Пример::

    from visualize.protocol import Standard2DVisualizer

    class MyRule(Module):
        visualizer = Standard2DVisualizer("hue", name="my-hue")

Визуализатор не обязан быть сложным: достаточно переопределить ``scheme`` или
``render_frame``. Это позволяет правилу с фазами показать их по кругу тонов, а
правилу старения — по возрастной шкале, не меняя общий код галереи.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from casweep.module import Role

from .standard import render_frame as standard_render_frame


class Visualizer:
    """Базовый визуализатор одного кадра автомата."""

    name = "standard2d"
    scheme = "age"
    supported: tuple[str, ...] = ()  # роли модулей, которые этот визуализатор поддерживает

    def render_frame(
        self,
        grid: Any,
        *,
        scale: int = 8,
        scheme: str | None = None,
        grid_lines: bool = False,
    ) -> np.ndarray:
        """Нарисовать кадр ``(height*scale, width*scale, 3)`` uint8."""

        return standard_render_frame(
            grid,
            scale,
            scheme=scheme or self.scheme,
            grid_lines=grid_lines,
        )

    def describe(self) -> str:
        return f"{self.name} ({self.scheme})"


class Standard2DVisualizer(Visualizer):
    """Общий 2D-стандарт с выбираемой палитрой."""

    def __init__(self, scheme: str = "age", *, name: str | None = None) -> None:
        if scheme not in ("age", "hue"):
            raise ValueError(f"Неизвестная схема визуализатора: {scheme!r}")
        self.scheme = scheme
        self.name = name or f"standard2d-{scheme}"
        self.supported = (Role.RULE, Role.NEIGHBORHOOD)  # роли, которые поддерживает этот визуализатор


def _coerce(candidate: Any) -> Visualizer | None:
    """Привести атрибут модуля к экземпляру ``Visualizer``."""

    if candidate is None:
        return None
    if isinstance(candidate, Visualizer):
        return candidate
    if isinstance(candidate, type) and issubclass(candidate, Visualizer):
        return candidate()
    # Допускаем фабрику без аргументов: visualizer = MyVisualizer.
    if callable(candidate):
        try:
            value = candidate()
        except TypeError:
            return None
        if value is not candidate:
            return _coerce(value)
    return None


def visualizer_of(module: Any) -> Visualizer | None:
    """Визуализатор, который объявил модуль, или ``None``."""

    if module is None:
        return None
    try:
        candidate = getattr(module, "visualizer", None)
    except Exception:  # noqa: BLE001 - битое свойство не должно ломать просмотр
        return None
    return _coerce(candidate)


def has_visualizer(module: Any) -> bool:
    return visualizer_of(module) is not None


def visualizer_name(module: Any) -> str | None:
    visualizer = visualizer_of(module)
    return None if visualizer is None else visualizer.describe()


def resolve_visualizer(assembled: Any) -> Visualizer:
    """Выбрать визуализатор: правило, затем окрестность, затем данные.

    Фильтры намеренно не участвуют: они не рисуют сетку. Если ни один модуль
    не объявил визуализатор, используется общий 2D-стандарт.
    """

    modules = getattr(assembled, "modules", {}) or {}
    for role in (Role.RULE, Role.NEIGHBORHOOD, Role.DATA):
        visualizer = visualizer_of(modules.get(role))
        if visualizer is not None:
            return visualizer
    return Standard2DVisualizer()
