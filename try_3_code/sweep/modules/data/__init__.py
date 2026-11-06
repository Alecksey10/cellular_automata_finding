"""Модули данных: сетки, на которых работают правила.

Одна реализация — один файл. Новая сетка (например, 3D или гексагональная)
добавляется файлом рядом и реэкспортом отсюда; движок и эксперименты
менять не нужно.
"""

from __future__ import annotations

from .grid2d import Grid2D
from .symmetric_grid2d import SymmetricGrid2D

__all__ = ["Grid2D", "SymmetricGrid2D"]
