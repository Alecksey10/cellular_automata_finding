"""Простая дискретная сетка для клеточных автоматов.

Архитектура:
1. AbstractGridDef - описание размеров и свойств (без массива данных)
2. GridArray - сам numpy массив со состояниями
3. DiscreteGrid - готовый объект с удобными интерфейсами (ZeroState и др.)
4. Все параметры перебора - просто функции/конфиги, никаких "оберток ради оберток"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 1. Abstract description of grid dimensions and properties
# ---------------------------------------------------------------------------

@dataclass
class GridDef:
    """Описывает структуру grids — размеры, состояния, фон.

    Это "конфиг" для создания массива. Никакой логики перебора здесь нет —
    просто данные, которые можно передавать вокруг.
    """

    width: int
    height: int
    states: int = 2
    zero_state: int = 0

    def validate(self) -> None:
        """Базовая валидация."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width и height должны быть > 0")
        if self.states < 1:
            raise ValueError("states должно быть >= 1")
        if not (0 <= self.zero_state < self.states):
            raise ValueError(
                f"zero_state={self.zero_state} fuera диапаста [0, {self.states})"
            )


# ---------------------------------------------------------------------------
# 2. The array itself — numpy-backed storage
# ---------------------------------------------------------------------------

class GridArray:
    """Нumpy-массив с состояниями клеток.

    Содержит raw данные. Все вычисления numba работают с этим классом
    напрямую через атрибут .array (который есть и есть numpy).
    """

    def __init__(self, defn: GridDef) -> None:
        self.defn = defn
        # Создаем массив: (height, width) заполненный zero_state
        self.array = np.full(
            (defn.height, defn.width), defn.zero_state, dtype=np.int32
        )
        # Экспонируем полезные атрибуты numpy для удобства
        self.size = self.array.size
        self.shape = self.array.shape

    def like(self) -> "GridArray":
        """Создать пустой массив того же размера/типа."""
        return GridArray(self.defn)

    def __repr__(self) -> str:
        return f"GridArray(shape=({self.defn.height}, {self.defn.width}), " f"states={self.defn.states}, zero={self.defn.zero_state})"


# ---------------------------------------------------------------------------
# 3. Ready-to-use grid with interfaces (ZeroState, MultiState, etc.)
# ---------------------------------------------------------------------------

@dataclass
class DiscreteGrid:
    """Готовый объект сетки с удобными интерфейсами.

    Содержит массив и метаданные. Создается из GridDef + optionally
    initialized array.
    """

    array: GridArray
    states: int
    zero_state: int

    # --- Properties for compatibility with rules/visualizers ---

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.array.shape  # (height, width)

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    def live_fraction(self) -> float:
        """Доля клеток, отличных от zero_state."""
        return float(
            np.count_nonzero(self.array.array != self.zero_state)
            / float(self.array.size)
        )

    def count_of(self, state: int) -> int:
        return int(np.count_nonzero(self.array.array == state))

    def __repr__(self) -> str:
        return f"DiscreteGrid(shape={self.shape}, states={self.states}, zero={self.zero_state})"


# ---------------------------------------------------------------------------
# 4. Factory — creates DiscreteGrid from GridDef
# ---------------------------------------------------------------------------

def create_grid(defn: GridDef, *, init_array: np.ndarray | None = None) -> DiscreteGrid:
    """Сreate a DiscreteGrid from a GridDef.

    Args:
        defn: Grid dimensions and properties
        init_array: Optional pre-initialized array. If None, creates one
                   filled with zero_state.

    Returns:
        DiscreteGrid ready for use
    """
    defn.validate()
    if init_array is None:
        array = GridArray(defn)
    else:
        if init_array.shape != (defn.height, defn.width):
            raise ValueError(
                f"Array shape {init_array.shape} != expected ({defn.height}, {defn.width})"
            )
        array = GridArray.__new__(GridArray)
        array.defn = defn
        array.array = np.asarray(init_array, dtype=np.int32)

    return DiscreteGrid(
        array=array,
        states=defn.states,
        zero_state=defn.zero_state,
    )


# ---------------------------------------------------------------------------
# 5. Simple parameter generation (NO class-spread sweeping logic)
# ---------------------------------------------------------------------------

def make_default_grid_def() -> GridDef:
    """Create a GridDef with reasonable default ranges.

    This replaces the old `namespace.range()` calls with simple functions.
    The caller decides how to vary these values — via experiments, loops, etc.
    """
    return GridDef(width=64, height=64, states=2, zero_state=0)


def generate_random_def(rng: np.random.Generator, *, seed: int | None = None) -> GridDef:
    """Generate a random GridDef.

    Useful for random search / evolution strategies.
    """
    if seed is not None:
        rng = rng.model_reset(seed=seed) if hasattr(rng, "model_reset") else rng

    # Simple ranges - user can customize
    width = int(rng.integers(16, 129))  # 16..128
    height = int(rng.integers(max(16, width), 130))  # >= width, 16..129
    states = int(rng.integers(2, 11))  # 2..10
    zero_state = int(rng.integers(0, states))

    return GridDef(width=width, height=height, states=states, zero_state=zero_state)


# ---------------------------------------------------------------------------
# 6. Interface checks (duck-typing, no inheritance required)
# ---------------------------------------------------------------------------

def has_zero_state(obj: Any) -> bool:
    """Check if object has zero_state interface."""
    return hasattr(obj, "zero_state") and isinstance(
        getattr(obj, "zero_state", None), int
    )


def has_multistate(obj: Any) -> bool:
    """Check if object has multi-state interface."""
    return hasattr(obj, "states") and isinstance(getattr(obj, "states", None), int) and obj.states >= 2


def is_discrete_2d(obj: Any) -> bool:
    """Check if object is a 2D discrete grid (duck-type check)."""
    return (
        hasattr(obj, "shape")
        and len(obj.shape) == 2
        and hasattr(obj, "zero_state")
        and hasattr(obj, "states")
    )


# Example usage pattern:
# ----------------------
# def1 = make_default_grid_def()
# grid = create_grid(def1)
# print(grid.width, grid.height)
# print(grid.live_fraction())
# print(is_discrete_2d(grid))