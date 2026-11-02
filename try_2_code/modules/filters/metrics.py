"""Общие метрики фильтров: симметрия, энтропия, доли состояний.

Здесь только numpy, без numba: считается один раз на проверку, а проверки
фильтров дешёвые по сравнению с итерацией ядра.
"""

from __future__ import annotations

import numpy as np


def live_mask(grid) -> np.ndarray:
    """Маска «клетка не в нулевом состоянии»."""

    return np.asarray(grid.array != grid.zero_state)


def symmetry_score(live: np.ndarray) -> float:
    """IoU маски с лучшим её поворотом/отражением.

    1.0 — маска переходит в себя при каком-то повороте или отражении.
    Сравнивается объединение, а не все клетки: иначе одна живая клетка давала
    бы 0.98 (она совпадает с собственным отражением почти везде).
    Для неквадратной сетки повороты на 90° меняют форму и не учитываются.
    """

    height, width = live.shape
    total = int(np.count_nonzero(live))
    if total == 0:
        return 0.0
    best = 0.0
    variants: list[np.ndarray] = []
    for turns in (1, 2, 3):
        rotated = np.rot90(live, turns)
        if rotated.shape == live.shape:
            variants.append(rotated)
    variants.append(np.fliplr(live))
    variants.append(np.flipud(live))
    if height == width:
        variants.append(live.T)
        variants.append(np.fliplr(live.T))
    for variant in variants:
        if variant.shape != live.shape:
            continue
        intersection = int(np.count_nonzero(variant & live))
        union = int(np.count_nonzero(variant | live))
        if union == 0:
            continue
        score = intersection / union
        if score > best:
            best = score
    return best


def normalized_entropy(grid) -> float:
    """Энтропия распределения состояний, нормированная на ``log(states)``."""

    states = int(grid.states)
    if states < 2:
        return 0.0
    counts = np.bincount(grid.array.ravel(), minlength=states)[:states]
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return entropy / float(np.log(states))


def phase_shares(grid) -> np.ndarray:
    """Доля каждой фазы (состояния) в сетке."""

    states = int(grid.states)
    counts = np.bincount(grid.array.ravel(), minlength=states)[:states]
    total = float(grid.array.size)
    if total <= 0:
        return np.zeros(states, dtype=float)
    return counts.astype(float) / total
