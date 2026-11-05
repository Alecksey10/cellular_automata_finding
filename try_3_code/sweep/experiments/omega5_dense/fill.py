"""Инициализация сетки по профилю долей.

Заполнение не обязано быть случайным «по одному состоянию»: профиль задаёт
допустимые доли, из них строится совместное распределение, и каждая клетка
получает состояние по этому распределению. Так поиск плотных автоматов
начинается уже в нужном режиме, а не тратит почти всё время на вымирание.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...module import RandomizerModule


def feasible_distribution(ranges: list[tuple[float, float]]) -> np.ndarray:
    """Распределение внутри границ, сумма ровно 1.

    Сначала берём минимумы, остаток делим пропорционально свободному запасу
    ``hi - lo``. Валидатор профиля гарантирует, что остаток неотрицателен.
    """

    lows = np.asarray([lo for lo, _ in ranges], dtype=np.float64)
    highs = np.asarray([hi for _, hi in ranges], dtype=np.float64)
    room = highs - lows
    remaining = 1.0 - float(lows.sum())
    if remaining <= 0.0 or float(room.sum()) <= 0.0:
        probs = lows.copy()
    else:
        probs = lows + room * (remaining / float(room.sum()))
    total = float(probs.sum())
    if total <= 0.0:
        probs = np.full(len(ranges), 1.0 / len(ranges), dtype=np.float64)
    else:
        probs = probs / total
    return probs


class ProfileFill(RandomizerModule):
    id = "randomizer.profile"
    label = "заливка по профилю долей"

    def declare(self, ns, view) -> None:
        # Профиль уже объявлен экспериментом как оси dense.share*.
        return None

    def fill(
        self,
        data: np.ndarray,
        fixed: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        zero_state: int,
        states: int,
    ) -> None:
        ranges = [
            (
                float(fixed[f"dense.share{state}_lo"]),
                float(fixed[f"dense.share{state}_hi"]),
            )
            for state in range(states)
        ]
        probs = feasible_distribution(ranges)
        choices = rng.choice(states, size=data.size, p=probs)
        data[:] = choices.reshape(data.shape).astype(np.int32)


__all__ = ["ProfileFill", "feasible_distribution"]
