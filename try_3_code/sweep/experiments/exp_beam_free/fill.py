"""Инициализация для дуэли двух групп."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...module import RandomizerModule


class BiasedDuelFill(RandomizerModule):
    """Слева преимущественно группа A, справа — группа B."""

    id = "randomizer.biased_duel"
    label = "боковая уплотнённая заливка"

    def declare(self, ns, view) -> None:
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
        height, width = data.shape
        split = _split_from_fixed(fixed, states)
        zero_prob = 0.12
        for y in range(height):
            for x in range(width):
                frac = x / max(1, width - 1)
                p_a = 0.62 * (1.0 - frac) + 0.08
                p_b = 0.62 * frac + 0.08
                p_zero = zero_prob
                total = p_a + p_b + p_zero
                p_a /= total
                p_b /= total
                p_zero /= total
                roll = float(rng.random())
                if roll < p_zero:
                    data[y, x] = zero_state
                elif roll < p_zero + p_a:
                    data[y, x] = int(rng.integers(1, split + 1))
                else:
                    data[y, x] = int(rng.integers(split + 1, states))


def _split_from_fixed(fixed: Mapping[str, Any], states: int) -> int:
    for key, value in fixed.items():
        if key.endswith(".split"):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return max(5, (states - 1) // 2 + 1)


__all__ = ["BiasedDuelFill"]
