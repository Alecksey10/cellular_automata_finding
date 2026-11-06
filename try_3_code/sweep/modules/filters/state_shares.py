"""Фильтр долей состояний: границы каждой доли задаются осями профиля."""

from __future__ import annotations

from typing import Any, Mapping

from ...kernels import state_histogram
from ...module import BuildView, FilterModule, FilterRuntime, Observation, Verdict
from .base import REQUIRES_MULTI


class StateSharesRuntime(FilterRuntime):
    """Проверить долю каждой клетки в заданных пользователем границах."""

    def __init__(self, ranges: list[tuple[float, float]], states: int) -> None:
        self.ranges = tuple((float(lo), float(hi)) for lo, hi in ranges)
        self.states = int(states)

    def check(self, observation: Observation) -> Verdict:
        histogram = state_histogram(observation.data, self.states)
        total = float(observation.data.size) or 1.0
        shares = [float(histogram[state]) / total for state in range(self.states)]
        details = {
            "shares": shares,
            "ranges": [[lo, hi] for lo, hi in self.ranges],
        }
        for state, (lo, hi) in enumerate(self.ranges):
            share = shares[state]
            if share < lo - 1e-9:
                details["state"] = state
                details["share"] = share
                return Verdict(False, f"state{state}_low", details)
            if share > hi + 1e-9:
                details["state"] = state
                details["share"] = share
                return Verdict(False, f"state{state}_high", details)
        return Verdict(True, "", details)


class StateSharesFilter(FilterModule):
    """Доли состояний, заданные осями ``dense.share<N>_lo/hi``.

    Профиль попадает в конфиг как обычные скалярные оси: его видно в
    ``successes.jsonl``, и старый прогон воспроизводится даже после правки
    файла профиля.
    """

    id = "filter.state_shares"
    label = "доли состояний"
    requires = REQUIRES_MULTI

    def __init__(self, prefix: str = "dense") -> None:
        self.prefix = str(prefix)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        states = int(fixed["grid2d.states"])
        ranges = [
            (
                float(fixed[f"{self.prefix}.share{state}_lo"]),
                float(fixed[f"{self.prefix}.share{state}_hi"]),
            )
            for state in range(states)
        ]
        return StateSharesRuntime(ranges, states)


__all__ = ["StateSharesFilter", "StateSharesRuntime"]
