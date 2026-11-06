"""Фильтр фаз: каждое состояние должно занимать заметную долю кадра."""

from __future__ import annotations

from typing import Any, Mapping

from ...kernels import state_histogram
from ...module import BuildView, FilterModule, FilterRuntime, Observation, Verdict
from .base import REQUIRES_MULTI


class PhasesRuntime(FilterRuntime):
    def __init__(self, min_share: float, states: int) -> None:
        self.min_share = float(min_share)
        self.states = int(states)

    def check(self, observation: Observation) -> Verdict:
        histogram = state_histogram(observation.data, self.states)
        total = float(observation.data.size) or 1.0
        shares = [float(histogram[state]) / total for state in range(self.states)]
        if any(share < self.min_share for share in shares):
            return Verdict(False, "missing_phase", {"shares": shares})
        return Verdict(True, "", {"shares": shares})


class PhasesFilter(FilterModule):
    id = "filter.phases"
    label = "все фазы представлены"
    requires = REQUIRES_MULTI

    def __init__(self, min_share: float = 0.02) -> None:
        self.min_share = float(min_share)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return PhasesRuntime(self.min_share, int(fixed["grid2d.states"]))


__all__ = ["PhasesFilter", "PhasesRuntime"]
