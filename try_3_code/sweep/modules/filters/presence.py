"""Фильтр присутствия: сколько разных состояний видно в кадре."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...module import BuildView, FilterModule, FilterRuntime, Observation, Verdict
from .base import REQUIRES_MULTI


class PresenceRuntime(FilterRuntime):
    """Требовать, чтобы в кадре присутствовало достаточно разных состояний."""

    def __init__(self, min_states: int) -> None:
        self.min_states = int(min_states)

    def check(self, observation: Observation) -> Verdict:
        states = int(np.unique(np.asarray(observation.data)).size)
        details = {"present_states": states, "min_states": self.min_states}
        if states < self.min_states:
            return Verdict(False, "too_few_present_states", details)
        return Verdict(True, "", details)


class PresenceFilter(FilterModule):
    id = "filter.presence"
    label = "разнообразие состояний в кадре"
    requires = REQUIRES_MULTI

    def __init__(self, prefix: str = "dense") -> None:
        self.prefix = str(prefix)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return PresenceRuntime(int(fixed[f"{self.prefix}.min_present_states"]))


__all__ = ["PresenceFilter", "PresenceRuntime"]
