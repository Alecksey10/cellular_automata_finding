"""Фильтр групп: в кадре должны быть обе группы клеток дуэли."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...module import BuildView, FilterModule, FilterRuntime, Observation, Verdict
from .base import REQUIRES_MULTI


class GroupPresenceRuntime(FilterRuntime):
    """Проверить, что в кадре присутствуют обе группы клеток."""

    def __init__(self, split: int) -> None:
        self.split = int(split)

    def check(self, observation: Observation) -> Verdict:
        data = np.asarray(observation.data)
        groups = {1 if 1 <= int(value) <= self.split else 2 for value in data.flat if int(value) != 0}
        details = {"groups": sorted(groups), "split": self.split}
        if len(groups) < 2:
            return Verdict(False, "single_group", details)
        return Verdict(True, "", details)


class GroupPresenceFilter(FilterModule):
    id = "filter.group_presence"
    label = "обе группы присутствуют"
    requires = REQUIRES_MULTI

    def __init__(self, split_key: str = "rule.duel.split") -> None:
        self.split_key = str(split_key)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        split = int(fixed.get(self.split_key, max(1, int(fixed["grid2d.states"]) // 2)))
        return GroupPresenceRuntime(split)


__all__ = ["GroupPresenceFilter", "GroupPresenceRuntime"]
