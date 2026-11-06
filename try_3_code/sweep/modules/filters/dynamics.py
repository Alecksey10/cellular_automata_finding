"""Фильтр динамики: состояния должны реально меняться, но не хаотично."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...module import BuildView, FilterModule, FilterRuntime, Observation, PASS, Verdict
from .base import REQUIRES_MULTI


class DynamicsRuntime(FilterRuntime):
    """Требовать реального изменения состояний на каждом шаге.

    В отличие от ``StabilityRuntime`` сравниваются не только маски «живая /
    мёртвая», а сами номера состояний. Поэтому автомат, у которого все клетки
    живы, но цвета ежешагово перекрашиваются, здесь не считается статичным.
    """

    def __init__(
        self,
        min_change: float = 0.0,
        max_change: float = 1.0,
        min_states: int = 0,
        min_transitions: int = 0,
    ) -> None:
        self.min_change = float(min_change)
        self.max_change = float(max_change)
        self.min_states = int(min_states)
        self.min_transitions = int(min_transitions)
        self._previous: np.ndarray | None = None

    def check(self, observation: Observation) -> Verdict:
        data = np.asarray(observation.data)
        if self._previous is None:
            self._previous = np.array(data, copy=True)
            return PASS  # сравнивать пока не с чем

        previous = self._previous
        self._previous = np.array(data, copy=True)
        changed_mask = data != previous
        changed = int(np.count_nonzero(changed_mask))
        total = int(data.size) or 1
        fraction = changed / total

        if changed:
            old_states = previous[changed_mask].astype(np.int64)
            new_states = data[changed_mask].astype(np.int64)
            states = int(np.unique(np.concatenate((old_states, new_states))).size)
            transitions = int(np.unique(old_states * 256 + new_states).size)
        else:
            states = 0
            transitions = 0

        details = {
            "changed_fraction": fraction,
            "changed_cells": changed,
            "changed_states": states,
            "transitions": transitions,
        }
        if fraction < self.min_change - 1e-12:
            return Verdict(False, "static", details)
        if fraction > self.max_change + 1e-12:
            return Verdict(False, "too_chaotic", details)
        if states < self.min_states:
            return Verdict(False, "few_changed_states", details)
        if transitions < self.min_transitions:
            return Verdict(False, "few_transitions", details)
        return Verdict(True, "", details)


class DynamicsFilter(FilterModule):
    id = "filter.dynamics"
    label = "динамика состояний между итерациями"
    requires = REQUIRES_MULTI

    def __init__(
        self,
        prefix: str | None = None,
        min_change: float = 0.02,
        max_change: float = 0.30,
        min_states: int = 0,
        min_transitions: int = 0,
    ) -> None:
        self.prefix = prefix
        self.min_change = float(min_change)
        self.max_change = float(max_change)
        self.min_states = int(min_states)
        self.min_transitions = int(min_transitions)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        if self.prefix:
            min_change = float(fixed.get(f"{self.prefix}.min_change", self.min_change))
            max_change = float(fixed.get(f"{self.prefix}.max_change", self.max_change))
            min_states = int(fixed.get(f"{self.prefix}.min_states", self.min_states))
            min_transitions = int(fixed.get(f"{self.prefix}.min_transitions", self.min_transitions))
        else:
            min_change = self.min_change
            max_change = self.max_change
            min_states = self.min_states
            min_transitions = self.min_transitions
        return DynamicsRuntime(min_change, max_change, min_states, min_transitions)


__all__ = ["DynamicsFilter", "DynamicsRuntime"]
