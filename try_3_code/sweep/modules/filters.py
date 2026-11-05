"""Модули фильтрации: решают, продолжать ли выполнение конфига.

Фильтр не знает о стадиях и итерациях: он получает
:class:`~sweep.module.Observation` и возвращает :class:`~sweep.module.Verdict`.
Когда именно вызывать ``check`` — решает движок по ``Phase.frequency``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..interfaces import CapabilityKind, Requirement
from ..kernels import as_mask, cluster_stats, mask_difference, mask_iou, state_histogram
from ..module import (
    BuildView,
    FilterModule,
    FilterRuntime,
    Observation,
    PASS,
    Role,
    Verdict,
)

_REQUIRES = (
    Requirement(CapabilityKind.GRID_2D),
    Requirement(CapabilityKind.ZERO_STATE),
)


class DensityRuntime(FilterRuntime):
    def __init__(self, min_fill: float, max_fill: float) -> None:
        self.min_fill = float(min_fill)
        self.max_fill = float(max_fill)

    def check(self, observation: Observation) -> Verdict:
        density = float(observation.metrics.get("alive_density", 0.0))
        if density < self.min_fill:
            return Verdict(False, "too_sparse", {"alive_density": density})
        if density > self.max_fill:
            return Verdict(False, "too_dense", {"alive_density": density})
        return PASS


class DensityFilter(FilterModule):
    id = "filter.density"
    label = "допустимое заполнение"
    requires = _REQUIRES

    def __init__(
        self,
        min_fill: float = 0.02,
        max_fill: float = 0.95,
        prefix: str | None = None,
    ) -> None:
        self.min_fill = float(min_fill)
        self.max_fill = float(max_fill)
        self.prefix = prefix

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        if self.prefix:
            min_fill = float(fixed[f"{self.prefix}.density_min"])
            max_fill = float(fixed[f"{self.prefix}.density_max"])
        else:
            min_fill = self.min_fill
            max_fill = self.max_fill
        return DensityRuntime(min_fill, max_fill)


class StabilityRuntime(FilterRuntime):
    def __init__(self, max_change: float, forbid_period2: bool) -> None:
        self.max_change = float(max_change)
        self.forbid_period2 = bool(forbid_period2)
        self._previous: np.ndarray | None = None
        self._before_previous: np.ndarray | None = None

    def check(self, observation: Observation) -> Verdict:
        zero = int(observation.metrics.get("zero", 0))
        mask = as_mask(observation.data, zero)
        if self._previous is None:
            self._previous = mask
            return PASS
        difference = float(mask_difference(self._previous, mask))
        period2 = self._before_previous is not None and np.array_equal(self._before_previous, mask)
        self._before_previous = self._previous
        self._previous = mask
        if period2 and self.forbid_period2:
            return Verdict(False, "period2", {"difference": difference})
        if difference > self.max_change:
            return Verdict(False, "unstable", {"difference": difference})
        return PASS


class StabilityFilter(FilterModule):
    id = "filter.stability"
    label = "стабилизация"
    requires = _REQUIRES

    def __init__(self, max_change: float = 0.3, forbid_period2: bool = True) -> None:
        self.max_change = float(max_change)
        self.forbid_period2 = bool(forbid_period2)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return StabilityRuntime(self.max_change, self.forbid_period2)


class ClustersRuntime(FilterRuntime):
    def __init__(self, min_components: int, min_mean_size: float, max_largest_share: float) -> None:
        self.min_components = int(min_components)
        self.min_mean_size = float(min_mean_size)
        self.max_largest_share = float(max_largest_share)

    def check(self, observation: Observation) -> Verdict:
        zero = int(observation.metrics.get("zero", 0))
        mask = as_mask(observation.data, zero)
        components, mean_size, largest_share = cluster_stats(mask)
        details = {
            "components": int(components),
            "mean_size": float(mean_size),
            "largest_share": float(largest_share),
        }
        if components < self.min_components:
            return Verdict(False, "too_few_components", details)
        if mean_size < self.min_mean_size:
            return Verdict(False, "small_clusters", details)
        if largest_share > self.max_largest_share:
            return Verdict(False, "one_huge_cluster", details)
        return Verdict(True, "", details)


class ClustersFilter(FilterModule):
    id = "filter.clusters"
    label = "структура кластеров"
    requires = _REQUIRES

    def __init__(
        self,
        min_components: int = 1,
        min_mean_size: float = 3.0,
        max_largest_share: float = 0.95,
    ) -> None:
        self.min_components = int(min_components)
        self.min_mean_size = float(min_mean_size)
        self.max_largest_share = float(max_largest_share)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return ClustersRuntime(self.min_components, self.min_mean_size, self.max_largest_share)


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
    requires = _REQUIRES + (Requirement(CapabilityKind.MULTI_STATE),)

    def __init__(self, min_share: float = 0.02) -> None:
        self.min_share = float(min_share)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return PhasesRuntime(self.min_share, int(fixed["grid2d.states"]))


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
    requires = _REQUIRES + (Requirement(CapabilityKind.MULTI_STATE),)

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
    requires = _REQUIRES + (Requirement(CapabilityKind.MULTI_STATE),)

    def __init__(self, prefix: str = "dense") -> None:
        self.prefix = str(prefix)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return PresenceRuntime(int(fixed[f"{self.prefix}.min_present_states"]))


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
    requires = _REQUIRES + (Requirement(CapabilityKind.MULTI_STATE),)

    def __init__(self, split_key: str = "rule.duel.split") -> None:
        self.split_key = str(split_key)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        split = int(fixed.get(self.split_key, max(1, int(fixed["grid2d.states"]) // 2)))
        return GroupPresenceRuntime(split)


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
    requires = _REQUIRES + (Requirement(CapabilityKind.MULTI_STATE),)

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


class SymmetryRuntime(FilterRuntime):
    def __init__(self, zero: int, min_iou: float) -> None:
        self.zero = int(zero)
        self.min_iou = float(min_iou)

    def check(self, observation: Observation) -> Verdict:
        mask = as_mask(observation.data, self.zero)
        # 0.0, а не 1.0: иначе max() всегда даёт >= 1 и фильтр не может отказать
        best = 0.0
        candidates = [mask[:, ::-1], mask[::-1, :], mask[::-1, ::-1]]
        for turns in (1, 2, 3):
            rotated = np.ascontiguousarray(np.rot90(mask, turns), dtype=np.uint8)
            if rotated.shape != mask.shape:
                continue  # неквадратную сетку можно сравнивать только с отражениями
            candidates.append(rotated)
            candidates.append(np.ascontiguousarray(np.fliplr(rotated), dtype=np.uint8))
        for candidate in candidates:
            best = max(best, float(mask_iou(mask, np.ascontiguousarray(candidate, dtype=np.uint8))))
        if best < self.min_iou:
            return Verdict(False, "asymmetric", {"iou": best})
        return PASS


class SymmetryFilter(FilterModule):
    id = "filter.symmetry"
    label = "симметрия"
    requires = _REQUIRES

    def __init__(self, min_iou: float = 0.9) -> None:
        self.min_iou = float(min_iou)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> FilterRuntime:
        return SymmetryRuntime(int(fixed["grid2d.zero"]), self.min_iou)


ALL_FILTERS = [
    DensityFilter,
    StabilityFilter,
    ClustersFilter,
    PhasesFilter,
    SymmetryFilter,
    StateSharesFilter,
    DynamicsFilter,
    PresenceFilter,
]

__all__ = [
    "ALL_FILTERS",
    "ClustersFilter",
    "ClustersRuntime",
    "DensityFilter",
    "DensityRuntime",
    "DynamicsFilter",
    "DynamicsRuntime",
    "PhasesFilter",
    "PhasesRuntime",
    "PresenceFilter",
    "PresenceRuntime",
    "StabilityFilter",
    "StabilityRuntime",
    "StateSharesFilter",
    "StateSharesRuntime",
    "SymmetryFilter",
    "SymmetryRuntime",
]
