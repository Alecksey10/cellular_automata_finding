"""Фильтр кластеров: связные компоненты и их размеры."""

from __future__ import annotations

from typing import Any, Mapping

from ...kernels import as_mask, cluster_stats
from ...module import BuildView, FilterModule, FilterRuntime, Observation, Verdict
from .base import REQUIRES


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
    requires = REQUIRES

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


__all__ = ["ClustersFilter", "ClustersRuntime"]
