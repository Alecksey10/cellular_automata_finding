"""Фильтры структуры: симметрия, энтропия, баланс фаз, кластеры."""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import FilterVerdict

from .kernels import component_stats
from .metrics import live_mask, normalized_entropy, phase_shares, symmetry_score


class SymmetryFilter(Module):
    """Требует, чтобы маска была близка к симметричной.

    Правила «симметричны по построению», но начальное заполнение — нет; фильтр
    проверяет, что динамика пришла к симметричной структуре (кристалл, домен,
    кольцо), а не осталась случайным шумом.
    """

    id = "filter.symmetry"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(self, *, min_score: float = 0.75) -> None:
        self.min_score = float(min_score)

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        namespace.value("min_score", self.min_score)

    def build(self, fixed: Fixed, view: BuildView) -> "SymmetryRuntime":
        return SymmetryRuntime(min_score=float(view.value("min_score")))


class SymmetryRuntime:
    __slots__ = ("min_score",)

    def __init__(self, min_score: float) -> None:
        self.min_score = min_score

    def check(self, grid) -> FilterVerdict:
        score = symmetry_score(live_mask(grid))
        metrics = {"symmetry": round(score, 4)}
        if score < self.min_score:
            return FilterVerdict(
                False,
                f"нет симметрии: {score:.2f} < {self.min_score:.2f}",
                metrics,
            )
        return FilterVerdict(True, metrics=metrics)


class EntropyFilter(Module):
    """Нормированная энтропия распределения фаз должна лежать в полосе."""

    id = "filter.entropy"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(self, *, min_entropy: float = 0.5, max_entropy: float = 0.999) -> None:
        self.criteria = {"min_entropy": min_entropy, "max_entropy": max_entropy}

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        for name, value in self.criteria.items():
            namespace.value(name, value)

    def build(self, fixed: Fixed, view: BuildView) -> "EntropyRuntime":
        return EntropyRuntime(
            min_entropy=float(view.value("min_entropy")),
            max_entropy=float(view.value("max_entropy")),
        )


class EntropyRuntime:
    __slots__ = ("min_entropy", "max_entropy")

    def __init__(self, min_entropy: float, max_entropy: float) -> None:
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

    def check(self, grid) -> FilterVerdict:
        entropy = normalized_entropy(grid)
        metrics = {"entropy": round(entropy, 4)}
        if entropy < self.min_entropy:
            return FilterVerdict(
                False,
                f"мало фаз: энтропия {entropy:.2f} < {self.min_entropy:.2f}",
                metrics,
            )
        if entropy > self.max_entropy:
            return FilterVerdict(
                False,
                f"шум: энтропия {entropy:.2f} > {self.max_entropy:.2f}",
                metrics,
            )
        return FilterVerdict(True, metrics=metrics)


class PhaseBalanceFilter(Module):
    """Все фазы цикла должны присутствовать: иначе волна выродилась."""

    id = "filter.phases"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 3),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(self, *, min_share: float = 0.02) -> None:
        self.min_share = float(min_share)

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        namespace.value("min_share", self.min_share)

    def build(self, fixed: Fixed, view: BuildView) -> "PhaseBalanceRuntime":
        return PhaseBalanceRuntime(min_share=float(view.value("min_share")))


class PhaseBalanceRuntime:
    __slots__ = ("min_share",)

    def __init__(self, min_share: float) -> None:
        self.min_share = min_share

    def check(self, grid) -> FilterVerdict:
        shares = phase_shares(grid)
        # Фаза 0 — «смерть»; для волны важны остальные фазы цикла.
        live_shares = shares[1:]
        metrics = {
            "phase_min": round(float(live_shares.min()), 4) if live_shares.size else 0.0,
            "phase_max": round(float(live_shares.max()), 4) if live_shares.size else 0.0,
        }
        if live_shares.size:
            worst = int(np.argmin(live_shares))
            share = float(live_shares[worst])
            if share < self.min_share:
                return FilterVerdict(
                    False,
                    f"фаза {worst + 1} выродилась: доля {share:.3f} "
                    f"< {self.min_share:.2f}",
                    metrics,
                )
        return FilterVerdict(True, metrics=metrics)


class ClusterFilter(Module):
    """Кластеры: число 8-связных компонент и их средний размер."""

    id = "filter.clusters"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(
        self,
        *,
        min_components: int = 1,
        max_components: int = 60,
        min_mean_component: float = 4.0,
    ) -> None:
        self.criteria = {
            "min_components": int(min_components),
            "max_components": int(max_components),
            "min_mean_component": float(min_mean_component),
        }

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        for name, value in self.criteria.items():
            namespace.value(name, value)

    def build(self, fixed: Fixed, view: BuildView) -> "ClusterRuntime":
        return ClusterRuntime(
            min_components=int(view.value("min_components")),
            max_components=int(view.value("max_components")),
            min_mean_component=float(view.value("min_mean_component")),
        )


class ClusterRuntime:
    __slots__ = ("min_components", "max_components", "min_mean_component")

    def __init__(
        self, min_components: int, max_components: int, min_mean_component: float
    ) -> None:
        self.min_components = min_components
        self.max_components = max_components
        self.min_mean_component = min_mean_component

    def check(self, grid) -> FilterVerdict:
        live = live_mask(grid)
        count = int(np.count_nonzero(live))
        if count == 0:
            return FilterVerdict(False, "нет живых клеток", {"components": 0})
        components, _ = component_stats(live, False)
        mean_component = float(count) / float(components)
        metrics = {
            "components": int(components),
            "mean_component": round(mean_component, 2),
        }
        if components < self.min_components:
            return FilterVerdict(
                False,
                f"слишком мало кластеров: {components} < {self.min_components}",
                metrics,
            )
        if components > self.max_components:
            return FilterVerdict(
                False,
                f"пыль: кластеров {components} > {self.max_components}",
                metrics,
            )
        if mean_component < self.min_mean_component:
            return FilterVerdict(
                False,
                f"мелкие кластеры: средний {mean_component:.1f} "
                f"< {self.min_mean_component:.1f}",
                metrics,
            )
        return FilterVerdict(True, metrics=metrics)
