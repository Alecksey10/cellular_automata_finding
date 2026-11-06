"""Базовые классы плагинов: общий модуль и по классу на роль."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ..interfaces import Capability, Requirement
from .roles import Role
from .spec import NeighborhoodSpec

if TYPE_CHECKING:  # pragma: no cover
    from ..space import Namespace


class Module:
    """Базовый плагин."""

    id: str = "module"
    label: str = ""
    role: Role = Role.DATA
    requires: tuple[Requirement, ...] = ()

    # ------------------------------------------------------------------
    # описание осей
    # ------------------------------------------------------------------
    def declare(self, ns: "Namespace", view: "DeclareView") -> None:
        """Объявить оси модуля. Значения появятся только при привязке."""

    # ------------------------------------------------------------------
    # доступ к собственным параметрам
    # ------------------------------------------------------------------
    def key(self, name: str) -> str:
        return f"{self.id}.{name}"

    def get(self, fixed: Mapping[str, Any], name: str) -> Any:
        return fixed[self.key(name)]

    # ------------------------------------------------------------------
    # интерфейсы
    # ------------------------------------------------------------------
    def capabilities(self, fixed: Mapping[str, Any]) -> tuple[Capability, ...]:
        return ()

    def derived_value(self, name: str, fixed: Mapping[str, Any]) -> Any:
        raise KeyError(f"{self.id}: нет производного значения {name!r}")

    def describe(self) -> str:
        return self.label or self.id


class DataModule(Module):
    role = Role.DATA

    def create(
        self,
        fixed: Mapping[str, Any],
        rng: np.random.Generator,
        randomizer: "RandomizerModule",
    ) -> np.ndarray:
        raise NotImplementedError

    def metrics(self, data: np.ndarray, fixed: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def alive_count(self, data: np.ndarray, fixed: Mapping[str, Any]) -> int:
        raise NotImplementedError


class RandomizerModule(Module):
    role = Role.RANDOMIZER

    def fill(
        self,
        data: np.ndarray,
        fixed: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        zero_state: int,
        states: int,
    ) -> None:
        raise NotImplementedError


class NeighborhoodModule(Module):
    role = Role.NEIGHBORHOOD

    def spec(self, fixed: Mapping[str, Any]) -> NeighborhoodSpec:
        raise NotImplementedError

    def derived_value(self, name: str, fixed: Mapping[str, Any]) -> Any:
        spec = self.spec(fixed)
        if name == "max_count":
            return spec.max_count
        if name == "inner_max_count":
            return spec.inner_max_count
        raise KeyError(f"{self.id}: нет производного значения {name!r}")


class RuleRuntime:
    """Исполняемое правило: один шаг и прогрев numba."""

    def step(self, src: np.ndarray, dst: np.ndarray, iteration: int, seed: int) -> None:
        raise NotImplementedError

    def precompile(self) -> None:
        src = np.zeros((2, 2), dtype=np.int32)
        dst = np.zeros((2, 2), dtype=np.int32)
        self.step(src, dst, 0, 0)


class RuleModule(Module):
    role = Role.RULE

    def build(self, fixed: Mapping[str, Any], view: "BuildView") -> RuleRuntime:
        raise NotImplementedError


class FilterRuntime:
    def check(self, observation: "Observation") -> "Verdict":
        raise NotImplementedError


class FilterModule(Module):
    role = Role.FILTER

    def build(self, fixed: Mapping[str, Any], view: "BuildView") -> FilterRuntime:
        raise NotImplementedError


__all__ = [
    "DataModule",
    "FilterModule",
    "FilterRuntime",
    "Module",
    "NeighborhoodModule",
    "RandomizerModule",
    "RuleModule",
    "RuleRuntime",
]
