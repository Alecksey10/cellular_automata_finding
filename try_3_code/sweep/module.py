"""База модулей: роли, требования, объявление осей, сборка.

Модуль — самостоятельный плагин. Он объявляет свои оси в
:class:`~sweep.space.Namespace` (только описание, без вычислений) и умеет
построить рабочий объект по зафиксированному конфигу. Ядро не знает, что
такое клетка, сосед или возраст: обо всём этом знают модули.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .interfaces import Capability, Requirement

if TYPE_CHECKING:  # pragma: no cover
    from .experiment import Experiment
    from .space import Namespace


class Role(enum.Enum):
    DATA = "data"
    RANDOMIZER = "randomizer"
    NEIGHBORHOOD = "neighborhood"
    RULE = "rule"
    FILTER = "filter"


# Порядок объявления важен: параметры данных и окрестности должны быть
# привязаны раньше, чем правило объявит зависящие от них пороги.
ROLE_ORDER: tuple[Role, ...] = (
    Role.DATA,
    Role.RANDOMIZER,
    Role.NEIGHBORHOOD,
    Role.RULE,
)


@dataclass(frozen=True)
class NeighborhoodSpec:
    """Готовое описание соседства: только данные, совместимые с numba."""

    name: str
    dy: np.ndarray
    dx: np.ndarray
    inner_dy: np.ndarray
    inner_dx: np.ndarray
    wrap: bool
    max_count: int
    inner_max_count: int = 0

    @property
    def wrap_flag(self) -> int:
        return 1 if self.wrap else 0


@dataclass(frozen=True)
class Observation:
    """Снимок, который фильтр получает на проверку."""

    stage: str
    iteration: int
    data: np.ndarray
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    """Результат фильтра."""

    ok: bool
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


PASS = Verdict(True)


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
    def check(self, observation: Observation) -> Verdict:
        raise NotImplementedError


class FilterModule(Module):
    role = Role.FILTER

    def build(self, fixed: Mapping[str, Any], view: "BuildView") -> FilterRuntime:
        raise NotImplementedError


class PeerProxy:
    """Ссылка на модуль другой роли, выбранный в текущем конфиге.

    ``view.peer(Role.NEIGHBORHOOD).derived("max_count")`` даёт ленивый handle:
    значение посчитается, когда окрестность уже привязана.
    """

    def __init__(self, experiment: "Experiment", role: Role) -> None:
        self.experiment = experiment
        self.role = role

    def derived(self, name: str):
        from .handles import Derived

        def resolve(fixed: Mapping[str, Any]) -> Any:
            module = self.experiment.selected(self.role, fixed)
            return module.derived_value(name, fixed)

        return Derived(f"{self.role.value}.{name}", resolve)

    def __repr__(self) -> str:  # pragma: no cover
        return f"PeerProxy({self.role.value})"


class DeclareView:
    """То, что доступно модулю на фазе описания."""

    def __init__(self, experiment: "Experiment", role: Role) -> None:
        self.experiment = experiment
        self.role = role

    def peer(self, role: Role) -> PeerProxy:
        return PeerProxy(self.experiment, role)


class BuildView:
    """То, что доступно модулю на фазе сборки конкретного конфига."""

    def __init__(self, fixed: Mapping[str, Any], modules: Mapping[Role, Module]) -> None:
        self.fixed = fixed
        self.modules = modules
        self._neighborhood: NeighborhoodSpec | None = None

    def module(self, role: Role) -> Module:
        return self.modules[role]

    def neighborhood(self) -> NeighborhoodSpec:
        if self._neighborhood is None:
            module = self.modules[Role.NEIGHBORHOOD]
            assert isinstance(module, NeighborhoodModule)
            self._neighborhood = module.spec(self.fixed)
        return self._neighborhood


__all__ = [
    "BuildView",
    "DataModule",
    "DeclareView",
    "FilterModule",
    "FilterRuntime",
    "Module",
    "NeighborhoodModule",
    "NeighborhoodSpec",
    "Observation",
    "PASS",
    "PeerProxy",
    "ROLE_ORDER",
    "RandomizerModule",
    "Role",
    "RuleModule",
    "RuleRuntime",
    "Verdict",
]
