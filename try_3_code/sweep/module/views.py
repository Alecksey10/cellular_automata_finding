"""То, что модуль видит на фазах описания и сборки."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from .base import NeighborhoodModule
from .roles import Role
from .spec import NeighborhoodSpec

if TYPE_CHECKING:  # pragma: no cover
    from ..experiment import Experiment


class PeerProxy:
    """Ссылка на модуль другой роли, выбранный в текущем конфиге.

    ``view.peer(Role.NEIGHBORHOOD).derived("max_count")`` даёт ленивый handle:
    значение посчитается, когда окрестность уже привязана.
    """

    def __init__(self, experiment: "Experiment", role: Role) -> None:
        self.experiment = experiment
        self.role = role

    def derived(self, name: str):
        from ..handles import Derived

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

    def __init__(self, fixed: Mapping[str, Any], modules: Mapping[Role, Any]) -> None:
        self.fixed = fixed
        self.modules = modules
        self._neighborhood: NeighborhoodSpec | None = None

    def module(self, role: Role) -> Any:
        return self.modules[role]

    def neighborhood(self) -> NeighborhoodSpec:
        if self._neighborhood is None:
            module = self.modules[Role.NEIGHBORHOOD]
            assert isinstance(module, NeighborhoodModule)
            self._neighborhood = module.spec(self.fixed)
        return self._neighborhood


__all__ = ["BuildView", "DeclareView", "PeerProxy"]
