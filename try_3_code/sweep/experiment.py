"""Описание эксперимента: оси, модули, фазы, фильтры.

Эксперимент — декларация, а не исполнение. Он собирает дерево осей
(:mod:`sweep.space`) и набор модулей, а конкретные значения фиксируются в
:class:`~sweep.config.Config` уже на этапе запуска. Один и тот же объект
описания переиспользуется для всех конфигов, поэтому ``build_*`` обязан
возвращать свежий эксперимент при каждом вызове.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .handles import Handle, Ref, as_handle, resolve
from .interfaces import Capability, Requirement, check_requirements
from .module import (
    DataModule,
    DeclareView,
    FilterModule,
    Module,
    NeighborhoodModule,
    RandomizerModule,
    ROLE_ORDER,
    Role,
    RuleModule,
)
from .space import Namespace, Space


@dataclass
class Phase:
    name: str
    iterations: Handle
    filters: tuple[FilterModule, ...] = ()
    frequency: str = "end"
    # как часто вызывать фильтры при frequency="each": каждые ``every`` шагов
    every: int = 1


class Experiment:
    def __init__(self, name: str, description: str = "", *, stop_on_extinct: bool = False) -> None:
        self.name = name
        self.description = description
        self.stop_on_extinct = stop_on_extinct
        self.space = Space()
        self._modules: dict[Role, list[Module]] = {role: [] for role in Role}
        self._phases: list[Phase] = []
        self._filters: list[FilterModule] = []
        self.visualizer: Any = None
        self._finalized = False

    # ------------------------------------------------------------------
    # выбор модулей
    # ------------------------------------------------------------------
    def select(self, role: Role, modules: Module | Iterable[Module]) -> Module | list[Module]:
        if isinstance(modules, Module):
            selected = [modules]
        else:
            selected = list(modules)
        if not selected:
            raise ValueError(f"для роли {role.value} не выбран ни один модуль")
        for module in selected:
            if module.role is not role:
                raise TypeError(f"{module.id}: роль {module.role.value}, ожидалась {role.value}")
            if not any(existing.id == module.id for existing in self._modules[role]):
                self._modules[role].append(module)
        self.space.add_selector(role, selected)
        self._finalized = False
        return selected[0] if len(selected) == 1 else selected

    def filter(self, *filters: FilterModule) -> FilterModule:
        for module in filters:
            if module.role is not Role.FILTER:
                raise TypeError(f"{module.id}: ожидался фильтр")
            if not any(existing.id == module.id for existing in self._filters):
                self._filters.append(module)
        self._finalized = False
        return filters[0] if filters else None  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # оси
    # ------------------------------------------------------------------
    def value(self, key: str, value: Any) -> Ref:
        return self.space.add_namespace().value(key, value)

    def range(self, key: str, lo: Any, hi: Any, step: Any = 1) -> Ref:
        return self.space.add_namespace().range(key, lo, hi, step)

    def choice(self, key: str, options: Iterable[Any]) -> Ref:
        return self.space.add_namespace().choice(key, options)

    def constrain(
        self,
        check: Callable[[Mapping[str, Any]], bool],
        keys: Iterable[str] | None = None,
        name: str = "",
    ) -> None:
        self.space.constrain(check, keys, name)

    # ------------------------------------------------------------------
    # фазы и итерация
    # ------------------------------------------------------------------
    def phase(
        self,
        name: str,
        iterations: Any,
        filters: Sequence[FilterModule] = (),
        frequency: str = "end",
        every: int = 1,
    ) -> Phase:
        if frequency not in ("end", "each"):
            raise ValueError("frequency должен быть 'end' или 'each'")
        if int(every) < 1:
            raise ValueError("every должен быть положительным")
        attached = tuple(filters)
        for module in attached:
            self.filter(module)
        phase = Phase(name, as_handle(iterations), attached, frequency, int(every))
        self._phases.append(phase)
        return phase

    def iterate(
        self,
        step: Callable[..., None],
        *,
        module_id: str | None = None,
        label: str = "",
        requires: Sequence[Requirement] = (),
        declare: Callable[..., None] | None = None,
        bind: Callable[..., tuple[Any, ...]] | None = None,
        with_control: bool = False,
    ):
        """Подключить итерацию, написанную прямо в этом эксперименте."""

        from .inline import InlineRule

        if self._modules[Role.RULE]:
            raise ValueError(
                "правило уже выбрано: используйте либо select(Role.RULE, ...), "
                "либо experiment.iterate(...), но не оба сразу"
            )
        rule = InlineRule(
            step,
            module_id=module_id or f"rule.{self.name}",
            label=label,
            requires=requires,
            declare=declare,
            bind=bind,
            with_control=with_control,
        )
        self.select(Role.RULE, rule)
        return rule

    # ------------------------------------------------------------------
    # фиксация описания
    # ------------------------------------------------------------------
    def finalize(self) -> None:
        if self._finalized:
            return
        if not self._modules[Role.RULE]:
            raise ValueError(f"{self.name}: не выбрано правило (select(Role.RULE, ...) или iterate(...))")
        for role in ROLE_ORDER:
            for module in self._modules[role]:
                ns = Namespace(self.space, prefix=module.id)
                module.declare(ns, DeclareView(self, role))
        for module in self._filters:
            ns = Namespace(self.space, prefix=module.id)
            module.declare(ns, DeclareView(self, Role.FILTER))
        if not self._phases:
            self.phase("main", self.range("phase.iterations", 10, 11), frequency="end")
        self._finalized = True

    # ------------------------------------------------------------------
    # работа с конкретной привязкой
    # ------------------------------------------------------------------
    def selected(self, role: Role, fixed: Mapping[str, Any]) -> Module:
        module_id = fixed.get(f"select.{role.value}")
        for module in self._modules[role]:
            if module.id == module_id:
                return module
        if self._modules[role]:
            return self._modules[role][0]
        raise KeyError(f"{self.name}: роль {role.value} не сконфигурирована")

    def instantiate(self, fixed: Mapping[str, Any]) -> dict[Role, Module]:
        return {role: self.selected(role, fixed) for role in ROLE_ORDER if self._modules[role]}

    def capabilities(self, fixed: Mapping[str, Any]) -> tuple[Capability, ...]:
        caps: list[Capability] = []
        for role in ROLE_ORDER:
            if not self._modules[role]:
                continue
            caps.extend(self.selected(role, fixed).capabilities(fixed))
        return tuple(caps)

    def check(self, fixed: Mapping[str, Any]) -> tuple[bool, str | None]:
        self.finalize()
        capabilities = self.capabilities(fixed)
        requirements: list[Requirement] = []
        rule = self.selected(Role.RULE, fixed)
        requirements.extend(rule.requires)
        for module in self._filters:
            requirements.extend(module.requires)
        missing = check_requirements(capabilities, requirements)
        if missing:
            return False, "; ".join(missing)
        return True, None

    def phase_iterations(self, phase: Phase, fixed: Mapping[str, Any]) -> int:
        return int(resolve(phase.iterations, fixed))

    @property
    def phases(self) -> tuple[Phase, ...]:
        return tuple(self._phases)

    @property
    def filters(self) -> tuple[FilterModule, ...]:
        return tuple(self._filters)

    # ------------------------------------------------------------------
    # перебор только совместимых конфигов
    # ------------------------------------------------------------------
    def iter_configs(
        self,
        *,
        strategy: str = "product",
        seed: int = 0,
        max_configs: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Ленивый поток привязок, прошедших проверку интерфейсов.

        Несовместимые комбинации (например, правило жизни на 5 состояниях)
        отбрасываются здесь, поэтому ``--limit`` считает уже исполнимые тесты.
        """

        self.finalize()
        if strategy == "random":
            source: Iterable[Mapping[str, Any]] = self.space.random_bindings(
                limit=None, seed=seed, feasible=True
            )
        elif strategy in ("product", "diagonal"):
            source = self.space.iter_bindings(strategy="product", feasible=True)
        else:
            raise ValueError(f"неизвестная стратегия: {strategy!r}")
        produced = 0
        for binding in source:
            ok, _ = self.check(binding)
            if not ok:
                continue
            yield dict(binding)
            produced += 1
            if max_configs is not None and produced >= max_configs:
                return

    def count(
        self,
        *,
        strategy: str = "product",
        feasible: bool = True,
        limit: int | None = None,
        seed: int = 0,
    ) -> int:
        self.finalize()
        if strategy == "random" and limit is None:
            raise ValueError("для случайной стратегии укажите limit")
        total = 0
        for _ in self.iter_configs(strategy=strategy, seed=seed, max_configs=limit):
            total += 1
            if limit is not None and total >= limit:
                break
        return total

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Experiment {self.name} phases={len(self._phases)}>"


__all__ = ["Experiment", "Phase"]
