"""Сборка автомата: проверка совместимости модулей и создание рабочих объектов."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .fixed import Fixed
from .interfaces import Capability, Requirement, first_match
from .module import ROLE_ORDER, BuildView, FilterBinding, Module, Role
from .space import IncompatibleModules, Selection


@dataclass(frozen=True)
class CheckLine:
    module_id: str
    role: Role
    requirement: Requirement
    matched: Capability | None

    @property
    def ok(self) -> bool:
        return self.matched is not None

    def text(self) -> str:
        status = "OK  " if self.ok else "FAIL"
        got = f"<- {self.matched}" if self.matched is not None else "<- нет подходящей возможности"
        return f"[{status}] {self.role.value:<12} {self.module_id:<24} требует {self.requirement.describe():<48} {got}"


@dataclass
class CompatibilityReport:
    """Подробный отчёт о совместимости выбранных модулей."""

    lines: list[CheckLine] = field(default_factory=list)
    provided: list[Capability] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(line.ok for line in self.lines)

    @property
    def problems(self) -> list[CheckLine]:
        return [line for line in self.lines if not line.ok]

    def text(self) -> str:
        head = "\n".join(line.text() for line in self.lines)
        if not head:
            head = "(нет требований)"
        if self.ok:
            return head
        return head + "\nНесовместимо:\n" + "\n".join(
            f"  - {line.module_id}: {line.requirement.describe()}" for line in self.problems
        )


@dataclass
class Assembled:
    """Результат сборки одного теста."""

    fixed: Fixed
    modules: Mapping[Role, Module]
    data_factory: Any
    randomizer: Any
    neighborhood: Any
    rule: Any
    filters: Mapping[str, Any]
    report: CompatibilityReport

    def precompile(self) -> None:
        """Прогреть numba-ядра до запуска пула процессов."""

        precompile = getattr(self.rule, "precompile", None)
        if callable(precompile):
            precompile()


class Assembly:
    """Сборка по выбранным модулям и подключённым фильтрам."""

    def __init__(
        self,
        selections: Mapping[Role, Selection],
        filters: Iterable[FilterBinding] = (),
    ) -> None:
        self.selections = dict(selections)
        self.filters = tuple(filters)

    # --- выбор модулей ---------------------------------------------------
    def module(self, fixed: Fixed, role: Role) -> Module:
        try:
            selection = self.selections[role]
        except KeyError as exc:
            raise IncompatibleModules(f"Роль {role.value!r} не выбрана в эксперименте") from exc
        return selection.selected_module(fixed)

    # --- проверка --------------------------------------------------------
    def report(self, fixed: Fixed) -> CompatibilityReport:
        provided: list[Capability] = []
        lines: list[CheckLine] = []
        for role in ROLE_ORDER:
            module = self.module(fixed, role)
            if role is not Role.DATA:
                for requirement in module.requires:
                    lines.append(
                        CheckLine(module.id, role, requirement, first_match(requirement, provided))
                    )
            else:
                for requirement in module.requires:
                    lines.append(CheckLine(module.id, role, requirement, first_match(requirement, [])))
            provided.extend(module.capabilities(fixed))
        for binding in self.filters:
            for requirement in binding.module.requires:
                lines.append(
                    CheckLine(
                        binding.module.id,
                        Role.FILTER,
                        requirement,
                        first_match(requirement, provided),
                    )
                )
        return CompatibilityReport(lines=lines, provided=provided)

    def compatible(self, fixed: Fixed) -> bool:
        return self.report(fixed).ok

    def partial_signature(self, binding: Mapping[str, Any]) -> tuple[Any, ...] | None:
        """Подпись совместимости, если она уже определена частичной привязкой.

        Возможности модуля считаются неизменными, если он не переопределяет
        ``capabilities`` (базовая реализация возвращает статический
        ``provides``). Иначе нужны все его параметры и производные значения.
        """

        parts: list[Any] = []
        for role in ROLE_ORDER:
            selection = self.selections.get(role)
            if selection is None:
                return None
            if selection.choice.key not in binding:
                return None
            module = selection.modules.get(binding[selection.choice.key])
            if module is None:
                return None
            if type(module).capabilities is not Module.capabilities:
                for name in selection.declared_params.get(module.id, ()):
                    if f"{role.value}.{name}" not in binding:
                        return None
                for name in getattr(module, "derived", ()):
                    if f"{role.value}#{name}" not in binding:
                        return None
                capabilities = module.capabilities(Fixed(binding))
            else:
                capabilities = module.provides
            parts.append((module.id, capabilities))
        parts.append(tuple(binding_module.module.id for binding_module in self.filters))
        return tuple(parts)

    def signature(self, fixed: Fixed) -> tuple[Any, ...]:
        """Ключ совместимости: выбранные модули + их возможности.

        Отчёт зависит только от этого, поэтому его можно кэшировать: тысячи
        конфигов, отличающихся лишь порогами правила, дают одну подпись.
        """

        parts: list[Any] = []
        for role in ROLE_ORDER:
            module = self.module(fixed, role)
            parts.append((module.id, module.capabilities(fixed)))
        parts.append(tuple(binding.module.id for binding in self.filters))
        return tuple(parts)

    # --- сборка ----------------------------------------------------------
    def build(self, fixed: Fixed) -> Assembled:
        report = self.report(fixed)
        if not report.ok:
            raise IncompatibleModules(report.text())

        modules: dict[Role, Module] = {}
        built: dict[Role, Any] = {}
        for role in ROLE_ORDER:
            module = self.module(fixed, role)
            modules[role] = module
            view = BuildView(fixed, role.value, built, modules)
            built[role] = module.build(fixed, view)

        filter_objects: dict[str, Any] = {}
        for binding in self.filters:
            view = BuildView(fixed, binding.prefix, built, modules)
            filter_objects[binding.name] = binding.module.build(fixed, view)

        return Assembled(
            fixed=fixed,
            modules=modules,
            data_factory=built[Role.DATA],
            randomizer=built[Role.RANDOMIZER],
            neighborhood=built[Role.NEIGHBORHOOD],
            rule=built[Role.RULE],
            filters=filter_objects,
            report=report,
        )
