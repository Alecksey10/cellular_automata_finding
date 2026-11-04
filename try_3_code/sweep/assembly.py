"""Сборка рабочих объектов из зафиксированного конфига.

Сборка — единственное место, где проверяются требования модулей и создаются
numba-совместимые рантаймы. Если конфиг несовместим, бросается
:class:`~sweep.errors.Incompatible` до запуска итераций.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import Config
from .errors import Incompatible
from .experiment import Experiment, Phase
from .interfaces import Capability
from .module import (
    BuildView,
    DataModule,
    FilterModule,
    Module,
    NeighborhoodModule,
    RandomizerModule,
    Role,
    RuleModule,
    RuleRuntime,
)


@dataclass
class Assembly:
    experiment: Experiment
    config: Config
    fixed: Mapping[str, Any]
    modules: Mapping[Role, Module]
    capabilities: tuple[Capability, ...]
    rule_runtime: RuleRuntime
    filter_modules: tuple[FilterModule, ...]

    @property
    def data(self) -> DataModule:
        module = self.modules[Role.DATA]
        assert isinstance(module, DataModule)
        return module

    @property
    def randomizer(self) -> RandomizerModule:
        module = self.modules[Role.RANDOMIZER]
        assert isinstance(module, RandomizerModule)
        return module

    @property
    def neighborhood(self) -> NeighborhoodModule:
        module = self.modules[Role.NEIGHBORHOOD]
        assert isinstance(module, NeighborhoodModule)
        return module

    @property
    def rule(self) -> RuleModule:
        module = self.modules[Role.RULE]
        assert isinstance(module, RuleModule)
        return module

    def iterations(self, phase: Phase) -> int:
        return self.experiment.phase_iterations(phase, self.fixed)


def assemble(experiment: Experiment, config: Config) -> Assembly:
    experiment.finalize()
    fixed = dict(config.values)
    ok, reason = experiment.check(fixed)
    if not ok:
        raise Incompatible(reason or "несовместимая конфигурация")
    modules = experiment.instantiate(fixed)
    view = BuildView(fixed, modules)

    rule = modules[Role.RULE]
    assert isinstance(rule, RuleModule)
    rule_runtime = rule.build(fixed, view)

    capabilities = experiment.capabilities(fixed)
    filter_modules = tuple(experiment.filters)
    for module in filter_modules:
        module.build(fixed, view)  # проверяем, что фильтр вообще собирается
    return Assembly(
        experiment=experiment,
        config=config,
        fixed=fixed,
        modules=modules,
        capabilities=capabilities,
        rule_runtime=rule_runtime,
        filter_modules=filter_modules,
    )


def preflight(experiment: Experiment, config: Config) -> Assembly:
    """Собрать и прогреть numba-ядра до запуска пула процессов."""

    assembly = assemble(experiment, config)
    assembly.rule_runtime.precompile()
    return assembly


__all__ = ["Assembly", "assemble", "preflight"]
