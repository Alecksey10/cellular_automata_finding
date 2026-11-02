"""Движок перебора клеточных автоматов.

Планирование (декларация осей) отделено от исполнения (фиксированный конфиг),
а предметная логика вынесена в модули-плагины.
"""

from .assembly import Assembled, Assembly, CompatibilityReport, CheckLine
from .experiment import Experiment, FilterUse, Phase
from .fixed import Fixed
from .handles import Handle, Shift
from .interfaces import (
    Capability,
    CapabilityKind,
    Comparison,
    Requirement,
    at_least,
    at_most,
    capability,
    eq,
)
from .module import BuildView, FilterBinding, Module, Role
from .space import (
    SWEEP_STRATEGIES,
    DeclarationError,
    DomainError,
    IncompatibleModules,
    Namespace,
    Selection,
    Space,
    SweepError,
)
from .trial import FilterVerdict, StepOutcome, TrialResult, TrialStatus, run_trial

__all__ = [
    "Assembled",
    "Assembly",
    "BuildView",
    "Capability",
    "CapabilityKind",
    "CheckLine",
    "Comparison",
    "CompatibilityReport",
    "DeclarationError",
    "DomainError",
    "Experiment",
    "FilterBinding",
    "FilterUse",
    "FilterVerdict",
    "Fixed",
    "Handle",
    "IncompatibleModules",
    "Module",
    "Namespace",
    "Phase",
    "Requirement",
    "Role",
    "Selection",
    "Shift",
    "Space",
    "SWEEP_STRATEGIES",
    "StepOutcome",
    "SweepError",
    "TrialResult",
    "TrialStatus",
    "at_least",
    "at_most",
    "capability",
    "eq",
    "run_trial",
]
