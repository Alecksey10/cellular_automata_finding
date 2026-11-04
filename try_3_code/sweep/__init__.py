"""sweep — движок перебора клеточных автоматов.

Слои:

* описание — :class:`~sweep.experiment.Experiment`, модули, оси;
* фиксация — :class:`~sweep.config.Config` (values + seed) и дайджест;
* исполнение — :mod:`sweep.runtime` (один тест) и :mod:`sweep.parallel` (пул);
* хранение — :mod:`sweep.storage`;
* визуализация — :mod:`sweep.visualize` (общий интерфейс + визуализатор
  конкретного эксперимента).
"""

from .config import Config
from .errors import Cancelled, Incompatible, SweepError, UnboundParameter
from .experiment import Experiment, Phase
from .handles import Const, Derived, Handle, Ref
from .interfaces import (
    Capability,
    CapabilityKind,
    Requirement,
    always,
    between,
    eq,
    ge,
    le,
)
from .module import Observation, PASS, Role, Verdict
from .parallel import RunOptions, execute
from .runtime import ERROR, REJECTED, STOPPED, SUCCESS, Trial, TrialResult
from .space import Namespace, Space

__version__ = "3.0.0"

__all__ = [
    "Cancelled",
    "Capability",
    "CapabilityKind",
    "Config",
    "Const",
    "Derived",
    "ERROR",
    "Experiment",
    "Handle",
    "Incompatible",
    "Namespace",
    "Observation",
    "PASS",
    "Phase",
    "REJECTED",
    "Ref",
    "Requirement",
    "Role",
    "RunOptions",
    "STOPPED",
    "SUCCESS",
    "Space",
    "SweepError",
    "Trial",
    "TrialResult",
    "UnboundParameter",
    "Verdict",
    "always",
    "between",
    "eq",
    "execute",
    "ge",
    "le",
]
