"""Снимок состояния для фильтра и его вердикт."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


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

__all__ = ["Observation", "PASS", "Verdict"]
