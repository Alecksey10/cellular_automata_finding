"""Исполнение одного теста: инициализация, итерации, фильтры, остановки."""

from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from .experiment import Experiment, FilterUse
from .fixed import Fixed
from .space import IncompatibleModules

FrameCallback = Callable[[str, int, Any], None]


class TrialStatus(Enum):
    PASSED = "passed"
    REJECTED = "rejected"
    STOPPED = "stopped"
    INFEASIBLE = "infeasible"
    ERROR = "error"


@dataclass(frozen=True)
class FilterVerdict:
    """Результат проверки фильтра."""

    ok: bool
    reason: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepOutcome:
    """Что вернуло правило итерации (может попросить остановку)."""

    stop: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    config: Fixed
    status: TrialStatus
    reason: str = ""
    phase: str | None = None
    iterations: int = 0
    seed: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    state_digest: str | None = None
    duration: float = 0.0
    error: str | None = None
    state: Any = None

    @property
    def ok(self) -> bool:
        return self.status is TrialStatus.PASSED

    def summary(self) -> str:
        detail = f" ({self.reason})" if self.reason else ""
        return f"{self.status.value:<10} {self.config.digest()}  {self.iterations} итер.{detail}"

    def to_record(self, include_state: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "config": self.config.to_dict(),
            "config_digest": self.config.digest(),
            "status": self.status.value,
            "reason": self.reason,
            "phase": self.phase,
            "iterations": self.iterations,
            "seed": self.seed,
            "metrics": dict(self.metrics),
            "state_digest": self.state_digest,
            "duration": round(self.duration, 6),
            "error": self.error,
        }
        if include_state and self.state is not None:
            state = self.state
            array = getattr(state, "array", state)
            record["state"] = array.tolist() if hasattr(array, "tolist") else array
        return record


def derive_seed(base_seed: int, config_digest: str) -> int:
    """Детерминированный seed из базового seed и конфига."""

    material = f"{base_seed}:{config_digest}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little")


def _state_digest(grid: Any) -> str | None:
    array = getattr(grid, "array", None)
    if array is None:
        return None
    h = hashlib.sha256()
    h.update(str(array.shape).encode("ascii"))
    h.update(array.tobytes())
    return h.hexdigest()[:16]


def _metrics_of(grid: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    live_fraction = getattr(grid, "live_fraction", None)
    if callable(live_fraction):
        values["fill"] = round(float(live_fraction()), 6)
    return values


def run_trial(
    experiment: Experiment,
    fixed: Fixed,
    *,
    seed: int = 0,
    keep_state: bool = False,
    on_frame: FrameCallback | None = None,
    max_iterations: int | None = None,
) -> TrialResult:
    """Полный прогон одного конфига.

    Стадии исполняются по порядку. Внутри стадии правило делает итерацию,
    затем срабатывают фильтры с режимом ``each``; после стадии — фильтры с
    режимом ``end``. Правило или фильтр могут остановить тест досрочно.

    ``max_iterations`` — общий бюджет итераций на тест: стадии урезаются по
    остатку бюджета, фильтры конца стадии всё равно проверяются. Нужен, чтобы
    длинные конфиги не занимали пул целиком.
    """

    started = time.perf_counter()
    result = TrialResult(config=fixed, status=TrialStatus.ERROR, seed=seed)

    try:
        assembled = experiment.assemble(fixed)
    except IncompatibleModules as exc:
        result.status = TrialStatus.INFEASIBLE
        result.reason = str(exc).splitlines()[-1] if str(exc) else "несовместимые модули"
        result.duration = time.perf_counter() - started
        return result

    try:
        import numpy as np  # локально: ядро движка не требует numpy

        rng = np.random.default_rng(seed)
        binder = getattr(assembled.rule, "bind_rng", None)
        if callable(binder):
            binder(rng)
        grid = assembled.data_factory.create()
        assembled.randomizer.fill(grid, rng)
        scratch = assembled.data_factory.like(grid)
        result.metrics.update(_metrics_of(grid))

        remaining = None if max_iterations is None else max(0, int(max_iterations))
        capped = False
        for phase in experiment.phases:
            planned = int(fixed[phase.iterations])
            iterations = planned
            if remaining is not None:
                if iterations > remaining:
                    iterations = remaining
                    capped = True
                remaining -= iterations
            for step in range(iterations):
                outcome = assembled.rule.step(grid, scratch)
                grid, scratch = scratch, grid
                result.iterations += 1
                if outcome is not None:
                    result.metrics.update(outcome.metrics)
                    if outcome.stop:
                        result.status = TrialStatus.STOPPED
                        result.reason = f"автомат: {outcome.stop}"
                        result.phase = phase.name
                        result.state_digest = _state_digest(grid)
                        result.state = grid if keep_state else None
                        result.duration = time.perf_counter() - started
                        return result
                if on_frame is not None:
                    on_frame(phase.name, step, grid)
                for use in phase.filters:
                    if use.mode != "each":
                        continue
                    verdict = _check_filter(assembled, use, grid)
                    result.metrics.update(verdict.metrics)
                    if not verdict.ok:
                        return _reject(result, phase, use, verdict, grid, started, keep_state)

            for use in phase.filters:
                if use.mode != "end":
                    continue
                verdict = _check_filter(assembled, use, grid)
                result.metrics.update(verdict.metrics)
                if not verdict.ok:
                    return _reject(result, phase, use, verdict, grid, started, keep_state)

        result.status = TrialStatus.PASSED
        result.phase = experiment.phases[-1].name if experiment.phases else None
        if capped:
            result.metrics["capped"] = int(max_iterations or 0)
        result.metrics.update(_metrics_of(grid))
        result.state_digest = _state_digest(grid)
        result.state = grid if keep_state else None
        result.duration = time.perf_counter() - started
        return result

    except Exception as exc:  # noqa: BLE001 - ошибка теста не должна ронять раннер
        result.status = TrialStatus.ERROR
        result.reason = f"{type(exc).__name__}: {exc}"
        result.error = traceback.format_exc(limit=8)
        result.duration = time.perf_counter() - started
        return result


def _check_filter(assembled: Any, use: FilterUse, grid: Any) -> FilterVerdict:
    runtime = assembled.filters[use.binding.name]
    return runtime.check(grid)


def _reject(
    result: TrialResult,
    phase: Any,
    use: FilterUse,
    verdict: FilterVerdict,
    grid: Any,
    started: float,
    keep_state: bool,
) -> TrialResult:
    result.status = TrialStatus.REJECTED
    result.phase = phase.name
    result.reason = f"{use.binding.name}: {verdict.reason}"
    result.state_digest = _state_digest(grid)
    result.state = grid if keep_state else None
    result.duration = time.perf_counter() - started
    return result
