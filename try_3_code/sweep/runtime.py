"""Исполнение одного конфига.

``Trial`` — это весь жизненный цикл теста: инициализация, фазы, метрики,
фильтры, остановки. Рантайм не знает, что такое клетка: он работает с
модулями из :class:`~sweep.assembly.Assembly`.

Детерминизм: начальные данные создаются из ``numpy.random.Generator(seed)``,
а правила получают ``seed``/``iteration`` явно; повтор того же конфига даёт
тот же результат.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .assembly import Assembly
from .errors import Incompatible
from .module import BuildView, Observation

SUCCESS = "success"
REJECTED = "rejected"
STOPPED = "stopped"
ERROR = "error"


@dataclass
class Frame:
    iteration: int
    stage: str
    data: np.ndarray
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    status: str
    metrics: Mapping[str, Any]
    reason: str | None = None
    iterations: int = 0
    frames: list[Frame] = field(default_factory=list)
    error: str | None = None
    duration: float = 0.0
    details: Mapping[str, Any] = field(default_factory=dict)


class Trial:
    def __init__(
        self,
        assembly: Assembly,
        seed: int,
        *,
        record: bool = False,
        max_iterations: int | None = None,
        frame_stride: int = 1,
    ) -> None:
        self.assembly = assembly
        self.seed = int(seed)
        self.record = record
        self.max_iterations = max_iterations
        # шаг записи кадров: длинный прогон не держит в памяти каждый шаг
        self.frame_stride = max(1, int(frame_stride))
        self.frames: list[Frame] = []

    # ------------------------------------------------------------------
    def run(self) -> TrialResult:
        started = time.perf_counter()
        experiment = self.assembly.experiment
        data_module = self.assembly.data
        fixed = dict(self.assembly.fixed)
        rng = np.random.default_rng(self.seed)
        try:
            data = data_module.create(fixed, rng, self.assembly.randomizer)
            data = np.ascontiguousarray(data, dtype=np.int32)
            buffer = np.empty_like(data)
            zero = int(fixed.get("grid2d.zero", 0))
            metrics = dict(data_module.metrics(data, fixed))
            metrics["zero"] = zero

            total_iterations = 0
            if self.record:
                self._push(0, "init", data, metrics)

            for phase in experiment.phases:
                iterations = self.assembly.iterations(phase)
                view = BuildView(fixed, self.assembly.modules)
                runtimes = [(module, module.build(fixed, view)) for module in phase.filters]
                for index in range(iterations):
                    if experiment.stop_on_extinct and data_module.alive_count(data, fixed) == 0:
                        return self._result(STOPPED, metrics, "extinct", total_iterations, started)
                    # правилу передаётся общий счётчик шагов: он не сбрасывается
                    # на каждой фазе, поэтому случайность и старение не повторяются
                    self.assembly.rule_runtime.step(data, buffer, total_iterations, self.seed)
                    data, buffer = buffer, data
                    total_iterations += 1
                    metrics = dict(data_module.metrics(data, fixed))
                    metrics["zero"] = zero
                    metrics["iteration"] = total_iterations
                    metrics["phase"] = phase.name
                    if self.record and (
                        self.frame_stride == 1
                        or total_iterations % self.frame_stride == 0
                        or index == iterations - 1
                    ):
                        self._push(total_iterations, phase.name, data, metrics)
                    if phase.frequency == "each" and (index + 1) % phase.every == 0:
                        observation = Observation(phase.name, index, data, metrics)
                        for module, runtime in runtimes:
                            verdict = runtime.check(observation)
                            if not verdict.ok:
                                details = dict(verdict.details)
                                details["filter"] = module.id
                                return self._result(
                                    STOPPED,
                                    metrics,
                                    verdict.reason or "filter_failed",
                                    total_iterations,
                                    started,
                                    details,
                                )
                    if self.max_iterations is not None and total_iterations >= self.max_iterations:
                        return self._result(STOPPED, metrics, "max_iterations", total_iterations, started)
                if phase.frequency == "end":
                    observation = Observation(phase.name, iterations, data, metrics)
                    for module, runtime in runtimes:
                        verdict = runtime.check(observation)
                        if not verdict.ok:
                            details = dict(verdict.details)
                            details["filter"] = module.id
                            return self._result(
                                STOPPED,
                                metrics,
                                verdict.reason or "filter_failed",
                                total_iterations,
                                started,
                                details,
                            )
                if experiment.stop_on_extinct and data_module.alive_count(data, fixed) == 0:
                    return self._result(STOPPED, metrics, "extinct", total_iterations, started)

            metrics = dict(data_module.metrics(data, fixed))
            metrics["zero"] = zero
            metrics["iteration"] = total_iterations
            self._final_data = data
            return self._result(SUCCESS, metrics, None, total_iterations, started)
        except Incompatible as exc:
            return TrialResult(REJECTED, {}, str(exc), 0, self.frames, None, time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - ошибка одного теста не валит прогон
            return TrialResult(
                ERROR,
                {},
                type(exc).__name__,
                0,
                self.frames,
                f"{type(exc).__name__}: {exc}",
                time.perf_counter() - started,
            )

    # ------------------------------------------------------------------
    def _push(self, iteration: int, stage: str, data: np.ndarray, metrics: Mapping[str, Any]) -> None:
        self.frames.append(Frame(iteration, stage, np.array(data, dtype=np.int32, copy=True), dict(metrics)))

    def _result(
        self,
        status: str,
        metrics: Mapping[str, Any],
        reason: str | None,
        iterations: int,
        started: float,
        details: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        return TrialResult(
            status=status,
            metrics=dict(metrics),
            reason=reason,
            iterations=iterations,
            frames=list(self.frames),
            duration=time.perf_counter() - started,
            details=dict(details or {}),
        )


__all__ = ["ERROR", "Frame", "REJECTED", "STOPPED", "SUCCESS", "Trial", "TrialResult"]
