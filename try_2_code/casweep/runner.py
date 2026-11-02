"""Раннер: перебор конфигов, отбраковка несовместимых, прогресс и сводка."""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .experiment import Experiment
from .fixed import Fixed
from .parallel import Dispatcher, Record, Task
from .storage import ResultStore
from .trial import TrialStatus


def _chain(first: Task | None, rest: Iterator[Task]) -> Iterator[Task]:
    if first is not None:
        yield first
    yield from rest


@dataclass
class RunOptions:
    limit: int | None = None
    skip: int = 0
    seed: int = 0
    workers: int = 1
    mode: str = "auto"  # auto | serial | thread | process
    out: str | Path | None = None
    resume: bool = False
    random_start: bool = False
    keep_state: bool = False
    only_feasible: bool = True
    progress: bool = True
    #: Бюджет итераций на один тест (стадии урезаются).
    max_iterations: int | None = None
    #: Остановиться, набрав N прошедших тестов.
    until: int | None = None
    #: Писать полный лог trials.jsonl (по умолчанию только successes.jsonl).
    save_all: bool = False
    #: Файл-флаг: пока он существует, новые задачи не берутся.
    stop_file: str | Path | None = None
    #: Порядок обхода: product, shuffle или diagonal (разреженный срез).
    order: str = "product"
    #: Брать каждый N-й конфиг (разреженная выборка).
    stride: int = 1


@dataclass
class RunSummary:
    requested: int = 0
    executed: int = 0
    passed: int = 0
    rejected: int = 0
    stopped: int = 0
    errors: int = 0
    infeasible: int = 0
    skipped: int = 0
    interrupted: bool = False
    target_reached: bool = False
    stopped_by_file: bool = False
    duration: float = 0.0
    by_reason: Counter = field(default_factory=Counter)
    visualizers: Counter = field(default_factory=Counter)

    def add(self, record: dict[str, Any]) -> None:
        status = record.get("status")
        if status == TrialStatus.PASSED.value:
            self.passed += 1
        elif status == TrialStatus.REJECTED.value:
            self.rejected += 1
            self.by_reason[str(record.get("reason", "")).split(":")[0]] += 1
        elif status == TrialStatus.STOPPED.value:
            self.stopped += 1
            self.by_reason[str(record.get("reason", ""))] += 1
        elif status == TrialStatus.ERROR.value:
            self.errors += 1
        elif status == TrialStatus.INFEASIBLE.value:
            self.infeasible += 1
        # Track visualizer used for this record
        visualizer = record.get("visualizer", None)
        if visualizer is not None:
            self.visualizers[visualizer] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "executed": self.executed,
            "passed": self.passed,
            "rejected": self.rejected,
            "stopped": self.stopped,
            "errors": self.errors,
            "infeasible": self.infeasible,
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "target_reached": self.target_reached,
            "stopped_by_file": self.stopped_by_file,
            "duration": round(self.duration, 3),
            "by_reason": dict(self.by_reason.most_common()),
            "visualizers": dict(self.visualizers.most_common()),
        }

    def text(self) -> str:
        lines = [
            f"Тестов исполнено: {self.executed}  (запрошено: {self.requested}, "
            f"пропущено при возобновлении: {self.skipped}, "
            f"несовместимых: {self.infeasible})",
            f"  passed: {self.passed} | rejected: {self.rejected} | "
            f"stopped: {self.stopped} | errors: {self.errors}",
            f"  время: {self.duration:.2f} с",
        ]
        if self.interrupted:
            lines.append("  ПРЕРВАНО пользователем — успешные результаты сохранены")
        if self.target_reached:
            lines.append("  ЦЕЛЬ ДОСТИГНУТА — новых задач не берётся")
        if self.stopped_by_file:
            lines.append("  ОСТАНОВКА по stop-файлу — успешные результаты сохранены")
        if self.by_reason:
            lines.append("  причины:")
            for reason, count in self.by_reason.most_common(8):
                lines.append(f"    {count:>6}  {reason}")
        if self.visualizers:
            lines.append("  визуализаторы:")
            for vis, count in self.visualizers.most_common(8):
                lines.append(f"    {count:>6}  {vis}")
        return "\n".join(lines)


class Runner:
    """Перебирает конфиги, отбрасывает несовместимые и запускает тесты."""

    def __init__(self, experiment: Experiment, options: RunOptions | None = None) -> None:
        self.experiment = experiment
        self.options = options or RunOptions()
        self.summary = RunSummary()
        self.store = ResultStore(
            self.options.out,
            resume=self.options.resume,
            save_all=self.options.save_all,
        )

    # --- источники конфигов ---------------------------------------------
    def _configs(self) -> Iterator[Fixed]:
        options = self.options
        if options.random_start:
            rng = random.Random(options.seed)
            produced = 0
            seen: set[str] = set()
            attempts = 0
            while options.limit is None or produced < options.limit * 8:
                attempts += 1
                if attempts > max(1000, produced * 100):
                    # Пространство исчерпано: не крутимся вечно на дублях.
                    return
                try:
                    binding = self.experiment.space.random_binding(rng)
                except Exception:  # noqa: BLE001 - редкий тупик ограничений
                    continue
                fixed = Fixed.of(binding)
                digest = fixed.digest()
                if digest in seen:
                    continue
                seen.add(digest)
                produced += 1
                yield fixed
            return
        skipped = 0
        for fixed in self.experiment.iter_configs(
            feasible=options.only_feasible,
            strategy=options.order,
            seed=options.seed,
            stride=options.stride,
        ):
            if skipped < options.skip:
                skipped += 1
                continue
            yield fixed

    def _tasks(self) -> Iterator[Task]:
        options = self.options
        executed = 0
        for fixed in self._configs():
            if options.limit is not None and executed >= options.limit:
                return
            digest = fixed.digest()
            if self.store.seen(digest):
                self.summary.skipped += 1
                continue
            reason = self.experiment.feasible(fixed)
            if reason is not None:
                self.summary.infeasible += 1
                if options.save_all:
                    self.store.append(
                        {
                            "config": fixed.to_dict(),
                            "config_digest": digest,
                            "status": TrialStatus.INFEASIBLE.value,
                            "reason": reason,
                            "phase": None,
                            "iterations": 0,
                            "seed": 0,
                            "metrics": {},
                            "state_digest": None,
                            "duration": 0.0,
                            "error": None,
                        }
                    )
                if options.progress and self.summary.infeasible <= 3:
                    print(f"  infeasible {digest}: {reason}")
                continue
            executed += 1
            self.summary.requested += 1
            yield (executed, fixed.to_dict())

    # --- запуск ----------------------------------------------------------
    def _should_stop(self) -> bool:
        options = self.options
        if options.until is not None and self.summary.passed >= options.until:
            self.summary.target_reached = True
            return True
        if options.stop_file is not None and Path(options.stop_file).exists():
            self.summary.stopped_by_file = True
            return True
        return False

    def run(self, on_result: Callable[[Record], None] | None = None) -> RunSummary:
        options = self.options
        started = time.perf_counter()
        workers = options.workers if options.mode != "serial" else 1
        dispatcher = Dispatcher(
            self.experiment,
            workers=workers,
            mode=options.mode,
            base_seed=options.seed,
            keep_state=options.keep_state,
            max_iterations=options.max_iterations,
            should_stop=self._should_stop,
        )

        task_stream = self._tasks()
        first = next(task_stream, None)
        if first is not None:
            self._warmup(first)

        def handle(record: Record) -> None:
            self.summary.executed += 1
            self.summary.add(record)
            self.store.append(record)
            if on_result is not None:
                on_result(record)
            elif options.progress:
                self._print_record(record)

        dispatcher.run(_chain(first, task_stream), handle)
        self.summary.interrupted = dispatcher.interrupted
        if dispatcher.stopped_early and options.until is not None:
            self.summary.target_reached = self.summary.passed >= options.until
        self.summary.duration = time.perf_counter() - started
        self._write_summary()
        self.store.close()
        return self.summary

    def _warmup(self, task: Task) -> None:
        """Прогреть numba до пула, чтобы компиляция не шла в каждом процессе."""

        if self.options.mode == "serial":
            return
        fixed = Fixed.from_dict(task[1])
        try:
            self.experiment.preflight(fixed)
        except Exception:  # noqa: BLE001 - прогрев не должен мешать прогону
            pass

    def _print_record(self, record: Record) -> None:
        status = str(record.get("status", "?"))
        digest = str(record.get("config_digest", ""))
        duration = float(record.get("duration", 0.0))
        reason = str(record.get("reason", ""))
        tail = f"  {reason}" if reason else ""
        index = record.get("index", "?")
        print(
            f"[{index:>6}] {status:<10} {digest}  {record.get('iterations', 0):>4} итер. "
            f"{duration:6.3f} с{tail}"
        )

    def _write_summary(self) -> None:
        if self.store.out_dir is None:
            return
        import json

        path = Path(self.store.out_dir) / "summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.summary.to_dict(), handle, ensure_ascii=False, indent=2)


def run(
    experiment: Experiment,
    options: RunOptions | None = None,
    on_result: Callable[[Record], None] | None = None,
) -> RunSummary:
    return Runner(experiment, options).run(on_result=on_result)
