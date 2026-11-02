"""Планировщик исполнения тестов.

Тесты независимы, поэтому один конфиг = одна задача. Поддерживаются три режима:

* ``serial``  — в текущем процессе (отладка, визуализация);
* ``thread``  — потоки; numba-ядра собраны с ``nogil=True`` и отпускают GIL;
* ``process`` — процессы через ``fork`` (компиляция numba наследуется потомками).

Задачи — это пары ``(index, config_dict)``, результаты — JSON-совместимые записи,
поэтому между процессами не нужно передавать модули с лямбдами.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from typing import Any, Callable, Iterable

from .experiment import Experiment
from .fixed import Fixed
from .trial import derive_seed, run_trial

Task = tuple[int, dict[str, Any]]
Record = dict[str, Any]

_CONTEXT: dict[str, Any] = {}


def _process_init(
    experiment: Experiment, base_seed: int, keep_state: bool, max_iterations: int | None
) -> None:
    _CONTEXT["experiment"] = experiment
    _CONTEXT["base_seed"] = base_seed
    _CONTEXT["keep_state"] = keep_state
    _CONTEXT["max_iterations"] = max_iterations


def _execute(
    experiment: Experiment,
    base_seed: int,
    keep_state: bool,
    max_iterations: int | None,
    task: Task,
) -> Record:
    index, config = task
    fixed = Fixed.from_dict(config)
    seed = derive_seed(base_seed, fixed.digest())
    result = run_trial(
        experiment,
        fixed,
        seed=seed,
        keep_state=keep_state,
        max_iterations=max_iterations,
    )
    record = result.to_record()
    record["index"] = index
    return record


def _process_job(task: Task) -> Record:
    return _execute(
        _CONTEXT["experiment"],
        _CONTEXT["base_seed"],
        _CONTEXT["keep_state"],
        _CONTEXT["max_iterations"],
        task,
    )


class Dispatcher:
    """Исполняет поток задач, вызывая ``on_result`` по мере готовности."""

    def __init__(
        self,
        experiment: Experiment,
        *,
        workers: int = 1,
        mode: str = "serial",
        base_seed: int = 0,
        keep_state: bool = False,
        window: int | None = None,
        max_iterations: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.experiment = experiment
        self.workers = max(1, workers)
        self.mode = mode
        self.base_seed = base_seed
        self.keep_state = keep_state
        self.window = window or self.workers * 4
        self.max_iterations = max_iterations
        self.should_stop = should_stop
        self.interrupted = False
        self.stopped_early = False

    # --- публичный API ---------------------------------------------------
    def run(self, tasks: Iterable[Task], on_result: Callable[[Record], None]) -> bool:
        """Вернуть ``True``, если работа была прервана (Ctrl+C)."""

        if self.mode == "auto":
            self.mode = "process" if self.workers > 1 and self._supports_fork() else (
                "thread" if self.workers > 1 else "serial"
            )
        if self.mode == "process" and not self._supports_fork():
            raise RuntimeError(
                "Режим process требует start method 'fork'; используйте --mode thread"
            )
        try:
            if self.mode == "serial":
                self._run_serial(tasks, on_result)
            elif self.mode == "thread":
                self._run_threads(tasks, on_result)
            else:
                self._run_processes(tasks, on_result)
        except KeyboardInterrupt:
            self.interrupted = True
        return self.interrupted

    # --- режимы ----------------------------------------------------------
    def _run_serial(self, tasks: Iterable[Task], on_result: Callable[[Record], None]) -> None:
        for task in tasks:
            if self._stop_requested():
                self.stopped_early = True
                return
            on_result(
                _execute(
                    self.experiment,
                    self.base_seed,
                    self.keep_state,
                    self.max_iterations,
                    task,
                )
            )

    def _run_threads(self, tasks: Iterable[Task], on_result: Callable[[Record], None]) -> None:
        iterator = iter(tasks)
        inflight: dict[Any, int] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            try:
                while True:
                    while len(inflight) < self.window and not self._stop_requested():
                        try:
                            task = next(iterator)
                        except StopIteration:
                            break
                        inflight[
                            executor.submit(
                                _execute,
                                self.experiment,
                                self.base_seed,
                                self.keep_state,
                                self.max_iterations,
                                task,
                            )
                        ] = task[0]
                    if self._stop_requested():
                        self.stopped_early = True
                        break
                    if not inflight:
                        break
                    ready = [future for future in inflight if future.done()]
                    if not ready:
                        time.sleep(0.004)
                        continue
                    for future in ready:
                        del inflight[future]
                        on_result(future.result())
            finally:
                for future in inflight:
                    future.cancel()

    def _run_processes(self, tasks: Iterable[Task], on_result: Callable[[Record], None]) -> None:
        # fork: эксперимент с лямбдами наследуется потомками без pickle,
        # а скомпилированные numba-ядра не компилируются заново.
        context = mp.get_context("fork")
        executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_process_init,
            initargs=(self.experiment, self.base_seed, self.keep_state, self.max_iterations),
        )
        iterator = iter(tasks)
        inflight: dict[Any, int] = {}
        try:
            while True:
                while len(inflight) < self.window and not self._stop_requested():
                    try:
                        task = next(iterator)
                    except StopIteration:
                        break
                    inflight[executor.submit(_process_job, task)] = task[0]
                if self._stop_requested():
                    self.stopped_early = True
                    break
                if not inflight:
                    break
                ready, _ = wait(inflight, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in ready:
                    del inflight[future]
                    on_result(future.result())
        finally:
            for future in inflight:
                future.cancel()
            # Не ждём уже запущенные задачи: частичные результаты уже сохранены.
            executor.shutdown(wait=False, cancel_futures=True)

    # --- вспомогательное -------------------------------------------------
    def _stop_requested(self) -> bool:
        return bool(self.should_stop is not None and self.should_stop())

    @staticmethod
    def _supports_fork() -> bool:
        return "fork" in mp.get_all_start_methods()

    @staticmethod
    def default_workers() -> int:
        return max(1, (os.cpu_count() or 1))
