"""Параллельное исполнение перебора.

Три режима:

* ``serial`` — один процесс, удобно для отладки;
* ``thread`` — numba отпускает GIL, поэтому потоки реально работают;
* ``process`` — пул процессов (fork), изоляция ошибок и настоящий параллелизм.

Раннер не знает, что исполняется: он получает ``Config`` и возвращает
``TrialResult``. Отмена по Ctrl+C выставляет событие, работники завершают
текущий тест и выходят.
"""

from __future__ import annotations

import concurrent.futures as futures
import multiprocessing as mp
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .assembly import preflight
from .config import Config
from .errors import Cancelled, Incompatible
from .experiment import Experiment
from .runtime import ERROR, REJECTED, STOPPED, SUCCESS, Trial, TrialResult
from .storage import RunStats, RunStore

# ----------------------------------------------------------------------
# состояние работника
# ----------------------------------------------------------------------
_WORKER_CANCEL: Any = None
_WORKER_EXPERIMENT: Experiment | None = None
_WORKER_RECORD: bool = True
_WORKER_MAX_ITERATIONS: int | None = None


@dataclass
class RunOptions:
    experiment: str = ""
    strategy: str = "random"
    limit: int | None = None
    until: int | None = None
    workers: int = 1
    mode: str = "serial"
    seed: int = 0
    out: str = "runs/default"
    resume: bool = False
    max_seconds: float | None = None
    max_iterations: int | None = None
    skip: int = 0
    quiet: bool = False
    # кадры при переборе не используются: прогон хранит только конфиг
    record_frames: bool = False


def _worker_init(
    cancel: Any,
    experiment: Experiment | None,
    record: bool,
    max_iterations: int | None,
    ignore_sigint: bool = False,
) -> None:
    global _WORKER_CANCEL, _WORKER_EXPERIMENT, _WORKER_RECORD, _WORKER_MAX_ITERATIONS
    _WORKER_CANCEL = cancel
    _WORKER_EXPERIMENT = experiment
    _WORKER_RECORD = record
    _WORKER_MAX_ITERATIONS = max_iterations
    if ignore_sigint:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_one(config: Config) -> TrialResult | None:
    if _WORKER_CANCEL is not None and _WORKER_CANCEL.is_set():
        return None
    from .assembly import assemble

    experiment = _WORKER_EXPERIMENT
    if experiment is None:  # pragma: no cover - serial путь
        raise RuntimeError("работник не инициализирован")
    try:
        assembly = assemble(experiment, config)
    except Incompatible as exc:
        return TrialResult(REJECTED, {}, str(exc), 0, [], None, 0.0)
    except Exception as exc:  # noqa: BLE001 - ошибка сборки не валит прогон
        return TrialResult(ERROR, {}, type(exc).__name__, 0, [], f"{type(exc).__name__}: {exc}", 0.0)
    trial = Trial(
        assembly,
        config.seed,
        record=_WORKER_RECORD,
        max_iterations=_WORKER_MAX_ITERATIONS,
    )
    return trial.run()


# ----------------------------------------------------------------------
# генерация конфигов
# ----------------------------------------------------------------------
def _config_source(experiment: Experiment, options: RunOptions) -> Iterator[Config]:
    experiment.finalize()
    bindings = experiment.iter_configs(strategy=options.strategy, seed=options.seed)
    produced = 0
    ordinal = 0
    for binding in bindings:
        if ordinal < options.skip:
            ordinal += 1
            continue
        if options.limit is not None and produced >= options.limit:
            return
        produced += 1
        yield Config(experiment.name, binding, options.seed + ordinal)
        ordinal += 1


# ----------------------------------------------------------------------
# исполнение
# ----------------------------------------------------------------------
def execute(experiment: Experiment, options: RunOptions, store: RunStore) -> RunStats:
    experiment.finalize()
    if options.mode not in ("serial", "thread", "process"):
        raise ValueError(f"неизвестный режим: {options.mode!r}")
    stats = RunStats()
    done = store.done_digests() if options.resume else set()

    if options.max_seconds is not None:
        deadline = time.time() + float(options.max_seconds)
    else:
        deadline = None

    if options.mode == "serial":
        _execute_serial(experiment, options, store, stats, done, deadline)
    else:
        _execute_pool(experiment, options, store, stats, done, deadline)

    summary = stats.summary()
    summary.update(
        {
            "experiment": experiment.name,
            "description": experiment.description,
            "strategy": options.strategy,
            "seed": options.seed,
            "mode": options.mode,
            "workers": options.workers,
            "limit": options.limit,
            "until": options.until,
            "out": str(store.directory),
        }
    )
    store.write_summary(summary)
    return stats


def _record(stats: RunStats, store: RunStore, config: Config, result: TrialResult, options: RunOptions) -> None:
    store.save(config, result)
    stats.add(result.status)
    if result.status == SUCCESS:
        stats.success_records += 1


def _skip(config: Config, done: set[str], stats: RunStats) -> bool:
    if config.digest() in done:
        stats.resumed += 1
        return True
    return False


def _execute_serial(
    experiment: Experiment,
    options: RunOptions,
    store: RunStore,
    stats: RunStats,
    done: set[str],
    deadline: float | None,
) -> None:
    _worker_init(None, experiment, options.record_frames, options.max_iterations)
    try:
        for config in _config_source(experiment, options):
            if _skip(config, done, stats):
                continue
            if deadline is not None and time.time() >= deadline:
                break
            result = _run_one(config)
            if result is None:
                break
            _record(stats, store, config, result, options)
            if not options.quiet:
                _progress(stats)
            if options.until is not None and stats.success_records >= options.until:
                break
    except KeyboardInterrupt:
        raise Cancelled("прогон прерван пользователем") from None
    finally:
        if not options.quiet:
            _clear_progress()


def _execute_pool(
    experiment: Experiment,
    options: RunOptions,
    store: RunStore,
    stats: RunStats,
    done: set[str],
    deadline: float | None,
) -> None:
    workers = max(1, int(options.workers))
    cancel: Any
    executor: futures.Executor
    if options.mode == "thread":
        cancel = threading.Event()
        executor = futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sweep")
        _worker_init(cancel, experiment, options.record_frames, options.max_iterations)
    else:
        context = mp.get_context("fork")
        cancel = context.Event()
        # прогрев numba в родителе: дети наследуют скомпилированный код
        try:
            first = next(_config_source(experiment, options))
            preflight(experiment, first)
        except StopIteration:
            return
        executor = futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(cancel, experiment, options.record_frames, options.max_iterations, True),
        )
        _worker_init(cancel, experiment, options.record_frames, options.max_iterations)

    pending: dict[futures.Future, Config] = {}
    last_report = 0.0
    interrupted = False
    stop = False
    source = _config_source(experiment, options)

    def submit_next() -> bool:
        for config in source:
            if _skip(config, done, stats):
                continue
            pending[executor.submit(_run_one, config)] = config
            return True
        return False

    try:
        while not stop:
            while len(pending) < workers * 2 and submit_next():
                pass
            if not pending:
                break
            if deadline is not None and time.time() >= deadline:
                break
            timeout = 0.25
            if deadline is not None:
                timeout = max(0.01, min(timeout, deadline - time.time()))
            ready, _ = futures.wait(pending, timeout=timeout, return_when=futures.FIRST_COMPLETED)
            if not ready:
                continue
            for future in ready:
                config = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = TrialResult(ERROR, {}, type(exc).__name__, 0, [], f"{type(exc).__name__}: {exc}")
                if result is None:
                    stop = True
                    continue
                _record(stats, store, config, result, options)
                if options.until is not None and stats.success_records >= options.until:
                    stop = True
                    break
            if not options.quiet:
                now = time.time()
                if now - last_report > 0.25:
                    _progress(stats)
                    last_report = now
    except KeyboardInterrupt:
        interrupted = True
        cancel.set()
    finally:
        if interrupted:
            if not options.quiet:
                _clear_progress()
                print("прерывание: останавливаю работников...", file=sys.stderr)
            executor.shutdown(wait=False, cancel_futures=True)
            raise Cancelled("прогон прерван пользователем")
        executor.shutdown(wait=True, cancel_futures=True)
        if not options.quiet:
            _clear_progress()


# ----------------------------------------------------------------------
def _progress(stats: RunStats) -> None:
    line = (
        f"тестов: {stats.trials} | успехов: {stats.success_records} | "
        f"отклонено: {stats.rejected} | остановлено: {stats.stopped} | ошибок: {stats.errors}"
    )
    print("\r" + line.ljust(96), end="", file=sys.stderr, flush=True)


def _clear_progress() -> None:
    print("\r" + " " * 96 + "\r", end="", file=sys.stderr, flush=True)


__all__ = ["RunOptions", "execute"]
