"""CLI движка: ``python -m sweep <команда>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .errors import Cancelled, Incompatible, SweepError
from .experiments import get_experiment, list_experiments
from .module import Role
from .parallel import RunOptions, execute
from .runtime import ERROR, REJECTED, STOPPED, SUCCESS
from .replay import frames_for, states_of
from .storage import RunStore, read_jsonl
from .visualize.engine import render_grid
from .visualize.server import make_server
from .visualize.terminal import render_terminal


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _experiment(name: str):
    try:
        return get_experiment(name)
    except KeyError as exc:
        raise SystemExit(str(exc.args[0] if exc.args else exc)) from None


def _use_profile(args: argparse.Namespace) -> None:
    """Профиль плотного эксперимента до его сборки."""

    path = getattr(args, "profile", None)
    if path:
        os.environ["SWEEP_OMEGA5_PROFILE"] = str(path)


def _states(experiment, record: dict[str, Any] | None = None) -> int:
    if record:
        return states_of(record)
    module = experiment._modules[Role.DATA][0]  # noqa: SLF001
    try:
        values = list(experiment.iter_configs(max_configs=1))
        if values:
            return int(values[0][module.key("states")])
    except Exception:  # noqa: BLE001
        pass
    return 2


def _pick(records: list[dict[str, Any]], index: int | None, digest: str | None) -> dict[str, Any]:
    if not records:
        raise SystemExit("в прогоне нет успешных конфигов")
    if digest:
        for record in records:
            if record.get("digest") == digest:
                return record
        raise SystemExit(f"конфиг {digest} не найден")
    if index is None:
        index = 0
    if index < 0 or index >= len(records):
        raise SystemExit(f"индекс {index} вне диапазона 0..{len(records) - 1}")
    return records[index]


# ----------------------------------------------------------------------
# команды
# ----------------------------------------------------------------------
def cmd_modules(args: argparse.Namespace) -> int:
    _use_profile(args)
    names = [args.experiment] if args.experiment else list_experiments()
    for name in names:
        experiment = _experiment(name)
        experiment.finalize()
        print(f"\n=== {experiment.name} ===")
        if experiment.description:
            print(experiment.description)
        for role in Role:
            modules = experiment._modules.get(role, [])  # noqa: SLF001
            if not modules:
                continue
            print(f"\n{role.value}:")
            for module in modules:
                print(f"  {module.id:<30} {module.describe()}")
                for requirement in module.requires:
                    print(f"      требует: {requirement.describe()}")
        print("\nфазы:")
        for phase in experiment.phases:
            filters = ", ".join(module.id for module in phase.filters) or "—"
            print(f"  {phase.name:<12} итерации={phase.iterations} частота={phase.frequency} фильтры={filters}")
    return 0


def cmd_space(args: argparse.Namespace) -> int:
    _use_profile(args)
    experiment = _experiment(args.experiment)
    experiment.finalize()
    print(f"эксперимент: {experiment.name}")
    for line in experiment.space.describe():
        print("  " + line)
    if args.count:
        limit = args.limit if args.limit is not None else 10_000
        try:
            total = experiment.count(strategy=args.strategy, limit=limit, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"не удалось посчитать: {exc}")
            return 1
        suffix = "+" if total >= limit else ""
        print(f"\nконфигов (с учётом совместимости): {total}{suffix}")
    if args.samples:
        source = experiment.iter_configs(
            strategy=args.strategy,
            seed=args.seed,
            max_configs=args.samples,
        )
        print("\nпримеры:")
        for index, binding in enumerate(source):
            values = ", ".join(f"{key}={value}" for key, value in sorted(binding.items()))
            print(f"  [{index}] {values}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _use_profile(args)
    experiment = _experiment(args.experiment)
    options = RunOptions(
        experiment=experiment.name,
        strategy=args.strategy,
        limit=args.limit,
        until=args.until,
        workers=args.workers,
        mode=args.mode,
        seed=args.seed,
        out=args.out,
        resume=args.resume,
        max_seconds=args.max_seconds,
        max_iterations=args.max_iterations,
        skip=args.skip,
        quiet=args.quiet,
    )
    store = RunStore(args.out)
    try:
        stats = execute(experiment, options, store)
    except Cancelled as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except Incompatible as exc:
        print(f"несовместимая конфигурация: {exc}", file=sys.stderr)
        return 1
    except SweepError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"Тестов исполнено: {stats.trials}  "
        f"(успехов: {stats.success_records}, возобновлено: {stats.resumed}, "
        f"отклонено: {stats.rejected}, остановлено: {stats.stopped}, ошибок: {stats.errors})"
    )
    print(f"Успешные конфиги: {Path(args.out) / 'successes.jsonl'}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    records = store.successes()
    if not records:
        print("успешных конфигов нет")
        return 0
    limit = args.limit or 20
    for index, record in enumerate(records[:limit]):
        metrics = record.get("metrics", {})
        dense = " ".join(f"{key}={value:.3g}" for key, value in sorted(metrics.items()) if isinstance(value, float))
        print(f"[{index:>4}] {record['digest']}  status={record.get('status')}  {dense}")
        values = record.get("config", {}).get("values", {})
        print("       " + ", ".join(f"{key}={value}" for key, value in sorted(values.items())))
    if len(records) > limit:
        print(f"... всего {len(records)}; используйте --limit")
    summary = store.summary()
    if summary:
        print(
            f"\nитого: {summary.get('trials')} тестов, "
            f"{summary.get('successes', len(records))} успехов, {summary.get('elapsed')} с"
        )
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    records = store.successes()
    record = _pick(records, args.index, args.digest)
    experiment = _experiment(record["config"]["experiment"])
    frames = frames_for(record, limit=args.frames)
    if not frames:
        raise SystemExit("конфиг не дал кадров")
    name = args.digest or record["digest"]
    visualizer = experiment.visualizer
    states = _states(experiment, record)
    out_dir = args.out or "viz"
    rendered = render_grid(
        frames,
        visualizer,
        states=states,
        out_dir=out_dir,
        name=name,
        cols=args.cols,
        scale=args.scale,
        captions=[f"кадр {i}" for i in range(len(frames))],
        write_html=not args.no_html,
    )
    print(f"кадров: {len(frames)} · PNG: {rendered.png}")
    if rendered.html:
        print(f"HTML: {rendered.html}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    records = store.successes()
    record = _pick(records, args.index, args.digest)
    experiment = _experiment(record["config"]["experiment"])
    frames = frames_for(record)
    if not frames:
        raise SystemExit("конфиг не дал кадров")
    states = _states(experiment, record)
    visualizer = experiment.visualizer
    try:
        while True:
            for frame in frames:
                print("\x1b[2J\x1b[H", end="")
                print(render_terminal(frame, visualizer, states=states, scale=args.scale))
                time.sleep(1.0 / max(1.0, args.fps))
            if args.once:
                break
    except KeyboardInterrupt:
        pass
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    record = _pick(store.successes(), args.index, args.digest)
    print(f"конфиг {record['digest']}: {record['config']['values']}")
    print(
        f"статус {record['status']} · итераций {record['iterations']} · "
        f"кадров {record.get('frames', 0)}"
    )
    print("метрики:", {k: v for k, v in record.get("metrics", {}).items() if not k.startswith("share_")})
    if args.trace:
        frames = frames_for(record)
        print(f"повторная сборка: кадров {len(frames)}")
        for frame in frames:
            print(f"  кадр {frame.shape} · states≈{int(frame.max()) + 1}")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Показать и проверить профиль долей плотного эксперимента."""

    _use_profile(args)
    from .experiments.omega5_dense.profile import PROFILE_PATH, describe_profile, load_density, load_profile

    try:
        profile = load_profile(args.profile)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    source = args.profile or os.environ.get("SWEEP_OMEGA5_PROFILE") or PROFILE_PATH
    print(f"профиль: {source}")
    print(describe_profile(profile))
    print(f"плотность: {load_density(args.profile)[0]:.3f}..{load_density(args.profile)[1]:.3f}")
    return 0


def cmd_rule(args: argparse.Namespace) -> int:
    """Напечатать формальный паспорт правила (JSON)."""

    _use_profile(args)
    if args.run_dir:
        store = RunStore(args.run_dir)
        record = _pick(store.successes(), args.index, args.digest)
        experiment = _experiment(record["config"]["experiment"])
        values = record["config"]["values"]
    else:
        experiment = _experiment(args.experiment)
        experiment.finalize()
        source = experiment.iter_configs(
            strategy="random",
            seed=args.seed,
            max_configs=1,
        )
        try:
            values = next(source)
        except StopIteration:
            raise SystemExit("не удалось получить ни одного совместимого конфига") from None
    info = experiment.visualizer.rule_info(values) if experiment.visualizer else None
    if info is None:
        raise SystemExit(f"эксперимент {experiment.name} не умеет описывать правило")
    print(
        json.dumps(
            info,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":") if args.compact else None,
            indent=None if args.compact else 2,
        )
    )
    return 0


def _stores(args: argparse.Namespace) -> dict[str, RunStore]:
    """Прогоны из ``--run-dir`` и подкаталогов ``--runs`` без пустых."""

    stores: dict[str, RunStore] = {}
    explicit: set[str] = set()
    for raw in getattr(args, "run_dir", None) or []:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"нет такого прогона: {path}")
        stores[path.name] = RunStore(path)
        explicit.add(path.name)
    root = getattr(args, "runs", None)
    if root:
        base = Path(root)
        if not base.is_dir():
            raise SystemExit(f"нет такого каталога: {base}")
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "successes.jsonl").exists():
                stores.setdefault(child.name, RunStore(child))
    if not stores:
        raise SystemExit("нечего показывать: укажите --run-dir или --runs")
    for name in list(stores):
        if stores[name].successes():
            continue
        if name in explicit:
            raise SystemExit(f"в прогоне {name} нет успешных конфигов")
        del stores[name]
    if not stores:
        raise SystemExit("ни в одном прогоне нет успешных конфигов")
    return stores


def cmd_view(args: argparse.Namespace) -> int:
    stores = _stores(args)
    server = make_server(stores, host=args.host, port=args.port)
    host, port = server.server_address[:2]
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    total = sum(len(store.successes()) for store in stores.values())
    print(
        f"http://{host}:{port}/  ·  прогонов: {len(stores)}  ·  конфигов: {total}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("просмотр остановлен", file=sys.stderr)
    finally:
        server.server_close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Статическая галерея для GitHub Pages: всё считается в браузере."""

    from .visualize.build import build_site

    stores = _stores(args)
    summary = build_site(stores, args.out, title=args.title, page_size=args.page_size)
    size = summary["bytes"] / (1024 * 1024)
    print(
        f"экспорт: {Path(args.out)}  ·  прогонов: {summary['runs']}  ·  карточек: {summary['cards']}"
        f"  ·  {size:.1f} МБ"
    )
    print("эксперименты: " + ", ".join(summary["experiments"]))
    if summary["skipped"]:
        print("пропущены (нет web-модуля): " + ", ".join(sorted(set(summary["skipped"]))))
    print(f"открыть локально: {Path(args.out) / 'index.html'}")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sweep", description="перебор клеточных автоматов")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("modules", help="модули, требования и фазы")
    p.add_argument("-e", "--experiment")
    p.add_argument("--profile", help="JSON-профиль долей для omega5_dense")
    p.set_defaults(func=cmd_modules)

    p = sub.add_parser("space", help="карта осей и размер пространства")
    p.add_argument("-e", "--experiment", default="life")
    p.add_argument("--profile", help="JSON-профиль долей для omega5_dense")
    p.add_argument("--strategy", default="random", choices=["product", "random"])
    p.add_argument("--samples", type=int, default=0)
    p.add_argument("--count", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_space)

    p = sub.add_parser("run", help="запустить перебор")
    p.add_argument("-e", "--experiment", default="life")
    p.add_argument("--profile", help="JSON-профиль долей для omega5_dense")
    p.add_argument("--strategy", default="random", choices=["product", "random"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--until", type=int, help="остановиться после N успехов")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--mode", default="serial", choices=["serial", "thread", "process"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/default")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--max-iterations", type=int)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("show", help="показать сохранённые успехи")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("viz", help="нарисовать кадры успешного конфига")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--index", type=int)
    p.add_argument("--digest")
    p.add_argument("--frames", type=int)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--out")
    p.add_argument("--no-html", action="store_true")
    p.set_defaults(func=cmd_viz)

    p = sub.add_parser("live", help="проиграть кадры в терминале")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--index", type=int)
    p.add_argument("--digest")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--scale", type=int, default=1)
    p.add_argument("--once", action="store_true", help="не зацикливать")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("replay", help="текстовая сводка конфига")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--index", type=int)
    p.add_argument("--digest")
    p.add_argument("--trace", action="store_true")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("profile", help="показать и проверить профиль omega5_dense")
    p.add_argument("--profile", help="путь к JSON-профилю долей")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("rule", help="формальный JSON-паспорт правила")
    p.add_argument("-e", "--experiment", default="omega5")
    p.add_argument("--profile", help="JSON-профиль долей для omega5_dense")
    p.add_argument("--run-dir")
    p.add_argument("--index", type=int)
    p.add_argument("--digest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compact", action="store_true")
    p.set_defaults(func=cmd_rule)

    p = sub.add_parser("view", help="галерея прогонов в браузере")
    p.add_argument("--run-dir", action="append", help="каталог прогона (можно несколько)")
    p.add_argument("--runs", help="каталог с прогонами-подкаталогами")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("export", help="статическая галерея для GitHub Pages")
    p.add_argument("--run-dir", action="append", help="каталог прогона (можно несколько)")
    p.add_argument("--runs", help="каталог с прогонами-подкаталогами")
    p.add_argument("--out", default="site")
    p.add_argument("--page-size", type=int, default=24, help="карточек на странице по умолчанию")
    p.add_argument("--title", default="sweep")
    p.set_defaults(func=cmd_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:  # pragma: no cover
        return 0


__all__ = ["build_parser", "main"]
