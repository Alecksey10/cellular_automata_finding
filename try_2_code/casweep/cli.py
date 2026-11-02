"""CLI движка перебора: просмотр пространства, запуск, повтор, визуализация."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .fixed import Fixed
from .module import ROLE_ORDER, FilterBinding, Module, Role
from .runner import RunOptions, run
from .space import SWEEP_STRATEGIES
from .storage import find_record, read_records
from .trial import run_trial

DEFAULT_EXPERIMENT = "experiments.two_rings_rot:build_experiment"


# ---------------------------------------------------------------------------
# Загрузка эксперимента
# ---------------------------------------------------------------------------
def load_experiment(spec: str = DEFAULT_EXPERIMENT) -> Any:
    """Загрузить эксперимент по строке ``module.path:factory``."""

    module_path, _, attribute = spec.partition(":")
    attribute = attribute or "build_experiment"
    module = importlib.import_module(module_path)
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise SystemExit(f"В модуле {module_path!r} нет {attribute!r}") from exc
    experiment = factory()
    experiment.source = module_path
    if not getattr(experiment, "phases", None):
        raise SystemExit(f"Эксперимент {spec!r} не объявил ни одной стадии")
    return experiment


def infer_experiment_spec(run_dir: str | Path) -> str | None:
    """Определить эксперимент по сохранённому конфигу.

    ``view``/``compare``/``render`` читают конфиг из ``successes.jsonl``, но
    сам файл не хранит имя эксперимента. По идентификаторам выбранных модулей
    можно восстановить фабрику: это спасает от запуска ``runs/two_rings``
    через эксперимент по умолчанию ``two_rings_rot``.
    """

    records = read_records(run_dir, statuses=None, limit=1)
    if not records:
        return None
    config = records[0].get("config") or {}
    rule = config.get("rule.module")
    neighborhood = config.get("neighborhood.module")
    filters = {
        key.split(".", 2)[1]
        for key in config
        if key.startswith("filter.") and key.count(".") >= 1
    }

    if rule == "rule.aging_fortress":
        return "experiments.two_rings:build_experiment"
    if rule == "rule.rot_aging":
        return "experiments.two_rings_rot:build_experiment"
    if neighborhood == "neighborhood.moore5":
        return "experiments.cyclic5:build_experiment"
    if neighborhood == "neighborhood.moore3":
        # multi3 отличается волнами фаз; moore3 — симметрией/активностью.
        if {"phases", "entropy"} & filters:
            return "experiments.multi3:build_experiment"
        return "experiments.moore3:build_experiment"
    if neighborhood in (
        "neighborhood.square",
        "neighborhood.radius",
        "neighborhood.ring",
    ):
        return "experiments.life2d:build_experiment"
    return None


def load_experiment_for_run(args: argparse.Namespace) -> Any:
    """Загрузить эксперимент, при необходимости — по конфигу из ``--run-dir``.

    Явный ``--experiment`` имеет приоритет; подстановка работает только когда
    оставлено значение по умолчанию и указан каталог прогона.
    """

    spec = args.experiment
    run_dir = getattr(args, "run_dir", None)
    explicit_config = bool(getattr(args, "config", None))
    if run_dir and not explicit_config and spec == DEFAULT_EXPERIMENT:
        inferred = infer_experiment_spec(run_dir)
        if inferred:
            spec = inferred
    return load_experiment(spec)


def visual_dir(experiment: Any, subdir: str | None = None) -> Path:
    """Каталог визуализации рядом с пакетом эксперимента."""

    home = getattr(experiment, "home_dir", None)
    base = (home / "_viewer") if home is not None else Path("viewer")
    return base / subdir if subdir else base


def _all_modules(experiment: Any) -> dict[str, Module]:
    modules: dict[str, Module] = {}
    for selection in experiment.selections.values():
        for module in selection.candidates:
            modules[module.id] = module
    for binding in experiment.filters.values():
        modules[binding.module.id] = binding.module
    return modules


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------
def cmd_modules(args: argparse.Namespace) -> int:
    from visualize.protocol import visualizer_name

    experiment = load_experiment(args.experiment)
    modules = _all_modules(experiment)
    print(f"Эксперимент: {experiment.name}")
    roles: list[Role] = list(ROLE_ORDER) + [Role.FILTER]
    for role in roles:
        group = [module for module in modules.values() if module.role is role]
        if not group:
            continue
        print(f"\n{role.value}:")
        for module in group:
            print(f"  {module.id}")
            print(f"    параметры: {_declared_params(experiment, module)}")
            visualizer = visualizer_name(module)
            if visualizer:
                print(f"    визуализатор: {visualizer}")
            if module.derived:
                print(f"    derived:   {', '.join(module.derived)}")
            if module.requires:
                print("    требует:")
                for requirement in module.requires:
                    print(f"      - {requirement.describe()}")
            if module.provides:
                print("    даёт:")
                for capability in module.provides:
                    print(f"      - {capability}")
    return 0


def _declared_params(experiment: Any, module: Module) -> str:
    for selection in experiment.selections.values():
        if module.id in selection.declared_params:
            return ", ".join(sorted(selection.declared_params[module.id])) or "нет"
    for binding in experiment.filters.values():
        if binding.module is module:
            axes = [
                axis.key.split(".", 1)[-1]
                for axis in experiment.space._registered.values()
                if axis.key.startswith(binding.prefix + ".")
            ]
            return ", ".join(sorted(axes)) or "нет"
    return "нет"


def cmd_space(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    print(experiment.describe())
    print()
    estimate = experiment.estimate(samples=args.samples)
    print(f"Оценка числа конфигураций (выборка {args.samples}, сверху): {estimate:,}")
    limit = args.limit
    exact = experiment.count(limit=limit)
    suffix = f" (подсчёт остановлен на {limit:,})" if limit and exact >= limit else ""
    print(f"Число конфигураций с ограничениями: {exact:,}{suffix}")
    if experiment.space.dimension > 1:
        diagonal = experiment.count(strategy="diagonal")
        print(f"Диагональный срез (≤ длины самой длинной оси): {diagonal:,}")
    if args.feasible:
        feasible = experiment.count(limit=limit, feasible=True)
        suffix = f" (подсчёт остановлен на {limit:,})" if limit and feasible >= limit else ""
        print(f"Из них совместимых по интерфейсам: {feasible:,}{suffix}")
    if args.preview:
        print()
        print(f"Первые {args.preview} конфигураций:")
        shown = 0
        for fixed in experiment.iter_configs(
            feasible=args.feasible, strategy=args.order
        ):
            if shown >= args.preview:
                break
            shown += 1
            reason = experiment.feasible(fixed)
            mark = "OK " if reason is None else "BAD"
            print(f"  [{mark}] {fixed.digest()}  {fixed.canonical()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    if args.random and args.limit is None and args.until is None:
        raise SystemExit("--random требует --limit или --until, иначе перебор бесконечен")
    if args.until is not None and args.until <= 0:
        raise SystemExit("--until должно быть положительным")
    if args.max_iterations is not None and args.max_iterations < 1:
        raise SystemExit("--max-iterations должно быть положительным")
    if args.stride < 1:
        raise SystemExit("--stride должно быть >= 1")
    options = RunOptions(
        limit=args.limit,
        skip=args.skip,
        seed=args.seed,
        workers=args.workers,
        mode=args.mode,
        out=args.out,
        resume=args.resume,
        random_start=args.random,
        keep_state=args.keep_state,
        only_feasible=not args.all_configs,
        progress=not args.quiet,
        max_iterations=args.max_iterations,
        until=args.until,
        save_all=args.save_all,
        stop_file=args.stop_file,
        order=args.order,
        stride=args.stride,
    )
    summary = run(experiment, options)
    print(summary.text())
    if args.out:
        saved = Path(args.out) / "successes.jsonl"
        print(f"Успешные конфиги: {saved}")
        if args.save_all:
            print(f"Полный лог: {Path(args.out) / 'trials.jsonl'}")
    return 0 if summary.errors == 0 else 1


def cmd_replay(args: argparse.Namespace) -> int:
    experiment = load_experiment_for_run(args)
    config, seed = _resolve_config(args, args.seed)
    fixed = Fixed.from_dict(config)
    print(f"Конфиг {fixed.digest()}  seed={seed}")
    print(fixed.pretty())
    report = experiment.assembly.report(fixed)
    print()
    print("Совместимость:")
    print(report.text())

    frames: list[str] = []
    on_frame = None
    if args.trace:
        from visualize.grid2d import ascii_grid

        def on_frame(phase: str, step: int, grid: Any) -> None:
            if step % args.every == 0:
                frames.append(f"--- {phase} / итерация {step} ---\n{ascii_grid(grid)}")

    result = run_trial(experiment, fixed, seed=seed, keep_state=True, on_frame=on_frame)
    print()
    print(result.summary())
    print(json.dumps(dict(result.metrics), ensure_ascii=False, indent=2))
    if result.error:
        print(result.error)
    if args.trace and frames:
        frames_to_show = frames if args.frames is None else frames[: args.frames]
        print("\n" + "\n".join(frames_to_show))
    return 0 if result.ok else 1


def cmd_view(args: argparse.Namespace) -> int:
    experiment = load_experiment_for_run(args)
    from visualize.live import (
        TerminalStyle,
        play,
        resolve_view_config,
        terminal_color_default,
        truecolor_default,
    )

    fixed, seed = resolve_view_config(
        experiment,
        config=args.config,
        digest=args.digest,
        run_dir=args.run_dir,
        index=args.index,
        seed=args.seed,
        first=args.first,
    )
    style = TerminalStyle(
        scheme=args.scheme,
        color=False if args.no_color else terminal_color_default(),
        truecolor=truecolor_default(),
        clear=not args.no_clear and terminal_color_default(),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    return play(experiment, fixed, seed=seed, steps=args.steps, style=style)


def cmd_compare(args: argparse.Namespace) -> int:
    experiment = load_experiment_for_run(args)
    from visualize.live import (
        TerminalStyle,
        compare,
        pick_interesting,
        terminal_color_default,
        truecolor_default,
    )

    entries: list[tuple[Fixed, int]] = []
    if args.run_dir:
        records = read_records(args.run_dir, statuses=("passed",), limit=args.limit)
        for record in records:
            entries.append(
                (Fixed.from_dict(record["config"]), int(record.get("seed", 0)))
            )
    elif args.random:
        for index in range(args.random):
            entries.append(
                (
                    pick_interesting(
                        experiment, args.seed + index, candidates=4, probe_steps=4
                    ),
                    args.seed + index,
                )
            )
    else:
        raise SystemExit("Укажите --run-dir с успехами или --random N")
    if not entries:
        raise SystemExit("Нет автоматов для сравнения")

    style = TerminalStyle(
        scheme=args.scheme,
        color=False if args.no_color else terminal_color_default(),
        truecolor=truecolor_default(),
        clear=not args.no_clear and terminal_color_default(),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    return compare(experiment, entries, steps=args.steps, style=style)


def cmd_browse(args: argparse.Namespace) -> int:
    """Браузерная галерея: много автоматов с анимацией на одной странице."""

    experiment = load_experiment_for_run(args)
    from visualize.browser import capture_items, write_browser
    from visualize.live import pick_interesting

    entries: list[tuple[Fixed, int]] = []
    if args.run_dir:
        records = read_records(args.run_dir, statuses=("passed",), limit=args.limit)
        for record in records:
            entries.append(
                (Fixed.from_dict(record["config"]), int(record.get("seed", 0)))
            )
    elif args.random:
        for index in range(args.random):
            entries.append(
                (
                    pick_interesting(
                        experiment, args.seed + index, candidates=4, probe_steps=4
                    ),
                    args.seed + index,
                )
            )
    else:
        raise SystemExit("Укажите --run-dir с успехами или --random N")
    if not entries:
        raise SystemExit("Нет автоматов для галереи")

    out = Path(args.out) if args.out else visual_dir(experiment)
    items = capture_items(
        experiment,
        entries,
        steps=args.steps,
        every=args.every,
        scale=args.scale,
        scheme=args.scheme,
        max_frames=args.max_frames,
        fps=args.fps,
        on_error=lambda fixed, exc: print(f"  пропуск {fixed.digest()[:8]}: {exc}"),
    )
    if not items:
        raise SystemExit("Не удалось собрать ни одного автомата")
    index = write_browser(
        out / "index.html",
        items,
        title=f"Автоматы: {experiment.name}",
        subtitle=f"{len(items)} автоматов · визуализатор выбран по модулям",
        fps=args.fps,
    )
    print(f"Карточек: {len(items)}")
    print(f"Галерея: {index}")
    if args.open_browser:
        import webbrowser

        try:
            webbrowser.open(index.resolve().as_uri())
        except Exception:  # noqa: BLE001 - запуск браузера не должен ломать команду
            pass
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    experiment = load_experiment_for_run(args)
    config, seed = _resolve_config(args, args.seed)
    fixed = Fixed.from_dict(config)
    out = Path(args.out) if args.out else visual_dir(experiment, fixed.digest())
    print(f"Конфиг {fixed.digest()}  seed={seed}")
    from visualize.gallery import capture

    result = capture(
        experiment,
        fixed,
        out,
        every=args.every,
        scale=args.scale,
        seed=seed,
        scheme=args.scheme,
        grid_lines=args.grid_lines,
        viewer=not args.no_viewer,
        sheet=not args.no_sheet,
        sheet_columns=args.sheet_columns,
    )
    print(f"Сохранено кадров: {len(result.frame_paths)} -> {out}")
    for label, path in (
        ("лента", result.strip),
        ("лист", result.sheet),
        ("вьюер", result.viewer),
        ("финал", result.final),
    ):
        if path is not None:
            print(f"  {label}: {path}")
    return 0


def cmd_gallery(args: argparse.Namespace) -> int:
    experiment = load_experiment_for_run(args)
    records = read_records(args.run_dir, statuses=("passed",), limit=args.limit)
    if not records:
        raise SystemExit(
            f"В {args.run_dir} нет успешных записей (successes.jsonl); сначала запустите run"
        )
    from visualize.gallery import capture, describe_config
    from visualize.viewer import write_gallery

    home = experiment.home_dir
    out = (
        Path(args.out)
        if args.out
        else ((home / "_viewer" / "gallery") if home is not None else Path(args.run_dir) / "gallery")
    )
    out.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for record in records:
        fixed = Fixed.from_dict(record["config"])
        digest = fixed.digest()
        result = capture(
            experiment,
            fixed,
            out / digest,
            every=args.every,
            scale=args.scale,
            seed=int(record.get("seed", 0)),
            scheme=args.scheme,
            grid_lines=args.grid_lines,
            strip=False,
            sheet=False,
            title=f"Автомат {digest}",
        )
        items.append(
            {
                "digest": digest,
                "seed": record.get("seed", 0),
                "image": result.final_pixels,
                "viewer": f"{digest}/viewer.html" if result.viewer else None,
                "summary": describe_config(fixed),
            }
        )
    index = write_gallery(
        out / "index.html", items, title=f"Галерея автоматов: {experiment.name}"
    )
    print(f"Автоматов в галерее: {len(items)}")
    print(f"Галерея: {index}")
    return 0


# ---------------------------------------------------------------------------
# Разбор конфига
# ---------------------------------------------------------------------------
def _resolve_config(
    args: argparse.Namespace, explicit_seed: int | None
) -> tuple[dict[str, Any], int]:
    """Вернуть конфиг и seed, при котором он был получен.

    Если конфиг взят из сохранённого прогона, берётся его seed — иначе повтор
    не воспроизведёт траекторию.
    """

    if args.config:
        raw = args.config
        if raw.lstrip().startswith("{"):
            return json.loads(raw), 0 if explicit_seed is None else explicit_seed
        path = Path(raw)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".jsonl":
                lines = [line for line in text.splitlines() if line.strip()]
                if not lines:
                    raise SystemExit(f"Пустой файл: {path}")
                record = json.loads(lines[-1])
                seed = record.get("seed", 0) if explicit_seed is None else explicit_seed
                return record["config"], int(seed)
            record = json.loads(text)
            if "config" in record:
                seed = record.get("seed", 0) if explicit_seed is None else explicit_seed
                return record["config"], int(seed)
            return record, 0 if explicit_seed is None else explicit_seed
        try:
            return json.loads(raw), 0 if explicit_seed is None else explicit_seed
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Не удалось разобрать конфиг: {exc}") from exc
    if args.digest:
        if not args.run_dir:
            raise SystemExit("--digest требует --run-dir")
        record = find_record(args.run_dir, args.digest)
        if record is None:
            raise SystemExit(f"Конфиг с дайджестом {args.digest!r} не найден в {args.run_dir}")
        seed = record.get("seed", 0) if explicit_seed is None else explicit_seed
        return record["config"], int(seed)
    raise SystemExit("Укажите --config <json|файл> или --digest вместе с --run-dir")


# ---------------------------------------------------------------------------
# Парсер
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="casweep",
        description="Перебор клеточных автоматов: пространство параметров, прогоны, визуализация.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_experiment(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--experiment",
            default=DEFAULT_EXPERIMENT,
            help=f"эксперимент в виде module:factory (по умолчанию {DEFAULT_EXPERIMENT})",
        )

    modules = subparsers.add_parser("modules", help="список модулей и их требований")
    add_experiment(modules)
    modules.set_defaults(func=cmd_modules)

    space = subparsers.add_parser("space", help="описание осей и числа конфигураций")
    add_experiment(space)
    space.add_argument("--limit", type=int, default=200_000, help="предел точного подсчёта")
    space.add_argument(
        "--samples", type=int, default=64, help="сколько путей усреднять в оценке"
    )
    space.add_argument("--feasible", action="store_true", help="считать только совместимые")
    space.add_argument("--preview", type=int, default=3, help="сколько конфигов показать")
    space.add_argument(
        "--order",
        choices=SWEEP_STRATEGIES,
        default="product",
        help="в каком порядке показывать preview (product, shuffle, diagonal)",
    )
    space.set_defaults(func=cmd_space)

    run_parser = subparsers.add_parser("run", help="запустить перебор")
    add_experiment(run_parser)
    run_parser.add_argument("--limit", type=int, default=None, help="сколько тестов исполнить")
    run_parser.add_argument("--skip", type=int, default=0, help="пропустить первые N конфигов")
    run_parser.add_argument("--seed", type=int, default=0, help="базовый seed инициализации")
    run_parser.add_argument("--workers", type=int, default=1, help="число потоков/процессов")
    run_parser.add_argument(
        "--mode",
        choices=("auto", "serial", "thread", "process"),
        default="auto",
        help="режим исполнения",
    )
    run_parser.add_argument(
        "--out", default=None, help="каталог для successes.jsonl (+ trials.jsonl с --save-all)"
    )
    run_parser.add_argument("--resume", action="store_true", help="пропускать уже посчитанные")
    run_parser.add_argument("--random", action="store_true", help="случайные конфиги вместо обхода")
    run_parser.add_argument(
        "--order",
        choices=SWEEP_STRATEGIES,
        default="product",
        help="порядок обхода: декартов, перемешанный или диагональный срез (не используется с --random)",
    )
    run_parser.add_argument(
        "--stride", type=int, default=1, help="брать каждый N-й конфиг (разреженная выборка)"
    )
    run_parser.add_argument("--keep-state", action="store_true", help="хранить финальную сетку")
    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="бюджет итераций на один тест (стадии урезаются)",
    )
    run_parser.add_argument(
        "--until",
        type=int,
        default=None,
        help="остановиться, набрав N прошедших тестов",
    )
    run_parser.add_argument(
        "--save-all",
        action="store_true",
        help="писать полный trials.jsonl (по умолчанию только успешные)",
    )
    run_parser.add_argument(
        "--stop-file",
        default=None,
        help="остановиться, как только этот файл появится (touch из другого терминала)",
    )
    run_parser.add_argument(
        "--all-configs",
        action="store_true",
        help="перебирать и несовместимые конфиги (с --save-all они пишутся как infeasible)",
    )
    run_parser.add_argument("--quiet", action="store_true", help="без построчного вывода")
    run_parser.set_defaults(func=cmd_run)

    def add_live_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--seed", type=int, default=0, help="seed инициализации")
        sub.add_argument("--steps", type=int, default=None, help="бюджет итераций")
        sub.add_argument("--fps", type=float, default=8.0, help="кадров в секунду")
        sub.add_argument("--width", type=int, default=0, help="ширина в символах (0 — авто)")
        sub.add_argument("--height", type=int, default=0, help="высота в строках (0 — авто)")
        sub.add_argument(
            "--scheme", choices=("age", "hue"), default="age", help="палитра состояний"
        )
        sub.add_argument("--no-color", action="store_true", help="ASCII без ANSI-цвета")
        sub.add_argument("--no-clear", action="store_true", help="не перерисовывать экран")

    view_parser = subparsers.add_parser(
        "view", help="живой просмотр одного автомата в терминале (без файлов)"
    )
    add_experiment(view_parser)
    add_live_args(view_parser)
    view_parser.add_argument("--config", default=None, help="JSON-конфиг строкой или файлом")
    view_parser.add_argument("--digest", default=None, help="дайджест конфига")
    view_parser.add_argument("--run-dir", default=None, help="каталог прогона с successes.jsonl")
    view_parser.add_argument("--index", type=int, default=0, help="какой успех взять")
    view_parser.add_argument("--first", action="store_true", help="первый совместимый конфиг")
    view_parser.set_defaults(func=cmd_view)

    compare_parser = subparsers.add_parser(
        "compare", help="покадровое сравнение нескольких автоматов рядом (без файлов)"
    )
    add_experiment(compare_parser)
    add_live_args(compare_parser)
    compare_parser.add_argument("--run-dir", default=None, help="каталог прогона с успехами")
    compare_parser.add_argument("--limit", type=int, default=6, help="сколько успехов взять")
    compare_parser.add_argument(
        "--random", type=int, default=0, help="взять N случайных совместимых конфигов"
    )
    compare_parser.set_defaults(func=cmd_compare)

    render_parsers: dict[str, argparse.ArgumentParser] = {}
    for name, func, help_text in (
        ("replay", cmd_replay, "пересчитать один конфиг и показать трассу"),
        ("render", cmd_render, "сохранить кадры, ленту, лист и HTML-вьюер автомата"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        add_experiment(sub)
        sub.add_argument("--config", default=None, help="JSON-конфиг строкой или файлом")
        sub.add_argument("--digest", default=None, help="дайджест конфига из прогона")
        sub.add_argument("--run-dir", default=None, help="каталог прогона для поиска по дайджесту")
        sub.add_argument(
            "--seed", type=int, default=None, help="seed (по умолчанию — из сохранённого теста)"
        )
        sub.add_argument("--every", type=int, default=1, help="рисовать каждый N-й шаг")
        sub.add_argument("--frames", type=int, default=None, help="сколько кадров показать")
        sub.add_argument("--scale", type=int, default=8, help="масштаб пикселя")
        sub.add_argument(
            "--out", default=None, help="каталог кадров (по умолчанию рядом с экспериментом)"
        )
        sub.add_argument("--trace", action="store_true", help="печатать ASCII-кадры (replay)")
        sub.set_defaults(func=func)
        render_parsers[name] = sub

    render = render_parsers["render"]
    render.add_argument(
        "--scheme",
        choices=("auto", "age", "hue"),
        default="auto",
        help="палитра: auto — из визуализатора модуля, age/hue — принудительно",
    )
    render.add_argument("--grid-lines", action="store_true", help="тонкие границы клеток")
    render.add_argument(
        "--sheet-columns", type=int, default=5, help="колонок в контактном листе"
    )
    render.add_argument("--no-sheet", action="store_true", help="не писать sheet.png")
    render.add_argument("--no-viewer", action="store_true", help="не писать viewer.html")

    gallery = subparsers.add_parser(
        "gallery", help="HTML-галерея успешных автоматов сохранённого прогона"
    )
    add_experiment(gallery)
    gallery.add_argument("--run-dir", required=True, help="каталог прогона с successes.jsonl")
    gallery.add_argument(
        "--out", default=None, help="каталог галереи (по умолчанию рядом с экспериментом)"
    )
    gallery.add_argument("--limit", type=int, default=12, help="сколько автоматов показать")
    gallery.add_argument("--every", type=int, default=2, help="сохранять каждый N-й кадр")
    gallery.add_argument("--scale", type=int, default=4, help="масштаб пикселя")
    gallery.add_argument(
        "--scheme",
        choices=("auto", "age", "hue"),
        default="auto",
        help="палитра: auto — из визуализатора модуля",
    )
    gallery.add_argument("--grid-lines", action="store_true", help="тонкие границы клеток")
    gallery.set_defaults(func=cmd_gallery)

    browse = subparsers.add_parser(
        "browse", help="браузерная галерея: много автоматов с анимацией"
    )
    add_experiment(browse)
    browse.add_argument("--run-dir", default=None, help="каталог прогона с successes.jsonl")
    browse.add_argument(
        "--random", type=int, default=0, help="взять N случайных совместимых конфигов"
    )
    browse.add_argument("--limit", type=int, default=9, help="сколько успехов взять")
    browse.add_argument("--seed", type=int, default=0, help="базовый seed для --random")
    browse.add_argument("--steps", type=int, default=24, help="бюджет итераций на автомат")
    browse.add_argument("--every", type=int, default=2, help="брать каждый N-й кадр")
    browse.add_argument("--scale", type=int, default=4, help="масштаб пикселя")
    browse.add_argument("--max-frames", type=int, default=32, help="кадров на карточку")
    browse.add_argument("--fps", type=float, default=4.0, help="скорость анимации")
    browse.add_argument(
        "--scheme",
        choices=("auto", "age", "hue"),
        default="auto",
        help="палитра: auto — из визуализатора модуля",
    )
    browse.add_argument(
        "--out", default=None, help="каталог галереи (по умолчанию рядом с экспериментом)"
    )
    browse.add_argument(
        "--no-open", dest="open_browser", action="store_false", help="не открывать браузер"
    )
    browse.set_defaults(func=cmd_browse, open_browser=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nПрервано пользователем", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # Вывод перенаправили в head/less и закрыли трубу — это не ошибка.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
