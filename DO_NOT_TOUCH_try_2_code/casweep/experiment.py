"""Эксперимент: пространство параметров + выбранные модули + стадии прогона."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .assembly import Assembly, Assembled
from .fixed import Fixed
from .handles import Handle
from .module import ROLE_ORDER, FilterBinding, Module, Role
from .space import Selection, Space

_NOT_CACHED = object()


@dataclass(frozen=True)
class FilterUse:
    """Использование фильтра в стадии."""

    binding: FilterBinding
    mode: str = "each"  # "each" — каждую итерацию, "end" — после стадии

    def __post_init__(self) -> None:
        if self.mode not in ("each", "end"):
            raise ValueError(f"Неизвестный режим фильтра: {self.mode!r}")


@dataclass(frozen=True)
class Phase:
    """Стадия прогона: сколько итераций и какие фильтры проверять."""

    name: str
    iterations: Handle
    filters: tuple[FilterUse, ...] = ()


class Experiment:
    """Описание перебора и сборки автомата.

    Пример::

        exp = Experiment("life")
        exp.select(Role.DATA, [DiscreteGrid2D()])
        exp.select(Role.RULE, [ThresholdRule()])
        exp.phase("observe", exp.space.range("stage.observe", 10, 30, 10))
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        #: Пакет, из которого загружен эксперимент (ставит ``cli.load_experiment``).
        self.source: str | None = None
        self.space = Space()
        self.phases: list[Phase] = []
        self._selections: dict[Role, Selection] = {}
        self._filters: dict[str, FilterBinding] = {}
        self._feasibility_cache: dict[tuple[Any, ...], str | None] = {}
        self._assembly: Assembly | None = None

    # --- описание --------------------------------------------------------
    def select(self, role: Role, candidates: Sequence[Module]) -> Selection:
        """Добавить выбор модуля для роли (порядок вызовов задаёт порядок осей)."""

        selection = self.space.select(role, candidates)
        self._selections[role] = selection
        self._assembly = None
        return selection

    def filter(self, module: Module, name: str | None = None) -> FilterBinding:
        """Подключить фильтр. Параметры фильтра объявляются один раз."""

        if module.role is not Role.FILTER:
            raise ValueError(f"Модуль {module.id!r} не является фильтром")
        name = name or module.id.split(".")[-1]
        if name in self._filters:
            raise ValueError(f"Фильтр {name!r} уже подключён")
        binding = FilterBinding(module, name)
        # Фильтр объявляет свои параметры под префиксом filter.<name>.
        from .space import DeclareView, Namespace

        module.declare(Namespace(self.space, binding.prefix), DeclareView(self.space))
        self._filters[name] = binding
        self._assembly = None
        return binding

    def phase(
        self,
        name: str,
        iterations: Handle,
        filters: Sequence[tuple[FilterBinding, str]] = (),
    ) -> Phase:
        """Добавить стадию. ``filters`` — пары (фильтр, режим "each"/"end")."""

        if name in {phase.name for phase in self.phases}:
            raise ValueError(f"Стадия {name!r} уже существует")
        if not isinstance(iterations, Handle):
            raise TypeError("Число итераций стадии должно быть Handle из space")
        uses = tuple(FilterUse(binding, mode) for binding, mode in filters)
        phase = Phase(name, iterations, uses)
        self.phases.append(phase)
        return phase

    @property
    def selections(self) -> Mapping[Role, Selection]:
        return dict(self._selections)

    @property
    def home_dir(self) -> Path | None:
        """Каталог пакета эксперимента — рядом с его ``__init__.py``.

        Используется визуализацией, чтобы артефакты лежали вместе с кодом
        автомата, а не в произвольном ``runs/``.
        """

        if not self.source:
            return None
        try:
            module = importlib.import_module(self.source)
        except ImportError:
            return None
        file = getattr(module, "__file__", None)
        return None if not file else Path(file).resolve().parent

    @property
    def filters(self) -> Mapping[str, FilterBinding]:
        return dict(self._filters)

    # --- сборка ----------------------------------------------------------
    @property
    def assembly(self) -> Assembly:
        if self._assembly is None:
            self._assembly = Assembly(self._selections, self._filters.values())
        return self._assembly

    def fixed(self, binding: Mapping[str, Any]) -> Fixed:
        return Fixed.of(binding)

    def feasible(self, fixed: Fixed) -> str | None:
        """Причина несовместимости или ``None``, если конфиг исполним.

        Проверка кэшируется по подписи сборки (модули + их возможности),
        поэтому обход сотен тысяч конфигов не пересчитывает отчёт каждый раз.
        """

        return self._reason(self.assembly.signature(fixed), fixed)

    def _reason(self, signature: tuple[Any, ...], fixed: Fixed) -> str | None:
        cached = self._feasibility_cache.get(signature, _NOT_CACHED)
        if cached is not _NOT_CACHED:
            return cached  # type: ignore[return-value]
        report = self.assembly.report(fixed)
        if report.ok:
            reason: str | None = None
        else:
            problems = report.problems
            reason = "; ".join(
                f"{line.module_id}: {line.requirement.describe()}" for line in problems[:3]
            ) or "несовместимая конфигурация"
        if len(self._feasibility_cache) > 50_000:
            self._feasibility_cache.clear()
        self._feasibility_cache[signature] = reason
        return reason

    def _prune(self, binding: Mapping[str, Any]) -> bool:
        """Отсечь поддерево, если несовместимость уже доказана."""

        signature = self.assembly.partial_signature(binding)
        if signature is None:
            return False
        return self._reason(signature, Fixed.of(binding)) is not None

    def assemble(self, fixed: Fixed) -> Assembled:
        return self.assembly.build(fixed)

    def preflight(self, fixed: Fixed) -> None:
        """Собрать конфиг и прогреть компиляцию (для запуска параллельного пула)."""

        self.assemble(fixed).precompile()

    # --- перебор ---------------------------------------------------------
    def iter_configs(
        self,
        *,
        feasible: bool = True,
        strategy: str = "product",
        seed: int = 0,
        stride: int = 1,
    ) -> Iterator[Fixed]:
        """Поток конфигов; по умолчанию только совместимые.

        ``strategy``/``seed``/``stride`` передаются в :meth:`Space.enumerate`:
        можно идти декартовым порядком (``product``), перемешать значения осей
        (``shuffle``) или взять разреженный диагональный срез (``diagonal``).
        Совместимость проверяется до выдачи конфига.
        """

        if not feasible:
            for binding in self.space.enumerate(
                strategy=strategy, seed=seed, stride=stride
            ):
                yield Fixed.of(binding)
            return
        for binding in self.space.enumerate(
            prune=self._prune,
            accept=lambda item: self.feasible(Fixed.of(item)) is None,
            strategy=strategy,
            seed=seed,
            stride=stride,
        ):
            yield Fixed.of(binding)

    def count(
        self,
        limit: int | None = None,
        *,
        feasible: bool = False,
        strategy: str = "product",
    ) -> int:
        """Точное число конфигов или размер выборки для данной стратегии."""

        if not feasible:
            return self.space.count(limit=limit, strategy=strategy)
        return self.space.count(
            limit=limit,
            accept=lambda binding: self.feasible(Fixed.of(binding)) is None,
            prune=self._prune,
            strategy=strategy,
        )

    def estimate(self, samples: int = 64, seed: int = 0) -> int:
        return self.space.estimate(samples=samples, seed=seed)

    # --- описание --------------------------------------------------------
    def describe(self) -> str:
        lines = [f"Эксперимент: {self.name}"]
        if self.description:
            lines.append(self.description)
        lines.append("")
        lines.append("Модули:")
        for role in ROLE_ORDER:
            selection = self._selections.get(role)
            if selection is None:
                lines.append(f"  {role.value:<12} (не выбран)")
                continue
            chosen = ", ".join(module.id for module in selection.candidates)
            lines.append(f"  {role.value:<12} {chosen}")
        if self._filters:
            lines.append(
                "  фильтры      " + ", ".join(binding.module.id for binding in self._filters.values())
            )
        lines.append("")
        lines.append("Оси перебора:")
        lines.extend("  " + line for line in self.space.describe())
        lines.append("")
        lines.append("Стадии:")
        for phase in self.phases:
            filters = ", ".join(
                f"{use.binding.name}[{use.mode}]" for use in phase.filters
            ) or "без фильтров"
            lines.append(f"  {phase.name:<10} итерации={phase.iterations.key}  {filters}")
        return "\n".join(lines)
