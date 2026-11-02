"""Пространство параметров: план перебора, который ещё не исполняется.

Ключевая идея
-------------
Эксперимент описывается декларативно: оси перебора (значения, диапазоны,
выбор модуля), зависимости между ними, ограничения и ветвления. Из этого дерева
лениво порождаются конфигурации (словари ``ключ -> значение``), каждая из
которых затем исполняется отдельным тестом.

Почему не «магическое» исполнение обычного кода с ``.get()``
------------------------------------------------------------
Императивный код вида ``height = Range(width.get(), 128, 32)`` в plan-фазе
требует символьной арифметики и символьных ветвлений — это хрупко и не даёт
надёжного числа вариаций. Здесь вместо этого:

* ``ns.range("height", width, 129, 32)`` — нижняя граница является ссылкой на
  другую ось и вычисляется при разворачивании;
* ``space.select(role, candidates)`` — выбор модуля; параметры каждого модуля
  объявляются только в его ветке;
* ``space.constrain(...)`` — отсечение нежелательных комбинаций ещё в момент
  генерации.

Сам «код автомата» пишется как модуль, который получает уже зафиксированный
конфиг (:class:`casweep.fixed.Fixed`) — см. ``casweep.module``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .fixed import Fixed
from .handles import Handle, referenced_keys, resolve

Binding = Mapping[str, Any]

#: Поддерживаемые способы обхода пространства.
SWEEP_STRATEGIES = ("product", "shuffle", "diagonal")


class SweepError(Exception):
    """Базовая ошибка движка перебора."""


class DeclarationError(SweepError):
    """Некорректно объявлено пространство параметров."""


class DomainError(SweepError):
    """Ось не может выдать значения при текущей привязке."""


class IncompatibleModules(SweepError):
    """Выбранные модули несовместимы."""


# ---------------------------------------------------------------------------
# Оси
# ---------------------------------------------------------------------------
class Axis:
    """Ось перебора: одно измерение пространства параметров."""

    kind = "axis"

    def __init__(self, key: str) -> None:
        if not key:
            raise DeclarationError("У оси не может быть пустого ключа")
        self.key = key

    def values(self, binding: Binding) -> tuple[Any, ...]:
        raise NotImplementedError

    def describe(self, probe: Binding) -> str:
        return f"{self.key:<30} {self.kind}"


class ValueAxis(Axis):
    """Ось с единственным значением: параметр фиксируется тестом."""

    kind = "value"

    def __init__(self, key: str, value: Any) -> None:
        super().__init__(key)
        self.value = value

    def values(self, binding: Binding) -> tuple[Any, ...]:
        return (self.value,)

    def describe(self, probe: Binding) -> str:
        return f"{self.key:<30} value {self.value!r}"


class ChoiceAxis(Axis):
    """Ось выбора из дискретного набора (например, идентификаторов модулей)."""

    kind = "choice"

    def __init__(self, key: str, values: Sequence[Any]) -> None:
        super().__init__(key)
        unique = tuple(values)
        if not unique:
            raise DeclarationError(f"Ось {key!r}: пустой список вариантов")
        if len(set(unique)) != len(unique):
            raise DeclarationError(f"Ось {key!r}: повторяющиеся варианты {unique!r}")
        self.options = unique

    def values(self, binding: Binding) -> tuple[Any, ...]:
        return self.options

    def describe(self, probe: Binding) -> str:
        return f"{self.key:<30} choose({', '.join(map(str, self.options))})"


class RangeAxis(Axis):
    """Числовой диапазон с полуоткрытым правым концом (как ``range``).

    Границы и шаг могут быть ссылками на другие оси, поэтому диапазон
    вычисляется в момент разворачивания. Это заменяет
    ``Range(width.get(), 128, 32)`` без символьной магии.

    ``inclusive=True`` добавляет верхнюю границу, если она попадает на шаг
    (для дробных шагов — с точностью до 1e-9 от шага); если не попадает,
    остаётся последняя точка сетки. Для точного числа точек используйте
    :class:`LinspaceAxis`.
    """

    kind = "range"

    def __init__(
        self, key: str, lo: Any, hi: Any, step: Any = 1, *, inclusive: bool = False
    ) -> None:
        super().__init__(key)
        self.lo, self.hi, self.step = lo, hi, step
        self.inclusive = bool(inclusive)

    def values(self, binding: Binding) -> tuple[Any, ...]:
        lo = resolve(self.lo, binding)
        hi = resolve(self.hi, binding)
        step = resolve(self.step, binding)
        if step == 0:
            raise DeclarationError(f"Ось {self.key!r}: шаг диапазона не может быть нулём")
        integral = all(isinstance(v, int) and not isinstance(v, bool) for v in (lo, hi, step))
        if integral:
            start, stop, stride = int(lo), int(hi), int(step)
            if start == stop:
                # Пустой полуоткрытый диапазон, но включительная граница —
                # это одна точка, а не ноль.
                return (start,) if self.inclusive else ()
            values = list(range(start, stop, stride))
            # Верхняя граница добавляется только если шаг действительно
            # ведёт к ней: range(1, 10, -1) пуст, и 10 в него не входит.
            if (
                self.inclusive
                and (stop - start) * stride > 0
                and (stop - start) % stride == 0
            ):
                values.append(stop)
            return tuple(values)

        values = []
        value = float(lo)
        limit = float(hi)
        step_value = float(step)
        epsilon = abs(step_value) * 1e-9
        while (step_value > 0 and value < limit - epsilon) or (
            step_value < 0 and value > limit + epsilon
        ):
            values.append(round(value, 12))
            value += step_value
        if self.inclusive and abs(value - limit) <= epsilon:
            values.append(round(limit, 12))
        return tuple(values)

    def describe(self, probe: Binding) -> str:
        bound = "[]" if self.inclusive else "[)"
        text = (
            f"{self.key:<30} range{bound[0]}{self.lo!r} .. {self.hi!r}{bound[1]} "
            f"шаг {self.step!r}"
        )
        if any(referenced_keys(item) for item in (self.lo, self.hi, self.step)):
            text += " (зависит)"
        try:
            values = self.values(probe)
            return f"{text}  -> {len(values)} шт. на пробе"
        except (KeyError, DomainError):
            return text


class LinspaceAxis(Axis):
    """Ровно ``count`` точек от ``lo`` до ``hi`` включительно.

    В отличие от :class:`RangeAxis`, верхняя граница всегда входит в набор,
    а шаг вычисляется сам. Для целых границ значения округляются и дубликаты
    убираются (например, при узком диапазоне и большом ``count``).
    """

    kind = "linspace"

    def __init__(
        self, key: str, lo: Any, hi: Any, count: Any, *, integer: bool = False
    ) -> None:
        super().__init__(key)
        self.lo, self.hi, self.count = lo, hi, count
        self.integer = bool(integer)

    def values(self, binding: Binding) -> tuple[Any, ...]:
        lo = resolve(self.lo, binding)
        hi = resolve(self.hi, binding)
        count = int(resolve(self.count, binding))
        if count < 1:
            raise DeclarationError(f"Ось {self.key!r}: count должен быть >= 1")
        if count == 1:
            return (lo,)
        points: list[Any] = []
        for index in range(count):
            value = lo + (hi - lo) * index / (count - 1)
            points.append(int(round(value)) if self.integer else round(float(value), 12))
        # При integer=True соседние точки могут совпасть; не раздуваем ось.
        return tuple(dict.fromkeys(points))

    def describe(self, probe: Binding) -> str:
        text = f"{self.key:<30} linspace({self.lo!r} .. {self.hi!r}, {self.count!r})"
        try:
            return f"{text}  -> {len(self.values(probe))} шт."
        except (KeyError, DomainError):
            return text


class LogspaceAxis(Axis):
    """Геометрическая сетка от ``lo`` до ``hi`` включительно.

    Полезна для плотностей, радиусов и вероятностей, где шаг по порядку
    величины осмысленнее линейного. Границы должны быть положительными.
    """

    kind = "logspace"

    def __init__(self, key: str, lo: Any, hi: Any, count: Any) -> None:
        super().__init__(key)
        self.lo, self.hi, self.count = lo, hi, count

    def values(self, binding: Binding) -> tuple[Any, ...]:
        import math

        lo = float(resolve(self.lo, binding))
        hi = float(resolve(self.hi, binding))
        count = int(resolve(self.count, binding))
        if count < 1:
            raise DeclarationError(f"Ось {self.key!r}: count должен быть >= 1")
        if lo <= 0 or hi <= 0:
            # В зависимости от ветки границы могут стать неположительными;
            # это вырожденный домен, а не ошибка всего эксперимента.
            raise DomainError(f"Ось {self.key!r}: logspace требует положительных границ")
        if count == 1:
            return (lo,)
        ratio = math.log(hi / lo)
        points: list[Any] = []
        for index in range(count):
            points.append(round(math.exp(math.log(lo) + ratio * index / (count - 1)), 12))
        return tuple(dict.fromkeys(points))

    def describe(self, probe: Binding) -> str:
        text = f"{self.key:<30} logspace({self.lo!r} .. {self.hi!r}, {self.count!r})"
        try:
            return f"{text}  -> {len(self.values(probe))} шт."
        except (KeyError, DomainError):
            return text


class DerivedAxis(Axis):
    """Вычисляемое значение: не перебирается, а выводится из привязки."""

    kind = "derived"

    def __init__(self, key: str, fn: Callable[[Binding], Any]) -> None:
        super().__init__(key)
        self.fn = fn

    def values(self, binding: Binding) -> tuple[Any, ...]:
        return (self.fn(binding),)

    def describe(self, probe: Binding) -> str:
        # Описание не должно падать из-за неполной привязки: это диагностика,
        # а не исполнение эксперимента.
        try:
            value = self.fn(probe)
        except Exception:  # noqa: BLE001 - любая ошибка означает «не вычислить»
            return f"{self.key:<30} derived"
        return f"{self.key:<30} derived = {value!r}"


class ConditionalAxis(Axis):
    """Ветвление: набор осей зависит от уже выбранного значения."""

    kind = "conditional"

    def __init__(self, key: str, selector_key: str, branches: Mapping[str, tuple[Axis, ...]]) -> None:
        super().__init__(key)
        self.selector = selector_key
        self.branches = {name: tuple(axes) for name, axes in branches.items()}

    def branch(self, binding: Binding) -> tuple[Axis, ...]:
        try:
            selector = binding[self.selector]
        except KeyError as exc:  # pragma: no cover - защита от неверного порядка осей
            raise DeclarationError(
                f"Ветвление {self.key!r} объявлено раньше выбора {self.selector!r}"
            ) from exc
        try:
            return self.branches[selector]
        except KeyError as exc:
            raise DomainError(
                f"Ветвление {self.key!r}: нет ветки для {selector!r}"
            ) from exc

    def values(self, binding: Binding) -> tuple[Any, ...]:  # pragma: no cover - не используется
        raise DeclarationError("ConditionalAxis разворачивается через branch(), а не values()")


# ---------------------------------------------------------------------------
# Ограничения
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Constraint:
    """Предикат над подмножеством ключей; срабатывает, как только они заданы."""

    name: str
    keys: tuple[str, ...]
    predicate: Callable[[Binding], bool]

    def applies(self, binding: Binding) -> bool:
        return all(key in binding for key in self.keys)

    def holds(self, binding: Binding) -> bool:
        return bool(self.predicate({key: binding[key] for key in self.keys}))

    def describe(self) -> str:
        return f"{self.name}: {', '.join(self.keys)}"


# ---------------------------------------------------------------------------
# Пространство
# ---------------------------------------------------------------------------
class Space:
    """Дерево осей и ограничений, из которого порождаются конфигурации."""

    def __init__(self) -> None:
        self._axes: list[Axis] = []
        self._registered: dict[str, Axis] = {}
        self._constraints: list[Constraint] = []
        self._selections: dict[Any, "Selection"] = {}

    # --- объявление -----------------------------------------------------
    def value(self, key: str, value: Any) -> Handle:
        return self._add_axis(ValueAxis(key, value))

    def range(
        self, key: str, lo: Any, hi: Any, step: Any = 1, *, inclusive: bool = False
    ) -> Handle:
        return self._add_axis(RangeAxis(key, lo, hi, step, inclusive=inclusive))

    def linspace(
        self, key: str, lo: Any, hi: Any, count: Any, *, integer: bool = False
    ) -> Handle:
        """Ровно ``count`` точек, включая обе границы.

        По умолчанию точки дробные (``0, 0.25, 0.5, ...``); ``integer=True``
        округляет их и убирает дубликаты.
        """

        return self._add_axis(LinspaceAxis(key, lo, hi, count, integer=integer))

    def logspace(self, key: str, lo: Any, hi: Any, count: Any) -> Handle:
        """Геометрическая сетка от ``lo`` до ``hi`` включительно."""

        return self._add_axis(LogspaceAxis(key, lo, hi, count))

    def choose(self, key: str, values: Sequence[Any]) -> Handle:
        return self._add_axis(ChoiceAxis(key, values))

    def derive(self, key: str, fn: Callable[[Binding], Any]) -> Handle:
        return self._add_axis(DerivedAxis(key, fn))

    def constrain(
        self,
        predicate: Callable[[Mapping[str, Any]], bool],
        keys: Iterable[str],
        name: str | None = None,
    ) -> None:
        """Оставить только комбинации, где ``predicate`` истинен.

        ``predicate`` получает словарь из перечисленных ключей. Ограничение
        проверяется сразу, как только все его ключи получили значения, поэтому
        отсекает ветки дерева, а не только готовые конфигурации.
        """

        keys = tuple(keys)
        if not keys:
            raise DeclarationError("Ограничение без ключей не имеет смысла")
        unknown = [key for key in keys if key not in self._registered]
        if unknown:
            raise DeclarationError(f"Ограничение ссылается на неизвестные ключи: {unknown}")
        constraint = Constraint(name or f"constraint#{len(self._constraints)}", keys, predicate)
        self._constraints.append(constraint)

    def select(self, role: Any, candidates: Sequence[Any]) -> "Selection":
        """Добавить выбор модуля для роли; параметры модулей — в их ветках."""

        if role in self._selections:
            raise DeclarationError(f"Роль {role!r} уже имеет выбор модуля")
        selection = Selection(self, role, tuple(candidates))
        self._selections[role] = selection
        return selection

    def selection(self, role: Any) -> "Selection":
        try:
            return self._selections[role]
        except KeyError as exc:
            raise DeclarationError(f"Роль {role!r} ещё не выбрана") from exc

    @property
    def selections(self) -> Mapping[Any, "Selection"]:
        return dict(self._selections)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        return tuple(self._constraints)

    @property
    def axes(self) -> tuple[Axis, ...]:
        return tuple(self._axes)

    @property
    def dimension(self) -> int:
        """Число осей верхнего уровня (ветвление модуля — одна ось)."""

        return len(self._axes)

    def axis(self, key: str) -> Axis:
        """Ось по ключу, включая параметры веток выбора модулей."""

        try:
            return self._registered[key]
        except KeyError as exc:
            raise DeclarationError(f"Неизвестная ось {key!r}") from exc

    def keys(self) -> tuple[str, ...]:
        """Все зарегистрированные ключи, включая оси во всех ветках."""

        return tuple(self._registered)

    def values(self, key: str, binding: Binding | None = None) -> tuple[Any, ...]:
        """Значения оси при частичной привязке (по умолчанию — пустой)."""

        return self.axis(key).values(binding or {})

    # --- внутреннее -----------------------------------------------------
    def _add_axis(self, axis: Axis) -> Handle:
        self._append(axis)
        self._register_shared(axis)
        return Handle(axis.key)

    def _append(self, axis: Axis) -> None:
        if axis.key in {a.key for a in self._axes}:
            raise DeclarationError(f"Ключ оси {axis.key!r} уже используется")
        self._axes.append(axis)

    def _register_shared(self, axis: Axis) -> None:
        existing = self._registered.get(axis.key)
        if existing is not None and existing is not axis:
            # Ключи внутри взаимоисключающих веток могут совпадать намеренно.
            return
        self._registered[axis.key] = axis

    def _add_conditional(
        self, key: str, selector_key: str, branches: Mapping[str, tuple[Axis, ...]]
    ) -> ConditionalAxis:
        axis = ConditionalAxis(key, selector_key, branches)
        self._append(axis)
        for sub_axes in branches.values():
            for sub_axis in sub_axes:
                self._register_shared(sub_axis)
        return axis

    def _add_constraint(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)

    # --- генерация ------------------------------------------------------
    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.enumerate()

    def enumerate(
        self,
        accept: Callable[[Binding], bool] | None = None,
        prune: Callable[[Binding], bool] | None = None,
        *,
        strategy: str = "product",
        seed: int = 0,
        stride: int = 1,
    ) -> Iterator[dict[str, Any]]:
        """Ленивый обход конфигураций.

        * ``prune(binding)`` вызывается на каждой частичной привязке и, если
          возвращает ``True``, отсекает всё поддерево (например, когда выбор
          модулей уже доказал несовместимость);
        * ``accept(binding)`` вызывается на полной привязке и фильтрует листья;
        * ``strategy="product"`` — обычный декартов порядок; ``"shuffle"`` —
          значения каждой оси перемешиваются с ``seed`` (тот же набор
          конфигов, другой порядок);
        * ``strategy="diagonal"`` — разреженный обход: на каждом раунде каждая
          ось берёт очередное значение, а не все комбинации. Вместо
          произведения размеров получается не больше самого длинного домена
          (полезно для больших пространств, где нужен покрытый срез, а не
          полный перебор);
        * ``stride`` берёт каждый N-й лист уже после ``accept``; вместе с
          ``shuffle`` это дешёвый способ получить разреженную выборку.
        """

        if strategy not in SWEEP_STRATEGIES:
            raise DeclarationError(
                f"Неизвестная стратегия обхода: {strategy!r} "
                f"(доступны: {', '.join(SWEEP_STRATEGIES)})"
            )
        if stride < 1:
            raise DeclarationError("stride должен быть >= 1")
        if strategy == "diagonal":
            stream = self._diagonal(accept, prune, seed)
        else:
            rng = random.Random(seed) if strategy == "shuffle" else None
            stream = self._walk(tuple(self._axes), {}, accept, prune, rng)
        if stride > 1:
            stream = islice(stream, 0, None, stride)
        yield from stream

    def _walk(
        self,
        pending: tuple[Axis, ...],
        binding: dict[str, Any],
        accept: Callable[[Binding], bool] | None = None,
        prune: Callable[[Binding], bool] | None = None,
        rng: random.Random | None = None,
    ) -> Iterator[dict[str, Any]]:
        if prune is not None and prune(binding):
            return
        if not pending:
            if accept is None or accept(binding):
                yield dict(binding)
            return
        axis, rest = pending[0], pending[1:]
        if isinstance(axis, ConditionalAxis):
            try:
                branch = axis.branch(binding)
            except DomainError:
                return
            yield from self._walk(branch + rest, binding, accept, prune, rng)
            return
        try:
            values = axis.values(binding)
        except DomainError:
            return
        if rng is not None:
            shuffled = list(values)
            rng.shuffle(shuffled)
            values = shuffled
        for value in values:
            extended = dict(binding)
            extended[axis.key] = value
            if self._constraints_hold(extended, axis.key):
                yield from self._walk(rest, extended, accept, prune, rng)

    def _diagonal_rounds(self) -> int:
        """Сколько раундов покрывает самый длинный домен на первом пути.

        Длины зависимых осей вычисляются по первым значениям предыдущих осей;
        для диапазонов, которые сужаются с ростом параметра, это даёт верхнюю
        оценку. Ограничения и ветвления могут только уменьшить число
        выданных конфигов.
        """

        probe: dict[str, Any] = {}
        pending = list(self._axes)
        rounds = 1
        while pending:
            axis = pending.pop(0)
            if isinstance(axis, ConditionalAxis):
                try:
                    pending = list(axis.branch(probe)) + pending
                except DomainError:
                    break
                continue
            try:
                values = axis.values(probe)
            except (DomainError, KeyError):
                break
            if not values:
                break
            rounds = max(rounds, len(values))
            probe[axis.key] = values[0]
        return max(1, rounds)

    def _walk_diagonal(
        self,
        pending: tuple[Axis, ...],
        binding: dict[str, Any],
        step: int,
        accept: Callable[[Binding], bool] | None,
        prune: Callable[[Binding], bool] | None,
        seed: int,
    ) -> Iterator[dict[str, Any]]:
        if prune is not None and prune(binding):
            return
        if not pending:
            if accept is None or accept(binding):
                yield dict(binding)
            return
        axis, rest = pending[0], pending[1:]
        if isinstance(axis, ConditionalAxis):
            try:
                branch = axis.branch(binding)
            except DomainError:
                return
            yield from self._walk_diagonal(branch + rest, binding, step, accept, prune, seed)
            return
        try:
            values = axis.values(binding)
        except DomainError:
            return
        if not values:
            return
        # Начинаем с «диагонального» значения, но если ограничение его
        # запрещает, берём ближайшее допустимое по кругу. Иначе парные оси
        # вроде r1/r2 с условием r2 > r1 всегда шли бы вместе и давали ноль
        # конфигов.
        start = (step + seed) % len(values)
        for offset in range(len(values)):
            extended = dict(binding)
            extended[axis.key] = values[(start + offset) % len(values)]
            if self._constraints_hold(extended, axis.key):
                yield from self._walk_diagonal(rest, extended, step, accept, prune, seed)
                return

    def _diagonal(
        self,
        accept: Callable[[Binding], bool] | None,
        prune: Callable[[Binding], bool] | None,
        seed: int,
    ) -> Iterator[dict[str, Any]]:
        seen: set[str] = set()
        for step in range(self._diagonal_rounds()):
            for binding in self._walk_diagonal(
                tuple(self._axes), {}, step, accept, prune, seed
            ):
                key = Fixed.of(binding).digest()
                if key in seen:
                    continue
                seen.add(key)
                yield binding

    def _constraints_hold(self, binding: Binding, just_set: str) -> bool:
        for constraint in self._constraints:
            if just_set not in constraint.keys or not constraint.applies(binding):
                continue
            if not constraint.holds(binding):
                return False
        return True

    def first(self) -> dict[str, Any] | None:
        for binding in self:
            return binding
        return None

    def count(
        self,
        limit: int | None = None,
        accept: Callable[[Binding], bool] | None = None,
        prune: Callable[[Binding], bool] | None = None,
        *,
        strategy: str = "product",
    ) -> int:
        """Число конфигураций (с учётом ограничений и ``prune``).

        ``limit`` останавливает подсчёт досрочно. Для независимых «хвостов»
        дерева используется произведение размеров доменов, поэтому подсчёт не
        перебирает все листья (кроме ``prune`` и ``strategy="diagonal"``, где
        нужен обход). ``shuffle`` даёт тот же набор, что и ``product``.
        """

        if strategy == "diagonal":
            total = 0
            for _ in self.enumerate(accept=accept, prune=prune, strategy=strategy):
                total += 1
                if limit is not None and total >= limit:
                    break
            return total
        if strategy not in SWEEP_STRATEGIES:
            raise DeclarationError(
                f"Неизвестная стратегия обхода: {strategy!r} "
                f"(доступны: {', '.join(SWEEP_STRATEGIES)})"
            )
        return self._count(tuple(self._axes), {}, limit, accept, prune)

    def _count(
        self,
        pending: tuple[Axis, ...],
        binding: dict[str, Any],
        limit: int | None,
        accept: Callable[[Binding], bool] | None,
        prune: Callable[[Binding], bool] | None = None,
    ) -> int:
        if prune is not None and prune(binding):
            return 0
        if not pending:
            return 1 if accept is None or accept(binding) else 0
        if accept is None and prune is None:
            product = self._static_product(pending, binding)
            if product is not None:
                if limit is not None:
                    return min(product, max(0, limit))
                return product
        axis, rest = pending[0], pending[1:]
        if isinstance(axis, ConditionalAxis):
            try:
                branch = axis.branch(binding)
            except DomainError:
                return 0
            return self._count(branch + rest, binding, limit, accept, prune)
        try:
            values = axis.values(binding)
        except DomainError:
            return 0
        total = 0
        for value in values:
            extended = dict(binding)
            extended[axis.key] = value
            if not self._constraints_hold(extended, axis.key):
                continue
            remaining = None if limit is None else max(0, limit - total)
            total += self._count(rest, extended, remaining, accept, prune)
            if limit is not None and total >= limit:
                return total
        return total

    def _static_product(self, pending: tuple[Axis, ...], binding: Binding) -> int | None:
        """Произведение доменов, если все они вычислимы на текущей привязке."""

        pending_keys = {axis.key for axis in pending}
        for constraint in self._constraints:
            if any(key in pending_keys for key in constraint.keys):
                return None
        total = 1
        for axis in pending:
            if isinstance(axis, ConditionalAxis):
                return None
            try:
                values = axis.values(binding)
            except (DomainError, KeyError):
                return None
            if not values:
                return 0
            total *= len(values)
        return total

    def estimate(self, samples: int = 64, seed: int = 0) -> int:
        """Статистическая оценка числа конфигов (без полного обхода).

        Делает несколько случайных путей по дереву и усредняет произведение
        размеров доменов вдоль каждого пути. Так учитываются зависимые
        диапазоны и ветки модулей, чего старая оценка «по первым значениям»
        не умела. Ограничения при выборе значений учитываются, но их влияние
        на «хвосты» не подсчитывается точно, поэтому оценка сверху.
        """

        if samples < 1:
            raise DeclarationError("samples должен быть >= 1")
        rng = random.Random(seed)
        totals: list[int] = []
        attempts = 0
        while len(totals) < samples and attempts < samples * 8:
            attempts += 1
            binding: dict[str, Any] = {}
            pending = list(self._axes)
            product = 1
            ok = True
            while pending:
                axis = pending.pop(0)
                if isinstance(axis, ConditionalAxis):
                    try:
                        pending = list(axis.branch(binding)) + pending
                    except DomainError:
                        ok = False
                        break
                    continue
                try:
                    values = axis.values(binding)
                except (DomainError, KeyError):
                    ok = False
                    break
                if not values:
                    ok = False
                    break
                product *= len(values)
                if self._constraints:
                    start = rng.randrange(len(values))
                    chosen: dict[str, Any] | None = None
                    for offset in range(len(values)):
                        value = values[(start + offset) % len(values)]
                        extended = dict(binding)
                        extended[axis.key] = value
                        if self._constraints_hold(extended, axis.key):
                            chosen = extended
                            break
                    if chosen is None:
                        ok = False
                        break
                    binding = chosen
                else:
                    binding[axis.key] = values[rng.randrange(len(values))]
            if ok:
                totals.append(product)
        if not totals:
            return 0
        return int(round(sum(totals) / len(totals)))

    def random_binding(self, rng: random.Random | None = None, attempts: int = 128) -> dict[str, Any]:
        """Случайная валидная конфигурация (обходом дерева со взвешенным выбором)."""

        rng = rng or random.Random()
        for _ in range(attempts):
            binding: dict[str, Any] = {}
            pending = list(self._axes)
            ok = True
            while pending:
                axis = pending.pop(0)
                if isinstance(axis, ConditionalAxis):
                    try:
                        pending = list(axis.branch(binding)) + pending
                    except DomainError:
                        ok = False
                        break
                    continue
                try:
                    values = axis.values(binding)
                except DomainError:
                    ok = False
                    break
                if not values:
                    ok = False
                    break
                # Пробуем значения этой оси, пока ограничение не пропустит
                # одно из них: иначе при широких диапазонах и трёх
                # упорядоченных порогах почти каждая попытка уходила бы в
                # перезапуск всего дерева.
                order = list(values)
                rng.shuffle(order)
                chosen: dict[str, Any] | None = None
                for value in order:
                    extended = dict(binding)
                    extended[axis.key] = value
                    if self._constraints_hold(extended, axis.key):
                        chosen = extended
                        break
                if chosen is None:
                    ok = False
                    break
                binding = chosen
            if ok:
                return binding
        raise SweepError("Не удалось получить случайную валидную конфигурацию")

    def random_bindings(
        self, count: int, seed: int = 0, *, unique: bool = False
    ) -> Iterator[dict[str, Any]]:
        """Поток случайных конфигов; ``unique`` отбрасывает повторы.

        Повторы маловероятны на больших пространствах, но ``--random`` без
        ``unique`` может потратить тесты на один и тот же конфиг. Повтор
        определяется по полной привязке (канонические ключи и значения).
        """

        rng = random.Random(seed)
        seen: set[tuple[Any, ...]] = set()
        produced = 0
        attempts = 0
        limit = max(1, count) * 100
        while produced < count and attempts < limit:
            attempts += 1
            binding = self.random_binding(rng)
            if unique:
                try:
                    key = tuple(sorted(binding.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                except TypeError:
                    # Экзотические нехешируемые значения — сравниваем repr.
                    key = tuple(sorted((k, repr(v)) for k, v in binding.items()))
                    if key in seen:
                        continue
                    seen.add(key)
            produced += 1
            yield binding

    def sample(
        self, count: int, seed: int = 0, *, strategy: str = "product"
    ) -> list[dict[str, Any]]:
        """Резервуарная выборка из обхода (для ``product`` — по всему пространству)."""

        rng = random.Random(seed)
        reservoir: list[dict[str, Any]] = []
        for index, binding in enumerate(self.enumerate(strategy=strategy, seed=seed)):
            if len(reservoir) < count:
                reservoir.append(binding)
            else:
                swap = rng.randrange(index + 1)
                if swap < count:
                    reservoir[swap] = binding
        return reservoir

    # --- описание -------------------------------------------------------
    def describe(self) -> list[str]:
        lines: list[str] = []
        probe: dict[str, Any] = {}
        for axis in self._axes:
            if isinstance(axis, ConditionalAxis):
                try:
                    selected = probe.get(axis.selector)
                except Exception:  # pragma: no cover
                    selected = None
                lines.append(f"{axis.key:<30} ветвление по {axis.selector} (выбрано: {selected!r})")
                for name, sub_axes in axis.branches.items():
                    lines.append(f"  [{name}]")
                    branch_probe = dict(probe)
                    for sub_axis in sub_axes:
                        lines.append("    " + sub_axis.describe(branch_probe))
                        try:
                            values = sub_axis.values(branch_probe)
                        except (DomainError, KeyError):
                            continue
                        if values:
                            branch_probe[sub_axis.key] = values[0]
                # Описание главной ветки продолжает общий probe, но только
                # после того, как все ветки отрисованы со своими значениями.
                if selected in axis.branches:
                    for sub_axis in axis.branches[selected]:
                        try:
                            values = sub_axis.values(probe)
                        except (DomainError, KeyError):
                            continue
                        if values:
                            probe[sub_axis.key] = values[0]
                continue
            lines.append(axis.describe(probe))
            try:
                values = axis.values(probe)
            except (DomainError, KeyError):
                continue
            if values:
                probe[axis.key] = values[0]
        if self._constraints:
            lines.append("Ограничения:")
            lines.extend("  " + constraint.describe() for constraint in self._constraints)
        return lines


# ---------------------------------------------------------------------------
# Namespace: удобный способ объявлять параметры модуля
# ---------------------------------------------------------------------------
class Namespace:
    """Объявление параметров модуля под общим префиксом.

    ``ns.range("r2", ns.ref("r1") + 1, 8)`` создаёт ось ``<prefix>.r2``,
    зависящую от ``<prefix>.r1``.
    """

    def __init__(self, space: Space, prefix: str, *, shared: bool = False) -> None:
        self.space = space
        self.prefix = prefix
        self.shared = shared
        self.axes: list[Axis] = []
        self.names: set[str] = set()

    def key(self, name: str) -> str:
        return f"{self.prefix}.{name}"

    def _push(self, axis: Axis, name: str) -> Handle:
        if name in self.names:
            raise DeclarationError(f"{self.prefix}: параметр {name!r} объявлен дважды")
        self.names.add(name)
        self.axes.append(axis)
        if self.shared:
            self.space._register_shared(axis)
        else:
            self.space._add_axis(axis)
        return Handle(axis.key)

    def value(self, name: str, value: Any) -> Handle:
        return self._push(ValueAxis(self.key(name), value), name)

    def range(
        self, name: str, lo: Any, hi: Any, step: Any = 1, *, inclusive: bool = False
    ) -> Handle:
        return self._push(
            RangeAxis(self.key(name), lo, hi, step, inclusive=inclusive), name
        )

    def linspace(
        self, name: str, lo: Any, hi: Any, count: Any, *, integer: bool = False
    ) -> Handle:
        return self._push(
            LinspaceAxis(self.key(name), lo, hi, count, integer=integer), name
        )

    def logspace(self, name: str, lo: Any, hi: Any, count: Any) -> Handle:
        return self._push(LogspaceAxis(self.key(name), lo, hi, count), name)

    def choose(self, name: str, values: Sequence[Any]) -> Handle:
        return self._push(ChoiceAxis(self.key(name), values), name)

    def derive(self, name: str, fn: Callable[[Binding], Any]) -> Handle:
        """Вычисляемый параметр модуля (не перебирается, а выводится)."""

        return self._push(DerivedAxis(self.key(name), fn), name)

    def ref(self, name: str) -> Handle:
        return Handle(self.key(name))

    def constrain(
        self,
        predicate: Callable[..., bool],
        *names: str,
        name: str | None = None,
    ) -> None:
        keys = tuple(self.key(item) for item in names)
        unknown = [key for key in keys if key not in {axis.key for axis in self.axes}]
        if unknown:
            raise DeclarationError(f"{self.prefix}: ограничение ссылается на чужие ключи {unknown}")

        def wrapped(binding: Binding, predicate=predicate, names=names, keys=keys) -> bool:
            return bool(predicate(**{n: binding[k] for n, k in zip(names, keys)}))

        self.space._add_constraint(Constraint(name or f"{self.prefix}#constraint", keys, wrapped))


# ---------------------------------------------------------------------------
# Выбор модуля
# ---------------------------------------------------------------------------
class _DerivedProxy:
    """Вызов ``module.derive`` в момент разворачивания оси."""

    __slots__ = ("module", "name")

    def __init__(self, module: Any, name: str) -> None:
        self.module = module
        self.name = name

    def __call__(self, binding: Binding) -> Any:
        values = self.module.derive(Fixed.of(binding))
        try:
            return values[self.name]
        except KeyError as exc:
            raise DeclarationError(
                f"Модуль {self.module.id!r} не вычислил {self.name!r} в derive()"
            ) from exc


class Selection:
    """Выбор модуля для роли + ветки с параметрами каждого кандидата."""

    def __init__(self, space: Space, role: Any, candidates: Sequence[Any]) -> None:
        if not candidates:
            raise DeclarationError("Нельзя выбрать роль без кандидатов")
        ids = [module.id for module in candidates]
        if len(set(ids)) != len(ids):
            raise DeclarationError(f"Роль {role!r}: повторяющиеся id модулей {ids!r}")
        self.space = space
        self.role = role
        self.candidates = candidates
        self.modules = dict(zip(ids, candidates))
        self.choice = space.choose(f"{role.value}.module", ids)

        declared: dict[str, set[str]] = {}
        branches: dict[str, tuple[Axis, ...]] = {}
        for module in candidates:
            namespace = Namespace(space, role.value, shared=True)
            module.declare(namespace, DeclareView(space))
            axes: list[Axis] = list(namespace.axes)
            for derived_name in getattr(module, "derived", ()):
                axes.append(DerivedAxis(f"{role.value}#{derived_name}", _DerivedProxy(module, derived_name)))
            branches[module.id] = tuple(axes)
            declared[module.id] = set(namespace.names)

        self.declared_params = declared
        for derived_name in getattr(candidates[0], "derived", ()):
            missing = [m.id for m in candidates if derived_name not in getattr(m, "derived", ())]
            if missing:
                raise DeclarationError(
                    f"Роль {role!r}: производное значение {derived_name!r} объявлено не всеми "
                    f"кандидатами (нет у {missing!r})"
                )
        self.branch_axis = space._add_conditional(f"{role.value}::params", self.choice.key, branches)

    # --- доступ к параметрам -------------------------------------------
    def param(self, name: str) -> Handle:
        """Ссылка на параметр выбранного модуля (имя общее для роли)."""

        return Handle(f"{self.role.value}.{name}")

    def derived(self, name: str) -> Handle:
        """Ссылка на производное значение выбранного модуля."""

        return Handle(f"{self.role.value}#{name}")

    def selected_module(self, fixed: Fixed) -> Any:
        return self.modules[fixed[self.choice]]

    def __repr__(self) -> str:
        return f"Selection({self.role!r}, candidates={list(self.modules)!r})"


class DeclareView:
    """Что модуль может видеть при объявлении своих параметров."""

    def __init__(self, space: Space) -> None:
        self.space = space

    def peer(self, role: Any) -> Selection:
        return self.space.selection(role)
