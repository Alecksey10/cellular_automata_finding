"""Дерево осей перебора.

Пространство — это набор селекторов модулей и осей параметров. Обход
порождает привязки (``dict[str, value]``) лениво, поэтому огромное
пространство не материализуется в память.

Ключи:

* ``select.<role>`` — выбранный модуль (его ``id``);
* ``<module>.<param>`` — параметр модуля;
* произвольные ключи экспериментальных осей (``phase.observe`` и т.п.).

Оси модуля объявляются внутри его ветки, поэтому параметр ``radius`` есть
только у выбранной окрестности. Зависимые диапазоны
(``born_max = range(born_min, max_count + 1)``) разрешаются в момент обхода,
когда нужные ключи уже привязаны.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .domains import Domain, choice_domain, range_domain
from .errors import UnboundParameter
from .handles import Handle, Ref, as_handle, resolve
from .module import Module, Role


@dataclass(frozen=True)
class Axis:
    key: str
    domain: Domain
    owner: str | None = None
    options: tuple[Module, ...] | None = None


@dataclass(frozen=True)
class Constraint:
    name: str
    keys: frozenset[str]
    check: Callable[[Mapping[str, Any]], bool]

    def ready(self, bound: Mapping[str, Any]) -> bool:
        return self.keys <= bound.keys()

    def allows(self, bound: Mapping[str, Any]) -> bool:
        return self.check(bound)


class Namespace:
    """Объявление осей внутри модуля или эксперимента.

    ``prefix`` отделяет параметры модуля: ``Grid2D`` с id ``grid2d`` и
    ``ns.range("states", ...)`` даёт ключ ``grid2d.states``.
    """

    def __init__(self, space: "Space", prefix: str = "") -> None:
        self.space = space
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}.{key}" if self.prefix else key

    def key(self, key: str) -> str:
        """Полный ключ оси (с префиксом модуля)."""

        return self._key(key)

    def value(self, key: str, value: Any) -> Ref:
        full = self._key(key)
        self.space.add_axis(Axis(full, choice_domain([value]), owner=self.prefix or None))
        return Ref(full)

    def range(self, key: str, lo: Any, hi: Any, step: Any = 1) -> Ref:
        full = self._key(key)
        self.space.add_axis(Axis(full, range_domain(lo, hi, step), owner=self.prefix or None))
        return Ref(full)

    def choice(self, key: str, options: Iterable[Any]) -> Ref:
        full = self._key(key)
        self.space.add_axis(Axis(full, choice_domain(options), owner=self.prefix or None))
        return Ref(full)

    def handle(self, key: str) -> Ref:
        return Ref(self._key(key))


class Space:
    """Корень дерева перебора: селекторы, подоси модулей и корневые оси."""

    def __init__(self) -> None:
        self.selectors: list[Axis] = []
        self.sub_axes: dict[str, list[Axis]] = {}
        self.axes: list[Axis] = []
        self.constraints: list[Constraint] = []
        self._by_key: dict[str, Axis] = {}
        self._current_prefix: str | None = None
        self._selector_by_role: dict[Role, Axis] = {}

    # ------------------------------------------------------------------
    # построение
    # ------------------------------------------------------------------
    def add_namespace(self, prefix: str = "") -> Namespace:
        return Namespace(self, prefix)

    def add_selector(self, role: Role, modules: Sequence[Module]) -> Axis:
        options = tuple(modules)
        key = f"select.{role.value}"
        axis = Axis(key, choice_domain([module.id for module in options]), options=options)
        # повторный select заменяет прежний выбор
        old = self._selector_by_role.get(role)
        if old is not None:
            self.selectors.remove(old)
            for module in old.options or ():
                self.sub_axes.pop(module.id, None)
                self._by_key.pop(old.key, None)
        self.selectors.append(axis)
        self._selector_by_role[role] = axis
        self._by_key[key] = axis
        return axis

    def add_axis(self, axis: Axis) -> None:
        if axis.key in self._by_key:
            return  # повторное объявление того же ключа игнорируем
        self._by_key[axis.key] = axis
        if axis.owner:
            self.sub_axes.setdefault(axis.owner, []).append(axis)
        else:
            self.axes.append(axis)

    def constrain(
        self,
        check: Callable[[Mapping[str, Any]], bool],
        keys: Iterable[str] | None = None,
        name: str = "",
    ) -> None:
        key_set = frozenset(keys or ())
        if not key_set:
            key_set = frozenset(self.all_keys())
        self.constraints.append(Constraint(name or f"constraint_{len(self.constraints)}", key_set, check))

    # ------------------------------------------------------------------
    # обход
    # ------------------------------------------------------------------
    def _constraints_allow(self, bound: Mapping[str, Any], key: str, candidate: Any) -> bool:
        trial = dict(bound)
        trial[key] = candidate
        for constraint in self.constraints:
            if constraint.ready(trial) and not constraint.allows(trial):
                return False
        return True

    def _leaf_ok(self, bound: Mapping[str, Any], feasible: bool) -> bool:
        if not feasible:
            return True
        for constraint in self.constraints:
            if constraint.ready(bound) and not constraint.allows(bound):
                return False
        return True

    def _bindings(self, feasible: bool) -> Iterator[dict[str, Any]]:
        def walk(bound: dict[str, Any], pending: list[Axis]) -> Iterator[dict[str, Any]]:
            if not pending:
                if self._leaf_ok(bound, feasible):
                    yield dict(bound)
                return
            axis = pending[0]
            rest = pending[1:]
            if axis.options is not None:
                for module in axis.options:
                    nxt = dict(bound)
                    nxt[axis.key] = module.id
                    queue_rest = list(rest)
                    # подоси выбранного модуля сразу после селектора
                    queue_rest[0:0] = list(self.sub_axes.get(module.id, ()))
                    yield from walk(nxt, queue_rest)
                return
            try:
                values = axis.domain.values(bound)
            except UnboundParameter:
                yield from walk(bound, rest)
                return
            for value in values:
                if not self._constraints_allow(bound, axis.key, value):
                    continue
                nxt = dict(bound)
                nxt[axis.key] = value
                yield from walk(nxt, rest)

        # Селекторы модулей + их подоси + корневые оси.
        # Проходим все селекторы, вставляя подоси выбранного модуля.
        def walk_all(bound: dict[str, Any], selectors: list[Axis], ordered: list[Axis]):
            if not selectors:
                yield from walk(bound, list(ordered) + list(self.axes))
                return
            axis = selectors[0]
            rest = selectors[1:]
            assert axis.options is not None
            for module in axis.options:
                nxt = dict(bound)
                nxt[axis.key] = module.id
                subs = list(self.sub_axes.get(module.id, ()))
                yield from walk_all(nxt, rest, ordered + subs)

        yield from walk_all({}, list(self.selectors), [])

    def iter_bindings(
        self,
        strategy: str = "product",
        feasible: bool = True,
        max_bindings: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        if strategy in ("product", "diagonal"):
            seen = 0
            for bound in self._bindings(feasible):
                yield bound
                seen += 1
                if max_bindings is not None and seen >= max_bindings:
                    return
            return
        if strategy == "random":
            raise ValueError("для случайного перебора используйте random_bindings")
        raise ValueError(f"неизвестная стратегия: {strategy!r}")

    def random_bindings(
        self,
        limit: int | None = None,
        *,
        seed: int = 0,
        feasible: bool = True,
        max_attempts: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        rng = _random.Random(seed)
        produced = 0
        attempts = 0
        limit_attempts = max_attempts or (limit * 200 if limit else 1_000_000)
        while limit is None or produced < limit:
            attempts += 1
            if attempts > limit_attempts:
                return
            bound = self._random_binding(rng, feasible)
            if bound is None:
                continue
            produced += 1
            yield bound

    def _random_binding(self, rng: _random.Random, feasible: bool) -> dict[str, Any] | None:
        bound: dict[str, Any] = {}
        queue: list[Axis] = list(self.selectors)
        index = 0
        while index < len(queue):
            axis = queue[index]
            index += 1
            if axis.options is not None:
                module = rng.choice(axis.options)
                bound[axis.key] = module.id
                queue[index:index] = list(self.sub_axes.get(module.id, ()))
                continue
            try:
                values = axis.domain.values(bound)
            except UnboundParameter:
                continue
            allowed = [v for v in values if self._constraints_allow(bound, axis.key, v)]
            if not allowed:
                return None
            bound[axis.key] = rng.choice(allowed)
        for axis in self.axes:
            if axis.key in bound:
                continue
            try:
                values = axis.domain.values(bound)
            except UnboundParameter:
                continue
            allowed = [v for v in values if self._constraints_allow(bound, axis.key, v)]
            if not allowed:
                return None
            bound[axis.key] = rng.choice(allowed)
        return bound if self._leaf_ok(bound, feasible) else None

    def count(self, *, feasible: bool = True, limit: int | None = None) -> int:
        total = 0
        for _ in self._bindings(feasible):
            total += 1
            if limit is not None and total >= limit:
                break
        return total

    # ------------------------------------------------------------------
    # интроспекция
    # ------------------------------------------------------------------
    def all_keys(self) -> list[str]:
        keys = [axis.key for axis in self.selectors]
        for axes in self.sub_axes.values():
            keys.extend(axis.key for axis in axes)
        keys.extend(axis.key for axis in self.axes)
        return keys

    def axis(self, key: str) -> Axis | None:
        return self._by_key.get(key)

    def describe(self) -> list[str]:
        """Человекочитаемая карта осей для CLI."""

        lines: list[str] = []
        first: dict[str, Any] = {}
        for axis in self.selectors:
            options = ", ".join(module.id for module in axis.options or ())
            lines.append(f"{axis.key:<28} выбор: {options}")
            if axis.options:
                first[axis.key] = axis.options[0].id
        for module_id, axes in self.sub_axes.items():
            for axis in axes:
                try:
                    text = axis.domain.describe(first)
                except (UnboundParameter, KeyError):
                    text = axis.domain.describe(None)
                lines.append(f"  {axis.key:<26} {text}")
                first[axis.key] = _first_value(axis, first)
        for axis in self.axes:
            try:
                text = axis.domain.describe(first)
            except (UnboundParameter, KeyError):
                text = axis.domain.describe(None)
            lines.append(f"{axis.key:<28} {text}")
            first[axis.key] = _first_value(axis, first)
        return lines


def _first_value(axis: Axis, fixed: Mapping[str, Any]) -> Any:
    if axis.options is not None:
        return axis.options[0].id if axis.options else None
    try:
        values = axis.domain.values(fixed)
    except (UnboundParameter, KeyError):
        return None
    return values[0] if values else None


def resolve_handle(value: Any, bound: Mapping[str, Any], *, what: str = "значение") -> Any:
    return resolve(as_handle(value), bound)


__all__ = [
    "Axis",
    "Constraint",
    "Namespace",
    "Space",
    "resolve_handle",
]
