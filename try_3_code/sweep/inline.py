"""Правило, написанное прямо в эксперименте.

Эксперимент может не выбирать готовый ``RuleModule``, а передать свою
numba-функцию итерации через :meth:`sweep.experiment.Experiment.iterate`.
Тогда цикл автомата живёт в файле эксперимента, а не в общем каталоге правил.

Контракт ядра:

* ``step(src, dst, *args)`` — обычная итерация;
* ``step(src, dst, iteration, seed, *args)`` — если ``with_control=True``
  (нужно правилам со случайностью или зависимостью от номера итерации);
* ``src``/``dst`` — ``np.int32`` массивы ``(height, width)``;
* ``args`` — кортеж, который возвращает ``bind(fixed, view)``; только массивы
  и скаляры, совместимые с numba.

Никаких Python-объектов внутри ``step`` быть не должно: функция передаётся в
numba как есть.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .interfaces import Requirement
from .module import BuildView, DeclareView, RuleModule, RuleRuntime
from .space import Namespace

Step = Callable[..., None]
Bind = Callable[[Mapping[str, Any], BuildView], tuple[Any, ...]]
Declare = Callable[[Namespace, DeclareView], None]


class InlineRuntime(RuleRuntime):
    def __init__(self, step: Step, args: tuple[Any, ...], with_control: bool) -> None:
        self._step = step
        self._args = args
        self._with_control = with_control

    def step(self, src: np.ndarray, dst: np.ndarray, iteration: int, seed: int) -> None:
        if self._with_control:
            self._step(src, dst, iteration, seed, *self._args)
        else:
            self._step(src, dst, *self._args)

    def precompile(self) -> None:
        src = np.zeros((2, 2), dtype=np.int32)
        dst = np.zeros((2, 2), dtype=np.int32)
        self.step(src, dst, 0, 0)


class InlineRule(RuleModule):
    """Адаптер: превращает функцию эксперимента в обычный модуль правила."""

    def __init__(
        self,
        step: Step,
        *,
        module_id: str,
        label: str = "",
        requires: Sequence[Requirement] = (),
        declare: Declare | None = None,
        bind: Bind | None = None,
        with_control: bool = False,
    ) -> None:
        if not callable(step):
            raise TypeError("шаг итерации должен быть вызываемой функцией")
        self.id = module_id
        self.label = label or module_id
        self.requires = tuple(requires)
        self._step = step
        self._declare = declare
        self._bind = bind
        self._with_control = with_control

    def declare(self, ns: Namespace, view: DeclareView) -> None:
        if self._declare is not None:
            self._declare(ns, view)

    def build(self, fixed: Mapping[str, Any], view: BuildView) -> RuleRuntime:
        args = tuple(self._bind(fixed, view)) if self._bind is not None else ()
        return InlineRuntime(self._step, args, self._with_control)


__all__ = ["InlineRule", "InlineRuntime"]
