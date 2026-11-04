"""Фильтры плавности: доля мёртвых и структура последнего состояния.

Оба фильтра смотрят на кадры подряд и ловят именно «частичное гниение»:
картинка меняется понемногу, мёртвые клетки не появляются/исчезают массово,
а последнее (самое твёрдое) состояние присутствует, но не захватывает всё.
"""

from __future__ import annotations

import numpy as np

from casweep.fixed import Fixed
from casweep.interfaces import CapabilityKind, Requirement, at_least
from casweep.module import BuildView, Module, Role
from casweep.space import DeclareView, Namespace
from casweep.trial import FilterVerdict


class ZeroShareFilter(Module):
    """Доля ``zero_state`` не должна прыгать больше чем на ``max_shift`` за кадр."""

    id = "filter.zero_balance"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(self, *, max_shift: float = 0.08) -> None:
        self.max_shift = float(max_shift)

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        namespace.value("max_shift", self.max_shift)

    def build(self, fixed: Fixed, view: BuildView) -> "ZeroShareRuntime":
        return ZeroShareRuntime(max_shift=float(view.value("max_shift")))


class ZeroShareRuntime:
    """Хранит долю мёртвых на предыдущем кадре (живёт один тест)."""

    __slots__ = ("max_shift", "previous")

    def __init__(self, max_shift: float) -> None:
        self.max_shift = max_shift
        self.previous: float | None = None

    def check(self, grid) -> FilterVerdict:
        array = np.asarray(grid.array)
        share = float(np.count_nonzero(array == grid.zero_state)) / float(array.size)
        if self.previous is None:
            self.previous = share
            return FilterVerdict(
                True, metrics={"zero_share": round(share, 6), "zero_shift": 0.0}
            )

        shift = abs(share - self.previous)
        self.previous = share
        metrics = {
            "zero_share": round(share, 6),
            "zero_shift": round(shift, 6),
        }
        if shift > self.max_shift:
            return FilterVerdict(
                False,
                f"доля мёртвых прыгнула на {shift:.3f} > {self.max_shift:.3f} за кадр",
                metrics,
            )
        return FilterVerdict(True, metrics=metrics)


class RotStructureFilter(Module):
    """Доля клеток последнего состояния должна быть в полосе.

    Слишком мало — гниения не видно; слишком много — «твёрдое» состояние
    захватило сетку и гниёт всё сразу.
    """

    id = "filter.rot_structure"
    role = Role.FILTER
    requires = (
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.ZERO_STATE),
    )

    def __init__(self, *, min_last_share: float = 0.01, max_last_share: float = 0.5) -> None:
        self.criteria = {
            "min_last_share": float(min_last_share),
            "max_last_share": float(max_last_share),
        }

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        for name, value in self.criteria.items():
            namespace.value(name, value)

    def build(self, fixed: Fixed, view: BuildView) -> "RotStructureRuntime":
        return RotStructureRuntime(
            **{name: float(view.value(name)) for name in self.criteria}
        )


class RotStructureRuntime:
    __slots__ = ("min_last_share", "max_last_share")

    def __init__(self, min_last_share: float, max_last_share: float) -> None:
        self.min_last_share = min_last_share
        self.max_last_share = max_last_share

    def check(self, grid) -> FilterVerdict:
        array = np.asarray(grid.array)
        last_state = int(grid.states) - 1
        count = int(np.count_nonzero(array == last_state))
        share = float(count) / float(array.size)
        metrics = {
            "last_state": last_state,
            "last_count": count,
            "last_share": round(share, 6),
        }
        if share < self.min_last_share:
            return FilterVerdict(
                False,
                f"нет гниения: последнее состояние {share:.3f} < {self.min_last_share:.3f}",
                metrics,
            )
        if share > self.max_last_share:
            return FilterVerdict(
                False,
                f"затвердело целиком: последнее состояние {share:.3f} > {self.max_last_share:.3f}",
                metrics,
            )
        return FilterVerdict(True, metrics=metrics)
