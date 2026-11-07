"""Геометрия автомата «симбиоз».

Автомат устроен вокруг двух опорных состояний с крупными радиусами (20–30
клеток) и двух зависимых состояний, которые считают опору в круге радиуса 5.
Круг радиуса 5 — это 80 клеток, поэтому «радиус» здесь не равен числу клеток.

Вспомогательные функции этого файла считают только геометрию: смещения,
маски крупных радиусов и номера секторов круга. Пороговые правила живут в
:mod:`sweep.experiments.symbiosis.kernel`.
"""

from __future__ import annotations

import math
from typing import Sequence

from ...neighbors.geometry import disc_offsets

#: Идентификаторы состояний: ровно семь, ноль — пустота.
VOID = 0
ANCHOR = 1
MATRIX = 2
GRAZER = 3
WEAVER = 4
SPORE = 5
PULSE = 6

STATES = 7

#: Радиусы опорных состояний и зависимой логики (в клетках).
ANCHOR_RADIUS = 20
MATRIX_RADIUS = 30
LOGIC_RADIUS = 5
#: Опоры рождаются только на расстоянии от уже стоящих опор: колония ветвится.
SPACING_RADIUS = 2

#: Способы счёта секторов круга зависимой логики.
SECTOR_MODES = ("spread", "ring", "gaps", "parity")
#: Считать ли опору всем кругом или только внешним кольцом.
WINDOW_MODES = ("disc", "ring")

METRICS = ("euclidean", "chebyshev")


def _distance(dy: int, dx: int, metric: str) -> float:
    if metric == "chebyshev":
        return float(max(abs(dy), abs(dx)))
    return math.hypot(dy, dx)


def support_offsets(matrix_radius: int, metric: str) -> tuple[tuple[int, int], ...]:
    """Смещения внешнего круга: по ним считаются оба опорных состояния."""

    return disc_offsets(int(matrix_radius), metric, 2)


def support_masks(
    offsets: Sequence[tuple[int, int]],
    anchor_radius: int,
    matrix_radius: int,
    mode: str,
    metric: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Маски «считать как ANCHOR» и «считать как MATRIX» для каждого смещения.

    В режиме ``ring`` крупный радиус считается только внешним кольцом
    (от 0.72 радиуса до радиуса), в режиме ``disc`` — всем кругом.
    """

    if mode not in WINDOW_MODES:
        raise ValueError(f"неизвестный режим окна опоры: {mode!r}")
    anchor_cut = 0.72 * float(anchor_radius)
    matrix_cut = 0.72 * float(matrix_radius)
    in_anchor = []
    in_matrix = []
    for dy, dx in offsets:
        distance = _distance(dy, dx, metric)
        if mode == "ring":
            in_anchor.append(int(anchor_cut < distance <= anchor_radius))
            in_matrix.append(int(matrix_cut < distance <= matrix_radius))
        else:
            in_anchor.append(int(distance <= anchor_radius))
            in_matrix.append(int(distance <= matrix_radius))
    return tuple(in_anchor), tuple(in_matrix)


def spacing_offsets(metric: str = "chebyshev") -> tuple[tuple[int, int], ...]:
    """Короткая окрестность запрета: опора не рождается вплотную к опоре."""

    return disc_offsets(SPACING_RADIUS, metric, 2)


def logic_offsets(mode: str, metric: str) -> tuple[tuple[int, int], ...]:
    """Смещения зависимой логики: круг радиуса 5 или его внешнее кольцо."""

    if mode not in WINDOW_MODES:
        raise ValueError(f"неизвестный режим окна логики: {mode!r}")
    offsets = disc_offsets(LOGIC_RADIUS, metric, 2)
    if mode == "disc":
        return offsets
    cut = 0.72 * float(LOGIC_RADIUS)
    return tuple(
        (dy, dx)
        for dy, dx in offsets
        if cut < _distance(dy, dx, metric) <= LOGIC_RADIUS
    )


def sector_ids(
    offsets: Sequence[tuple[int, int]],
    sectors: int,
    offset: float = 0.5,
) -> tuple[int, ...]:
    """Номер сектора для каждого смещения: 2 половины, 4 четверти, 6 или 8 долей.

    Угол ``offset`` (в долях сектора) поворачивает границы секторов, чтобы
    они не проходили ровно по осям сетки: иначе половины и четверти получаются
    вырожденными на квадратной решётке.
    """

    if sectors < 2:
        raise ValueError("секторов должно быть не меньше двух")
    turn = float(offset) / float(sectors) * 2.0 * math.pi
    ids = []
    for dy, dx in offsets:
        angle = (math.atan2(dy, dx) + turn) % (2.0 * math.pi)
        index = int(angle / (2.0 * math.pi) * sectors)
        ids.append(min(sectors - 1, max(0, index)))
    return tuple(ids)


def sector_counts(ids: Sequence[int], sectors: int) -> tuple[int, ...]:
    """Сколько смещений попало в каждый сектор (для проверок и тестов)."""

    counts = [0] * int(sectors)
    for index in ids:
        counts[int(index)] += 1
    return tuple(counts)


__all__ = [
    "ANCHOR",
    "ANCHOR_RADIUS",
    "GRAZER",
    "LOGIC_RADIUS",
    "MATRIX",
    "MATRIX_RADIUS",
    "METRICS",
    "PULSE",
    "SECTOR_MODES",
    "SPORE",
    "STATES",
    "VOID",
    "WINDOW_MODES",
    "WEAVER",
    "logic_offsets",
    "sector_counts",
    "sector_ids",
    "support_masks",
    "SPACING_RADIUS",
    "spacing_offsets",
    "support_offsets",
]
