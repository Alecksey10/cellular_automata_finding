"""Формальное правило эксперимента ``omega5``.

Правило намеренно описано данными, а не только кодом ядра:

* ``MECHANISMS`` — способы вычислить один счётчик соседей в окне 5×5;
* ``CATEGORIES`` — готовые интервалы «рождение / выживание / смерть»;
* ``TRICKS`` — около двадцати нестандартных реакций состояний.

Конфиг прогона хранит лишь строковые/числовые коды. Из них функция
:func:`rule_descriptor` собирает полный JSON-паспорт автомата, а
:func:`serialize_rule` — его каноническую строку. Этого паспорта вместе с
``seed`` достаточно, чтобы восстановить и нарисовать автомат.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

WINDOW_RADIUS = 2
WINDOW_SIZE = 2 * WINDOW_RADIUS + 1
WINDOW_CELLS = WINDOW_SIZE * WINDOW_SIZE
WINDOW_NEIGHBORS = WINDOW_CELLS - 1
STATES = 8

# Все смещения окна 5×5, кроме центра. Порядок не важен: ядро получает
# ``group``-массив, согласованный с конкретной окрестностью.
WINDOW_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (dy, dx)
    for dy in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1)
    for dx in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1)
    if not (dy == 0 and dx == 0)
)


def _sector(dy: int, dx: int) -> int:
    """Номер одного из 8 направлений (0 — восток, далее по часовой стрелке)."""

    angle = math.atan2(dy, dx)
    return int(round(angle / (math.pi / 4))) % 8


def _quadrant(dy: int, dx: int) -> int:
    """NE, NW, SE, SW; оси относятся к положительной стороне."""

    return (0 if dy >= 0 else 2) + (0 if dx >= 0 else 1)


def _row(dy: int, dx: int) -> int:
    return dy + WINDOW_RADIUS


def _col(dy: int, dx: int) -> int:
    return dx + WINDOW_RADIUS


def _group_ids(offsets: Sequence[tuple[int, int]], assign) -> list[int]:
    return [int(assign(dy, dx)) for dy, dx in offsets]


def group_ids(mechanism: str, offsets: Iterable[tuple[int, int]]) -> list[int]:
    """Сопоставить каждому смещению номер группы для механизма.

    ``-1`` означает «клетка не участвует в счёте» (например, для кольца).
    """

    pairs = tuple(offsets)
    if mechanism in ("full",):
        return _group_ids(pairs, lambda dy, dx: 0)
    if mechanism == "inner":
        return _group_ids(pairs, lambda dy, dx: 0 if max(abs(dy), abs(dx)) == 1 else -1)
    if mechanism == "outer":
        return _group_ids(pairs, lambda dy, dx: 0 if max(abs(dy), abs(dx)) == 2 else -1)
    if mechanism in ("quarters_max", "quarters_min", "quarters_opposite_min"):
        return _group_ids(pairs, _quadrant)
    if mechanism in ("sectors_max", "sectors_min", "sectors_opposite_min"):
        return _group_ids(pairs, _sector)
    if mechanism == "axis":
        return _group_ids(pairs, lambda dy, dx: 0 if dy == 0 or dx == 0 else -1)
    if mechanism == "diagonal":
        return _group_ids(pairs, lambda dy, dx: 0 if abs(dy) == abs(dx) else -1)
    if mechanism in ("chess_max", "chess_min"):
        return _group_ids(pairs, lambda dy, dx: (dy + dx) & 1)
    if mechanism in ("rows_max", "rows_min"):
        return _group_ids(pairs, _row)
    if mechanism in ("cols_max", "cols_min"):
        return _group_ids(pairs, _col)
    if mechanism == "corners":
        return _group_ids(
            pairs,
            lambda dy, dx: 0 if abs(dy) == WINDOW_RADIUS and abs(dx) == WINDOW_RADIUS else -1,
        )
    if mechanism == "edge_mid":
        return _group_ids(
            pairs,
            lambda dy, dx: 0
            if (abs(dy) == WINDOW_RADIUS and dx == 0) or (dy == 0 and abs(dx) == WINDOW_RADIUS)
            else -1,
        )
    # distinct_states / dominant_state / minority_state считаются по состояниям.
    return [-1 for _ in pairs]


def _group_sizes(mechanism: str) -> list[int]:
    groups = group_ids(mechanism, WINDOW_OFFSETS)
    count = max(groups) + 1 if groups else 0
    sizes = [0] * count
    for group in groups:
        if group >= 0:
            sizes[group] += 1
    return sizes


def _max_of_sizes(mechanism: str) -> int:
    sizes = _group_sizes(mechanism)
    return max(sizes) if sizes else 0


def _min_of_sizes(mechanism: str) -> int:
    sizes = _group_sizes(mechanism)
    return min(sizes) if sizes else 0


def _opposite_pair_max(mechanism: str) -> int:
    """Верхняя граница минимума по парам (все клетки живы)."""

    sizes = _group_sizes(mechanism)
    if mechanism == "quarters_opposite_min":
        # NE+SW и NW+SE
        pairs = [sizes[0] + sizes[3], sizes[1] + sizes[2]]
    else:
        pairs = [sizes[i] + sizes[i + 4] for i in range(4)]
    return min(pairs) if pairs else 0


@dataclass(frozen=True)
class Mechanism:
    id: str
    code: int
    label: str
    max_count: int

    def describe(self) -> str:
        return f"{self.label} (счётчик 0..{self.max_count})"


# Коды синхронизированы с ``kernel.py``. Не меняйте числа без правки ядра.
_MECHANISM_DEFS = [
    ("full", "сумма всей маски 5×5"),
    ("inner", "внутреннее кольцо 3×3"),
    ("outer", "внешнее кольцо 5×5"),
    ("quarters_max", "четверти: максимум"),
    ("quarters_min", "четверти: минимум"),
    ("quarters_opposite_min", "четверти: минимум по противоположным парам"),
    ("sectors_max", "8 направлений: максимум"),
    ("sectors_min", "8 направлений: минимум"),
    ("sectors_opposite_min", "8 направлений: минимум по противоположным парам"),
    ("axis", "крест: строка + столбец"),
    ("diagonal", "две диагонали"),
    ("chess_max", "шахматная маска: максимум"),
    ("chess_min", "шахматная маска: минимум"),
    ("rows_max", "пять строк: максимум"),
    ("rows_min", "пять строк: минимум"),
    ("cols_max", "пять столбцов: максимум"),
    ("cols_min", "пять столбцов: минимум"),
    ("corners", "четыре угла"),
    ("edge_mid", "четыре середины сторон"),
    ("distinct_states", "число разных состояний"),
    ("dominant_state", "самое частое состояние"),
    ("minority_state", "самое редкое живое состояние"),
]


def _mechanism_max(key: str) -> int:
    if key == "full":
        return WINDOW_NEIGHBORS
    if key == "inner":
        return _max_of_sizes("inner")
    if key == "outer":
        return _max_of_sizes("outer")
    if key == "distinct_states":
        return STATES - 1
    if key in ("dominant_state", "minority_state"):
        return WINDOW_NEIGHBORS
    if key in ("axis", "diagonal", "corners", "edge_mid"):
        return _max_of_sizes(key)
    if key.endswith("_opposite_min"):
        return _opposite_pair_max(key)
    if key.endswith("_max"):
        return _max_of_sizes(key)
    if key.endswith("_min"):
        return _min_of_sizes(key)
    return WINDOW_NEIGHBORS


MECHANISMS: "OrderedDict[str, Mechanism]" = OrderedDict(
    (key, Mechanism(key, code, label, _mechanism_max(key)))
    for code, (key, label) in enumerate(_MECHANISM_DEFS)
)


def mechanism_max(mechanism: str) -> int:
    return MECHANISMS[mechanism].max_count


@dataclass(frozen=True)
class Category:
    id: str
    label: str


# Порядок вариантов задаёт порядок полного перебора; ``none`` выключает
# соответствующее условие. Интервалы считаются от текущего механизма.
CATEGORIES: "OrderedDict[str, Category]" = OrderedDict(
    (key, Category(key, label))
    for key, label in [
        ("none", "нет условия"),
        ("any", "любое число ≥ 1"),
        ("empty", "ровно 0"),
        ("sparse", "1–2"),
        ("classic", "ровно 3"),
        ("wide", "2–4"),
        ("loose", "1–5"),
        ("balanced", "2–6"),
        ("dense", "4–12"),
        ("crowded", "8–20"),
        ("saturated", "13 и больше"),
        ("minority", "не больше четверти"),
        ("majority", "не меньше половины"),
    ]
)


def category_interval(category: str, max_count: int) -> tuple[int, int]:
    """Вернуть ``(lo, hi)``; ``lo > hi`` означает выключенное условие."""

    maximum = max(0, int(max_count))
    if category == "none":
        return (1, 0)
    if category == "any":
        return (1, maximum)
    if category == "empty":
        return (0, 0)
    if category == "sparse":
        return (1, min(2, maximum))
    if category == "classic":
        value = min(3, maximum)
        return (value, value)
    if category == "wide":
        return (2, min(4, maximum)) if maximum >= 2 else (1, 0)
    if category == "loose":
        return (1, min(5, maximum))
    if category == "balanced":
        return (2, min(6, maximum)) if maximum >= 2 else (1, 0)
    if category == "dense":
        return (4, min(12, maximum)) if maximum >= 4 else (1, 0)
    if category == "crowded":
        return (8, min(20, maximum)) if maximum >= 8 else (1, 0)
    if category == "saturated":
        return (13, maximum) if maximum >= 13 else (1, 0)
    if category == "minority":
        return (1, max(1, maximum // 4))
    if category == "majority":
        return (max(1, maximum // 2), maximum)
    raise KeyError(f"неизвестная категория: {category!r}")


@dataclass(frozen=True)
class Trick:
    id: str
    code: int
    name: str
    description: str


# Коды синхронизированы с ``kernel.py``.
TRICKS: "OrderedDict[str, Trick]" = OrderedDict(
    (
        key,
        Trick(key, code, name, description),
    )
    for code, (key, name, description) in enumerate(
        [
            (
                "stable6",
                "состояние 6 постоянно стабильно",
                "клетка 6 остаётся 6 независимо от соседей",
            ),
            (
                "predator4",
                "состояние 4 уничтожает 3 и становится 2",
                "4, увидевшая 3, сама переходит в 2; любая 3 рядом с 4 умирает",
            ),
            (
                "immortal7",
                "состояние 7 бессмертно",
                "клетка 7 остаётся 7 при любом окружении",
            ),
            (
                "food_chain",
                "3 ест 2 и растёт до 4",
                "3 при двух и более 2 становится 4; голодная 4 без 2 умирает",
            ),
            (
                "phoenix2",
                "феникс из 2 и 3",
                "мёртвая клетка рядом с 2 и 3 рождается как 2",
            ),
            (
                "anti6",
                "1 травит 6",
                "6 умирает, если рядом есть 1",
            ),
            (
                "mirror3",
                "зеркало 3 и распад 2",
                "3 при трёх 3 становится 2; 2 при четырёх 2 умирает",
            ),
            (
                "spread5",
                "споры 5",
                "мёртвая клетка с двумя 5 рождается как 5",
            ),
            (
                "immune5",
                "состояние 5 иммунно",
                "клетка 5 остаётся 5",
            ),
            (
                "convert5",
                "5 переходит в 1 под давлением",
                "5 становится 1, если 1 вокруг больше, чем 5",
            ),
            (
                "seed6",
                "семена 6",
                "мёртвая клетка с тремя 6 рождается как 6",
            ),
            (
                "bomb7",
                "взрыв 7",
                "7 при 12 и более живых соседях становится 1",
            ),
            (
                "hide2",
                "2 прячется в стае",
                "2 при пяти 2 остаётся 2",
            ),
            (
                "chain3",
                "цепочка 3",
                "любая 3 превращается в 2",
            ),
            (
                "eat6",
                "1 и 6 взаимно уничтожаются",
                "6 при 1 становится 2; 1 рядом с 6 умирает",
            ),
            (
                "quarantine7",
                "карантин 7",
                "7 при двух 7 остаётся 7; новые 7 не рождаются",
            ),
            (
                "fuse23",
                "слияние 2 и 3",
                "3 рядом с 2 становится 2; 2 рядом с 3 становится 3",
            ),
            (
                "decay4",
                "распад 4",
                "4 без своих соседей становится 3; при четырёх 3 умирает",
            ),
            (
                "burst1",
                "вспышка 1",
                "мёртвая клетка с четырьмя 1 рождается как 6; 6 с двумя 1 становится 1",
            ),
            (
                "swarm2",
                "рой 2",
                "мёртвая клетка с тремя 2 рождается как 1; 1 с двумя 2 становится 2",
            ),
        ],
        start=1,
    )
)


def trick_code(trick: str) -> int:
    if trick == "none":
        return 0
    return TRICKS[trick].code


def rule_descriptor(values: Mapping[str, Any]) -> dict[str, Any]:
    """Полное описание правила, пригодное для показа и сериализации."""

    prefix = "rule.omega5."
    mechanism = str(values.get(prefix + "mechanism", "full"))
    if mechanism not in MECHANISMS:
        mechanism = "full"
    maximum = mechanism_max(mechanism)
    categories: dict[str, Any] = {}
    for kind in ("birth", "survival", "death"):
        category = str(values.get(prefix + kind, "none"))
        if category not in CATEGORIES:
            category = "none"
        lo, hi = category_interval(category, maximum)
        categories[kind] = {
            "id": category,
            "label": CATEGORIES[category].label,
            "interval": [lo, hi],
            "enabled": lo <= hi,
        }
    birth_state = int(values.get(prefix + "birth_state", 1))
    categories["birth"]["state"] = birth_state

    trick = str(values.get(prefix + "trick", "none"))
    info = TRICKS.get(trick)
    return {
        "version": 1,
        "kernel": "omega5.v1",
        "experiment": "omega5",
        "states": STATES,
        "zero": int(values.get("grid2d.zero", 0)),
        "window": {
            "size": [WINDOW_SIZE, WINDOW_SIZE],
            "radius": WINDOW_RADIUS,
            "neighbors": WINDOW_NEIGHBORS,
            "wrap": bool(values.get("neighborhood.square5.wrap", True)),
            "center_excluded": True,
        },
        "mechanism": {
            "id": mechanism,
            "code": MECHANISMS[mechanism].code,
            "label": MECHANISMS[mechanism].label,
            "max_count": maximum,
        },
        "categories": categories,
        "trick": (
            {"id": "none", "code": 0, "name": "без особой реакции", "description": ""}
            if info is None
            else {
                "id": info.id,
                "code": info.code,
                "name": info.name,
                "description": info.description,
            }
        ),
        "phase": {
            "warmup": int(values.get("phase.warmup", 0)),
            "observe": int(values.get("phase.observe", 0)),
        },
    }


def serialize_rule(values: Mapping[str, Any]) -> str:
    """Каноническая JSON-строка правила (без seed и путей прогона)."""

    return json.dumps(
        rule_descriptor(values),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def describe_rule(values: Mapping[str, Any]) -> str:
    """Короткое человекочитаемое описание для карточки/терминала."""

    info = rule_descriptor(values)
    lines = [
        f"окно {WINDOW_SIZE}×{WINDOW_SIZE}, {WINDOW_NEIGHBORS} соседей, "
        f"{info['states']} состояний"
        + (" (тор)" if info["window"]["wrap"] else " (границы)"),
        f"счётчик: {info['mechanism']['label']} "
        f"0..{info['mechanism']['max_count']}",
    ]
    for kind, title in (("birth", "рождение"), ("survival", "выживание"), ("death", "смерть")):
        item = info["categories"][kind]
        interval = item["interval"]
        if item["enabled"]:
            text = f"{interval[0]}..{interval[1]}"
        else:
            text = "выключено"
        if kind == "birth" and item["enabled"]:
            text += f" → состояние {item['state']}"
        lines.append(f"{title}: {item['label']} [{text}]")
    trick = info["trick"]
    lines.append(f"реакция: {trick['name']}" + (f" — {trick['description']}" if trick["description"] else ""))
    return "\n".join(lines)


__all__ = [
    "CATEGORIES",
    "MECHANISMS",
    "STATES",
    "TRICKS",
    "WINDOW_NEIGHBORS",
    "WINDOW_OFFSETS",
    "WINDOW_RADIUS",
    "WINDOW_SIZE",
    "Category",
    "Mechanism",
    "Trick",
    "category_interval",
    "describe_rule",
    "group_ids",
    "mechanism_max",
    "rule_descriptor",
    "serialize_rule",
    "trick_code",
]
