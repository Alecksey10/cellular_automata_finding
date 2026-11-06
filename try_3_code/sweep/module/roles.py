"""Роли модулей и порядок их объявления.

Порядок важен: параметры данных и окрестности должны быть привязаны раньше,
чем правило объявит зависящие от них пороги.
"""

from __future__ import annotations

import enum


class Role(enum.Enum):
    DATA = "data"
    RANDOMIZER = "randomizer"
    NEIGHBORHOOD = "neighborhood"
    RULE = "rule"
    FILTER = "filter"


ROLE_ORDER: tuple[Role, ...] = (
    Role.DATA,
    Role.RANDOMIZER,
    Role.NEIGHBORHOOD,
    Role.RULE,
)

__all__ = ["ROLE_ORDER", "Role"]
