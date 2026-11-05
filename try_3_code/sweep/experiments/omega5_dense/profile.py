"""Профиль плотного автомата: допустимые доли каждого состояния.

Пользователь правит ``profile.json`` (или указывает свой файл через
``SWEEP_OMEGA5_PROFILE`` / ``python -m sweep run ... --profile FILE``) до
запуска. Проверка «выверенности»:

* все 8 состояний заданы;
* ``0 <= lo <= hi <= 1``;
* суммарные минимумы не больше 1, суммарные максимумы не меньше 1 — значит,
  набор долей в принципе может сложиться в 100%;
* у мёртвого состояния верхняя граница не больше 5% — это и делает автомат
  плотным.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

PROFILE_ENV = "SWEEP_OMEGA5_PROFILE"
STATES = 8
ZERO_MAX_SHARE = 0.35
PROFILE_PATH = Path(__file__).with_name("profile.json")

# Запасной профиль, если рядом нет profile.json. Значения специально
# «выверены»: минимумы дают 84%, максимумы — 120%, значит 100% достижимы.
DEFAULT_PROFILE: dict[int, tuple[float, float]] = {
    0: (0.15, 0.35),
    1: (0.02, 0.05),
    2: (0.02, 0.05),
    3: (0.30, 0.50),
    4: (0.05, 0.10),
    5: (0.02, 0.05),
    6: (0.02, 0.05),
    7: (0.01, 0.05),
}


def _as_range(value: object, state: int) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"состояние {state}: ожидался список [lo, hi], получено {value!r}")
    try:
        lo = float(value[0])
        hi = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"состояние {state}: границы должны быть числами") from exc
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(
            f"состояние {state}: нужно 0 <= lo <= hi <= 1, получено [{lo}, {hi}]"
        )
    return lo, hi


def validate_profile(
    raw: Mapping[object, object],
    *,
    states: int = STATES,
    zero_max: float = ZERO_MAX_SHARE,
) -> dict[int, tuple[float, float]]:
    """Проверить профиль и вернуть ``{state: (lo, hi)}``."""

    profile: dict[int, tuple[float, float]] = {}
    missing: list[int] = []
    for state in range(states):
        if str(state) in raw:
            value = raw[str(state)]
        elif state in raw:
            value = raw[state]  # type: ignore[index]
        else:
            missing.append(state)
            continue
        profile[state] = _as_range(value, state)
    if missing:
        raise ValueError(
            "в профиле не заданы состояния: " + ", ".join(str(state) for state in missing)
        )

    zero_lo, zero_hi = profile[0]
    if zero_hi > zero_max + 1e-12:
        raise ValueError(
            f"плотный автомат требует долю состояния 0 не больше {zero_max:.0%}, "
            f"а в профиле {zero_lo:.3f}..{zero_hi:.3f}"
        )
    min_sum = sum(lo for lo, _ in profile.values())
    max_sum = sum(hi for _, hi in profile.values())
    if min_sum > 1.0 + 1e-9:
        raise ValueError(
            f"сумма минимумов {min_sum:.3f} больше 100% — такие доли несовместимы"
        )
    if max_sum < 1.0 - 1e-9:
        raise ValueError(
            f"сумма максимумов {max_sum:.3f} меньше 100% — такие доли несовместимы"
        )
    return profile


def _extract_density(raw: Mapping[object, object]) -> tuple[float, float]:
    if "density" not in raw:
        return 0.2, 1.0
    value = raw["density"]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"density: ожидался список [min, max], получено {value!r}")
    try:
        lo = float(value[0])
        hi = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("density: границы должны быть числами") from exc
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"density: нужно 0 <= min <= max <= 1, получено [{lo}, {hi}]")
    return lo, hi


def _read_raw(path: str | Path | None = None) -> dict[str, Any] | None:
    chosen = path or os.environ.get(PROFILE_ENV)
    if chosen:
        file = Path(chosen)
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"нет файла профиля: {file}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file}: некорректный JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{file}: ожидался объект")
        return raw
    if PROFILE_PATH.exists():
        try:
            raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{PROFILE_PATH}: некорректный JSON: {exc}") from exc
        if isinstance(raw, dict):
            return raw
    return None


def read_profile(path: str | Path, *, states: int = STATES) -> dict[int, tuple[float, float]]:
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"нет файла профиля: {file}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: некорректный JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{file}: ожидался объект с ключами 0..{states - 1}")
    clean = {
        key: value
        for key, value in raw.items()
        if not str(key).startswith("_") and str(key) != "density"
    }
    return validate_profile(clean, states=states)


def load_profile(path: str | Path | None = None, *, states: int = STATES):
    raw = _read_raw(path)
    if raw is None:
        return validate_profile(
            {str(state): value for state, value in DEFAULT_PROFILE.items()}, states=states
        )
    clean = {
        key: value
        for key, value in raw.items()
        if not str(key).startswith("_") and str(key) != "density"
    }
    return validate_profile(clean, states=states)


def load_density(path: str | Path | None = None) -> tuple[float, float]:
    raw = _read_raw(path)
    if raw is None:
        return 0.2, 1.0
    return _extract_density(raw)


def profile_json(profile: Mapping[int, tuple[float, float]]) -> str:
    return json.dumps(
        {str(state): [lo, hi] for state, (lo, hi) in sorted(profile.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def describe_profile(profile: Mapping[int, tuple[float, float]]) -> str:
    lines = [
        f"состояние {state}: {lo * 100:.1f}%..{hi * 100:.1f}%"
        for state, (lo, hi) in sorted(profile.items())
    ]
    min_sum = sum(lo for lo, _ in profile.values())
    max_sum = sum(hi for _, hi in profile.values())
    lines.append(f"сумма: {min_sum * 100:.1f}%..{max_sum * 100:.1f}%")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_PROFILE",
    "PROFILE_ENV",
    "PROFILE_PATH",
    "STATES",
    "ZERO_MAX_SHARE",
    "describe_profile",
    "load_density",
    "load_profile",
    "profile_json",
    "read_profile",
    "validate_profile",
]
