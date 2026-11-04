"""Зафиксированная конфигурация одного теста.

Конфиг — это плоский словарь ``ключ параметра -> значение``. Он полностью
описывает прогон: по нему можно восстановить автомат и воспроизвести результат.
Поэтому конфиг сериализуется в канонический JSON и имеет стабильный дайджест.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .handles import Handle


def _json_default(value: Any) -> Any:
    try:
        import numpy as np  # локальный импорт: ядро работает и без numpy
    except ImportError:  # pragma: no cover - numpy есть в зависимостях
        np = None
    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    raise TypeError(f"Значение {value!r} не сериализуется в JSON")


@dataclass(frozen=True)
class Fixed:
    """Набор значений, при котором исполняется один тест."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))

    # --- доступ ---------------------------------------------------------
    def __getitem__(self, key: str | Handle) -> Any:
        return self.values[key.key if isinstance(key, Handle) else key]

    def get(self, key: str | Handle, default: Any = None) -> Any:
        return self.values.get(key.key if isinstance(key, Handle) else key, default)

    def __contains__(self, key: str | Handle) -> bool:
        return (key.key if isinstance(key, Handle) else key) in self.values

    def keys(self) -> tuple[str, ...]:
        return tuple(self.values)

    def subset(self, keys: Iterable[str]) -> "Fixed":
        return Fixed({key: self.values[key] for key in keys})

    def __len__(self) -> int:
        return len(self.values)

    # --- сериализация ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fixed":
        return cls(dict(data))

    @classmethod
    def of(cls, binding: Mapping[str, Any]) -> "Fixed":
        return cls(binding)

    def canonical(self) -> str:
        return json.dumps(
            self.values,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )

    def digest(self, length: int = 16) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:length]

    def __str__(self) -> str:
        return self.canonical()

    def pretty(self, indent: int = 2) -> str:
        return json.dumps(
            self.values, sort_keys=True, ensure_ascii=False, indent=indent, default=_json_default
        )
