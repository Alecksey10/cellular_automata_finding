"""Зафиксированный конфиг и его каноническая сериализация.

Конфиг — единственный источник воспроизводимости: он содержит только
скалярные значения (int/float/str/bool) и seed. Массивы, rng и скомпилированные
ядра в конфиг не попадают: они создаются заново на этапе исполнения.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def normalize_value(value: Any) -> Any:
    """Привести значение к JSON-типу, совместимому и с numba, и с хранилищем."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    # numpy-скаляры: int64/float32 и т.п.
    item = getattr(value, "item", None)
    if callable(item):
        return normalize_value(item())
    raise TypeError(f"параметр должен быть скаляром, получено {type(value).__name__}")


def normalize_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): normalize_value(value) for key, value in values.items()}


@dataclass(frozen=True)
class Config:
    experiment: str
    values: Mapping[str, Any]
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", normalize_values(self.values))
        object.__setattr__(self, "seed", int(self.seed))

    def canonical(self) -> str:
        return json.dumps(
            {"experiment": self.experiment, "seed": self.seed, "values": dict(sorted(self.values.items()))},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return hashlib.sha1(self.canonical().encode("utf-8")).hexdigest()[:16]

    def to_record(self) -> dict[str, Any]:
        return {"experiment": self.experiment, "seed": self.seed, "values": dict(sorted(self.values.items()))}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Config":
        return cls(
            experiment=str(record["experiment"]),
            values=dict(record["values"]),
            seed=int(record["seed"]),
        )


__all__ = ["Config", "normalize_value", "normalize_values"]
