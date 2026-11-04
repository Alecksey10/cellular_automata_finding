"""Хранение результатов прогона.

Прогон хранит ровно то, что нужно для отбора и воспроизведения:

* ``trials.jsonl`` — компактная запись о каждом исполненном конфиге;
* ``successes.jsonl`` — только успешные конфиги (настройки + метрики);
* ``summary.json`` — счётчики и параметры запуска.

Массивы и картинки не сохраняются. Конфиг вместе с seed — это и есть способ
заново собрать автомат (:mod:`sweep.replay`), поэтому визуализация всегда
строится из свежего исполнения и не занимает диск.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import Config
from .runtime import SUCCESS, TrialResult


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass
class RunStats:
    trials: int = 0
    passed: int = 0
    rejected: int = 0
    stopped: int = 0
    errors: int = 0
    resumed: int = 0
    success_records: int = 0
    started: float = field(default_factory=time.time)

    def add(self, status: str) -> None:
        self.trials += 1
        if status == SUCCESS:
            self.passed += 1
        elif status == "rejected":
            self.rejected += 1
        elif status == "stopped":
            self.stopped += 1
        else:
            self.errors += 1

    def summary(self) -> dict[str, Any]:
        return {
            "trials": self.trials,
            "passed": self.passed,
            "rejected": self.rejected,
            "stopped": self.stopped,
            "errors": self.errors,
            "resumed": self.resumed,
            "successes": self.success_records,
            "elapsed": round(time.time() - self.started, 3),
        }


class RunStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.successes_path = self.directory / "successes.jsonl"
        self.trials_path = self.directory / "trials.jsonl"
        self.summary_path = self.directory / "summary.json"

    # ------------------------------------------------------------------
    def done_digests(self) -> set[str]:
        return {record.get("digest") for record in read_jsonl(self.trials_path) if record.get("digest")}

    def successes(self) -> list[dict[str, Any]]:
        return read_jsonl(self.successes_path)

    def trials(self) -> list[dict[str, Any]]:
        return read_jsonl(self.trials_path)

    def summary(self) -> dict[str, Any]:
        if self.summary_path.exists():
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        return {}

    # ------------------------------------------------------------------
    def save(self, config: Config, result: TrialResult) -> dict[str, Any]:
        record: dict[str, Any] = {
            "digest": config.digest(),
            "config": config.to_record(),
            "status": result.status,
            "reason": result.reason,
            "iterations": result.iterations,
            "duration": round(result.duration, 4),
            "metrics": dict(result.metrics),
            "error": result.error,
            "details": dict(result.details),
            "frames": len(result.frames),
        }
        self._append(self.trials_path, record)
        if result.status == SUCCESS:
            self._append(self.successes_path, record)
        return record

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        write_json(self.summary_path, summary)

    @staticmethod
    def _append(path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


__all__ = ["RunStats", "RunStore", "read_jsonl", "write_json"]
