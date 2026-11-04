"""Хранилище результатов: JSONL с дозаписью и возобновлением прогона."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping


def _json_default(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    raise TypeError(f"Не сериализуется: {value!r}")


class ResultStore:
    """Пишет ``successes.jsonl``; полный лог ``trials.jsonl`` — только по запросу.

    По умолчанию отклонённые и несовместимые конфиги никуда не записываются:
    на больших переборах они занимают гигабайты и не нужны. ``save_all=True``
    включает полный лог (для отладки и анализа отказов).
    """

    def __init__(
        self,
        out_dir: str | Path | None,
        *,
        resume: bool = False,
        save_all: bool = False,
        success_statuses: Iterable[str] = ("passed",),
    ) -> None:
        self.enabled = out_dir is not None
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.save_all = bool(save_all)
        self.success_statuses = frozenset(success_statuses)
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._trials_file = None
        self._success_file = None
        self.written = 0
        if not self.enabled:
            return
        assert self.out_dir is not None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trials = self.out_dir / "trials.jsonl"
        successes = self.out_dir / "successes.jsonl"
        if resume:
            for path in (successes, trials):
                if path.exists():
                    self._seen |= _read_digests(path)
        if self.save_all:
            self._trials_file = trials.open("a", encoding="utf-8")
        self._success_file = successes.open("a", encoding="utf-8")

    # --- API -------------------------------------------------------------
    def seen(self, digest: str) -> bool:
        return digest in self._seen

    def mark(self, digest: str) -> None:
        self._seen.add(digest)

    def append(self, record: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        is_success = record.get("status") in self.success_statuses
        if not is_success and not self.save_all:
            return
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default)
        with self._lock:
            if self._trials_file is not None:
                self._trials_file.write(line + "\n")
                self._trials_file.flush()
            if is_success and self._success_file is not None:
                self._success_file.write(line + "\n")
                self._success_file.flush()
            self.written += 1
        digest = record.get("config_digest")
        if digest:
            self._seen.add(str(digest))

    def close(self) -> None:
        for handle in (self._trials_file, self._success_file):
            if handle is not None:
                handle.close()
        self._trials_file = None
        self._success_file = None

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- чтение ----------------------------------------------------------
    @property
    def trials_path(self) -> Path | None:
        return None if self.out_dir is None else self.out_dir / "trials.jsonl"

    def read_configs(self, *, statuses: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Прочитать конфиги из полного лога или из сохранённых успехов."""

        if self.out_dir is None:
            return []
        paths = [self.out_dir / "trials.jsonl", self.out_dir / "successes.jsonl"]
        allowed = set(statuses) if statuses is not None else None
        configs: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if allowed is not None and record.get("status") not in allowed:
                        continue
                    configs.append(record.get("config", {}))
        return configs


def _read_digests(path: Path) -> set[str]:
    digests: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            digest = record.get("config_digest")
            if digest:
                digests.add(str(digest))
    return digests


def find_record(run_dir: str | Path, digest: str) -> dict[str, Any] | None:
    """Найти запись теста по (префиксу) дайджеста в successes.jsonl или trials.jsonl."""

    base = Path(run_dir)
    for name in ("successes.jsonl", "trials.jsonl"):
        path = base / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if str(record.get("config_digest", "")).startswith(digest):
                    return record
    return None


def find_config(run_dir: str | Path, digest: str) -> dict[str, Any] | None:
    """Найти конфиг по дайджесту в сохранённом прогоне."""

    record = find_record(run_dir, digest)
    return None if record is None else record.get("config")


def read_records(
    run_dir: str | Path,
    *,
    statuses: Iterable[str] | None = ("passed",),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Прочитать записи прогона по порядку, без дублей по дайджесту.

    Сначала читается ``successes.jsonl``, затем (если он есть и дополняет)
    ``trials.jsonl``. Нужно для галереи: каждая запись — один автомат.
    """

    base = Path(run_dir)
    allowed = None if statuses is None else set(statuses)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in ("successes.jsonl", "trials.jsonl"):
        path = base / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if allowed is not None and record.get("status") not in allowed:
                    continue
                digest = str(record.get("config_digest", ""))
                if digest and digest in seen:
                    continue
                if digest:
                    seen.add(digest)
                records.append(record)
                if limit is not None and len(records) >= limit:
                    return records
    return records
