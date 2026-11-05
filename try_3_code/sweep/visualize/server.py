"""Просмотр прогонов в браузере карточками.

Сервер не пишет на диск ни кадров, ни картинок: страница запрашивает кадр,
сервер собирает автомат из сохранённого конфига (:mod:`sweep.replay`),
раскрашивает кадр визуализатором эксперимента и отдаёт PNG. Собранные кадры
кэшируются в памяти, поэтому листать карточки и проигрывать анимацию дёшево,
а на диске остаются только журналы прогона.

Можно открыть сразу несколько каталогов прогонов (``--runs runs`` или
несколько ``--run-dir``): страница получит переключатель экспериментов, а
выбранный эксперимент и номер страницы сохраняются в адресе.

Каждый запрос может нести мета-параметры (``phase.warmup``, ``phase.observe``,
``seed``, ``grid2d.size``): тот же автомат пересобирается с другими
длительностями фаз или разрешением.

Запуск: ``python -m sweep view --runs runs --port 8000``.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

import numpy as np

from ..replay import (
    META_MAX_ITERATIONS,
    META_PREFIX,
    MAX_SEED,
    SIZE_KEY,
    meta_parameters,
    rebuild,
    states_of,
    visualizer_for,
)
from ..storage import RunStore
from .engine import colorize, scale_nearest
from .page import PAGE
from .png import png_bytes

MAX_SCALE = 16
# метка сборки: после перезапуска сервера браузер не берёт старые PNG из кэша
BUILD = format(int(time.time()), "x")


def query_overrides(record: Mapping[str, Any], query: Mapping[str, list[str]]) -> dict[str, int]:
    """Мета-параметры из query-строки, допустимые для этого конфига."""

    values = record.get("config", {}).get("values", {})
    allowed = {
        key
        for key, value in values.items()
        if key.startswith(META_PREFIX) and isinstance(value, int) and not isinstance(value, bool)
    }
    if SIZE_KEY in values:
        allowed.add(SIZE_KEY)
    elif isinstance(values.get("grid2d.width"), int) and values.get("grid2d.width") == values.get("grid2d.height"):
        allowed.add(SIZE_KEY)
    overrides: dict[str, int] = {}
    for key in allowed | {"seed"}:
        raw = query.get(key)
        if not raw:
            continue
        try:
            number = int(raw[0])
        except (TypeError, ValueError):
            continue
        limit = MAX_SEED if key == "seed" else META_MAX_ITERATIONS
        overrides[key] = max(0, min(number, limit))
    return overrides


class Gallery:
    """Успешные конфиги нескольких прогонов и небольшой кэш собранных вариантов."""

    def __init__(self, stores: Mapping[str, RunStore], *, cache_size: int = 12) -> None:
        self.stores = dict(stores)
        self.records = {name: store.successes() for name, store in self.stores.items()}
        self.by_digest = {
            name: {record["digest"]: record for record in records}
            for name, records in self.records.items()
        }
        self._cache: dict[tuple[str, str, tuple[tuple[str, int], ...]], tuple[list[np.ndarray], Any]] = {}
        self._order: list[tuple[str, str, tuple[tuple[str, int], ...]]] = []
        self._cache_size = max(1, int(cache_size))
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _card(self, record: Mapping[str, Any], index: int) -> dict[str, Any]:
        config = record.get("config", {})
        return {
            "index": index,
            "digest": record["digest"],
            "status": record.get("status"),
            "reason": record.get("reason"),
            "iterations": record.get("iterations", 0),
            "metrics": {
                key: value
                for key, value in record.get("metrics", {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            "values": config.get("values", {}),
            "meta": meta_parameters(record),
        }

    def payload(self) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for name, records in self.records.items():
            experiment = ""
            for record in records:
                experiment = str(record.get("config", {}).get("experiment") or "")
                if experiment:
                    break
            runs.append(
                {
                    "name": name,
                    "experiment": experiment,
                    "directory": str(self.stores[name].directory),
                    "cards": [self._card(record, index) for index, record in enumerate(records)],
                }
            )
        return {"build": BUILD, "mode": "server", "runs": runs}

    # ------------------------------------------------------------------
    def _lookup(self, run: str, digest: str) -> Mapping[str, Any]:
        try:
            return self.by_digest[run][digest]
        except KeyError:
            raise KeyError(f"{run}/{digest}") from None

    def overrides(self, run: str, digest: str, query: Mapping[str, list[str]]) -> dict[str, int]:
        return query_overrides(self._lookup(run, digest), query)

    def _variant(self, run: str, digest: str, overrides: Mapping[str, int] | None) -> tuple[list[np.ndarray], Any]:
        record = self._lookup(run, digest)
        key = (run, digest, tuple(sorted((overrides or {}).items())))
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = rebuild(record, overrides=overrides)
        cached = ([frame.data for frame in result.frames], result)
        with self._lock:
            self._cache[key] = cached
            self._order.append(key)
            while len(self._order) > self._cache_size:
                stale = self._order.pop(0)
                self._cache.pop(stale, None)
        return cached

    def details(self, run: str, digest: str, overrides: Mapping[str, int] | None = None) -> dict[str, Any]:
        record = self._lookup(run, digest)
        frames, result = self._variant(run, digest, overrides)
        visualizer = visualizer_for(record.get("config", {}).get("experiment", ""))
        states = states_of(record)
        return {
            "digest": digest,
            "run": run,
            "frames": len(frames),
            "states": states,
            "palette": [list(visualizer.state_color(state, states)) for state in range(states)],
            "legend": [[list(color), label] for color, label in visualizer.legend(states)],
            "background": list(visualizer.background()),
            "rule": visualizer.rule_info(record.get("config", {}).get("values", {})),
            "status": result.status,
            "reason": result.reason,
            "iterations": result.iterations,
            "metrics": {
                key: float(value)
                for key, value in result.metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            "overrides": dict(overrides or {}),
        }

    def frame_png(
        self,
        run: str,
        digest: str,
        index: int,
        scale: int,
        overrides: Mapping[str, int] | None = None,
    ) -> bytes | None:
        record = self._lookup(run, digest)
        frames, _ = self._variant(run, digest, overrides)
        if index < 0 or index >= len(frames):
            return None
        visualizer = visualizer_for(record.get("config", {}).get("experiment", ""))
        image = colorize(frames[index], visualizer, states_of(record))
        return png_bytes(scale_nearest(image, max(1, min(int(scale), MAX_SCALE))))


class _Handler(BaseHTTPRequestHandler):
    gallery: Gallery
    server_version = "sweep-view"

    # ------------------------------------------------------------------
    def _send(self, status: int, content_type: str, body: bytes, *, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200, *, cache: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body, cache=cache)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - имя задано базовым классом
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        parts = [part for part in path.split("/") if part]
        try:
            if path in ("/", "/index.html"):
                page = PAGE.replace("__DATA__", json.dumps(self.gallery.payload(), ensure_ascii=False))
                self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
                return
            if parts == ["api", "runs"]:
                self._json(self.gallery.payload())
                return
            if len(parts) >= 5 and parts[:2] == ["api", "run"]:
                run, kind, digest = parts[2], parts[3], parts[4]
                if kind == "record" and len(parts) == 5:
                    overrides = self.gallery.overrides(run, digest, query)
                    self._json(self.gallery.details(run, digest, overrides))
                    return
                if kind == "frames" and len(parts) == 6 and parts[5].endswith(".png"):
                    self._frame(run, digest, parts[5][: -len(".png")], query)
                    return
            self._error(404, "не найдено")
        except KeyError as exc:
            self._error(404, f"конфиг {exc} не найден")
        except Exception as exc:  # noqa: BLE001 - страница должна показать причину
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _frame(self, run: str, digest: str, raw_index: str, query: Mapping[str, list[str]]) -> None:
        try:
            index = int(raw_index)
            scale = int(query.get("scale", ["4"])[0])
        except ValueError:
            self._error(400, "номер кадра и масштаб должны быть числами")
            return
        overrides = self.gallery.overrides(run, digest, query)
        png = self.gallery.frame_png(run, digest, index, scale, overrides)
        if png is None:
            self._error(404, f"кадр {index} недоступен")
            return
        self._send(200, "image/png", png, cache=True)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - подпись базового класса
        """Не мешаем выводу прогона сообщениями о каждом запросе."""


def make_server(
    stores: Mapping[str, RunStore],
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """Сервер галереи для одного или нескольких каталогов прогонов."""

    gallery = Gallery(stores)
    handler = type("GalleryHandler", (_Handler,), {"gallery": gallery})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


__all__ = ["BUILD", "Gallery", "MAX_SCALE", "make_server", "query_overrides"]
