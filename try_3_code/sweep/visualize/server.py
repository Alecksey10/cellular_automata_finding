"""Просмотр прогона в браузере карточками.

Сервер не пишет на диск ни кадров, ни картинок: страница запрашивает кадр,
сервер собирает автомат из сохранённого конфига (:mod:`sweep.replay`),
раскрашивает кадр визуализатором эксперимента и отдаёт PNG. Собранные кадры
кэшируются в памяти, поэтому листать карточки и проигрывать анимацию дёшево,
а на диске остаются только журналы прогона.

Каждый запрос может нести мета-параметры (`phase.warmup`, `phase.observe`,
`seed`): тот же автомат пересобирается с другими длительностями фаз.

Запуск: ``python -m sweep view --run-dir runs/life --port 8000``.
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
    meta_parameters,
    rebuild,
    states_of,
    visualizer_for,
)
from ..storage import RunStore
from .engine import colorize, scale_nearest
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
    """Успешные конфиги прогона и небольшой кэш собранных вариантов."""

    def __init__(self, store: RunStore, *, cache_size: int = 12) -> None:
        self.store = store
        self.records = store.successes()
        self.by_digest = {record["digest"]: record for record in self.records}
        self._cache: dict[tuple[str, tuple[tuple[str, int], ...]], tuple[list[np.ndarray], Any]] = {}
        self._order: list[tuple[str, tuple[tuple[str, int], ...]]] = []
        self._cache_size = max(1, int(cache_size))
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def payload(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for index, record in enumerate(self.records):
            config = record.get("config", {})
            entries.append(
                {
                    "index": index,
                    "digest": record["digest"],
                    "experiment": config.get("experiment", ""),
                    "seed": config.get("seed"),
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
            )
        return {
            "run": str(self.store.directory),
            "experiment": self._experiment_name(),
            "build": BUILD,
            "records": entries,
        }

    def _experiment_name(self) -> str:
        for record in self.records:
            name = record.get("config", {}).get("experiment")
            if name:
                return str(name)
        return ""

    # ------------------------------------------------------------------
    def overrides(self, digest: str, query: Mapping[str, list[str]]) -> dict[str, int]:
        record = self.by_digest.get(digest)
        if record is None:
            raise KeyError(digest)
        return query_overrides(record, query)

    def _variant(
        self, digest: str, overrides: Mapping[str, int] | None
    ) -> tuple[list[np.ndarray], Any]:
        record = self.by_digest.get(digest)
        if record is None:
            raise KeyError(digest)
        key = (digest, tuple(sorted((overrides or {}).items())))
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = rebuild(record, overrides=overrides)
        frames = [frame.data for frame in result.frames]
        cached = (frames, result)
        with self._lock:
            self._cache[key] = cached
            self._order.append(key)
            while len(self._order) > self._cache_size:
                self._cache.pop(self._order.pop(0), None)
        return cached

    def details(self, digest: str, overrides: Mapping[str, int] | None = None) -> dict[str, Any] | None:
        record = self.by_digest.get(digest)
        if record is None:
            return None
        frames, result = self._variant(digest, overrides)
        visualizer = visualizer_for(record.get("config", {}).get("experiment", ""))
        states = states_of(record)
        return {
            "digest": digest,
            "frames": len(frames),
            "states": states,
            "palette": [list(visualizer.state_color(state, states)) for state in range(states)],
            "legend": [[list(color), label] for color, label in visualizer.legend(states)],
            "background": list(visualizer.background()),
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

    def frame_png(self, digest: str, index: int, scale: int, overrides: Mapping[str, int] | None = None) -> bytes | None:
        record = self.by_digest.get(digest)
        if record is None:
            return None
        frames, _ = self._variant(digest, overrides)
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
        try:
            if path in ("/", "/index.html"):
                page = PAGE.replace("__DATA__", json.dumps(self.gallery.payload(), ensure_ascii=False))
                self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
                return
            if path == "/api/records":
                self._json(self.gallery.payload())
                return
            if path.startswith("/api/record/"):
                digest = path[len("/api/record/") :]
                details = self.gallery.details(digest, self.gallery.overrides(digest, query))
                # метаданные варианта дешёвые и меняются вместе с кодом — не кэшируем
                self._json(details)
                return
            if path.startswith("/api/frames/"):
                self._frame(path[len("/api/frames/") :], query)
                return
            self._error(404, "не найдено")
        except KeyError as exc:
            self._error(404, f"конфиг {exc} не найден")
        except Exception as exc:  # noqa: BLE001 - страница должна показать причину
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _frame(self, rest: str, query: Mapping[str, list[str]]) -> None:
        parts = rest.split("/")
        if len(parts) != 2 or not parts[1].endswith(".png"):
            self._error(404, "ожидался /api/frames/<digest>/<n>.png")
            return
        digest = parts[0]
        try:
            index = int(parts[1][: -len(".png")])
            scale = int(query.get("scale", ["4"])[0])
        except ValueError:
            self._error(400, "номер кадра и масштаб должны быть числами")
            return
        overrides = self.gallery.overrides(digest, query)
        png = self.gallery.frame_png(digest, index, scale, overrides)
        if png is None:
            self._error(404, f"кадр {index} недоступен")
            return
        self._send(200, "image/png", png, cache=True)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - подпись базового класса
        """Не мешаем выводу прогона сообщениями о каждом запросе."""


def make_server(store: RunStore, *, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Сервер галереи для каталога прогона."""

    gallery = Gallery(store)
    handler = type("GalleryHandler", (_Handler,), {"gallery": gallery})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


PAGE = """<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<title>sweep · карточки прогона</title>
<style>
 :root { color-scheme: dark; }
 * { box-sizing: border-box; }
 body { margin: 0; background: #0d0f13; color: #d7dbe3;
        font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
 header { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 12px;
          flex-wrap: wrap; padding: 10px 16px; background: #0f1218; border-bottom: 1px solid #232733; }
 header h1 { margin: 0; font-size: 15px; font-weight: 600; white-space: nowrap; }
 header .run { color: #7d879b; font-size: 12px; max-width: 320px; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }
 button { background: #1d2330; color: #dfe4ee; border: 1px solid #2e3648; border-radius: 7px;
          padding: 5px 11px; font: inherit; cursor: pointer; }
 button:hover { background: #243047; }
 button.primary { background: #2d5fbf; border-color: #3c74d8; }
 input[type=number] { width: 62px; background: #12161f; color: #e4e9f2; border: 1px solid #2e3648;
                      border-radius: 6px; padding: 3px 6px; font: inherit; }
 input[type=range] { width: 86px; vertical-align: -3px; }
 .hint { color: #7d879b; font-size: 12px; }
 .grow { flex: 1; }
 main { display: grid; grid-template-columns: repeat(auto-fill, minmax(258px, 1fr));
        gap: 14px; padding: 16px; }
 .card { background: #131824; border: 1px solid #232b3a; border-radius: 10px; overflow: hidden;
         display: flex; flex-direction: column; }
 .card.stopped { border-color: #4a2f36; }
 .head { display: flex; align-items: center; justify-content: space-between; gap: 8px;
         padding: 8px 10px; border-bottom: 1px solid #1d2430; }
 .head b { font: 12px ui-monospace, monospace; color: #cfd8e8; }
 .badge { font-size: 11px; color: #93a0b8; background: #1b2333; border-radius: 999px;
          padding: 1px 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 150px; }
 .card.stopped .badge { background: #35222a; color: #e2a0ad; }
 .shot { position: relative; background: #0b0d11; display: grid; place-items: center; min-height: 130px; }
 .shot img { width: 100%; height: auto; display: block; image-rendering: pixelated; }
 .shot .note { position: absolute; inset: 0; display: grid; place-items: center;
               color: #6f788c; font-size: 12px; }
 .line { padding: 7px 10px 0; color: #8b93a7; font: 11.5px ui-monospace, monospace; }
 .knobs { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 8px 10px 10px; }
 .knob { display: flex; align-items: center; gap: 4px; color: #8b93a7; font-size: 11.5px; }
 details { border-top: 1px solid #1d2430; }
 summary { padding: 6px 10px; color: #8b93a7; font-size: 11.5px; cursor: pointer; }
 details pre { margin: 0; padding: 0 10px 8px; color: #a9b4c8; font: 11px/1.5 ui-monospace, monospace;
               white-space: pre-wrap; max-height: 220px; overflow: auto; }
 .legend { display: flex; flex-wrap: wrap; gap: 6px 12px; padding: 0 10px 8px; font-size: 11.5px; color: #a9b4c8; }
 .legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block;
             vertical-align: -1px; margin-right: 5px; }
 #more { padding: 0 16px 28px; text-align: center; }
</style>
<body>
<header>
  <h1>sweep</h1>
  <span class="run" id="run"></span>
  <span class="hint" id="count"></span>
  <span class="grow"></span>
  <button id="play" class="primary">▶ все</button>
  <button id="pause">⏸</button>
  <label class="hint">fps <input id="fps" type="range" min="1" max="30" value="6"><b id="fpsVal">6</b></label>
  <label class="hint">× <input id="scale" type="range" min="1" max="8" value="3"><b id="scaleVal">3</b></label>
  <label class="hint">warmup <input id="warm" type="number" min="0" max="2000" placeholder="—"></label>
  <label class="hint">observe <input id="obs" type="number" min="0" max="2000" placeholder="—"></label>
  <button id="applyAll">применить ко всем</button>
</header>
<main id="grid"></main>
<div id="more"><button id="moreBtn">показать ещё</button></div>
<script>
const DATA = __DATA__;
const ALL = DATA.records;
const BUILD = DATA.build || "";
const PAGE_SIZE = 48;
const el = (id) => document.getElementById(id);
let fps = 6, scale = 3, timer = null, shown = 0;
const cards = [];

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function metricLine(r) {
  const m = r.metrics || {};
  const keys = ["alive_density", "zero_density", "occupied_states", "max_state_share"].filter((k) => k in m);
  return keys.map((k) => k + "=" + Number(m[k]).toFixed(3)).join("  ");
}

function params(state) {
  return Object.entries(state.overrides)
    .map(([key, value]) => encodeURIComponent(key) + "=" + encodeURIComponent(value))
    .join("&");
}

function frameUrl(state, index) {
  const query = params(state);
  return "/api/frames/" + state.record.digest + "/" + index + ".png?scale=" + scale +
    (query ? "&" + query : "") + "&v=" + BUILD;
}

function makeCard(record) {
  const node = document.createElement("article");
  node.className = "card";
  node.innerHTML =
    '<div class="head"><b>' + escapeHtml(record.digest) + '</b><span class="badge">' +
    escapeHtml(record.status || "") + '</span></div>' +
    '<div class="shot"><img alt="кадр"><span class="note">вне экрана</span></div>' +
    '<div class="line"></div>' +
    '<div class="knobs"></div>' +
    '<details><summary>конфиг и метрики</summary><div class="legend"></div><pre></pre></details>';
  const state = {
    node: node,
    record: record,
    overrides: {},
    frames: 0,
    frame: 0,
    loaded: false,
    busy: false,
    visible: false,
    img: node.querySelector("img"),
    note: node.querySelector(".note"),
    line: node.querySelector(".line"),
    pre: node.querySelector("pre"),
    legend: node.querySelector(".legend"),
    badge: node.querySelector(".badge"),
    inputs: {},
  };
  state.line.textContent = metricLine(record) + "  ·  " + record.iterations + " итер.";
  state.pre.textContent = Object.entries(record.values).map(([k, v]) => k + " = " + v).join("\\n");

  const knobs = node.querySelector(".knobs");
  for (const meta of record.meta || []) {
    const label = document.createElement("label");
    label.className = "knob";
    const name = document.createElement("span");
    name.textContent = meta.label;
    const input = document.createElement("input");
    input.type = "number";
    input.min = meta.min;
    input.max = meta.max;
    input.value = meta.value;
    label.append(name, input);
    knobs.append(label);
    state.inputs[meta.key] = { input: input, meta: meta };
  }
  const run = document.createElement("button");
  run.textContent = "пересобрать";
  run.addEventListener("click", () => applyCard(state));
  knobs.append(run);
  node.__state = state;
  return state;
}

function overridesFrom(state) {
  const overrides = {};
  for (const [key, item] of Object.entries(state.inputs)) {
    const value = Number(item.input.value);
    if (!Number.isFinite(value)) continue;
    const rounded = Math.max(item.meta.min, Math.min(item.meta.max, Math.round(value)));
    if (rounded !== item.meta.value) overrides[key] = rounded;
  }
  return overrides;
}

function applyCard(state) {
  state.overrides = overridesFrom(state);
  state.loaded = false;
  state.frames = 0;
  load(state);
}

async function load(state) {
  if (state.busy) return;
  state.busy = true;
  state.note.style.display = "inline";
  state.note.textContent = "сборка…";
  try {
    const query = params(state);
    // метка сборки в URL: после перезапуска сервера браузер не берёт старый ответ
    const response = await fetch(
      "/api/record/" + state.record.digest + "?" + (query ? query + "&" : "") + "v=" + BUILD
    );
    const details = await response.json();
    if (!response.ok || details.error) throw new Error(details.error || String(response.status));
    state.frames = details.frames;
    state.frame = 0;
    state.loaded = true;
    state.note.style.display = "none";
    state.line.textContent = metricLine(details) + "  ·  " + details.iterations + " итер.";
    state.badge.textContent = details.status + (details.reason ? " · " + details.reason : "");
    state.node.classList.toggle("stopped", details.status !== "success");
    state.legend.textContent = "";
    (details.legend || []).forEach(([rgb, label]) => {
      const span = document.createElement("span");
      const sw = document.createElement("i");
      sw.style.background = "rgb(" + rgb.join(",") + ")";
      span.append(sw, document.createTextNode(label));
      state.legend.append(span);
    });
    if (state.frames) {
      state.img.src = frameUrl(state, 0);
      preload(state);
    }
  } catch (error) {
    state.note.textContent = "ошибка: " + error.message;
  } finally {
    state.busy = false;
  }
}

function preload(state) {
  for (let i = 0; i < state.frames; i += 1) {
    const image = new Image();
    image.src = frameUrl(state, i);
  }
}

function tick() {
  for (const state of cards) {
    if (!state.visible || !state.loaded || state.frames < 2) continue;
    state.frame = (state.frame + 1) % state.frames;
    state.img.src = frameUrl(state, state.frame);
  }
}

function play() {
  if (timer || !cards.length) return;
  timer = setInterval(tick, 1000 / fps);
  el("play").textContent = "⏸ идёт";
}

function pause() {
  if (!timer) return;
  clearInterval(timer);
  timer = null;
  el("play").textContent = "▶ все";
}

const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const state = entry.target.__state;
    if (!state) continue;
    state.visible = entry.isIntersecting;
    if (entry.isIntersecting) {
      state.node.querySelector(".shot").classList.add("on");
      if (!state.loaded) load(state);
    }
  }
}, { rootMargin: "240px" });

function addCards(count) {
  const slice = ALL.slice(shown, shown + count);
  for (const record of slice) {
    const state = makeCard(record);
    cards.push(state);
    el("grid").append(state.node);
    observer.observe(state.node);
  }
  shown += slice.length;
  el("moreBtn").style.display = shown < ALL.length ? "inline-block" : "none";
  el("count").textContent = shown + " из " + ALL.length + " карточек";
}

el("play").addEventListener("click", play);
el("pause").addEventListener("click", pause);
el("fps").addEventListener("input", (event) => {
  fps = Number(event.target.value);
  el("fpsVal").textContent = fps;
  if (timer) { pause(); play(); }
});
el("scale").addEventListener("input", (event) => {
  scale = Number(event.target.value);
  el("scaleVal").textContent = scale;
  for (const state of cards) if (state.loaded && state.frames) state.img.src = frameUrl(state, state.frame);
});
el("applyAll").addEventListener("click", () => {
  const warm = el("warm").value, observe = el("obs").value;
  for (const state of cards) {
    if (warm !== "" && state.inputs["phase.warmup"]) state.inputs["phase.warmup"].input.value = warm;
    if (observe !== "" && state.inputs["phase.observe"]) state.inputs["phase.observe"].input.value = observe;
    applyCard(state);
  }
  play();
});
el("moreBtn").addEventListener("click", () => addCards(PAGE_SIZE));
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && event.target === document.body) {
    event.preventDefault();
    timer ? pause() : play();
  }
});
el("run").textContent = DATA.run;
el("count").textContent = ALL.length + " карточек";
addCards(PAGE_SIZE);
play();
</script>
</body>
</html>
"""

__all__ = ["Gallery", "MAX_SCALE", "PAGE", "make_server", "query_overrides"]
