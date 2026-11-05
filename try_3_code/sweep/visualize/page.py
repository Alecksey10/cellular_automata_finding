"""HTML-страница галереи, общая для локального сервера и статического экспорта.

Данные подставляются вместо ``__DATA__`` — это JSON вида::

    {"build": "...", "mode": "server" | "static",
     "runs": [{"name": "langton", "experiment": "langton", "cards": [...]}]}

В режиме ``server`` кадры и подробности карточки запрашиваются у сервера,
поэтому карточку можно пересобрать с другими мета-параметрами. В режиме
``static`` у карточки уже есть готовая картинка (контактный лист кадров), и
страница работает без бэкенда — так её можно положить на GitHub Pages.

Навигация хранится в адресе (``?run=...&page=...&size=...``), поэтому
перезагрузка страницы возвращает на ту же страницу и эксперимент.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sweep · карточки прогона</title>
<style>
 :root { color-scheme: dark; }
 * { box-sizing: border-box; }
 body { margin: 0; background: #0d0f13; color: #d7dbe3;
        font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
 header { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 10px;
          flex-wrap: wrap; padding: 10px 16px; background: #0f1218; border-bottom: 1px solid #232733; }
 header h1 { margin: 0; font-size: 15px; font-weight: 600; white-space: nowrap; }
 select, input[type=number] { background: #12161f; color: #e4e9f2; border: 1px solid #2e3648;
          border-radius: 6px; padding: 3px 6px; font: inherit; }
 select { max-width: 200px; }
 input[type=number] { width: 74px; }
 button { background: #1d2330; color: #dfe4ee; border: 1px solid #2e3648; border-radius: 7px;
          padding: 5px 11px; font: inherit; cursor: pointer; }
 button:hover { background: #243047; }
 button:disabled { opacity: .4; cursor: default; }
 button.primary { background: #2d5fbf; border-color: #3c74d8; }
 input[type=range] { width: 84px; vertical-align: -3px; }
 .hint { color: #7d879b; font-size: 12px; }
 .grow { flex: 1; }
 .sep { width: 1px; height: 22px; background: #232733; }
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
 footer { display: flex; align-items: center; justify-content: center; gap: 10px;
          padding: 4px 16px 28px; flex-wrap: wrap; }
</style>
<body>
<header>
  <h1>sweep</h1>
  <select id="runSel" title="эксперимент"></select>
  <span class="hint" id="count"></span>
  <span class="grow"></span>
  <button id="play" class="primary">▶ все</button>
  <button id="pause">⏸</button>
  <label class="hint">fps <input id="fps" type="range" min="1" max="30" value="6"><b id="fpsVal">6</b></label>
  <label class="hint">× <input id="scale" type="range" min="1" max="12" value="3"><b id="scaleVal">3</b></label>
  <span class="sep knobs-only"></span>
  <label class="hint knobs-only">warmup <input id="warm" type="number" min="0" max="20000" placeholder="—"></label>
  <label class="hint knobs-only">observe <input id="obs" type="number" min="0" max="20000" placeholder="—"></label>
  <label class="hint knobs-only">размер <input id="res" type="number" min="32" max="192" step="16" placeholder="—"></label>
  <button id="applyAll" class="knobs-only">применить ко всем</button>
</header>
<main id="grid"></main>
<footer>
  <button id="prev">‹ назад</button>
  <label class="hint">стр. <input id="pageNo" type="number" min="1" value="1"></label>
  <span class="hint" id="pageInfo"></span>
  <button id="next">вперёд ›</button>
  <label class="hint">по <select id="pageSize"></select></label>
  <label class="hint">смещение <input id="jump" type="text" placeholder="номер или digest" size="16"></label>
</footer>
<script>
const DATA = __DATA__;
const MODE = DATA.mode || "server";
const BUILD = DATA.build || "";
const RUNS = DATA.runs || [];
const PAGE_SIZES = [24, 48, 96, 192];
const META_MAX = 20000;
const el = (id) => document.getElementById(id);
let current = 0, pageIndex = 0, pageSize = 48;
let fps = 6, scale = 3, timer = null;
let cards = [];

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
  if (MODE === "static") return state.record.image || "";
  const query = params(state);
  return "/api/run/" + encodeURIComponent(state.run.name) + "/frames/" + state.record.digest + "/" + index +
    ".png?scale=" + scale + (query ? "&" + query : "") + "&v=" + BUILD;
}

function makeCard(record, run) {
  const node = document.createElement("article");
  node.className = "card";
  node.innerHTML =
    '<div class="head"><b>' + "#" + record.index + " · " + escapeHtml(record.digest.slice(0, 8)) + "</b><span class=\\"badge\\">" +
    escapeHtml(record.status || "") + "</span></div>" +
    '<div class="shot"><img alt="кадр"><span class="note">вне экрана</span></div>' +
    '<div class="line"></div>' +
    '<div class="knobs"></div>' +
    "<details><summary>конфиг и метрики</summary><div class=\\"legend\\"></div><pre></pre></details>" +
    "<details><summary>правило (JSON)</summary><pre class=\\"rule\\">—</pre></details>";
  const state = {
    node: node, record: record, run: run, overrides: {}, frames: 0, frame: 0,
    loaded: false, busy: false, visible: false,
    img: node.querySelector("img"), note: node.querySelector(".note"),
    line: node.querySelector(".line"), pre: node.querySelector("pre"),
    rule: node.querySelector("pre.rule"), legend: node.querySelector(".legend"),
    badge: node.querySelector(".badge"), inputs: {},
  };
  state.line.textContent = metricLine(record) + "  ·  " + record.iterations + " итер.";
  state.pre.textContent = Object.entries(record.values).map(([k, v]) => k + " = " + v).join("\\n");

  const knobs = node.querySelector(".knobs");
  if (MODE === "static") {
    knobs.style.display = "none";
  } else {
    for (const meta of record.meta || []) {
      const label = document.createElement("label");
      label.className = "knob";
      const name = document.createElement("span");
      name.textContent = meta.label;
      const input = document.createElement("input");
      input.type = "number";
      input.min = meta.min;
      input.max = meta.max;
      if (meta.step) input.step = meta.step;
      input.value = meta.value;
      label.append(name, input);
      knobs.append(label);
      state.inputs[meta.key] = { input: input, meta: meta };
    }
    const run_ = document.createElement("button");
    run_.textContent = "пересобрать";
    run_.addEventListener("click", () => applyCard(state));
    knobs.append(run_);
  }
  node.__state = state;
  node.dataset.index = String(record.index);
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

function applyDetails(state, details) {
  state.frames = details.frames || 0;
  state.frame = 0;
  state.loaded = true;
  state.note.style.display = "none";
  state.line.textContent = metricLine(details) + "  ·  " + details.iterations + " итер.";
  state.badge.textContent = (details.status || "") + (details.reason ? " · " + details.reason : "");
  state.node.classList.toggle("stopped", details.status !== "success");
  state.legend.textContent = "";
  (details.legend || []).forEach(([rgb, label]) => {
    const span = document.createElement("span");
    const sw = document.createElement("i");
    sw.style.background = "rgb(" + rgb.join(",") + ")";
    span.append(sw, document.createTextNode(label));
    state.legend.append(span);
  });
  if (state.rule) state.rule.textContent = details.rule ? JSON.stringify(details.rule, null, 2) : "—";
  if (state.frames > 0) {
    state.img.src = frameUrl(state, 0);
    if (MODE !== "static") preload(state);
  }
}

async function load(state) {
  if (state.busy) return;
  if (MODE === "static") {
    applyDetails(state, state.record);
    return;
  }
  state.busy = true;
  state.note.style.display = "inline";
  state.note.textContent = "сборка…";
  try {
    const query = params(state);
    const response = await fetch(
      "/api/run/" + encodeURIComponent(state.run.name) + "/record/" + state.record.digest +
      "?" + (query ? query + "&" : "") + "v=" + BUILD
    );
    const details = await response.json();
    if (!response.ok || details.error) throw new Error(details.error || String(response.status));
    applyDetails(state, details);
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
  if (timer || MODE === "static" || !cards.length) return;
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
    if (entry.isIntersecting && !state.loaded) load(state);
  }
}, { rootMargin: "240px" });

function currentRun() { return RUNS[current]; }

function pageCount() {
  const total = currentRun() ? currentRun().cards.length : 0;
  return Math.max(1, Math.ceil(total / pageSize));
}

function syncUrl() {
  const run = currentRun();
  const search = new URLSearchParams();
  if (run) search.set("run", run.name);
  search.set("from", String(pageIndex * pageSize));
  search.set("size", String(pageSize));
  history.replaceState(null, "", location.pathname + "?" + search.toString());
}

function readUrl() {
  const search = new URLSearchParams(location.search);
  const name = search.get("run");
  const found = RUNS.findIndex((run) => run.name === name);
  current = found >= 0 ? found : 0;
  const size = Number(search.get("size"));
  pageSize = PAGE_SIZES.includes(size) ? size : 48;
  const from = Number(search.get("from"));
  if (Number.isFinite(from) && from > 0) {
    pageIndex = Math.floor(from / pageSize);
  } else {
    const page = Number(search.get("page"));
    pageIndex = Number.isFinite(page) && page >= 1 ? Math.floor(page) - 1 : 0;
  }
}

function renderPage() {
  pause();
  const run = currentRun();
  cards = [];
  el("grid").textContent = "";
  if (!run) {
    el("count").textContent = "нет прогонов";
    return;
  }
  const total = run.cards.length;
  const pages = pageCount();
  pageIndex = Math.min(Math.max(0, pageIndex), pages - 1);
  const slice = run.cards.slice(pageIndex * pageSize, pageIndex * pageSize + pageSize);
  for (const record of slice) {
    const state = makeCard(record, run);
    cards.push(state);
    el("grid").append(state.node);
    observer.observe(state.node);
  }
  el("count").textContent = total + " карточек" + (run.experiment ? " · " + run.experiment : "");
  el("pageNo").value = String(pageIndex + 1);
  el("pageNo").max = String(pages);
  el("pageInfo").textContent = "из " + pages + " · " + (pageIndex * pageSize + 1) + "–" +
    (pageIndex * pageSize + slice.length);
  el("prev").disabled = pageIndex <= 0;
  el("next").disabled = pageIndex >= pages - 1;
  syncUrl();
  window.scrollTo({ top: 0 });
  if (MODE !== "static") play();
}

function goTo(index) {
  pageIndex = Math.max(0, Math.min(pageCount() - 1, index));
  renderPage();
}

function highlight(index) {
  const node = document.querySelector('.card[data-index="' + index + '"]');
  if (!node) return;
  node.style.outline = "2px solid #3c74d8";
  node.scrollIntoView({ block: "center", behavior: "smooth" });
}

// Перейти к карточке по номеру или по префиксу digest — без листания с начала
function jumpTo(raw) {
  const text = String(raw || "").trim().toLowerCase();
  const run = currentRun();
  if (!text || !run) return;
  let index = -1;
  if (/^\d+$/.test(text)) {
    index = Math.min(run.cards.length - 1, Number(text));
  } else {
    index = run.cards.findIndex((card) => card.digest.startsWith(text));
  }
  if (index < 0) {
    el("jump").style.borderColor = "#a33";
    return;
  }
  el("jump").style.borderColor = "";
  pageIndex = Math.floor(index / pageSize);
  renderPage();
  highlight(index);
}

el("play").addEventListener("click", play);
el("pause").addEventListener("click", pause);
el("prev").addEventListener("click", () => goTo(pageIndex - 1));
el("next").addEventListener("click", () => goTo(pageIndex + 1));
el("pageNo").addEventListener("change", (event) => goTo(Number(event.target.value) - 1));
el("pageSize").addEventListener("change", (event) => {
  pageSize = Number(event.target.value) || 48;
  pageIndex = 0;
  renderPage();
});
el("runSel").addEventListener("change", (event) => {
  current = Number(event.target.value) || 0;
  pageIndex = 0;
  el("jump").value = "";
  renderPage();
});
el("jump").addEventListener("change", (event) => jumpTo(event.target.value));
el("jump").addEventListener("keydown", (event) => {
  if (event.code === "Enter") { event.preventDefault(); jumpTo(event.target.value); }
});
el("fps").addEventListener("input", (event) => {
  fps = Number(event.target.value);
  el("fpsVal").textContent = fps;
  if (timer) { pause(); play(); }
});
el("scale").addEventListener("input", (event) => {
  scale = Number(event.target.value);
  el("scaleVal").textContent = scale;
  if (MODE === "static") return;
  for (const state of cards) if (state.loaded && state.frames) state.img.src = frameUrl(state, state.frame);
});
el("applyAll").addEventListener("click", () => {
  const warm = el("warm").value, observe = el("obs").value, size = el("res").value;
  for (const state of cards) {
    if (warm !== "" && state.inputs["phase.warmup"]) state.inputs["phase.warmup"].input.value = warm;
    if (observe !== "" && state.inputs["phase.observe"]) state.inputs["phase.observe"].input.value = observe;
    if (size !== "" && state.inputs["grid2d.size"]) state.inputs["grid2d.size"].input.value = size;
    applyCard(state);
  }
  play();
});
document.addEventListener("keydown", (event) => {
  if (event.target !== document.body) return;
  if (event.code === "Space") { event.preventDefault(); timer ? pause() : play(); }
  if (event.code === "ArrowLeft") goTo(pageIndex - 1);
  if (event.code === "ArrowRight") goTo(pageIndex + 1);
});

if (MODE === "static") {
  for (const id of ["play", "pause", "fps", "scale"]) {
    const node = el(id);
    if (node && node.closest("label")) node.closest("label").style.display = "none";
    else if (node) node.style.display = "none";
  }
  for (const node of document.querySelectorAll(".knobs-only")) node.style.display = "none";
}

for (const [index, run] of RUNS.entries()) {
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = run.name + (run.cards.length ? " (" + run.cards.length + ")" : "");
  el("runSel").append(option);
}
for (const size of PAGE_SIZES) {
  const option = document.createElement("option");
  option.value = String(size);
  option.textContent = String(size);
  el("pageSize").append(option);
}
readUrl();
el("runSel").value = String(current);
el("pageSize").value = String(pageSize);
renderPage();
</script>
</body>
</html>
"""

__all__ = ["PAGE"]
