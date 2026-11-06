/**
 * Галерея статической сборки: карточки, страницы и живой прогон в браузере.
 *
 * Кадры не запрашиваются у сервера: у каждой карточки есть сжатые сетки
 * ``init``/``final`` и модуль эксперимента, поэтому рост можно проиграть
 * заново. Универсальный контрол даёт четыре вещи для любого эксперимента:
 *
 *   - цикл: после ``iterations`` шагов начать заново;
 *   - размер: сжать/растянуть инициализирующую сетку;
 *   - итерации: сколько шагов считать «обзором»;
 *   - итер/с: сколько шагов делать в секунду (1 — совсем медленно).
 *
 * Анимация идёт только у карточек в зоне видимости — вне экрана
 * итерирование выключается. Вся логика шага живёт в ``Runner`` из
 * ``runtime.js``; здесь только DOM и таймеры.
 */

(function () {
  "use strict";

  const DATA = window.SWEEP_DATA || { runs: [] };
  const RULES = window.SweepRules || { experiments: {} };
  const paintGrid = RULES.paintGrid;
  const finalGrid = RULES.finalGrid;
  const PAGE_SIZES = [12, 24, 48];
  const el = (id) => document.getElementById(id);

  let current = 0;
  let pageIndex = 0;
  let pageSize = PAGE_SIZES.includes(Number(DATA.pageSize)) ? Number(DATA.pageSize) : 24;
  let defaultSpeed = 10;
  let defaultLoop = true;
  let views = [];
  let modalView = null;
  let raf = 0;
  let lastTick = 0;
  const active = new Set();

  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function metricLine(record) {
    const metrics = record.metrics || {};
    const keys = ["alive_density", "zero_density", "occupied_states", "max_state_share"].filter((key) => key in metrics);
    return keys.map((key) => key + "=" + Number(metrics[key]).toFixed(3)).join("  ");
  }

  function run() {
    return DATA.runs[current];
  }

  // ------------------------------------------------------------------ карточки
  function createView(record) {
    const node = document.createElement("article");
    node.className = "card";
    node.dataset.index = String(record.index);
    node.innerHTML =
      '<div class="head"><b>#' + record.index + " · " + escapeHtml(record.digest.slice(0, 8)) + "</b>" +
      '<span class="badge">' + escapeHtml(record.status || "") + "</span></div>" +
      '<div class="shot"><canvas aria-label="automaton"></canvas><span class="note"></span></div>' +
      '<div class="line"></div>' +
      '<div class="controls">' +
      '<button class="play" title="play growth">▶</button>' +
      '<button class="reset" title="reset to initial frame">↺</button>' +
      '<label class="knob" title="restart after the goal"><input class="loop" type="checkbox"> loop</label>' +
      '<label class="knob">size <input class="size" type="number" min="8" max="512" step="1"></label>' +
      '<label class="knob">steps <input class="iters" type="number" min="1" step="1"></label>' +
      '<label class="knob">steps/s <input class="speed" type="number" min="0.1" max="1000" step="1"></label>' +
      '<button class="open" title="enlarge">⛶</button>' +
      '<span class="step"></span>' +
      "</div>" +
      '<details><summary>config and metrics</summary><div class="legend"></div><pre></pre></details>' +
      '<details><summary>rule</summary><pre class="rule"></pre></details>';

    const view = {
      record: record,
      node: node,
      canvas: node.querySelector("canvas"),
      note: node.querySelector(".note"),
      line: node.querySelector(".line"),
      step: node.querySelector(".step"),
      badge: node.querySelector(".badge"),
      playButton: node.querySelector(".play"),
      resetButton: node.querySelector(".reset"),
      loopInput: node.querySelector(".loop"),
      sizeInput: node.querySelector(".size"),
      itersInput: node.querySelector(".iters"),
      speedInput: node.querySelector(".speed"),
      legend: node.querySelector(".legend"),
      pre: node.querySelector("pre"),
      rule: node.querySelector("pre.rule"),
      runner: null,
      playing: false,
      visible: false,
      accum: 0,
    };

    view.line.textContent = metricLine(record) + "  ·  " + record.iterations + " steps";
    view.pre.textContent = Object.entries(record.values || {}).map(([key, value]) => key + " = " + value).join("\n");
    view.rule.textContent = record.rule ? JSON.stringify(record.rule, null, 2) : "—";
    (record.legend || []).forEach(([color, label]) => {
      const span = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.background = color;
      span.append(swatch, document.createTextNode(label));
      view.legend.append(span);
    });

    view.runner = RULES.createRunner
      ? RULES.createRunner(record, {
          iterations: record.iterations,
          speed: defaultSpeed,
          loop: defaultLoop,
        })
      : null;

    if (!view.runner) {
      view.note.textContent = "no web module";
      view.playButton.disabled = true;
      view.sizeInput.disabled = true;
      view.itersInput.disabled = true;
      view.speedInput.disabled = true;
      view.loopInput.disabled = true;
      view.playButton.title = "this experiment has no sweep/experiments/<name>/web.js";
      view.step.textContent = record.iterations + " steps";
    } else {
      syncControls(view);
      paintFinal(view);
    }

    view.playButton.addEventListener("click", () => toggleView(view));
    view.resetButton.addEventListener("click", () => resetView(view, true));
    node.querySelector(".open").addEventListener("click", () => openModal(view));
    view.loopInput.addEventListener("change", () => {
      if (!view.runner) return;
      view.runner.loop = view.loopInput.checked;
      updateStep(view);
      if (modalView === view) syncModal(view);
    });
    view.sizeInput.addEventListener("change", () => {
      if (!view.runner) return;
      applySize(view, view.sizeInput.value);
    });
    view.itersInput.addEventListener("change", () => {
      if (!view.runner) return;
      applyIterations(view, view.itersInput.value);
    });
    view.speedInput.addEventListener("change", () => {
      if (!view.runner) return;
      view.runner.setSpeed(view.speedInput.value);
      syncControls(view);
      if (modalView === view) syncModal(view);
    });
    return view;
  }

  function syncControls(view) {
    const runner = view.runner;
    if (!runner) return;
    const focused = document.activeElement;
    if (focused !== view.sizeInput) view.sizeInput.value = String(runner.width);
    if (focused !== view.itersInput) view.itersInput.value = String(runner.iterations);
    if (focused !== view.speedInput) view.speedInput.value = String(runner.speed);
    view.loopInput.checked = Boolean(runner.loop);
    updateStep(view);
  }

  function updateStep(view) {
    if (!view.runner) return;
    const runner = view.runner;
    const tail = runner.loop && runner.done ? " ⟲" : "";
    view.step.textContent = runner.iteration + " / " + runner.iterations + " steps" + tail;
  }

  function paintFinal(view) {
    paintGrid(view.record, view.canvas, finalGrid(view.record), view.record.width, view.record.height);
    view.node.classList.remove("live");
    view.badge.textContent = view.record.status || "";
    view.step.textContent = view.record.iterations + " steps";
    if (modalView === view) paintModal(view);
  }

  function paintRunner(view) {
    const runner = view.runner;
    if (!runner || !runner.state) return;
    paintGrid(runner.card, view.canvas, runner.state.grid, runner.width, runner.height);
    view.node.classList.add("live");
    view.badge.textContent = "live";
    updateStep(view);
    if (modalView === view) paintModal(view);
  }

  function toggleView(view) {
    if (!view.runner) return;
    if (view.playing) stopView(view);
    else startView(view);
  }

  function startView(view) {
    const runner = view.runner;
    if (!runner) return;
    // после последнего шага ▶ начинает обзор заново
    if (runner.done) runner.reset();
    view.playing = true;
    view.accum = 0;
    view.playButton.textContent = "⏸";
    if (modalView === view) el("mplay").textContent = "⏸";
    active.add(view);
    paintRunner(view);
    schedule();
  }

  function stopView(view) {
    view.playing = false;
    view.playButton.textContent = "▶";
    active.delete(view);
    if (modalView === view) el("mplay").textContent = "▶";
  }

  function resetView(view, repaint) {
    stopView(view);
    if (!view.runner) return;
    view.runner.reset();
    view.accum = 0;
    syncControls(view);
    if (repaint !== false) paintRunner(view);
    if (modalView === view) {
      syncModal(view);
      paintModal(view);
    }
  }

  function applySize(view, raw) {
    if (!view.runner) return;
    const wasPlaying = view.playing;
    stopView(view);
    view.runner.setSize(raw);
    view.accum = 0;
    syncControls(view);
    paintRunner(view);
    if (modalView === view) syncModal(view);
    if (wasPlaying) startView(view);
  }

  function applyIterations(view, raw) {
    if (!view.runner) return;
    view.runner.setIterations(raw);
    syncControls(view);
    if (view.playing && !view.runner.loop && view.runner.done) stopView(view);
    if (modalView === view) syncModal(view);
  }

  // --------------------------------------------------------------- анимация
  function schedule() {
    if (raf) return;
    lastTick = 0;
    raf = requestAnimationFrame(frame);
  }

  function frame(now) {
    raf = 0;
    if (!lastTick) lastTick = now;
    const dt = Math.min(0.25, Math.max(0, (now - lastTick) / 1000));
    lastTick = now;
    const running = [...active].filter(
      (view) => view.playing && (view.visible || modalView === view)
    );
    if (!running.length) return;

    for (const view of running) {
      const runner = view.runner;
      view.accum += dt * runner.speed;
      const steps = Math.floor(view.accum);
      if (steps <= 0) continue;
      // защита от зависания при большой скорости и вкладке в фоне
      const consumed = Math.min(steps, 20000);
      view.accum -= consumed;
      runner.advance(consumed);
      paintRunner(view);
      if (!runner.loop && runner.done) stopView(view);
    }
    if ([...active].some((view) => view.playing && (view.visible || modalView === view))) {
      raf = requestAnimationFrame(frame);
    }
  }

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const view = entry.target.__view;
      if (!view) continue;
      view.visible = entry.isIntersecting;
      view.note.textContent = entry.isIntersecting ? "" : "off screen";
      if (entry.isIntersecting && view.playing) schedule();
    }
  }, { rootMargin: "160px" });

  // ------------------------------------------------------------------ крупно
  function paintModal(view) {
    if (!view || !view.runner || !view.runner.state) return;
    const runner = view.runner;
    paintGrid(runner.card, el("mcanvas"), runner.state.grid, runner.width, runner.height);
    el("mstep").textContent = runner.iteration + " / " + runner.iterations + " steps" +
      (runner.loop && runner.done ? " ⟲" : "");
  }

  function syncModal(view) {
    if (!view || !view.runner) return;
    const runner = view.runner;
    if (document.activeElement !== el("msize")) el("msize").value = String(runner.width);
    if (document.activeElement !== el("miters")) el("miters").value = String(runner.iterations);
    if (document.activeElement !== el("mspeed")) el("mspeed").value = String(runner.speed);
    el("mloop").checked = Boolean(runner.loop);
  }

  function openModal(view) {
    if (!view.runner) return;
    modalView = view;
    el("modal").hidden = false;
    el("mtitle").textContent = "#" + view.record.index + " · " + view.record.digest.slice(0, 8) +
      " · " + view.record.experiment;
    syncModal(view);
    paintModal(view);
    el("mplay").textContent = view.playing ? "⏸" : "▶";
    schedule();
  }

  function closeModal() {
    if (modalView && modalView.playing) stopView(modalView);
    el("modal").hidden = true;
    modalView = null;
  }

  // ----------------------------------------------------------- страницы/адрес
  function pageCount() {
    const cards = run() ? run().cards : [];
    return Math.max(1, Math.ceil(cards.length / pageSize));
  }

  function syncUrl() {
    const search = new URLSearchParams();
    if (run()) search.set("run", run().name);
    search.set("from", String(pageIndex * pageSize));
    search.set("size", String(pageSize));
    search.set("speed", String(defaultSpeed));
    search.set("loop", defaultLoop ? "1" : "0");
    try {
      history.replaceState(null, "", location.pathname + "?" + search.toString());
    } catch (error) {
      // file:// и строгие политики могут запретить replaceState — не критично
    }
  }

  function readUrl() {
    const search = new URLSearchParams(location.search);
    const name = search.get("run");
    const found = DATA.runs.findIndex((item) => item.name === name);
    current = found >= 0 ? found : 0;
    const size = Number(search.get("size"));
    if (PAGE_SIZES.includes(size)) pageSize = size;
    const from = Number(search.get("from"));
    pageIndex = Number.isFinite(from) && from > 0 ? Math.floor(from / pageSize) : 0;
    const speed = Number(search.get("speed"));
    if (Number.isFinite(speed) && speed > 0) defaultSpeed = Math.max(0.1, Math.min(1000, speed));
    defaultLoop = search.get("loop") !== "0";
  }

  function renderPage() {
    for (const view of views) {
      stopView(view);
      observer.unobserve(view.node);
    }
    active.clear();
    if (raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
    views = [];
    el("grid").textContent = "";
    const target = run();
    if (!target) {
      el("count").textContent = "no runs";
      return;
    }
    const pages = pageCount();
    pageIndex = Math.min(Math.max(0, pageIndex), pages - 1);
    const slice = target.cards.slice(pageIndex * pageSize, pageIndex * pageSize + pageSize);
    for (const record of slice) {
      const view = createView(record);
      view.node.__view = view;
      views.push(view);
      el("grid").append(view.node);
      observer.observe(view.node);
    }
    el("count").textContent = target.cards.length + " cards · " + (target.experiment || "");
    el("pageNo").value = String(pageIndex + 1);
    el("pageNo").max = String(pages);
    el("pageInfo").textContent = "of " + pages + " · " + (pageIndex * pageSize + 1) + "–" +
      (pageIndex * pageSize + slice.length);
    el("prev").disabled = pageIndex <= 0;
    el("next").disabled = pageIndex >= pages - 1;
    syncUrl();
    window.scrollTo({ top: 0 });
    // страница открывается уже анимированной: карточки стартуют сразу
    if (views.some((view) => view.runner)) playAll(true);
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

  function jumpTo(raw) {
    const text = String(raw || "").trim().toLowerCase();
    const target = run();
    if (!text || !target) return;
    let index = -1;
    if (/^\d+$/.test(text)) index = Math.min(target.cards.length - 1, Number(text));
    else index = target.cards.findIndex((card) => card.digest.startsWith(text));
    if (index < 0) {
      el("jump").style.borderColor = "#a33";
      return;
    }
    el("jump").style.borderColor = "";
    pageIndex = Math.floor(index / pageSize);
    renderPage();
    highlight(index);
  }

  function playAll(enable) {
    for (const view of views) {
      if (enable && view.runner) startView(view);
      else stopView(view);
    }
    el("playAll").textContent = enable ? "⏸ all" : "▶ all";
  }

  function applyAll() {
    defaultSpeed = Number(el("speedAll").value) || defaultSpeed;
    defaultLoop = el("loopAll").checked;
    const sizeRaw = el("sizeAll").value;
    const itersRaw = el("itersAll").value;
    for (const view of views) {
      if (!view.runner) continue;
      if (sizeRaw !== "") view.runner.setSize(sizeRaw);
      if (itersRaw !== "") view.runner.setIterations(itersRaw);
      view.runner.setSpeed(defaultSpeed);
      view.runner.loop = defaultLoop;
      view.accum = 0;
      syncControls(view);
      paintRunner(view);
    }
    if (modalView) {
      syncModal(modalView);
      paintModal(modalView);
    }
    syncUrl();
  }

  // ---------------------------------------------------------------- события
  el("playAll").addEventListener("click", () => {
    const anyPlaying = views.some((view) => view.playing);
    playAll(!anyPlaying);
  });
  el("pauseAll").addEventListener("click", () => playAll(false));
  el("applyAll").addEventListener("click", applyAll);
  el("speedAll").addEventListener("change", applyAll);
  el("loopAll").addEventListener("change", applyAll);
  el("prev").addEventListener("click", () => goTo(pageIndex - 1));
  el("next").addEventListener("click", () => goTo(pageIndex + 1));
  el("pageNo").addEventListener("change", (event) => goTo(Number(event.target.value) - 1));
  el("pageSize").addEventListener("change", (event) => {
    pageSize = Number(event.target.value) || 24;
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
    if (event.code === "Enter") {
      event.preventDefault();
      jumpTo(event.target.value);
    }
  });

  el("mclose").addEventListener("click", closeModal);
  el("modal").addEventListener("click", (event) => {
    if (event.target === el("modal")) closeModal();
  });
  el("mplay").addEventListener("click", () => {
    if (!modalView) return;
    toggleView(modalView);
    el("mplay").textContent = modalView.playing ? "⏸" : "▶";
  });
  el("mrestart").addEventListener("click", () => {
    if (!modalView) return;
    resetView(modalView, true);
    startView(modalView);
    el("mplay").textContent = "⏸";
  });
  el("mloop").addEventListener("change", () => {
    if (!modalView || !modalView.runner) return;
    modalView.runner.loop = el("mloop").checked;
    syncControls(modalView);
    updateStep(modalView);
  });
  el("msize").addEventListener("change", () => {
    if (!modalView) return;
    applySize(modalView, el("msize").value);
  });
  el("miters").addEventListener("change", () => {
    if (!modalView) return;
    applyIterations(modalView, el("miters").value);
  });
  el("mspeed").addEventListener("change", () => {
    if (!modalView || !modalView.runner) return;
    modalView.runner.setSpeed(el("mspeed").value);
    syncControls(modalView);
    syncModal(modalView);
  });
  document.addEventListener("keydown", (event) => {
    if (event.code === "Escape" && !el("modal").hidden) closeModal();
    if (event.target !== document.body) return;
    if (event.code === "Space") {
      event.preventDefault();
      el("playAll").click();
    }
    if (event.code === "ArrowLeft") goTo(pageIndex - 1);
    if (event.code === "ArrowRight") goTo(pageIndex + 1);
  });

  // ------------------------------------------------------------------ старт
  for (const [index, item] of DATA.runs.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = item.name + " (" + item.cards.length + ")";
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
  el("speedAll").value = String(defaultSpeed);
  el("loopAll").checked = defaultLoop;
  renderPage();
})();
