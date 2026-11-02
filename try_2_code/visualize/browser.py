"""Браузерная галерея: много автоматов на одной странице.

В отличие от терминального ``visualize.live``, здесь каждый автомат — это
небольшая карточка с анимацией. Все кадры встроены в один HTML, поэтому
страницу можно открыть без сервера и без Python.

Визуализатор выбирается по модулям собранного автомата
(:func:`visualize.protocol.resolve_visualizer`): правило с фазами покажет
палитру ``hue``, старение — ``age``, а модуль без своего визуализатора
получит общий 2D-стандарт.
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from casweep.fixed import Fixed
from casweep.module import Role
from casweep.trial import run_trial

from .png import encode_png
from .protocol import Visualizer, resolve_visualizer, _coerce, Standard2DVisualizer

_LABELS: tuple[tuple[str, str], ...] = (
    ("neighborhood.r1", "r1"),
    ("neighborhood.r2", "r2"),
    ("neighborhood.size", "size"),
    ("neighborhood.radius", "r"),
    ("data.width", "w"),
    ("data.height", "h"),
    ("data.states", "S"),
    ("rule.rot", "rot"),
    ("rule.born_inner", "bi"),
    ("rule.born_total", "bt"),
    ("rule.survive_lo", "lo"),
    ("rule.survive_hi", "hi"),
    ("rule.threshold", "thr"),
    ("rule.born_above", "born"),
)


@dataclass
class BrowserItem:
    """Один автомат в браузерной галерее."""

    title: str
    digest: str
    seed: int
    summary: str
    scheme: str
    visualizer: str
    status: str
    reason: str
    frames: list[np.ndarray] = field(default_factory=list)
    fps: float = 4.0


def describe_config(fixed: Fixed, *, max_parts: int = 8) -> str:
    """Короткая подпись конфига для карточки."""

    values = fixed.to_dict()
    parts: list[str] = []
    for key, label in _LABELS:
        if key in values:
            parts.append(f"{label}={values[key]}")
        if len(parts) >= max_parts:
            break
    return " ".join(parts) or fixed.digest()


def _data_uri(frame: np.ndarray) -> str:
    payload = base64.b64encode(encode_png(frame)).decode("ascii")
    return "data:image/png;base64," + payload


def _sample_frames(frames: list[np.ndarray], limit: int) -> list[np.ndarray]:
    if limit <= 0 or len(frames) <= limit:
        return frames
    indexes = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return [frames[int(index)] for index in indexes]


def _chosen_visualizer(
    experiment: Any, fixed: Fixed, visualizer: Visualizer | None = None
) -> Visualizer:
    """Выбрать визуализатор, уважая протокол ``supported`` модулей."""

    if visualizer is not None:
        return visualizer

    assembled = experiment.assemble(fixed)
    modules = getattr(assembled, "modules", {}) or {}

    # ИщемRule с visualizer, у которого есть поддержка роли rule
    for role in (Role.RULE, Role.NEIGHBORHOOD):
        mod = modules.get(role.value) if isinstance(modules, dict) else modules.get(role)
        if mod is not None and hasattr(mod, "visualizer"):
            v = mod.visualizer
            if v is not None:
                coerced = _coerce(v)
                if coerced is not None and Role.RULE in getattr(coerced, "supported", ()):
                    return coerced

    chosen = resolve_visualizer(assembled)
    # Если выбранный визуализатор не поддерживает ни одну роль — используем стандарт
    if chosen.supported and not any(
        role in chosen.supported for role in (Role.RULE, Role.NEIGHBORHOOD)
    ):
        return Standard2DVisualizer()
    return chosen


def capture_item(
    experiment: Any,
    fixed: Fixed,
    *,
    seed: int = 0,
    steps: int = 24,
    every: int = 2,
    scale: int = 4,
    scheme: str | None = None,
    max_frames: int = 32,
    fps: float = 4.0,
    visualizer: Visualizer | None = None,
) -> BrowserItem:
    """Прогонять конфиг и собрать кадры для одной карточки."""

    chosen = _chosen_visualizer(experiment, fixed, visualizer)
    actual_scheme = chosen.scheme if scheme in (None, "auto") else scheme
    # Ограничиваем кадры, чтобы не блокировать браузер
    effective_max = min(max_frames, 150)
    frames: list[np.ndarray] = []
    counter = {"value": 0}

    def on_frame(phase: str, step: int, grid: Any) -> None:
        index = counter["value"]
        counter["value"] += 1
        if every <= 1 or index % every == 0:
            frames.append(
                chosen.render_frame(grid, scale=scale, scheme=actual_scheme)
            )

    trial = run_trial(
        experiment,
        fixed,
        seed=seed,
        keep_state=True,
        on_frame=on_frame,
        max_iterations=steps,
    )
    if not frames and trial.state is not None:
        frames.append(
            chosen.render_frame(trial.state, scale=scale, scheme=actual_scheme)
        )
    frames = _sample_frames(frames, effective_max)
    return BrowserItem(
        title=f"{experiment.name} · {fixed.digest()[:8]}",
        digest=fixed.digest(),
        seed=seed,
        summary=describe_config(fixed),
        scheme=actual_scheme,
        visualizer=chosen.describe(),
        status=trial.status.value,
        reason=trial.reason,
        frames=frames,
        fps=fps,
    )


def capture_items(
    experiment: Any,
    entries: Iterable[tuple[Fixed, int]],
    *,
    steps: int = 24,
    every: int = 2,
    scale: int = 4,
    scheme: str | None = None,
    max_frames: int = 32,
    fps: float = 4.0,
    on_error: Callable[[Fixed, Exception], None] | None = None,
) -> list[BrowserItem]:
    """Собрать карточки для всех конфигов; сломанный конфиг не роняет галерею."""

    items: list[BrowserItem] = []
    for fixed, seed in entries:
        try:
            items.append(
                capture_item(
                    experiment,
                    fixed,
                    seed=seed,
                    steps=steps,
                    every=every,
                    scale=scale,
                    scheme=scheme,
                    max_frames=max_frames,
                    fps=fps,
                )
            )
        except Exception as exc:  # noqa: BLE001 - галерея не должна падать целиком
            if on_error is not None:
                on_error(fixed, exc)
    return items


_STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: #0b0e13; color: #d8dee9;
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 650; }
.sub { color: #8b98a9; margin-bottom: 18px; }
.bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 20px; }
button, select {
  background: #182030; color: #d8dee9; border: 1px solid #2a3648;
  border-radius: 8px; padding: 7px 12px; font: inherit; cursor: pointer;
}
button:hover, select:hover { background: #1f2a3d; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
.card { background: #10161f; border: 1px solid #1d2634; border-radius: 12px; padding: 12px; }
.head { display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 8px; }
.digest { color: #cdd6e4; font-family: ui-monospace, monospace; font-size: 12px; }
.badge { background: #141b26; border: 1px solid #222d3d; border-radius: 999px; padding: 2px 9px; color: #9aa7b8; font-size: 12px; }
.frame { display: block; width: 100%; image-rendering: pixelated; border-radius: 8px; background: #0c1016; cursor: zoom-in; }
.card.zoom { position: fixed; inset: 2vh 2vw; z-index: 10; overflow: auto; box-shadow: 0 24px 80px #000c; }
.card.zoom .frame { cursor: zoom-out; max-height: 70vh; object-fit: contain; }
.controls { display: flex; gap: 8px; align-items: center; margin-top: 9px; }
.controls input[type=range] { flex: 1 1 110px; accent-color: #d69b3c; }
.counter { color: #8b98a9; font-size: 12px; min-width: 52px; text-align: right; }
.summary { color: #9aa7b8; font-size: 12px; margin-top: 8px; }
.reason { color: #d69b3c; font-size: 12px; margin-top: 4px; }
.empty { color: #8b98a9; padding: 40px; text-align: center; }
"""


def browser_html(
    items: Sequence[BrowserItem],
    *,
    title: str = "Автоматы",
    subtitle: str = "",
    fps: float = 4.0,
) -> str:
    """Самодостаточная HTML-галерея с анимацией всех карточек."""

    payload = []
    cards = []
    for item in items:
        uris = [_data_uri(frame) for frame in item.frames]
        if not uris:
            uris = [""]
        payload.append(
            {
                "frames": uris,
                "fps": item.fps or fps,
                "title": item.title,
            }
        )
        reason = (
            f'<div class="reason">{html.escape(item.reason)}</div>'
            if item.reason
            else ""
        )
        cards.append(
            f"""<article class="card">
  <div class="head">
    <span class="digest">{html.escape(item.digest[:8])}</span>
    <span class="badge">{html.escape(item.status)}</span>
  </div>
  <img class="frame" alt="автомат {html.escape(item.digest[:8])}">
  <div class="controls">
    <input class="scrub" type="range" min="0" max="{max(0, len(item.frames) - 1)}" value="0">
    <span class="counter">1 / {max(1, len(item.frames))}</span>
  </div>
  <div class="summary">{html.escape(item.summary)}</div>
  <div class="summary">{html.escape(item.visualizer)} · seed {item.seed}</div>
  {reason}
</article>"""
        )

    body = "\n".join(cards) or '<div class="empty">Нет автоматов для показа.</div>'
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="sub">{html.escape(subtitle)}</div>
<div class="bar">
  <button id="toggle">❚❚ пауза</button>
  <select id="speed" title="скорость">
    <option value="1">1 кадр/с</option>
    <option value="2">2 кадра/с</option>
    <option value="4" selected>4 кадра/с</option>
    <option value="8">8 кадров/с</option>
    <option value="12">12 кадров/с</option>
  </select>
  <span class="sub" style="margin:0">карточек: {len(items)}</span>
</div>
<div class="cards">
{body}
</div>
<script>
const ITEMS = {data};
const cards = Array.from(document.querySelectorAll('.card'));
let tick = 0, playing = true, timer = null;
const speed = document.getElementById('speed');
const toggle = document.getElementById('toggle');
function render() {{
  cards.forEach((card, index) => {{
    const item = ITEMS[index];
    if (!item || !item.frames.length) return;
    const image = card.querySelector('.frame');
    const scrub = card.querySelector('.scrub');
    const counter = card.querySelector('.counter');
    const offset = Number(card.dataset.offset || 0);
    const current = (tick + offset) % item.frames.length;
    image.src = item.frames[current];
    scrub.value = current;
    counter.textContent = (current + 1) + ' / ' + item.frames.length;
  }});
}}
function start() {{
  stop();
  timer = setInterval(() => {{ tick += 1; render(); }}, 1000 / Number(speed.value));
}}
function stop() {{
  if (timer !== null) {{ clearInterval(timer); timer = null; }}
}}
toggle.addEventListener('click', () => {{
  playing = !playing;
  toggle.textContent = playing ? '❚❚ пауза' : '▶ играть';
  if (playing) start(); else stop();
}});
speed.addEventListener('change', () => {{ if (playing) start(); }});
cards.forEach((card) => {{
  const scrub = card.querySelector('.scrub');
  scrub.addEventListener('input', () => {{ card.dataset.offset = String(Number(scrub.value) - tick); render(); }});
  card.querySelector('.frame').addEventListener('click', () => card.classList.toggle('zoom'));
}});
render();
start();
</script>
</body>
</html>
"""



def write_browser(
    path: str | Path,
    items: Sequence[BrowserItem],
    *,
    title: str = "Автоматы",
    subtitle: str = "",
    fps: float = 4.0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        browser_html(items, title=title, subtitle=subtitle, fps=fps),
        encoding="utf-8",
    )
    return path