"""HTML-вьюер: один автомат с анимацией и галерея по сохранённым успехам.

Вьюер самодостаточен: кадры встроены как base64 PNG, внешних скриптов и
шрифтов нет. Он использует те же пиксели, что и обычный PNG-рендер, поэтому
картинка в браузере совпадает с файлами.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .png import encode_png
from .standard import BACKGROUND, palette

_STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: #0b0e13; color: #d8dee9;
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 15px; margin: 24px 0 8px; color: #9aa7b8; font-weight: 600; }
.sub { color: #8b98a9; margin-bottom: 18px; }
.stage { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
canvas, .frame { image-rendering: pixelated; border-radius: 8px; background: #0c1016; }
.frame { display: block; max-width: 100%; height: auto; box-shadow: 0 0 0 1px #1e2633, 0 16px 40px #0008; }
.controls { display: flex; align-items: center; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
button, select {
  background: #182030; color: #d8dee9; border: 1px solid #2a3648;
  border-radius: 8px; padding: 7px 12px; font: inherit; cursor: pointer;
}
button:hover, select:hover { background: #1f2a3d; }
input[type=range] { flex: 1 1 220px; accent-color: #d69b3c; }
code { color: #cdd6e4; background: #141b26; padding: 2px 5px; border-radius: 4px; }
.legend { display: flex; gap: 6px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
.swatch { width: 16px; height: 16px; border-radius: 4px; box-shadow: 0 0 0 1px #1e2633; }
.badge { display: inline-block; background: #141b26; border: 1px solid #222d3d; border-radius: 999px; padding: 2px 10px; margin: 0 6px 6px 0; color: #9aa7b8; }
pre { background: #10161f; border: 1px solid #1d2634; border-radius: 8px; padding: 12px; overflow: auto; font-size: 12px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }
.card { background: #10161f; border: 1px solid #1d2634; border-radius: 10px; padding: 12px; }
.card img { width: 100%; image-rendering: pixelated; border-radius: 6px; background: #0c1016; }
.card a { color: #e2b25c; text-decoration: none; }
.card a:hover { text-decoration: underline; }
"""


def _data_uri(pixels: np.ndarray) -> str:
    payload = base64.b64encode(encode_png(pixels)).decode("ascii")
    return "data:image/png;base64," + payload


def _legend_html(states: int, zero_state: int, scheme: str) -> str:
    colors = palette(states, zero_state, scheme)
    chips = []
    for state in range(states):
        red, green, blue = (int(value) for value in colors[state])
        name = "мёртвые" if state == zero_state else f"состояние {state}"
        chips.append(
            f'<span class="swatch" style="background: rgb({red},{green},{blue})" '
            f'title="{html.escape(name)}"></span>'
        )
    labels = " ".join(
        f'<code>{index}</code>' for index in range(states)
    )
    return f'<div class="legend">{"".join(chips)} <span style="margin-left:8px">{labels}</span></div>'


def viewer_html(
    frames: Iterable[np.ndarray],
    *,
    title: str,
    subtitle: str = "",
    config: Any = None,
    metrics: Any = None,
    states: int = 2,
    zero_state: int = 0,
    scheme: str = "age",
    fps: float = 4.0,
) -> str:
    """Собрать HTML одного автомата (кадры — уже отрисованные RGB-массивы)."""

    uris = [_data_uri(frame) for frame in frames]
    if not uris:
        uris = [
            _data_uri(
                np.zeros((16, 16, 3), dtype=np.uint8)
                + np.array(BACKGROUND, dtype=np.uint8)
            )
        ]
    config_text = json.dumps(config, ensure_ascii=False, indent=2, default=str) if config else ""
    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2, default=str) if metrics else ""
    extra = ""
    if config_text:
        extra += f"<h2>Конфиг</h2><pre>{html.escape(config_text)}</pre>"
    if metrics_text:
        extra += f"<h2>Метрики последнего кадра</h2><pre>{html.escape(metrics_text)}</pre>"

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
<div class="stage">
  <div>
    <img id="frame" class="frame" alt="кадр автомата">
    <div class="controls">
      <button id="play">▶ играть</button>
      <input id="slider" type="range" min="0" max="{len(uris) - 1}" value="0">
      <span id="counter">1 / {len(uris)}</span>
      <select id="speed" title="скорость">
        <option value="1">1 кадр/с</option>
        <option value="2">2 кадра/с</option>
        <option value="4" selected>4 кадра/с</option>
        <option value="8">8 кадров/с</option>
      </select>
    </div>
    {_legend_html(int(states), int(zero_state), scheme)}
  </div>
  <div style="flex:1 1 320px; min-width:260px">
    <div>
      <span class="badge">кадров: {len(uris)}</span>
      <span class="badge">схема: {html.escape(scheme)}</span>
    </div>
    {extra}
  </div>
</div>
<script>
const FRAMES = {json.dumps(uris)};
let index = 0, timer = null;
const image = document.getElementById('frame');
const slider = document.getElementById('slider');
const counter = document.getElementById('counter');
const button = document.getElementById('play');
const speed = document.getElementById('speed');
function show(next) {{
  index = (next + FRAMES.length) % FRAMES.length;
  image.src = FRAMES[index];
  slider.value = index;
  counter.textContent = (index + 1) + ' / ' + FRAMES.length;
}}
function stop() {{
  if (timer !== null) {{ clearInterval(timer); timer = null; }}
  button.textContent = '▶ играть';
}}
function play() {{
  stop();
  timer = setInterval(() => show(index + 1), 1000 / Number(speed.value));
  button.textContent = '❚❚ пауза';
}}
button.addEventListener('click', () => timer === null ? play() : stop());
slider.addEventListener('input', () => {{ stop(); show(Number(slider.value)); }});
speed.addEventListener('change', () => {{ if (timer !== null) play(); }});
show(0);
</script>
</body>
</html>
"""


def write_viewer(path: str | Path, frames: Iterable[np.ndarray], **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(viewer_html(frames, **kwargs), encoding="utf-8")
    return path


def gallery_html(items: list[dict[str, Any]], *, title: str = "Галерея автоматов") -> str:
    """Галерея: карточка на автомат со ссылкой на его вьюер."""

    cards = []
    for item in items:
        image = item.get("image")
        uri = _data_uri(image) if image is not None else ""
        digest = html.escape(str(item.get("digest", "")))
        seed = item.get("seed", 0)
        summary = item.get("summary", "")
        viewer = item.get("viewer")
        link = (
            f'<a href="{html.escape(str(viewer))}">открыть анимацию →</a>'
            if viewer
            else ""
        )
        cards.append(
            f"""<div class="card">
  <img src="{uri}" alt="автомат {digest}">
  <div><code>{digest}</code></div>
  <div class="sub" style="margin:6px 0">{html.escape(str(summary))}</div>
  <div><span class="badge">seed {seed}</span>{link}</div>
</div>"""
        )
    body = "\n".join(cards) or '<div class="sub">Нет сохранённых автоматов.</div>'
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
<div class="sub">Каждый автомат отрисован общим 2D-стандартом; ссылка открывает его анимацию.</div>
<div class="cards">
{body}
</div>
</body>
</html>
"""


def write_gallery(path: str | Path, items: list[dict[str, Any]], **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gallery_html(items, **kwargs), encoding="utf-8")
    return path
