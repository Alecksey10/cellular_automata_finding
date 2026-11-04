"""Раскладка кадров в сетку и запись PNG/HTML."""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .interface import Rendered, Visualizer
from .png import write_png

GAP = 6


def colorize(data: np.ndarray, visualizer: Visualizer, states: int) -> np.ndarray:
    height, width = data.shape
    out = np.empty((height, width, 3), dtype=np.uint8)
    for state in range(states):
        mask = data == state
        if mask.any():
            out[mask] = visualizer.state_color(state, states)
    # состояния вне диапазона
    extra = data >= states
    if extra.any():
        out[extra] = visualizer.state_color(states - 1, states)
    return out


def scale_nearest(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def compose(
    frames: Sequence[np.ndarray],
    visualizer: Visualizer,
    *,
    states: int,
    cols: int = 4,
    scale: int = 4,
) -> np.ndarray:
    if not frames:
        raise ValueError("нет кадров для визуализации")
    cols = max(1, min(cols, len(frames)))
    rows = math.ceil(len(frames) / cols)
    tiles = [scale_nearest(colorize(frame, visualizer, states), scale) for frame in frames]
    tile_h, tile_w = tiles[0].shape[:2]
    background = np.array(visualizer.background(), dtype=np.uint8)
    width = cols * tile_w + (cols + 1) * GAP
    height = rows * tile_h + (rows + 1) * GAP
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = background
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        y = GAP + row * (tile_h + GAP)
        x = GAP + col * (tile_w + GAP)
        canvas[y : y + tile_h, x : x + tile_w] = tile
    return canvas


def render(
    frames: Sequence[np.ndarray],
    visualizer: Visualizer,
    *,
    captions: Sequence[str] | None = None,
    states: int,
    out_dir: str | Path,
    name: str,
    cols: int = 4,
    scale: int = 4,
    write_html: bool = True,
) -> Rendered:
    canvas = compose(frames, visualizer, states=states, cols=cols, scale=scale)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = write_png(out / f"{name}.png", canvas)

    html_path: Path | None = None
    if write_html:
        encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
        legend = "".join(
            f'<span class="sw" style="background:rgb{color}"></span>{html.escape(label)} '
            for color, label in visualizer.legend(states)
        )
        caption_html = ""
        if captions:
            caption_html = "<ol>" + "".join(f"<li>{html.escape(text)}</li>" for text in captions) + "</ol>"
        document = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(name)}</title>
<style>
body {{ background:#111; color:#ddd; font:14px system-ui, sans-serif; margin:24px; }}
img {{ image-rendering: pixelated; max-width:100%; border:1px solid #333; }}
.sw {{ display:inline-block; width:12px; height:12px; margin:0 6px 0 14px; vertical-align:-1px; }}
ol {{ columns: 4; }}
</style>
<h1>{html.escape(name)}</h1>
<div>{legend}</div>
<img src="data:image/png;base64,{encoded}" alt="{html.escape(name)}">
{caption_html}
"""
        html_path = out / f"{name}.html"
        html_path.write_text(document, encoding="utf-8")
    return Rendered(png=png_path, html=html_path)


def render_grid(
    frames: Sequence[np.ndarray],
    visualizer: Visualizer,
    *,
    states: int,
    out_dir: str | Path,
    name: str,
    cols: int = 4,
    scale: int = 4,
    captions: Sequence[str] | None = None,
    write_html: bool = True,
) -> Rendered:
    return render(
        frames,
        visualizer,
        captions=captions,
        states=states,
        out_dir=out_dir,
        name=name,
        cols=cols,
        scale=scale,
        write_html=write_html,
    )


__all__ = ["colorize", "compose", "render", "render_grid", "scale_nearest"]
