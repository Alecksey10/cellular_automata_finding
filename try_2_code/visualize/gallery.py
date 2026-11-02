"""Сохранение прогона одного автомата в общем 2D-стандарте.

На выходе — отдельные PNG-кадры, ``final.png``, лента, контактный лист и
самодостаточный ``viewer.html`` с анимацией. Все картинки строятся одной
функцией :func:`visualize.standard.render_frame`, поэтому совпадают.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from casweep.fixed import Fixed
from casweep.trial import run_trial

from .png import write_png
from .protocol import resolve_visualizer
from .standard import save_sheet, save_strip
from .viewer import write_viewer


@dataclass
class Capture:
    """Что получилось из одного прогона."""

    frame_paths: list[Path] = field(default_factory=list)
    final: Path | None = None
    strip: Path | None = None
    sheet: Path | None = None
    viewer: Path | None = None
    record: dict = field(default_factory=dict)
    #: Пиксели последнего кадра — для галереи без повторного чтения PNG.
    final_pixels: np.ndarray | None = None


def describe_config(fixed: Fixed) -> str:
    """Короткая подпись автомата для галереи."""

    values = fixed.to_dict()

    def get(name: str, default: object = "?") -> object:
        return values.get(name, default)

    return (
        f"r1={get('neighborhood.r1')} r2={get('neighborhood.r2')} "
        f"states={get('data.states')} rot={get('rule.rot')} "
        f"born_inner={get('rule.born_inner')} born_total={get('rule.born_total')} "
        f"survive={get('rule.survive_lo')}..{get('rule.survive_hi')}"
    )


def capture(
    experiment,
    fixed: Fixed,
    out_dir: str | Path,
    *,
    every: int = 1,
    scale: int = 6,
    seed: int = 0,
    scheme: str | None = None,
    grid_lines: bool = False,
    viewer: bool = True,
    strip: bool = True,
    sheet: bool = True,
    sheet_columns: int = 5,
    title: str | None = None,
) -> Capture:
    """Прогнать конфиг и сохранить кадры, ленту, лист и вьюер."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = Capture()
    pixels: list[np.ndarray] = []
    visualizer = resolve_visualizer(experiment.assemble(fixed))
    actual_scheme = visualizer.scheme if scheme in (None, "auto") else scheme

    def render(grid) -> np.ndarray:
        return visualizer.render_frame(
            grid,
            scale=scale,
            scheme=actual_scheme,
            grid_lines=grid_lines,
        )

    def on_frame(phase: str, step: int, grid) -> None:
        frame = render(grid)
        index = len(pixels)
        pixels.append(frame)
        if every > 0 and index % every == 0:
            result.frame_paths.append(
                write_png(output / f"{phase}-{step:04d}.png", frame)
            )

    trial = run_trial(
        experiment,
        fixed,
        seed=seed,
        keep_state=True,
        on_frame=on_frame,
    )
    if trial.state is not None:
        result.final_pixels = render(trial.state)
        result.final = write_png(output / "final.png", result.final_pixels)

    if pixels:
        if len(pixels) > 8:
            indexes = np.linspace(0, len(pixels) - 1, 8).round().astype(int)
            sample = [pixels[int(index)] for index in indexes]
        else:
            sample = pixels
        if strip:
            result.strip = save_strip(sample, output / "strip.png")
        if sheet:
            result.sheet = save_sheet(pixels, output / "sheet.png", columns=sheet_columns)

    states = int(getattr(trial.state, "states", fixed.to_dict().get("data.states", 2)))
    zero_state = int(getattr(trial.state, "zero_state", 0))
    if viewer:
        result.viewer = write_viewer(
            output / "viewer.html",
            pixels,
            title=title or f"Автомат {fixed.digest()}",
            subtitle=describe_config(fixed),
            config=fixed.to_dict(),
            metrics=trial.metrics,
            states=states,
            zero_state=zero_state,
            scheme=actual_scheme,
        )

    result.record = trial.to_record()
    (output / "trial.json").write_text(
        json.dumps(result.record, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result
