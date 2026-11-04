"""Тесты живой визуализации: без файлов, с общим стандартом."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from casweep.fixed import Fixed
from experiments.two_rings_rot import build_experiment
from modules.data.discrete2d import Grid2D
from visualize.live import (
    TerminalStyle,
    compare,
    downsample,
    frame_lines,
    main_for,
    play,
    resolve_view_config,
)


def make_grid(shape=(10, 12), states=4) -> Grid2D:
    array = np.zeros(shape, dtype=np.int32)
    array[2:5, 2:5] = 1
    array[4:7, 6:9] = 2
    array[0, -1] = 3
    return Grid2D(array, states, 0)


def small_config(experiment) -> Fixed:
    values = next(experiment.iter_configs(feasible=True)).to_dict()
    values.update(
        {
            "data.width": 32,
            "data.height": 32,
            "stage.warmup": 2,
            "stage.observe": 3,
        }
    )
    return Fixed(values)


class TestFrameStandard(unittest.TestCase):
    def test_plain_frame_has_requested_size(self) -> None:
        style = TerminalStyle(color=False, clear=False)
        lines = frame_lines(make_grid(), style, width=8, height=4)
        self.assertEqual(len(lines), 4)
        self.assertEqual([len(line) for line in lines], [8, 8, 8, 8])
        self.assertNotIn("\x1b[", "".join(lines))

    def test_color_frame_uses_half_blocks(self) -> None:
        style = TerminalStyle(color=True, truecolor=False, clear=False)
        lines = frame_lines(make_grid(), style, width=8, height=4)
        self.assertEqual(len(lines), 4)
        self.assertIn("\x1b[38;5;", lines[0])
        self.assertIn("▀", lines[0])

    def test_downsample_shape(self) -> None:
        sampled = downsample(np.arange(100).reshape(10, 10), 4, 3)
        self.assertEqual(sampled.shape, (3, 4))

    def test_small_grid_in_large_frame(self) -> None:
        # 10×12 сетка в окне 8×12: индексы повторяются, кадр не падает.
        for color in (False, True):
            style = TerminalStyle(color=color, truecolor=False, clear=False)
            lines = frame_lines(make_grid(), style, width=8, height=12)
            self.assertEqual(len(lines), 12)


class TestPlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = build_experiment()
        cls.fixed = small_config(cls.experiment)

    def test_play_prints_frames_and_writes_nothing(self) -> None:
        style = TerminalStyle(color=False, clear=False)
        buffer = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                with contextlib.redirect_stdout(buffer):
                    code = play(
                        self.experiment,
                        self.fixed,
                        seed=4,
                        steps=2,
                        style=style,
                    )
                files = list(Path(tmp).iterdir())
            finally:
                os.chdir(previous)
        self.assertEqual(code, 0)
        self.assertEqual(files, [])
        text = buffer.getvalue()
        self.assertIn("итерация", text)
        self.assertIn("итерация   2", text)
        self.assertIn(self.fixed.digest()[:8], text)

    def test_main_for_first_config(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main_for(
                self.experiment,
                scheme="age",
                argv=[
                    "--first",
                    "--steps",
                    "1",
                    "--no-color",
                    "--no-clear",
                    "--width",
                    "12",
                    "--height",
                    "3",
                ],
            )
        self.assertEqual(code, 0)
        self.assertIn("two_rings_rot", buffer.getvalue())


class TestCompare(unittest.TestCase):
    def test_compare_prints_several_automata(self) -> None:
        experiment = build_experiment()
        first = small_config(experiment)
        second = Fixed({**first.to_dict(), "rule.rot": 0.05})
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = compare(
                experiment,
                [(first, 1), (second, 2)],
                steps=2,
                style=TerminalStyle(color=False, clear=False),
            )
        text = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("2 автоматов", text)
        self.assertIn(first.digest()[:8], text)
        self.assertIn(second.digest()[:8], text)
        self.assertIn("кадр 2", text)


class TestResolution(unittest.TestCase):
    def test_resolve_from_run_dir(self) -> None:
        experiment = build_experiment()
        fixed = small_config(experiment)
        with tempfile.TemporaryDirectory() as tmp:
            record = {"config": fixed.to_dict(), "seed": 17, "status": "passed"}
            Path(tmp, "successes.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            resolved, seed = resolve_view_config(experiment, run_dir=tmp, index=0)
        self.assertEqual(resolved.digest(), fixed.digest())
        self.assertEqual(seed, 17)

    def test_each_experiment_has_view_module(self) -> None:
        for name in ("life2d", "two_rings", "two_rings_rot", "moore3", "multi3", "cyclic5"):
            module = importlib.import_module(f"experiments.{name}.view")
            self.assertTrue(callable(module.main_for))
            self.assertIn(module.SCHEME, ("age", "hue"))


if __name__ == "__main__":
    unittest.main()
