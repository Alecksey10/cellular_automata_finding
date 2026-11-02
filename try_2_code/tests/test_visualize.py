"""Тесты визуализации: общий стандарт, лента/лист, вьюер и галерея."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from casweep.cli import load_experiment, visual_dir
from casweep.cli import load_experiment, visual_dir
from casweep.fixed import Fixed
from casweep.module import Role
from experiments.two_rings_rot import build_experiment
from modules.data.discrete2d import Grid2D
from visualize.browser import capture_item, write_browser
from visualize.gallery import capture, describe_config
from visualize.png import encode_png
from visualize.protocol import Standard2DVisualizer, has_visualizer, resolve_visualizer
from visualize.standard import compose, legend, palette, render_frame, save_sheet
from visualize.viewer import gallery_html, viewer_html


def make_grid(shape=(6, 8), states=4) -> Grid2D:
    array = np.zeros(shape, dtype=np.int32)
    array[1:3, 1:3] = 1
    array[3:5, 3:6] = 2
    array[0, 7] = 3
    return Grid2D(array, states, 0)


class TestStandard(unittest.TestCase):
    def test_palette_marks_zero_dark_and_ages_bright(self) -> None:
        colors = palette(4, 0, "age")
        self.assertEqual(tuple(colors[0]), (12, 16, 22))
        brightness = [int(colors[state].sum()) for state in range(1, 4)]
        self.assertEqual(brightness, sorted(brightness))
        self.assertGreater(brightness[-1], 600)

    def test_render_frame_shape_and_dtype(self) -> None:
        frame = render_frame(make_grid(), scale=3)
        self.assertEqual(frame.shape, (18, 24, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_encode_png_has_signature(self) -> None:
        payload = encode_png(render_frame(make_grid(), scale=1))
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_compose_strip_and_sheet(self) -> None:
        frame = render_frame(make_grid(), scale=2)
        strip = compose([frame, frame, frame])
        self.assertEqual(strip.shape[:2], (12, 3 * frame.shape[1] + 2 * 4))
        sheet = compose([frame, frame, frame], columns=2)
        self.assertGreater(sheet.shape[0], frame.shape[0])

    def test_legend_has_one_cell_per_state(self) -> None:
        bar = legend(4, 0, cell=10, gap=2)
        self.assertEqual(bar.shape, (10, 4 * 10 + 3 * 2, 3))


class TestProtocol(unittest.TestCase):
    def test_visual_files_live_next_to_experiment(self) -> None:
        experiment = load_experiment("experiments.two_rings")
        self.assertIsNotNone(experiment.home_dir)
        self.assertEqual(experiment.home_dir.name, "two_rings")
        self.assertEqual(visual_dir(experiment).name, "_viewer")
        self.assertEqual(visual_dir(experiment).parent, experiment.home_dir)

    def test_visual_files_live_next_to_experiment(self) -> None:
        experiment = load_experiment("experiments.two_rings")
        self.assertIsNotNone(experiment.home_dir)
        self.assertEqual(experiment.home_dir.name, "two_rings")
        self.assertEqual(visual_dir(experiment).name, "_viewer")
        self.assertEqual(visual_dir(experiment).parent, experiment.home_dir)

    def test_module_visualizer_is_detected_and_fallback_works(self) -> None:
        class FakeRule:
            visualizer = Standard2DVisualizer("hue", name="fake-hue")

        assembled = SimpleNamespace(
            modules={Role.RULE: FakeRule(), Role.NEIGHBORHOOD: object(), Role.DATA: object()}
        )
        self.assertTrue(has_visualizer(FakeRule()))
        self.assertFalse(has_visualizer(object()))
        chosen = resolve_visualizer(assembled)
        self.assertEqual(chosen.scheme, "hue")
        self.assertEqual(chosen.name, "fake-hue")

        fallback = resolve_visualizer(SimpleNamespace(modules={Role.RULE: object()}))
        self.assertEqual(fallback.scheme, "age")

    def test_rules_declare_palette(self) -> None:
        experiment = build_experiment()
        fixed = next(experiment.iter_configs(feasible=True))
        visualizer = resolve_visualizer(experiment.assemble(fixed))
        self.assertEqual(visualizer.scheme, "age")
        self.assertIn("age", visualizer.describe())

    def test_browser_gallery_embeds_several_frames(self) -> None:
        experiment = build_experiment()
        values = next(experiment.iter_configs(feasible=True)).to_dict()
        values.update(
            {
                "data.width": 16,
                "data.height": 16,
                "stage.warmup": 2,
                "stage.observe": 2,
            }
        )
        fixed = Fixed(values)
        item = capture_item(
            experiment, fixed, seed=3, steps=3, every=1, scale=1, max_frames=4
        )
        self.assertTrue(item.frames)
        self.assertEqual(item.scheme, "age")
        with tempfile.TemporaryDirectory() as tmp:
            path = write_browser(Path(tmp) / "index.html", [item], title="Тест")
            text = path.read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", text)
        self.assertIn("Тест", text)
        self.assertIn(item.digest[:8], text)


class TestViewer(unittest.TestCase):
    def test_viewer_embeds_frames(self) -> None:
        frame = render_frame(make_grid(), scale=2)
        html = viewer_html(
            [frame, frame],
            title="Тест",
            subtitle="два кадра",
            states=4,
            zero_state=0,
        )
        self.assertEqual(html.count("data:image/png;base64,"), 2)
        self.assertIn("кадров: 2", html)
        self.assertIn("FRAMES", html)
        self.assertIn("играть", html)

    def test_gallery_links_to_viewers(self) -> None:
        frame = render_frame(make_grid(), scale=2)
        html = gallery_html(
            [
                {
                    "digest": "abc123",
                    "seed": 7,
                    "image": frame,
                    "viewer": "abc123/viewer.html",
                    "summary": "r1=1 r2=2",
                }
            ],
            title="Галерея",
        )
        self.assertIn("abc123/viewer.html", html)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("Галерея", html)

    def test_capture_writes_all_artifacts(self) -> None:
        experiment = build_experiment()
        base = next(experiment.iter_configs(feasible=True))
        values = base.to_dict()
        values.update(
            {
                "data.width": 32,
                "data.height": 32,
                "stage.warmup": 3,
                "stage.observe": 10,
            }
        )
        fixed = Fixed(values)
        with tempfile.TemporaryDirectory() as tmp:
            result = capture(experiment, fixed, tmp, every=5, scale=2, seed=3)
            self.assertTrue(result.final is not None and result.final.exists())
            self.assertTrue(result.strip is not None and result.strip.exists())
            self.assertTrue(result.sheet is not None and result.sheet.exists())
            self.assertTrue(result.viewer is not None and result.viewer.exists())
            self.assertGreaterEqual(len(result.frame_paths), 1)
            self.assertIsNotNone(result.final_pixels)
            html = Path(result.viewer).read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", html)
            self.assertTrue(Path(tmp, "trial.json").exists())

    def test_describe_config_mentions_rings_and_rot(self) -> None:
        experiment = build_experiment()
        fixed = next(experiment.iter_configs(feasible=True))
        text = describe_config(fixed)
        self.assertIn("r1=", text)
        self.assertIn("rot=", text)


if __name__ == "__main__":
    unittest.main()
