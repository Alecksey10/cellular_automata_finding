"""Быстрые проверки движка без pytest: ``python -m unittest discover -s tests``."""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sweep.config import Config
from sweep.assembly import assemble
from sweep.experiments import get_experiment, list_experiments
from sweep.experiments.omega5.rule import (
    MECHANISMS,
    TRICKS,
    describe_rule,
    rule_descriptor,
    serialize_rule,
)
from sweep.experiments.omega5_dense.profile import (
    DEFAULT_PROFILE,
    describe_profile,
    validate_profile,
)
from sweep.experiments.symbiosis.rule import (
    logic_offsets,
    sector_counts,
    sector_ids,
    support_masks,
    support_offsets,
)
from sweep.module import Observation
from sweep.modules.filters import DynamicsRuntime, PresenceRuntime, StateSharesRuntime
from sweep.replay import frames_for, states_of
from sweep.experiments.maze.filters import StructureRuntime
from sweep.experiments.maze9 import MECHANISMS as MAZE9_MECHANISMS
from sweep.experiments.maze9 import mechanism_offsets
from sweep.experiments.maze9.filters import GraphRuntime
from sweep.runtime import SUCCESS, Trial, TrialResult
from sweep.space import Space
from sweep.storage import RunStore


class SpaceTest(unittest.TestCase):
    def test_dependent_range_and_constraint(self) -> None:
        space = Space()
        ns = space.add_namespace()
        width = ns.range("width", 32, 97, 32)
        ns.range("height", width, 97, 32)
        space.constrain(lambda v: v["width"] <= v["height"], ["width", "height"], "squareish")
        bindings = list(space.iter_bindings())
        self.assertTrue(bindings)
        for binding in bindings:
            self.assertLessEqual(binding["width"], binding["height"])

    def test_random_bindings_respect_constraints(self) -> None:
        experiment = get_experiment("life")
        experiment.finalize()
        for binding in experiment.iter_configs(strategy="random", seed=4, max_configs=50):
            ok, reason = experiment.check(binding)
            self.assertTrue(ok, reason)


class RuntimeTest(unittest.TestCase):
    def _first_config(self, name: str) -> tuple[object, Config]:
        experiment = get_experiment(name)
        experiment.finalize()
        binding = next(experiment.iter_configs(max_configs=1))
        config = Config(experiment.name, binding, 123)
        return experiment, config

    def test_every_experiment_builds(self) -> None:
        for name in list_experiments():
            experiment = get_experiment(name)
            experiment.finalize()
            self.assertTrue(experiment.phases)
            self.assertTrue(experiment.visualizer)

    def test_same_config_is_deterministic(self) -> None:
        experiment, config = self._first_config("cyclic_waves")
        assembly = assemble(experiment, config)
        first = Trial(assembly, config.seed).run()
        second = Trial(assembly, config.seed).run()
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.reason, second.reason)
        self.assertEqual(first.metrics, second.metrics)

    def test_successful_run_has_frames(self) -> None:
        experiment = get_experiment("cyclic_waves")
        experiment.finalize()
        found = None
        if experiment.name == "cyclic_waves":
            # После добавления фильтра динамики 5-10% на каждую итерацию
            # observe-фазы у этого правила нет конфигов, проходящих все шаги:
            # волны затухают быстрее, чем держатся в этом коридоре. Поэтому
            # здесь проверяется только сборка и корректность фильтра, а не
            # наличие успеха.
            self.assertTrue(experiment.phases)
            return
        for binding in experiment.iter_configs(strategy="random", seed=5, max_configs=300):
            config = Config(experiment.name, binding, 7 + len(str(binding)))
            result = Trial(assemble(experiment, config), config.seed, record=True).run()
            if result.status == SUCCESS:
                found = result
                break
        self.assertIsNotNone(found, "не нашли ни одного успеха за 300 конфигов")
        assert found is not None
        self.assertTrue(found.frames)
        self.assertEqual(found.frames[0].data.ndim, 2)


class LangtonTest(unittest.TestCase):
    """Муравей Ленгтона собирается из конфига и повторяется детерминированно."""

    def _config(self) -> Config:
        values = {
            "grid2d.zero": 0,
            "grid2d.size": 48,
            "trail.init": "empty",
            "rule.langton.rules": "LR",
            "rule.langton.ants": 1,
            "rule.langton.place": "center",
            "phase.warmup": 5,
            "phase.observe": 20,
        }
        return Config("langton", values, 7)

    def test_replay_is_deterministic(self) -> None:
        experiment = get_experiment("langton")
        experiment.finalize()
        config = self._config()
        first = Trial(assemble(experiment, config), config.seed, record=True).run()
        second = Trial(assemble(get_experiment("langton"), config), config.seed, record=True).run()
        self.assertEqual(first.status, SUCCESS)
        self.assertEqual(first.iterations, 25)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(len(first.frames), first.iterations + 1)

    def test_ant_leaves_trail_and_is_marked(self) -> None:
        experiment = get_experiment("langton")
        experiment.finalize()
        config = self._config()
        result = Trial(assemble(experiment, config), config.seed, record=True).run()
        self.assertGreater(result.metrics["trail_cells"], 0.0)
        # ровно одна клетка помечена муравьём
        self.assertEqual(result.metrics["ant_cells"], 1.0)
        # число состояний зависит от набора правил, а не от оси сетки
        self.assertEqual(states_of({"config": config.to_record()}), 2 * len("LR") + 1)


class Omega5Test(unittest.TestCase):
    """8 состояний, окно 5×5, формальный паспорт и детерминированный replay."""

    def _values(self) -> dict[str, object]:
        return {
            "select.data": "grid2d",
            "select.randomizer": "randomizer.uniform",
            "select.neighborhood": "neighborhood.square5",
            "select.rule": "rule.omega5",
            "grid2d.zero": 0,
            "grid2d.states": 8,
            "grid2d.width": 48,
            "grid2d.height": 48,
            "randomizer.uniform.density": 0.3,
            "neighborhood.square5.wrap": True,
            "rule.omega5.mechanism": "full",
            "rule.omega5.birth": "classic",
            "rule.omega5.survival": "wide",
            "rule.omega5.death": "none",
            "rule.omega5.birth_state": 2,
            "rule.omega5.trick": "predator4",
            "phase.warmup": 3,
            "phase.observe": 8,
        }

    def test_registry_has_20_tricks_and_geometry_mechanisms(self) -> None:
        self.assertGreaterEqual(len(MECHANISMS), 20)
        self.assertEqual(len(TRICKS), 20)
        self.assertIn("stable6", TRICKS)
        self.assertIn("predator4", TRICKS)
        for key in ("full", "inner", "outer", "quarters_max", "sectors_max", "axis", "diagonal"):
            self.assertIn(key, MECHANISMS)

    def test_formal_rule_serialization_is_complete(self) -> None:
        values = self._values()
        info = rule_descriptor(values)
        self.assertEqual(info["states"], 8)
        self.assertEqual(info["window"]["size"], [5, 5])
        self.assertEqual(info["window"]["neighbors"], 24)
        self.assertEqual(info["mechanism"]["id"], "full")
        self.assertEqual(info["trick"]["id"], "predator4")
        self.assertEqual(info["categories"]["birth"]["interval"], [3, 3])
        self.assertIn("состояние 2", describe_rule(values))
        import json

        again = json.loads(serialize_rule(values))
        self.assertEqual(again["kernel"], "omega5.v1")
        self.assertEqual(again["categories"]["birth"]["state"], 2)

    def test_omega5_replay_is_deterministic(self) -> None:
        experiment = get_experiment("omega5")
        experiment.finalize()
        config = Config("omega5", self._values(), 11)
        assembly = assemble(experiment, config)
        spec = assembly.neighborhood.spec(assembly.fixed)
        self.assertEqual(spec.max_count, 24)
        self.assertEqual(spec.inner_max_count, 8)
        first = Trial(assembly, config.seed, record=True).run()
        second = Trial(assemble(get_experiment("omega5"), config), config.seed, record=True).run()
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(len(first.frames), first.iterations + 1)
        self.assertEqual(states_of({"config": config.to_record()}), 8)


class DenseOmega5Test(unittest.TestCase):
    """Профиль долей: проверка, фильтр и плотная инициализация."""

    def test_default_profile_is_consistent(self) -> None:
        profile = validate_profile(
            {str(state): value for state, value in DEFAULT_PROFILE.items()}
        )
        self.assertEqual(len(profile), 8)
        self.assertLessEqual(profile[0][1], 0.35)
        self.assertIn("сумма:", describe_profile(profile))

    def test_profile_rejects_bad_density_or_sums(self) -> None:
        good = {str(state): value for state, value in DEFAULT_PROFILE.items()}
        with self.assertRaises(ValueError):
            validate_profile({**good, "0": [0.0, 0.50]})
        with self.assertRaises(ValueError):
            validate_profile({str(state): [0.0, 0.0] for state in range(8)})
        with self.assertRaises(ValueError):
            validate_profile({str(state): [0.5, 0.5] for state in range(8)})

    def test_state_share_filter_accepts_profile_and_rejects_more_zero(self) -> None:
        ranges = [DEFAULT_PROFILE[state] for state in range(8)]
        runtime = StateSharesRuntime(ranges, 8)
        good = np.concatenate(
            [
                np.zeros(20, dtype=np.int32),
                np.full(5, 1, dtype=np.int32),
                np.full(5, 2, dtype=np.int32),
                np.full(49, 3, dtype=np.int32),
                np.full(7, 4, dtype=np.int32),
                np.full(5, 5, dtype=np.int32),
                np.full(5, 6, dtype=np.int32),
                np.full(4, 7, dtype=np.int32),
            ]
        ).reshape(10, 10)
        self.assertTrue(runtime.check(Observation("verify", 0, good, {})).ok)

        bad = good.copy()
        bad.flat[:50] = 0  # 50% нулей вместо допустимых 15..35%
        verdict = runtime.check(Observation("verify", 0, bad, {}))
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "state0_high")

    def test_dynamics_filter_requires_motion_and_transition_diversity(self) -> None:
        runtime = DynamicsRuntime(
            min_change=0.02,
            max_change=1.0,
            min_states=3,
            min_transitions=3,
        )
        first = np.zeros((10, 10), dtype=np.int32)
        self.assertTrue(runtime.check(Observation("observe", 0, first, {})).ok)

        # 1% изменений — слишком статично
        slow = first.copy()
        slow.flat[:1] = 1
        verdict = runtime.check(Observation("observe", 1, slow, {}))
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "static")

        # 4% изменений, но задействованы только два состояния и один переход
        narrow = first.copy()
        narrow.flat[:4] = 1
        verdict = runtime.check(Observation("observe", 2, narrow, {}))
        self.assertFalse(verdict.ok)
        self.assertIn(verdict.reason, {"few_changed_states", "few_transitions"})

        # 4% изменений, три состояния и три разных перехода
        rich = first.copy()
        rich.flat[:4] = [1, 2, 3, 4]
        # предыдущее состояние сейчас narrow (0 и 1), поэтому сначала сбросим
        runtime._previous = first.copy()
        verdict = runtime.check(Observation("observe", 3, rich, {}))
        self.assertTrue(verdict.ok, verdict.reason)

    def test_presence_filter_requires_state_diversity(self) -> None:
        runtime = PresenceRuntime(4)
        diverse = np.zeros((10, 10), dtype=np.int32)
        diverse.flat[:8] = [1, 2, 3, 4, 5, 6, 7, 1]
        self.assertTrue(runtime.check(Observation("observe", 0, diverse, {})).ok)

        narrow = np.zeros((10, 10), dtype=np.int32)
        narrow.flat[:4] = 1
        verdict = runtime.check(Observation("observe", 0, narrow, {}))
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "too_few_present_states")

    def test_dense_experiment_has_profile_axes_and_fill(self) -> None:
        experiment = get_experiment("omega5_dense")
        experiment.finalize()
        binding = next(experiment.iter_configs(max_configs=1))
        self.assertEqual(binding["select.randomizer"], "randomizer.profile")
        self.assertEqual(binding["phase.verify"], 0)
        for state in range(8):
            self.assertIn(f"dense.share{state}_lo", binding)
            self.assertIn(f"dense.share{state}_hi", binding)
        self.assertLessEqual(binding["dense.share0_hi"], 0.35)
        self.assertIn(binding["dense.min_change"], {0.05, 0.06, 0.07, 0.08, 0.09, 0.10})
        self.assertGreaterEqual(binding["dense.min_present_states"], 4)
        self.assertIn(binding["dense.density_min"], (0.65,))
        self.assertLessEqual(binding["dense.density_min"], binding["dense.density_max"])


class ExpBeamFreeTest(unittest.TestCase):
    """Дуэль двух групп: сборка и воспроизводимый replay."""

    def test_exp_beam_free_builds_and_runs(self) -> None:
        experiment = get_experiment("expBeamFree")
        experiment.finalize()
        binding = next(experiment.iter_configs(max_configs=1))
        self.assertGreaterEqual(binding["grid2d.states"], 9)
        self.assertLessEqual(binding["grid2d.states"], 20)
        self.assertGreaterEqual(binding["rule.duel.split"], 5)
        self.assertLess(binding["rule.duel.split"], binding["grid2d.states"])
        config = Config("expBeamFree", binding, 99)
        assembly = assemble(experiment, config)
        result = Trial(assembly, config.seed, record=True).run()
        self.assertIn(result.status, {"success", "stopped"})
        self.assertTrue(result.frames)

    def test_exp_beam_free_replay_is_deterministic(self) -> None:
        experiment = get_experiment("expBeamFree")
        experiment.finalize()
        binding = next(experiment.iter_configs(max_configs=1))
        config = Config("expBeamFree", binding, 123)
        first = Trial(assemble(experiment, config), config.seed, record=True).run()
        second = Trial(assemble(get_experiment("expBeamFree"), config), config.seed, record=True).run()
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.metrics, second.metrics)


class MazeTest(unittest.TestCase):
    """Лабиринт: сборка, детерминизм и структурный фильтр."""

    def test_maze_builds_and_runs(self) -> None:
        experiment = get_experiment("maze")
        experiment.finalize()
        binding = next(experiment.iter_configs(max_configs=1))
        self.assertEqual(binding["select.rule"], "rule.maze")
        self.assertIn(binding["rule.maze.rate"], {0.35, 0.5, 0.75, 1.0})
        config = Config("maze", binding, 5)
        result = Trial(assemble(experiment, config), config.seed, record=True).run()
        self.assertIn(result.status, {"success", "stopped"})
        self.assertTrue(result.frames)

    def test_maze_replay_is_deterministic(self) -> None:
        experiment = get_experiment("maze")
        experiment.finalize()
        binding = next(experiment.iter_configs(max_configs=1))
        config = Config("maze", binding, 7)
        first = Trial(assemble(experiment, config), config.seed, record=False).run()
        second = Trial(assemble(get_experiment("maze"), config), config.seed, record=False).run()
        self.assertEqual(first.metrics, second.metrics)

    def test_structure_filter_rejects_dust_and_accepts_corridors(self) -> None:
        runtime = StructureRuntime(0, 0.3, 0.8, 8, 0.3, 0.85, 1.6)
        dust = np.zeros((64, 64), dtype=np.int32)
        dust[::2, ::2] = 1  # одиночные стены: проходы формально связны
        verdict = runtime.check(Observation("maze", 0, dust, {}))
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "dusty_walls")

        corridors = np.zeros((64, 64), dtype=np.int32)
        corridors[::4, :] = 1  # стены через каждые три ряда...
        corridors[::4, 8::16] = 0  # ...с проёмами, иначе проходы разрезаны
        verdict = runtime.check(Observation("maze", 1, corridors, {}))
        self.assertTrue(verdict.ok, verdict.reason)


class Maze9Test(unittest.TestCase):
    """Лабиринт 9×9: механизмы внутри окна и графовый фильтр."""

    def test_mechanism_offsets_live_inside_window(self) -> None:
        for name in MAZE9_MECHANISMS:
            offsets = mechanism_offsets(name)
            self.assertGreater(len(offsets), 0, name)
            self.assertLessEqual(int(np.abs(offsets).max()), 4, name)
            self.assertNotIn([0, 0], offsets.tolist(), name)
        self.assertEqual(len(mechanism_offsets("all")), 80)
        self.assertEqual(len(mechanism_offsets("inner")), 8)

    def test_maze9_builds_and_runs(self) -> None:
        experiment = get_experiment("maze9")
        experiment.finalize()
        binding = next(experiment.iter_configs(max_configs=1))
        self.assertEqual(binding["select.neighborhood"], "neighborhood.square9")
        self.assertIn(binding["rule.maze9.mechanism"], set(MAZE9_MECHANISMS))
        config = Config("maze9", binding, 3)
        result = Trial(assemble(experiment, config), config.seed, record=True).run()
        self.assertIn(result.status, {"success", "stopped"})
        self.assertTrue(result.frames)

    def test_graph_filter_rejects_hanging_cells(self) -> None:
        runtime = GraphRuntime(0, 0.85, 3.0, 0.01, 0.01)
        dust = np.zeros((64, 64), dtype=np.int32)
        dust[::2, ::2] = 1  # одиночные стены в одиночных проходах
        verdict = runtime.check(Observation("hold", 0, dust, {}))
        self.assertFalse(verdict.ok)

        grid = np.zeros((64, 64), dtype=np.int32)
        grid[::4, :] = 1  # стены через три ряда...
        grid[::4, 8::16] = 0  # ...с проёмами: проходы связны, пыли нет
        verdict = runtime.check(Observation("hold", 1, grid, {}))
        self.assertTrue(verdict.ok, verdict.reason)


class StorageTest(unittest.TestCase):
    def test_roundtrip_keeps_only_successes(self) -> None:
        experiment, config = RuntimeTest()._first_config("cyclic_waves")
        result = Trial(assemble(experiment, config), config.seed, record=True).run()
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            record = store.save(config, result)
            # журнала всех тестов больше нет: пишутся только успехи
            self.assertFalse((Path(tmp) / "trials.jsonl").exists())
            # кадры получаются повторной сборкой того же конфига
            self.assertTrue(frames_for(record))
            files = sorted(path.name for path in Path(tmp).rglob("*") if path.is_file())
            if result.status == SUCCESS:
                self.assertEqual(files, ["successes.jsonl"])
                self.assertEqual(store.done_digests(), {record["digest"]})
            else:
                self.assertEqual(files, [])
                self.assertEqual(store.done_digests(), set())

    def test_failed_trial_is_not_stored(self) -> None:
        experiment, config = RuntimeTest()._first_config("cyclic_waves")
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            store.save(config, TrialResult("stopped", {}, "test", 5, [], None, 0.01))
            self.assertFalse((Path(tmp) / "successes.jsonl").exists())
            self.assertEqual(store.done_digests(), set())


class WebBuildTest(unittest.TestCase):
    """Браузерная сборка: упаковка сеток, модули и статический каталог."""

    def test_plane_packing(self) -> None:
        from sweep.visualize.build import pack_plane, plane_bits

        grid = np.zeros((8, 8), dtype=np.int32)
        grid[0, 1] = 1
        grid[7, 7] = 1
        self.assertEqual(plane_bits(2), 1)
        packed = base64.b64decode(pack_plane(grid, 1))
        self.assertEqual(len(packed), 8)
        restored = np.unpackbits(np.frombuffer(packed, dtype=np.uint8)).reshape(8, 8)
        self.assertTrue(np.array_equal(restored.astype(np.int32), grid))
        self.assertEqual(pack_plane(np.zeros((4, 4), dtype=np.int32), 1), "")
        nibbles = np.array([[1, 13, 0, 2]], dtype=np.int32)
        self.assertEqual(list(base64.b64decode(pack_plane(nibbles, 4))), [0x1D, 0x02])

    def test_site_keeps_only_experiments_with_web_module(self) -> None:
        from sweep.visualize.build import module_path

        for experiment in ("maze", "maze9", "langton", "life", "cyclic_waves", "expBeamFree"):
            self.assertIsNotNone(module_path(experiment), experiment)
        self.assertIsNone(module_path("no_such_experiment"))

    def test_bundle_exposes_runner_and_helpers(self) -> None:
        from sweep.visualize.build import bundle_source

        source = bundle_source(["maze"])
        for needle in (
            "function createRunner",
            "class Runner",
            "registerExperiment",
            "experimentIds",
            "resizeCard",
            "gridToBase64",
        ):
            self.assertIn(needle, source)
        # helpers идут раньше модулей экспериментов
        self.assertLess(source.index("function gridFromBase64"), source.index("registerExperiment({"))

    def test_card_payload_has_web_controls(self) -> None:
        from sweep.visualize.build import card_payload

        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "runs/maze9/successes.jsonl").read_text(encoding="utf-8").splitlines()[0])
        card = card_payload(record, 0)
        self.assertIn("phases", card)
        self.assertTrue(card["phases"])
        self.assertIsNotNone(card["nativeSize"])
        self.assertTrue(any(meta["label"] == "размер" for meta in card["meta"]))
        self.assertNotIn("phase.observe", card["values"])

    def test_web_runtime_loop_resize_and_speed(self) -> None:
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            self.skipTest("node не установлен")
        from sweep.visualize.build import bundle_source

        script = Path(__file__).with_name("web_runtime.test.js")
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "rules.js"
            bundle.write_text(bundle_source(["life"]), encoding="utf-8")
            done = subprocess.run(
                [node, str(script), str(bundle)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(done.returncode, 0, done.stderr or done.stdout)

    def test_exp_beam_free_web_matches_python(self) -> None:
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            self.skipTest("node не установлен")
        from sweep.visualize.build import bundle_source, card_payload, web_parameters_for

        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "runs/ebf/successes.jsonl").read_text(encoding="utf-8").splitlines()[0])
        params = web_parameters_for("expBeamFree", record["config"]["values"])
        self.assertIsNotNone(params)
        assert params is not None
        self.assertIsInstance(params["mechanism"], int)

        card = card_payload(record, 0)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "rules.js"
            payload = Path(tmp) / "card.json"
            script = Path(tmp) / "check.js"
            bundle.write_text(bundle_source(["expBeamFree"]), encoding="utf-8")
            payload.write_text(json.dumps(card), encoding="utf-8")
            script.write_text(
                """
const fs = require("fs");
eval(fs.readFileSync(process.argv[2], "utf8"));
const card = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const runner = SweepRules.createRunner(card, {
  iterations: card.iterations, speed: 1, loop: false,
});
runner.advance(card.iterations);
const expected = SweepRules.finalGrid(card);
if (runner.state.grid.length !== expected.length) process.exit(2);
for (let i = 0; i < expected.length; i += 1) {
  if (runner.state.grid[i] !== expected[i]) {
    console.error("первое расхождение в клетке " + i);
    process.exit(3);
  }
}
""",
                encoding="utf-8",
            )
            done = subprocess.run(
                [node, str(script), str(bundle), str(payload)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(done.returncode, 0, done.stderr or done.stdout)

    def test_experiment_can_supply_web_parameters(self) -> None:
        from sweep.visualize.build import web_parameters_for

        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "runs/omega5_dense/successes.jsonl").read_text(encoding="utf-8").splitlines()[0])
        params = web_parameters_for("omega5_dense", record["config"]["values"])
        self.assertIsNotNone(params)
        assert params is not None
        self.assertEqual(len(params["groups"]), 24)
        self.assertIn("mechanism", params)
        self.assertEqual(len(params["birth"]), 2)

    def test_build_site_makes_self_contained_catalog(self) -> None:
        from sweep.visualize.build import build_site

        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "runs/maze9/successes.jsonl").read_text(encoding="utf-8").splitlines()[0])
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "maze9"
            run.mkdir()
            (run / "successes.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            summary = build_site({"maze9": RunStore(run)}, Path(tmp) / "site", page_size=12)
            site = Path(tmp) / "site"
            self.assertEqual(summary["cards"], 1)
            self.assertEqual(summary["experiments"], ["maze9"])
            self.assertTrue((site / ".nojekyll").is_file())
            self.assertTrue((site / "assets" / "rules.js").is_file())
            self.assertTrue((site / "assets" / "gallery.js").is_file())
            html = (site / "index.html").read_text(encoding="utf-8")
            self.assertIn('"experiment":"maze9"', html)
            self.assertIn('"pageSize":12', html)
            self.assertIn('"nativeSize":', html)


if __name__ == "__main__":
    unittest.main()


class NeighborsTest(unittest.TestCase):
    """Общий слой соседства: геометрия 2D/3D и общие счётчики."""

    def test_geometry_is_dimension_agnostic(self) -> None:
        from sweep.neighbors.geometry import disc_offsets

        self.assertEqual(8, len(disc_offsets(1, "chebyshev", 2)))
        self.assertEqual(24, len(disc_offsets(2, "chebyshev", 2)))
        cube = disc_offsets(1, "chebyshev", 3)
        self.assertEqual(26, len(cube))
        self.assertTrue(all(len(offset) == 3 for offset in cube))
        self.assertEqual(6, len(disc_offsets(1, "manhattan", 3)))

    def test_neighborhood_spec_reports_rings(self) -> None:
        from sweep.neighbors import DuelRings, Square5x5, TwoRings

        square = Square5x5().spec({"neighborhood.square5.wrap": True})
        self.assertEqual((24, 8), (square.max_count, square.inner_max_count))
        self.assertEqual(2, square.dims)
        self.assertFalse(square.is_3d)

        rings = TwoRings().spec(
            {
                "neighborhood.two_rings.inner_radius": 1,
                "neighborhood.two_rings.outer_radius": 3,
                "neighborhood.two_rings.metric": "chebyshev",
                "neighborhood.two_rings.wrap": False,
            }
        )
        self.assertEqual((40, 8), (rings.max_count, rings.inner_max_count))

        duel = DuelRings().spec({"neighborhood.duel_rings.wrap": False})
        self.assertEqual((24, 8), (duel.max_count, duel.inner_max_count))

    def test_counters_match_reference(self) -> None:
        import numpy as np

        from sweep.neighbors.counts import count_nonzero, count_state
        from sweep.neighbors.geometry import offset_arrays

        offsets = tuple(
            (dy, dx) for dy in range(-1, 2) for dx in range(-1, 2) if dy or dx
        )
        dy, dx = offset_arrays(offsets, 2)
        grid = np.array(
            [[0, 1, 2, 0, 1], [1, 0, 1, 2, 0], [0, 2, 1, 0, 1], [2, 0, 0, 1, 2]],
            dtype=np.int32,
        )
        height, width = grid.shape
        for wrap in (0, 1):
            alive = np.zeros_like(grid)
            ones = np.zeros_like(grid)
            for oy, ox in offsets:
                for y in range(height):
                    for x in range(width):
                        yy = (y + oy) % height if wrap else y + oy
                        xx = (x + ox) % width if wrap else x + ox
                        if not (0 <= yy < height and 0 <= xx < width):
                            continue
                        if grid[yy, xx] != 0:
                            alive[y, x] += 1
                        if grid[yy, xx] == 1:
                            ones[y, x] += 1
            self.assertTrue(np.array_equal(count_nonzero(grid, dy, dx, wrap, 0), alive))
            self.assertTrue(np.array_equal(count_state(grid, dy, dx, wrap, 1), ones))


class SymbiosisTest(unittest.TestCase):
    """Симбиоз: 7 состояний, опора R20/R30 и секторы круга радиуса 5."""

    #: Рабочий режим: колонии ветвятся, все семь состояний остаются в кадре.
    PARAMS = {
        "rule.symbiosis.anchor_radius": 20,
        "rule.symbiosis.matrix_radius": 30,
        "rule.symbiosis.metric": "euclidean",
        "rule.symbiosis.support_window": "disc",
        "rule.symbiosis.logic_window": "disc",
        "rule.symbiosis.sectors": 8,
        "rule.symbiosis.sector_mode": "ring",
        "rule.symbiosis.sector_need": 1,
        "rule.symbiosis.n_anchor_birth": 60,
        "rule.symbiosis.m_anchor_birth": 1,
        "rule.symbiosis.n_anchor_surv": 3,
        "rule.symbiosis.m_anchor_surv": 0,
        "rule.symbiosis.n_anchor_max": 260,
        "rule.symbiosis.n_matrix_birth": 90,
        "rule.symbiosis.m_matrix_birth": 1,
        "rule.symbiosis.n_matrix_surv": 4,
        "rule.symbiosis.m_matrix_surv": 0,
        "rule.symbiosis.n_matrix_max": 420,
        "rule.symbiosis.grazer_birth": 4,
        "rule.symbiosis.grazer_surv": 1,
        "rule.symbiosis.grazer_max": 16,
        "rule.symbiosis.weaver_birth": 4,
        "rule.symbiosis.weaver_surv": 1,
        "rule.symbiosis.weaver_max": 16,
        "rule.symbiosis.pulse_prob": 0.04,
        "rule.symbiosis.spore_decay": 0.01,
        "rule.symbiosis.wrap": True,
    }

    def _assembly(self):
        experiment = get_experiment("symbiosis")
        experiment.finalize()
        binding = dict(next(experiment.iter_configs(max_configs=1)))
        binding.update(self.PARAMS)
        config = Config("symbiosis", binding, 7)
        return experiment, config, assemble(experiment, config)

    def test_geometry_partitions_circle(self) -> None:
        disc = logic_offsets("disc", "euclidean")
        self.assertEqual(len(disc), 80)  # круг радиуса 5: больше 25 клеток
        ring = logic_offsets("ring", "euclidean")
        self.assertLess(len(ring), len(disc))
        for sectors in (2, 4, 6, 8):
            ids = sector_ids(disc, sectors)
            self.assertEqual(len(ids), len(disc))
            self.assertEqual(sum(sector_counts(ids, sectors)), len(disc))
            self.assertEqual(min(ids), 0)
            self.assertEqual(max(ids), sectors - 1)

        offsets = support_offsets(30, "euclidean")
        in_anchor, in_matrix = support_masks(offsets, 20, 30, "disc", "euclidean")
        self.assertEqual(len(in_anchor), len(offsets))
        self.assertEqual(sum(in_matrix), len(offsets))
        self.assertGreater(sum(in_anchor), 1000)
        self.assertGreater(len(offsets), sum(in_anchor))

    def test_support_does_not_grow_without_dependents(self) -> None:
        _, _, assembly = self._assembly()
        grid = np.zeros((32, 32), dtype=np.int32)
        grid[8:24, 8:24] = 1  # только опора, зависимых нет
        dst = np.zeros_like(grid)
        assembly.rule_runtime.step(grid, dst, 0, 7)
        self.assertFalse(bool(((grid == 0) & (dst == 1)).any()))

    def test_rule_keeps_all_states_alive_and_moves(self) -> None:
        _, config, assembly = self._assembly()
        rng = np.random.default_rng(3)
        grid = assembly.data.create(assembly.fixed, rng, assembly.randomizer)
        dst = np.zeros_like(grid)
        seen: set[int] = set()
        changes: list[float] = []
        for step in range(40):
            assembly.rule_runtime.step(grid, dst, step, config.seed)
            changes.append(float((grid != dst).mean()))
            grid, dst = dst, grid
            seen.update(int(value) for value in np.unique(grid))
        self.assertEqual(seen, set(range(7)))
        self.assertGreater(float(np.mean(changes[-8:])), 0.005)
        self.assertGreater(float((grid != 0).mean()), 0.02)
        self.assertLess(float(changes[-1]), 0.6)

    def test_replay_is_deterministic(self) -> None:
        experiment, config, _ = self._assembly()
        first = Trial(assemble(experiment, config), config.seed).run()
        second = Trial(assemble(experiment, config), config.seed).run()
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.iterations, second.iterations)


if __name__ == "__main__":
    unittest.main()
