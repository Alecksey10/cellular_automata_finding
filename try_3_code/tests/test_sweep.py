"""Быстрые проверки движка без pytest: ``python -m unittest discover -s tests``."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
