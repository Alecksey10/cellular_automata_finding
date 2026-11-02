"""Тесты новых правил (3×3, цикл, рефрактерность), фильтров и режимов прогона."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from casweep.runner import RunOptions, run
from casweep.storage import ResultStore
from casweep.trial import TrialStatus, run_trial
from experiments.cyclic5 import build_experiment as build_cyclic5
from experiments.moore3 import build_experiment as build_moore3
from experiments.multi3 import build_experiment as build_multi3
from modules.data.discrete2d import Grid2D
from modules.filters.band import ActivityRuntime, FillBandRuntime
from modules.filters.cycle import PeriodRuntime
from modules.filters.structure import (
    ClusterRuntime,
    EntropyRuntime,
    PhaseBalanceRuntime,
    SymmetryRuntime,
)
from modules.filters.metrics import symmetry_score
from modules.neighborhoods.base import offsets_from_predicate
from modules.rules.kernels import (
    cyclic_step,
    refractory_step,
    totalistic_step,
    vote_step,
)

MOORE3 = offsets_from_predicate(1, "square", 1, 1)


def make_grid(shape=(8, 8), states=3, zero_state=0) -> Grid2D:
    return Grid2D(np.zeros(shape, dtype=np.int32), states, zero_state)


def di_dj(offsets=MOORE3):
    di = np.fromiter((cell[0] for cell in offsets), dtype=np.int64, count=len(offsets))
    dj = np.fromiter((cell[1] for cell in offsets), dtype=np.int64, count=len(offsets))
    return di, dj


class TestTotalisticKernel(unittest.TestCase):
    def test_blinker_under_b3_s23(self) -> None:
        di, dj = di_dj()
        src = np.zeros((7, 7), dtype=np.int32)
        src[3, 2:5] = 1
        dst = np.empty_like(src)
        totalistic_step(src, dst, di, dj, 3, 3, 2, 3, 0, 1, False)
        expected = np.zeros_like(src)
        expected[2:5, 3] = 1
        np.testing.assert_array_equal(dst, expected)

    def test_born_band_is_respected(self) -> None:
        di, dj = di_dj()
        src = np.zeros((7, 7), dtype=np.int32)
        src[3, 2] = 1
        src[3, 4] = 1
        dst = np.empty_like(src)
        # Нижняя граница рождения 3: с двумя соседями клетка не рождается.
        totalistic_step(src, dst, di, dj, 3, 3, 2, 3, 0, 1, False)
        self.assertEqual(int(dst[3, 3]), 0)


class TestVoteKernel(unittest.TestCase):
    def test_majority_wins(self) -> None:
        di, dj = di_dj()
        src = np.zeros((7, 7), dtype=np.int32)
        src[2:5, 2:5] = 1  # центр окружён восемью единицами
        dst = np.empty_like(src)
        vote_step(src, dst, di, dj, 1, 2, 0, False)
        self.assertEqual(int(dst[3, 3]), 1)

    def test_quorum_above_counts_keeps_cell(self) -> None:
        di, dj = di_dj()
        src = np.zeros((7, 7), dtype=np.int32)
        src[3, 3] = 2
        dst = np.empty_like(src)
        vote_step(src, dst, di, dj, 9, 3, 0, False)
        self.assertEqual(int(dst[3, 3]), 2)


class TestCyclicKernel(unittest.TestCase):
    def test_phase_spreads_and_top_dies(self) -> None:
        di, dj = di_dj()
        src = np.zeros((7, 7), dtype=np.int32)
        src[3, 3] = 0  # клетка фазы 0 рядом с фазой 1
        src[3, 4] = 1
        dst = np.empty_like(src)
        cyclic_step(src, dst, di, dj, 1, 4, 0, False, False)
        self.assertEqual(int(dst[3, 3]), 1)
        self.assertEqual(int(dst[3, 4]), 1)

    def test_always_die_kills_top_phase(self) -> None:
        di, dj = di_dj()
        src = np.zeros((7, 7), dtype=np.int32)
        src[3, 3] = 3
        dst = np.empty_like(src)
        cyclic_step(src, dst, di, dj, 1, 4, 0, True, False)
        self.assertEqual(int(dst[3, 3]), 0)


class TestRefractoryKernel(unittest.TestCase):
    def test_excitation_and_refractory_aging(self) -> None:
        di, dj = di_dj()
        src = np.zeros((7, 7), dtype=np.int32)
        src[3, 3] = 1
        src[3, 4] = 4
        dst = np.empty_like(src)
        refractory_step(src, dst, di, dj, 1, 4, 0, False)
        self.assertEqual(int(dst[3, 3]), 2)  # возбуждённая -> рефрактерная
        self.assertEqual(int(dst[3, 2]), 1)  # сосед зажёгся
        self.assertEqual(int(dst[3, 4]), 0)  # последняя фаза -> смерть


class TestStructureFilters(unittest.TestCase):
    def test_symmetry_score_of_block_and_noise(self) -> None:
        block = np.zeros((10, 10), dtype=bool)
        block[3:7, 3:7] = True
        self.assertGreater(symmetry_score(block), 0.99)
        noise = np.zeros((10, 10), dtype=bool)
        noise[2, 7] = True
        self.assertLess(symmetry_score(noise), 0.6)

    def test_symmetry_filter_threshold(self) -> None:
        runtime = SymmetryRuntime(min_score=0.9)
        grid = make_grid((10, 10))
        grid.array[3:7, 3:7] = 1
        self.assertTrue(runtime.check(grid).ok)
        grid.array[3:7, 3:7] = 0
        grid.array[0, 5] = 1
        self.assertFalse(runtime.check(grid).ok)

    def test_entropy_filter_rejects_single_phase(self) -> None:
        runtime = EntropyRuntime(min_entropy=0.5, max_entropy=1.0)
        grid = make_grid((8, 8), states=4)
        grid.array[:] = 1
        self.assertFalse(runtime.check(grid).ok)
        grid.array[:] = np.arange(64).reshape(8, 8) % 4
        self.assertTrue(runtime.check(grid).ok)

    def test_phase_balance_filter(self) -> None:
        runtime = PhaseBalanceRuntime(min_share=0.01)
        grid = make_grid((8, 8), states=4)
        grid.array[:] = 1
        self.assertFalse(runtime.check(grid).ok)
        grid.array[0, 0] = 2
        grid.array[0, 1] = 3
        self.assertTrue(runtime.check(grid).ok)

    def test_cluster_filter_rejects_dust(self) -> None:
        runtime = ClusterRuntime(min_components=1, max_components=10, min_mean_component=4.0)
        grid = make_grid((10, 10))
        grid.array[::3, ::3] = 1
        self.assertFalse(runtime.check(grid).ok)
        grid.array[:] = 0
        grid.array[2:6, 2:6] = 1
        self.assertTrue(runtime.check(grid).ok)


class TestBandAndCycleFilters(unittest.TestCase):
    def test_fill_band(self) -> None:
        runtime = FillBandRuntime(min_fill=0.1, max_fill=0.9)
        grid = make_grid((8, 8))
        self.assertFalse(runtime.check(grid).ok)
        grid.array[:] = 1
        self.assertFalse(runtime.check(grid).ok)
        grid.array[:] = 0
        grid.array[:4, :] = 1
        self.assertTrue(runtime.check(grid).ok)

    def test_activity_band(self) -> None:
        runtime = ActivityRuntime(min_change=0.02, max_change=0.5)
        grid = make_grid((8, 8))
        self.assertTrue(runtime.check(grid).ok)
        grid.array[0, 0] = 1
        self.assertFalse(runtime.check(grid).ok)  # застыло
        grid.array[4:, :] = 1
        self.assertTrue(runtime.check(grid).ok)

    def test_period_filter_detects_oscillation(self) -> None:
        runtime = PeriodRuntime(max_period=1, window=8)
        grid = make_grid((8, 8))
        grid.array[2, 2] = 1
        self.assertTrue(runtime.check(grid).ok)
        grid.array[2, 2] = 0
        grid.array[2, 6] = 1
        self.assertTrue(runtime.check(grid).ok)
        grid.array[2, 2] = 1
        grid.array[2, 6] = 0
        verdict = runtime.check(grid)  # та же картинка, что на шаге 0
        self.assertFalse(verdict.ok)
        self.assertIn("цикл", verdict.reason)


class TestExperiments(unittest.TestCase):
    def test_moore3_uses_eight_neighbours_and_two_states(self) -> None:
        experiment = build_moore3()
        configs = []
        for fixed in experiment.iter_configs(feasible=True):
            configs.append(fixed)
            if len(configs) >= 25:
                break
        self.assertTrue(configs)
        for fixed in configs:
            self.assertEqual(fixed["data.states"], 2)
            self.assertEqual(fixed["neighborhood#max_count"], 8)
            self.assertLessEqual(fixed["rule.born_hi"], 8)
            self.assertLessEqual(fixed["rule.survive_hi"], 8)

    def test_cyclic5_needs_four_states_and_24_neighbours(self) -> None:
        experiment = build_cyclic5()
        configs = []
        for fixed in experiment.iter_configs(feasible=True):
            configs.append(fixed)
            if len(configs) >= 25:
                break
        self.assertTrue(configs)
        for fixed in configs:
            self.assertGreaterEqual(fixed["data.states"], 3)
            if fixed["rule.module"] == "rule.refractory":
                self.assertGreaterEqual(fixed["data.states"], 4)
            self.assertEqual(fixed["neighborhood#max_count"], 24)
            self.assertGreaterEqual(fixed["rule.threshold"], 1)
            self.assertLessEqual(fixed["rule.threshold"], 8)

    def test_multi3_allows_more_states(self) -> None:
        experiment = build_multi3()
        fixed = next(experiment.iter_configs(feasible=True))
        self.assertGreaterEqual(fixed["data.states"], 3)

    def test_trial_is_deterministic_with_iteration_cap(self) -> None:
        experiment = build_cyclic5()
        fixed = next(experiment.iter_configs(feasible=True))
        first = run_trial(experiment, fixed, seed=4, max_iterations=2)
        second = run_trial(experiment, fixed, seed=4, max_iterations=2)
        self.assertEqual(first.state_digest, second.state_digest)
        self.assertLessEqual(first.iterations, 2)
        self.assertIn(
            first.status,
            (TrialStatus.PASSED, TrialStatus.REJECTED, TrialStatus.STOPPED),
        )


class TestRunModes(unittest.TestCase):
    def test_default_store_writes_only_successes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(directory)
            store.append({"config_digest": "a", "status": "rejected"})
            store.append({"config_digest": "b", "status": "passed"})
            store.close()
            self.assertFalse((Path(directory) / "trials.jsonl").exists())
            lines = (Path(directory) / "successes.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["config_digest"], "b")

    def test_save_all_writes_full_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(directory, save_all=True)
            store.append({"config_digest": "a", "status": "rejected"})
            store.append({"config_digest": "b", "status": "passed"})
            store.close()
            trials = (Path(directory) / "trials.jsonl").read_text().splitlines()
            successes = (Path(directory) / "successes.jsonl").read_text().splitlines()
            self.assertEqual(len(trials), 2)
            self.assertEqual(len(successes), 1)

    def test_until_stops_after_target(self) -> None:
        experiment = build_cyclic5()
        with tempfile.TemporaryDirectory() as directory:
            options = RunOptions(
                random_start=True,
                seed=9,
                until=2,
                mode="serial",
                out=directory,
                progress=False,
            )
            summary = run(experiment, options)
            self.assertGreaterEqual(summary.passed, 2)
            self.assertTrue(summary.target_reached)
            self.assertFalse(summary.interrupted)

    def test_stop_file_blocks_new_tasks(self) -> None:
        experiment = build_cyclic5()
        with tempfile.TemporaryDirectory() as directory:
            flag = Path(directory) / "stop.flag"
            flag.write_text("stop", encoding="utf-8")
            options = RunOptions(
                random_start=True,
                limit=10,
                seed=1,
                mode="serial",
                out=directory,
                stop_file=flag,
                progress=False,
            )
            summary = run(experiment, options)
            self.assertEqual(summary.executed, 0)
            self.assertTrue(summary.stopped_by_file)

    def test_iteration_budget_is_shared_by_phases(self) -> None:
        experiment = build_cyclic5()
        fixed = next(experiment.iter_configs(feasible=True))
        result = run_trial(experiment, fixed, seed=3, max_iterations=4)
        self.assertLessEqual(result.iterations, 4)
        if result.status is TrialStatus.PASSED:
            self.assertEqual(result.metrics.get("capped"), 4)


if __name__ == "__main__":
    unittest.main()
