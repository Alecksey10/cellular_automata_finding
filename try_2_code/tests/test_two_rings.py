"""Тесты эксперимента «два кольца», правила старения и новых фильтров."""

from __future__ import annotations

import unittest

import numpy as np

from casweep.fixed import Fixed
from casweep.space import IncompatibleModules
from casweep.trial import TrialStatus, run_trial
from experiments.two_rings import build_experiment
from modules.data.discrete2d import Grid2D
from modules.filters.clouds import CloudRuntime
from modules.filters.stability import StabilityRuntime
from modules.neighborhoods.two_rings import two_ring_offsets


def make_grid(shape=(20, 20), states=3, zero_state=0) -> Grid2D:
    return Grid2D(np.zeros(shape, dtype=np.int32), states, zero_state)


class TestTwoRingNeighborhood(unittest.TestCase):
    def test_inner_and_outer_offsets_are_disjoint(self) -> None:
        combined, inner, outer = two_ring_offsets(2, 4)
        self.assertEqual(len(inner), 12)
        self.assertEqual(len(outer), 36)
        self.assertEqual(len(combined), 48)
        self.assertEqual(set(inner) & set(outer), set())
        self.assertEqual(set(inner) | set(outer), set(combined))


class TestTwoRingExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = build_experiment()
        cls.base = next(cls.experiment.iter_configs(feasible=True))

    def patched(self, **overrides) -> Fixed:
        values = self.base.to_dict()
        values.update(overrides)
        return Fixed(values)

    def test_radius_and_rule_axes_are_consistent(self) -> None:
        fixed = self.base
        r1 = fixed["neighborhood.r1"]
        r2 = fixed["neighborhood.r2"]
        self.assertGreaterEqual(r1, 1)
        self.assertLessEqual(r1, 5)
        self.assertGreater(r2, r1)
        self.assertLessEqual(r2, 8)

        self.assertGreaterEqual(fixed["rule.born_inner"], 1)
        self.assertLessEqual(fixed["rule.born_inner"], min(fixed["neighborhood#max_inner"], 5))
        self.assertGreaterEqual(fixed["rule.born_total"], 1)
        self.assertLessEqual(fixed["rule.born_total"], min(fixed["neighborhood#max_count"], 12))
        self.assertLessEqual(fixed["rule.born_total"], fixed["rule.crowd_limit"])
        self.assertLessEqual(fixed["rule.crowd_limit"], fixed["neighborhood#max_count"])
        self.assertGreaterEqual(fixed["rule.fortify"], 1)
        self.assertLessEqual(fixed["rule.fortify"], 3)

    def test_rule_needs_three_states(self) -> None:
        fixed = self.patched(**{"data.states": 2})
        self.assertIsNotNone(self.experiment.feasible(fixed))
        with self.assertRaises(IncompatibleModules):
            self.experiment.assemble(fixed)

    def test_trial_is_deterministic(self) -> None:
        first = run_trial(self.experiment, self.base, seed=11)
        second = run_trial(self.experiment, self.base, seed=11)
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first.state_digest, second.state_digest)
        self.assertIn(
            first.status,
            (TrialStatus.PASSED, TrialStatus.REJECTED, TrialStatus.STOPPED),
        )


class TestCloudFilter(unittest.TestCase):
    def runtime(self) -> CloudRuntime:
        return CloudRuntime(
            min_fill=0.05,
            max_fill=0.85,
            min_density=0.4,
            min_clustered=0.6,
            min_mean_component=6.0,
        )

    def test_dense_block_is_a_cloud(self) -> None:
        grid = make_grid()
        grid.array[5:11, 5:11] = 1
        verdict = self.runtime().check(grid)
        self.assertTrue(verdict.ok, verdict.reason)
        self.assertEqual(verdict.metrics["components"], 1)

    def test_full_grid_is_rejected(self) -> None:
        grid = make_grid()
        grid.array[:] = 1
        verdict = self.runtime().check(grid)
        self.assertFalse(verdict.ok)
        self.assertIn("залито", verdict.reason)

    def test_dust_is_rejected(self) -> None:
        grid = make_grid()
        grid.array[::4, ::4] = 1
        verdict = self.runtime().check(grid)
        self.assertFalse(verdict.ok)
        self.assertIn("пыль", verdict.reason)


class TestStabilityFilter(unittest.TestCase):
    def test_aging_is_not_flicker(self) -> None:
        runtime = StabilityRuntime(max_change=0.2)
        grid = make_grid(shape=(8, 8), states=4)
        grid.array[2:5, 2:5] = 1
        self.assertTrue(runtime.check(grid).ok)
        grid.array[2:5, 2:5] = 2  # те же живые клетки, изменился только возраст
        self.assertTrue(runtime.check(grid).ok)

    def test_mass_change_is_flicker(self) -> None:
        runtime = StabilityRuntime(max_change=0.2)
        grid = make_grid(shape=(8, 8), states=3)
        self.assertTrue(runtime.check(grid).ok)
        grid.array[0:4, 0:4] = 1  # 16 из 64 клеток = 25%
        verdict = runtime.check(grid)
        self.assertFalse(verdict.ok)
        self.assertIn("мелькание", verdict.reason)

    def test_period_two_is_flicker(self) -> None:
        runtime = StabilityRuntime(max_change=0.2)
        grid = make_grid(shape=(8, 8), states=3)
        self.assertTrue(runtime.check(grid).ok)
        grid.array[0, 0] = 1
        grid.array[0, 7] = 1
        self.assertTrue(runtime.check(grid).ok)
        grid.array[0, 0] = 0
        grid.array[0, 7] = 0
        verdict = runtime.check(grid)
        self.assertFalse(verdict.ok)
        self.assertIn("период 2", verdict.reason)


if __name__ == "__main__":
    unittest.main()
