"""Тесты «двух колец с частичным гниением»."""

from __future__ import annotations

import unittest

import numpy as np

from casweep.fixed import Fixed
from casweep.trial import TrialStatus, run_trial
from experiments.two_rings_rot import build_experiment
from modules.data.discrete2d import Grid2D
from modules.filters.drift import RotStructureRuntime, ZeroShareRuntime
from modules.neighborhoods.two_rings import two_ring_offsets
from modules.rules.kernels import rot_aging_step

INNER = ((1, 0), (-1, 0), (0, 1), (0, -1))
OUTER = ((1, 1), (-1, -1), (1, -1), (-1, 1))


def arrays(offsets):
    di = np.fromiter((cell[0] for cell in offsets), dtype=np.int64, count=len(offsets))
    dj = np.fromiter((cell[1] for cell in offsets), dtype=np.int64, count=len(offsets))
    return di, dj


def run_step(src, *, born_inner=1, born_total=1, survive_lo=1, survive_hi=4, rot=0.0, noise=None):
    idi, idj = arrays(INNER)
    odi, odj = arrays(OUTER)
    dst = np.empty_like(src)
    if noise is None:
        noise = np.zeros(src.shape, dtype=np.float32)
    rot_aging_step(
        src,
        dst,
        noise,
        idi,
        idj,
        odi,
        odj,
        born_inner,
        born_total,
        survive_lo,
        survive_hi,
        3,
        rot,
        0,
        False,
    )
    return dst


class TestRotKernel(unittest.TestCase):
    def grid(self) -> np.ndarray:
        return np.zeros((5, 5), dtype=np.int32)

    def test_birth_needs_inner_ring(self) -> None:
        src = self.grid()
        src[1, 2] = src[2, 1] = src[3, 2] = src[2, 3] = 1  # только внутреннее кольцо
        dst = run_step(src, born_inner=1, born_total=1)
        self.assertEqual(int(dst[2, 2]), 1)

        empty = self.grid()
        dst = run_step(empty, born_inner=1, born_total=1)
        self.assertEqual(int(dst[2, 2]), 0)

    def test_outer_ring_alone_does_not_give_birth(self) -> None:
        src = self.grid()
        for di, dj in OUTER:
            src[2 + di, 2 + dj] = 1
        dst = run_step(src, born_inner=1, born_total=1)
        self.assertEqual(int(dst[2, 2]), 0)

    def test_survival_ages_cell(self) -> None:
        src = self.grid()
        for di, dj in INNER:
            src[2 + di, 2 + dj] = 1
        src[2, 2] = 2
        dst = run_step(src, survive_lo=1, survive_hi=5)
        self.assertEqual(int(dst[2, 2]), 3)

    def test_starvation_removes_one_level(self) -> None:
        src = self.grid()
        src[2, 2] = 3
        dst = run_step(src, survive_lo=1, survive_hi=5)
        self.assertEqual(int(dst[2, 2]), 2)

    def test_last_state_rots_with_probability(self) -> None:
        src = self.grid()
        for di, dj in INNER:
            src[2 + di, 2 + dj] = 1
        src[2, 2] = 3
        never = np.ones(src.shape, dtype=np.float32)
        self.assertEqual(int(run_step(src, survive_lo=1, survive_hi=5, rot=1.0, noise=never)[2, 2]), 3)
        always = np.zeros(src.shape, dtype=np.float32)
        self.assertEqual(int(run_step(src, survive_lo=1, survive_hi=5, rot=1.0, noise=always)[2, 2]), 0)

    def test_noise_is_used_only_for_last_state(self) -> None:
        src = self.grid()
        for di, dj in INNER:
            src[2 + di, 2 + dj] = 1
        src[2, 2] = 2
        dst = run_step(src, survive_lo=1, survive_hi=5, rot=1.0, noise=np.zeros(src.shape, dtype=np.float32))
        self.assertEqual(int(dst[2, 2]), 3)


class TestDriftFilters(unittest.TestCase):
    def grid(self, shape=(10, 10), states=4) -> Grid2D:
        return Grid2D(np.zeros(shape, dtype=np.int32), states, 0)

    def test_zero_share_shift_above_limit_is_rejected(self) -> None:
        runtime = ZeroShareRuntime(max_shift=0.08)
        grid = self.grid()
        grid.array[:, :1] = 1  # 90% мёртвых
        self.assertTrue(runtime.check(grid).ok)
        grid.array[:, :2] = 1  # 80% мёртвых: прыжок 10%
        verdict = runtime.check(grid)
        self.assertFalse(verdict.ok)
        self.assertIn("прыгнула", verdict.reason)

    def test_zero_share_small_shift_passes(self) -> None:
        runtime = ZeroShareRuntime(max_shift=0.08)
        grid = self.grid()
        self.assertTrue(runtime.check(grid).ok)
        grid.array[0, 0] = 1  # 99% -> 100%: 1%
        self.assertTrue(runtime.check(grid).ok)

    def test_last_state_band(self) -> None:
        runtime = RotStructureRuntime(min_last_share=0.01, max_last_share=0.5)
        grid = self.grid()
        self.assertFalse(runtime.check(grid).ok)
        grid.array[0, :5] = 3  # 5%
        self.assertTrue(runtime.check(grid).ok)
        grid.array[:] = 3
        self.assertFalse(runtime.check(grid).ok)


class TestRotExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = build_experiment()
        cls.configs = []
        for index, fixed in enumerate(cls.experiment.iter_configs(feasible=True)):
            cls.configs.append(fixed)
            if index >= 99:
                break
        cls.base = cls.configs[0]

    def test_axes_match_requested_ranges(self) -> None:
        for fixed in self.configs:
            r1 = fixed["neighborhood.r1"]
            r2 = fixed["neighborhood.r2"]
            self.assertGreaterEqual(r1, 1)
            self.assertLessEqual(r1, 5)
            self.assertGreaterEqual(r2, 1)
            self.assertLessEqual(r2, 7)
            self.assertGreater(r2, r1)

            self.assertGreaterEqual(fixed["data.states"], 3)
            self.assertLessEqual(fixed["data.states"], 6)

            self.assertGreater(fixed["rule.rot"], 0)
            self.assertLessEqual(fixed["rule.survive_lo"], fixed["rule.survive_hi"])
            self.assertLessEqual(fixed["rule.born_total"], fixed["rule.survive_lo"])

            self.assertGreaterEqual(fixed["stage.warmup"], 3)
            self.assertLessEqual(fixed["stage.warmup"], 5)
            self.assertGreaterEqual(fixed["stage.observe"], 10)

    def test_two_states_are_incompatible(self) -> None:
        values = self.base.to_dict()
        values["data.states"] = 2
        self.assertIsNotNone(self.experiment.feasible(Fixed(values)))

    def test_trial_is_deterministic(self) -> None:
        first = run_trial(self.experiment, self.base, seed=13)
        second = run_trial(self.experiment, self.base, seed=13)
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.state_digest, second.state_digest)
        self.assertEqual(first.iterations, second.iterations)
        self.assertIn(
            first.status,
            (TrialStatus.PASSED, TrialStatus.REJECTED, TrialStatus.STOPPED),
        )

    def test_neighborhood_rings_are_disjoint(self) -> None:
        combined, inner, outer = two_ring_offsets(1, 7)
        self.assertEqual(set(inner) & set(outer), set())
        self.assertEqual(set(inner) | set(outer), set(combined))
        self.assertEqual(len(inner), 4)


if __name__ == "__main__":
    unittest.main()
