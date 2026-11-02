"""Тесты сборки автомата и исполнения одного теста."""

from __future__ import annotations

import unittest

from casweep.fixed import Fixed
from casweep.space import IncompatibleModules
from casweep.trial import TrialStatus, run_trial
from experiments.life2d import build_experiment


class TestAssembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = build_experiment()
        cls.base = next(cls.experiment.iter_configs(feasible=True))

    def patched(self, **overrides) -> Fixed:
        values = self.base.to_dict()
        values.update(overrides)
        return Fixed(values)

    def test_life_requires_exactly_two_states(self) -> None:
        fixed = self.patched(**{"data.states": 3, "rule.module": "rule.life"})
        reason = self.experiment.feasible(fixed)
        self.assertIsNotNone(reason)
        self.assertIn("states", reason or "")
        with self.assertRaises(IncompatibleModules):
            self.experiment.assemble(fixed)

    def test_life_is_feasible_on_binary_grid(self) -> None:
        fixed = self.patched(**{"data.states": 2, "rule.module": "rule.life"})
        self.assertIsNone(self.experiment.feasible(fixed))
        assembled = self.experiment.assemble(fixed)
        self.assertIn("rule.life", repr(assembled.modules))

    def test_config_is_serializable(self) -> None:
        fixed = self.base
        restored = Fixed.from_dict(fixed.to_dict())
        self.assertEqual(restored.digest(), fixed.digest())


class TestTrial(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = build_experiment()
        cls.base = next(cls.experiment.iter_configs(feasible=True))

    def patched(self, **overrides) -> Fixed:
        values = self.base.to_dict()
        values.update(overrides)
        return Fixed(values)

    def test_life_trial_is_deterministic(self) -> None:
        fixed = self.patched(**{"data.states": 2, "rule.module": "rule.life"})
        first = run_trial(self.experiment, fixed, seed=42)
        second = run_trial(self.experiment, fixed, seed=42)
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first.state_digest, second.state_digest)
        self.assertIn(
            first.status,
            (TrialStatus.PASSED, TrialStatus.REJECTED, TrialStatus.STOPPED),
        )

    def test_density_filter_rejects_sparse_binary_grid(self) -> None:
        fixed = self.patched(
            **{
                "data.states": 9,
                "data.width": 112,
                "data.height": 112,
                "rule.module": "rule.thresholds",
                "stage.warmup": 0,
                "filter.density.min_fill": 0.95,
            }
        )
        self.assertIsNone(self.experiment.feasible(fixed))
        result = run_trial(self.experiment, fixed, seed=7)
        self.assertEqual(result.status, TrialStatus.REJECTED, result.reason)
        self.assertIn("density", result.reason)
        self.assertLess(result.iterations, 10)

    def test_threshold_trial_runs(self) -> None:
        fixed = self.patched(**{"stage.warmup": 0, "stage.observe": 10})
        result = run_trial(self.experiment, fixed, seed=3)
        self.assertIn(
            result.status,
            (TrialStatus.PASSED, TrialStatus.REJECTED, TrialStatus.STOPPED),
        )
        self.assertGreaterEqual(result.iterations, 1)
        self.assertIsNotNone(result.state_digest)

    def test_feasible_configs_have_no_duplicates(self) -> None:
        import itertools

        configs = list(itertools.islice(self.experiment.iter_configs(feasible=True), 25))
        digests = {fixed.digest() for fixed in configs}
        self.assertEqual(len(digests), len(configs))


if __name__ == "__main__":
    unittest.main()
