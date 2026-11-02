"""Тесты раннера: процессный пул, запись результатов, возобновление."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from casweep.cli import DEFAULT_EXPERIMENT, infer_experiment_spec, load_experiment_for_run
from casweep.runner import RunOptions, run
from experiments.life2d import build_experiment


class TestRunner(unittest.TestCase):
    def test_parallel_run_and_resume(self) -> None:
        experiment = build_experiment()
        with tempfile.TemporaryDirectory() as directory:
            options = RunOptions(
                limit=2,
                workers=2,
                mode="process",
                seed=5,
                out=directory,
                save_all=True,
                progress=False,
            )
            summary = run(experiment, options)
            self.assertEqual(summary.executed, 2)
            self.assertEqual(summary.errors, 0)

            trials = Path(directory) / "trials.jsonl"
            self.assertTrue(trials.exists())
            records = [json.loads(line) for line in trials.read_text().splitlines() if line]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertIn("config", record)
                self.assertIn("config_digest", record)
                self.assertIn(record["status"], ("passed", "rejected", "stopped"))

            resume_options = RunOptions(
                limit=2,
                workers=1,
                mode="serial",
                seed=5,
                out=directory,
                resume=True,
                save_all=True,
                progress=False,
            )
            resumed = run(experiment, resume_options)
            self.assertEqual(resumed.skipped, 2)
            self.assertEqual(resumed.executed, 2)
            all_records = [
                json.loads(line)
                for line in trials.read_text().splitlines()
                if line
            ]
            digests = [record["config_digest"] for record in all_records]
            self.assertEqual(len(digests), 4)
            self.assertEqual(len(set(digests)), 4)

    def test_run_dir_infers_experiment(self) -> None:
        config = {
            "rule.module": "rule.aging_fortress",
            "neighborhood.module": "neighborhood.two_rings",
            "data.states": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "successes.jsonl").write_text(
                json.dumps({"config": config, "status": "passed", "config_digest": "test"}),
                encoding="utf-8",
            )
            self.assertEqual(
                infer_experiment_spec(directory),
                "experiments.two_rings:build_experiment",
            )

            moore_config = {
                "rule.module": "rule.vote",
                "neighborhood.module": "neighborhood.moore3",
                "filter.symmetry.min_score": 0.7,
            }
            Path(directory, "successes.jsonl").write_text(
                json.dumps({"config": moore_config, "status": "passed"}),
                encoding="utf-8",
            )
            self.assertEqual(
                infer_experiment_spec(directory),
                "experiments.moore3:build_experiment",
            )

            multi_config = {
                **moore_config,
                "rule.module": "rule.cyclic",
                "filter.phases.min_share": 0.02,
            }
            Path(directory, "successes.jsonl").write_text(
                json.dumps({"config": multi_config, "status": "passed"}),
                encoding="utf-8",
            )
            self.assertEqual(
                infer_experiment_spec(directory),
                "experiments.multi3:build_experiment",
            )

            # Возвращаем two_rings для проверки загрузки эксперимента.
            Path(directory, "successes.jsonl").write_text(
                json.dumps({"config": config, "status": "passed", "config_digest": "test"}),
                encoding="utf-8",
            )
            inferred = load_experiment_for_run(
                argparse.Namespace(experiment=DEFAULT_EXPERIMENT, run_dir=directory)
            )
            self.assertEqual(inferred.name, "two_rings")

            explicit = load_experiment_for_run(
                argparse.Namespace(
                    experiment="experiments.cyclic5:build_experiment",
                    run_dir=directory,
                )
            )
            self.assertEqual(explicit.name, "cyclic5")

    def test_diagonal_experiment_count_matches_iteration(self) -> None:
        experiment = build_experiment()
        expected = experiment.count(strategy="diagonal", feasible=True)
        actual = len(list(experiment.iter_configs(strategy="diagonal")))
        self.assertEqual(expected, actual)
        self.assertGreater(actual, 0)
        for fixed in experiment.iter_configs(strategy="diagonal"):
            self.assertIsNone(experiment.feasible(fixed))

    def test_serial_run_collects_results(self) -> None:
        experiment = build_experiment()
        options = RunOptions(limit=2, mode="serial", seed=1, progress=False)
        summary = run(experiment, options)
        self.assertEqual(summary.executed, 2)
        self.assertEqual(summary.executed, summary.passed + summary.rejected + summary.stopped + summary.errors)


if __name__ == "__main__":
    unittest.main()
