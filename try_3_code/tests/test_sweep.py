"""Быстрые проверки движка без pytest: ``python -m unittest discover -s tests``."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sweep.config import Config
from sweep.assembly import assemble
from sweep.experiments import get_experiment, list_experiments
from sweep.replay import frames_for
from sweep.runtime import SUCCESS, Trial
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


class StorageTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        experiment, config = RuntimeTest()._first_config("cyclic_waves")
        result = Trial(assemble(experiment, config), config.seed, record=True).run()
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            record = store.save(config, result)
            self.assertTrue((Path(tmp) / "trials.jsonl").exists())
            self.assertEqual(store.done_digests(), {record["digest"]})
            # массивы на диск не пишутся: в каталоге прогона только журналы
            files = sorted(path.name for path in Path(tmp).rglob("*") if path.is_file())
            self.assertEqual(files, ["trials.jsonl"])
            # кадры получаются повторной сборкой того же конфига
            frames = frames_for(record)
            self.assertEqual(len(frames), len(result.frames))


if __name__ == "__main__":
    unittest.main()
