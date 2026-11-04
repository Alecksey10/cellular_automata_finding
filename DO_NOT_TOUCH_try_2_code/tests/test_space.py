"""Тесты пространства параметров и фиксированного конфига."""

from __future__ import annotations

import unittest

from casweep.fixed import Fixed
from casweep.space import Space


class TestFixed(unittest.TestCase):
    def test_digest_is_order_independent(self) -> None:
        first = Fixed({"x": 1, "y": "wrap"})
        second = Fixed({"y": "wrap", "x": 1})
        self.assertEqual(first.digest(), second.digest())
        self.assertEqual(Fixed.from_dict(first.to_dict()).digest(), first.digest())

    def test_canonical_is_json(self) -> None:
        fixed = Fixed({"b": 2, "a": 1})
        self.assertEqual(fixed.canonical(), '{"a":1,"b":2}')


class TestSpace(unittest.TestCase):
    def test_dependent_range(self) -> None:
        space = Space()
        width = space.range("width", 16, 128, 32)
        space.range("height", width, 129, 32)
        bindings = list(space)
        # width: 16,48,80,112; height: 4,3,2,1 вариантов соответственно.
        self.assertEqual(len(bindings), 10)
        for binding in bindings:
            self.assertGreaterEqual(binding["height"], binding["width"])
            self.assertLess(binding["height"], 129)

    def test_constraint_prunes_and_count_matches(self) -> None:
        space = Space()
        start = space.range("a", 1, 5)
        space.range("b", start, 7)
        space.constrain(lambda values: values["b"] % 2 == 0, ["b"])
        bindings = list(space)
        self.assertEqual(len(bindings), 10)
        self.assertTrue(all(binding["b"] % 2 == 0 for binding in bindings))
        self.assertEqual(space.count(), len(bindings))

    def test_static_product_count(self) -> None:
        space = Space()
        start = space.range("a", 1, 4)
        space.range("b", start, 10)
        space.choose("c", ("x", "y"))
        # a=1 -> 9 значений b, a=2 -> 8, a=3 -> 7; всего 24 * 2
        self.assertEqual(space.count(), 48)
        self.assertEqual(len(list(space)), 48)

    def test_accept_filters_leaves(self) -> None:
        space = Space()
        space.range("a", 1, 5)
        bindings = list(space.enumerate(accept=lambda values: values["a"] % 2 == 0))
        self.assertEqual([binding["a"] for binding in bindings], [2, 4])
        self.assertEqual(space.count(accept=lambda values: values["a"] % 2 == 0), 2)

    def test_random_binding_respects_constraints(self) -> None:
        space = Space()
        space.range("a", 1, 5)
        space.range("b", 1, 5)
        space.constrain(lambda values: values["a"] < values["b"], ["a", "b"])
        for binding in space.random_bindings(20, seed=1):
            self.assertLess(binding["a"], binding["b"])

    def test_module_selection_branches_and_derived(self) -> None:
        class Alpha:
            id = "fake.alpha"
            derived = ("twice",)

            def declare(self, namespace, view):
                namespace.range("n", 1, 4)

            def derive(self, fixed):
                return {"twice": fixed["role.n"] * 2}

        class Beta:
            id = "fake.beta"
            derived = ("twice",)

            def declare(self, namespace, view):
                namespace.choose("k", ("u", "v"))

            def derive(self, fixed):
                return {"twice": 0}

        class FakeRole:
            value = "role"

        space = Space()
        role = FakeRole()
        selection = space.select(role, [Alpha(), Beta()])
        self.assertEqual(selection.derived("twice").key, "role#twice")

        description = "\n".join(space.describe())
        self.assertIn("[fake.alpha]", description)
        self.assertIn("role.n", description)
        self.assertIn("[fake.beta]", description)
        self.assertIn("role.k", description)

        bindings = list(space)
        self.assertEqual(len(bindings), 3 + 2)
        for binding in bindings:
            if binding["role.module"] == "fake.alpha":
                self.assertNotIn("role.k", binding)
                self.assertEqual(binding["role#twice"], binding["role.n"] * 2)
            else:
                self.assertNotIn("role.n", binding)
                self.assertEqual(binding["role#twice"], 0)

    def test_inclusive_range(self) -> None:
        integers = Space()
        integers.range("a", 1, 5, inclusive=True)
        self.assertEqual([binding["a"] for binding in integers], [1, 2, 3, 4, 5])

        floats = Space()
        floats.range("f", 0.4, 0.7, 0.1, inclusive=True)
        self.assertEqual(
            [binding["f"] for binding in floats], [0.4, 0.5, 0.6, 0.7]
        )

    def test_linspace_and_logspace(self) -> None:
        floats = Space()
        floats.linspace("f", 0, 1, 5)
        self.assertEqual(
            [binding["f"] for binding in floats], [0.0, 0.25, 0.5, 0.75, 1.0]
        )

        integers = Space()
        integers.linspace("i", 0, 10, 5, integer=True)
        points = [binding["i"] for binding in integers]
        self.assertEqual(points[0], 0)
        self.assertEqual(points[-1], 10)
        self.assertEqual(len(points), 5)

        geometric = Space()
        geometric.logspace("g", 1, 100, 3)
        self.assertEqual([round(binding["g"], 6) for binding in geometric], [1.0, 10.0, 100.0])

    def test_shuffle_and_stride_cover_the_same_leaves(self) -> None:
        space = Space()
        space.range("a", 1, 5)
        space.choose("b", ("x", "y"))
        product = list(space)
        shuffled = list(space.enumerate(strategy="shuffle", seed=3))
        key = lambda item: tuple(sorted(item.items()))
        self.assertEqual({key(item) for item in shuffled}, {key(item) for item in product})

        stride = Space()
        stride.range("a", 1, 6)
        self.assertEqual([item["a"] for item in stride.enumerate(stride=2)], [1, 3, 5])

    def test_estimate_tracks_dependent_ranges(self) -> None:
        space = Space()
        width = space.range("width", 16, 128, 32)
        space.range("height", width, 129, 32)
        exact = space.count()
        estimate = space.estimate(samples=128, seed=3)
        self.assertEqual(exact, 10)
        self.assertGreaterEqual(estimate, exact // 2)
        self.assertLessEqual(estimate, exact * 2)

    def test_random_bindings_can_be_unique(self) -> None:
        space = Space()
        space.range("a", 1, 4)
        space.range("b", 1, 4)
        bindings = list(space.random_bindings(20, seed=1, unique=True))
        self.assertEqual(len(bindings), 9)
        self.assertEqual(len({tuple(sorted(item.items())) for item in bindings}), 9)

    def test_constraints_are_described(self) -> None:
        space = Space()
        space.range("a", 1, 4)
        space.constrain(lambda values: values["a"] > 1, ["a"], name="a_positive")
        text = "\n".join(space.describe())
        self.assertIn("Ограничения:", text)
        self.assertIn("a_positive: a", text)

    def test_axis_accessors_and_inclusive_equal_bounds(self) -> None:
        space = Space()
        axis = space.range("a", 5, 5, inclusive=True)
        self.assertIs(space.axis("a"), space.axis(axis.key))
        self.assertEqual(space.keys(), ("a",))
        self.assertEqual(space.dimension, 1)
        self.assertEqual(space.values("a"), (5,))
        self.assertEqual(list(space), [{"a": 5}])

        exclusive = Space()
        exclusive.range("a", 5, 5)
        self.assertEqual(list(exclusive), [])

        descending = Space()
        descending.range("a", 5, 1, -1, inclusive=True)
        self.assertEqual([item["a"] for item in descending], [5, 4, 3, 2, 1])

        wrong_direction = Space()
        wrong_direction.range("a", 1, 10, -1, inclusive=True)
        self.assertEqual(list(wrong_direction), [])

    def test_diagonal_strategy_is_sparse_and_cycles_axes(self) -> None:
        space = Space()
        space.range("a", 1, 5)  # 4 значения
        space.range("b", 10, 13)  # 3 значения
        diagonal = list(space.enumerate(strategy="diagonal"))
        self.assertEqual(len(diagonal), 4)
        self.assertEqual([item["a"] for item in diagonal], [1, 2, 3, 4])
        self.assertEqual([item["b"] for item in diagonal], [10, 11, 12, 10])
        self.assertEqual(space.count(strategy="diagonal"), 4)
        self.assertEqual(space.count(), 12)

        shifted = list(space.enumerate(strategy="diagonal", seed=1))
        self.assertEqual([item["a"] for item in shifted], [2, 3, 4, 1])
        self.assertEqual(len(space.sample(2, seed=1, strategy="diagonal")), 2)

    def test_diagonal_respects_constraints_and_dependent_ranges(self) -> None:
        constrained = Space()
        constrained.range("a", 1, 5)
        constrained.constrain(lambda values: values["a"] % 2 == 0, ["a"])
        self.assertEqual(
            [item["a"] for item in constrained.enumerate(strategy="diagonal")],
            [2, 4],
        )
        self.assertEqual(constrained.count(strategy="diagonal"), 2)

        dependent = Space()
        width = dependent.range("width", 16, 128, 32)
        dependent.range("height", width, 129, 32)
        for binding in dependent.enumerate(strategy="diagonal"):
            self.assertGreaterEqual(binding["height"], binding["width"])
            self.assertLess(binding["height"], 129)
        self.assertEqual(
            dependent.count(strategy="diagonal"),
            len(list(dependent.enumerate(strategy="diagonal"))),
        )

    def test_branch_keys_cannot_be_declared_twice_in_one_module(self) -> None:
        class Broken:
            id = "fake.broken"
            derived = ()

            def declare(self, namespace, view):
                namespace.range("n", 1, 4)
                namespace.value("n", 1)

        class FakeRole:
            value = "role"

        space = Space()
        with self.assertRaises(Exception):
            space.select(FakeRole(), [Broken()])


if __name__ == "__main__":
    unittest.main()
