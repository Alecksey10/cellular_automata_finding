/**
 * Браузерный модуль эксперимента «лабиринт 9×9».
 *
 * Механизм (``inner``, ``middle``, ``outer``, ``rim``, ``cross``, …) выбирает
 * подмаску внутри окна 9×9, а пороги заданы в процентах от её размера.
 */

registerExperiment(makeMazeRule({
  id: "maze9",
  prefix: "rule.maze9",
  mechanism: true,
  label: "механизм в окне 9×9",
}));
