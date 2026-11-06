/**
 * Браузерный модуль эксперимента «лабиринт».
 *
 * Ядро роста стен живёт в ``sweep/visualize/web/engine.js``: ``maze`` и
 * ``maze9`` отличаются только подмаской внутри окна, поэтому модуль — это
 * одна строка настройки.
 */

registerExperiment(makeMazeRule({
  id: "maze",
  prefix: "rule.maze",
  mechanism: false,
  label: "рост стен лабиринта",
}));
