/**
 * Смоук-тест браузерного runtime без DOM.
 *
 * Запуск: node tests/web_runtime.test.js /tmp/rules.js
 * Первый аргумент — собранный ``bundle_source([...])`` из Python-теста.
 */

const fs = require("fs");
const assert = require("assert");

const bundle = process.argv[2];
if (!bundle) throw new Error("нужен путь к собранному rules.js");
eval(fs.readFileSync(bundle, "utf8"));

assert.ok(SweepRules && SweepRules.createRunner, "createRunner должен быть экспортирован");

const card = {
  experiment: "life",
  width: 8,
  height: 8,
  bits: 1,
  init: "",
  final: "",
  states: 2,
  iterations: 5,
  digest: "test-card",
  values: {
    "select.neighborhood": "neighborhood.moore",
    "neighborhood.moore.radius": 1,
    "neighborhood.moore.wrap": true,
    "grid2d.zero": 0,
    "rule.life.born_lo": 3,
    "rule.life.born_hi": 3,
    "rule.life.surv_lo": 2,
    "rule.life.surv_hi": 3,
  },
};

const runner = SweepRules.createRunner(card, { iterations: 3, speed: 1, loop: true });
assert.strictEqual(runner.width, 8);
assert.strictEqual(runner.height, 8);
assert.strictEqual(runner.state.grid.length, 64);
assert.strictEqual(runner.iteration, 0);

runner.advance(3);
assert.strictEqual(runner.iteration, 3);
assert.strictEqual(runner.done, true);

// цикл: цель достигнута, следующий шаг начинает состояние заново
runner.advance(1);
assert.strictEqual(runner.iteration, 1);

runner.setIterations(10);
assert.strictEqual(runner.iterations, 10);

// размер меняет инициализирующую сетку и пересоздаёт состояние
runner.setSize(16);
assert.strictEqual(runner.width, 16);
assert.strictEqual(runner.height, 16);
assert.strictEqual(runner.state.grid.length, 256);
assert.strictEqual(runner.iteration, 0);

// без цикла автомат останавливается на цели
runner.loop = false;
runner.setIterations(2);
runner.advance(2);
assert.strictEqual(runner.iteration, 2);
assert.strictEqual(runner.advance(5), 0);
assert.strictEqual(runner.iteration, 2);

// упаковка сетки переживает round-trip
const grid = new Uint8Array(16);
grid[0] = 1;
grid[15] = 1;
const packed = SweepRules.gridToBase64(grid, 1);
const restored = SweepRules.gridFromBase64(packed, 1, 16);
assert.deepStrictEqual(Array.from(restored), Array.from(grid));

console.log("web runtime: ok");
