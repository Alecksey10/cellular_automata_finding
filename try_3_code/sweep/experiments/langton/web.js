/**
 * Браузерный модуль эксперимента «муравей Ленгтона».
 *
 * Сетка хранит след: ``0`` — не посещалась, ``1..colors`` — цвет, числа
 * ``colors+1..2*colors`` — тот же цвет с отметкой муравья. Набор поворотов
 * (``turns``) и стартовые позиции муравьёв (``ants``) приходят в карточке:
 * их посчитал Python по тому же seed, поэтому браузерный прогон совпадает с
 * прогоном перебора.
 */

const DY = [0, 1, 1, 1, 0, -1, -1, -1];
const DX = [1, 1, 0, -1, -1, -1, 0, 1];

registerExperiment({
  id: "langton",
  label: "муравей Ленгтона",
  create(card) {
    const states = Number(card.states) || 3;
    return {
      width: card.width,
      height: card.height,
      grid: gridFromBase64(card.init, card.bits, card.width * card.height),
      next: new Uint8Array(card.width * card.height),
      colors: Math.max(1, (states - 1) >> 1),
      turns: (card.turns || []).slice(),
      ants: (card.ants || []).map((ant) => [ant[0], ant[1], ant[2]]),
      marked: [],
    };
  },
  step(state) {
    const width = state.width;
    const height = state.height;
    const grid = state.grid;
    const next = state.next;
    const colors = state.colors;
    const turns = state.turns;
    next.set(grid);
    // снять отметки прошлого шага
    for (let i = 0; i < state.marked.length; i += 1) {
      const index = state.marked[i];
      const value = next[index];
      if (value > colors) next[index] = value - colors;
    }
    state.marked.length = 0;
    for (const ant of state.ants) {
      let y = ant[0];
      let x = ant[1];
      let heading = ant[2];
      let value = grid[y * width + x];
      if (value > colors) value -= colors;
      const color = value > 0 ? value : 1;
      heading = (((heading + turns[color - 1]) % 8) + 8) % 8;
      next[y * width + x] = (color % colors) + 1;
      const nextY = (y + DY[heading] + height) % height;
      const nextX = (x + DX[heading] + width) % width;
      let target = grid[nextY * width + nextX];
      if (target > colors) target -= colors;
      if (target <= 0) target = 1;
      next[nextY * width + nextX] = target + colors;
      state.marked.push(nextY * width + nextX);
      ant[0] = nextY;
      ant[1] = nextX;
      ant[2] = heading;
    }
    state.grid = next;
    state.next = grid;
  },
});
