/**
 * Браузерный модуль эксперимента «жизнь»: B/S-пороги по числу соседей.
 *
 * Клетка не равна ``zero`` — живая. Рождение и выживание проверяются по
 * включительным границам из конфига; за краем сетки соседей нет.
 */

registerExperiment({
  id: "life",
  label: "B/S-пороги",
  create(card) {
    const values = card.values || {};
    const offsets = neighborhoodOffsets(values);
    return {
      width: card.width,
      height: card.height,
      grid: gridFromBase64(card.init, card.bits, card.width * card.height),
      next: new Uint8Array(card.width * card.height),
      dy: Int32Array.from(offsets.map((pair) => pair[0])),
      dx: Int32Array.from(offsets.map((pair) => pair[1])),
      wrap: neighborhoodWrap(values),
      zero: Number(values["grid2d.zero"] || 0),
      bornLo: Number(values["rule.life.born_lo"]),
      bornHi: Number(values["rule.life.born_hi"]),
      survLo: Number(values["rule.life.surv_lo"]),
      survHi: Number(values["rule.life.surv_hi"]),
    };
  },
  step(state) {
    const width = state.width;
    const height = state.height;
    const grid = state.grid;
    const next = state.next;
    const dy = state.dy;
    const dx = state.dx;
    const total = dy.length;
    const zero = state.zero;
    const wrap = state.wrap;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let count = 0;
        for (let k = 0; k < total; k += 1) {
          let yy = y + dy[k];
          let xx = x + dx[k];
          if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
            if (!wrap) continue;
            yy = ((yy % height) + height) % height;
            xx = ((xx % width) + width) % width;
          }
          if (grid[yy * width + xx] !== zero) count += 1;
        }
        const index = y * width + x;
        if (grid[index] === zero) {
          next[index] = count >= state.bornLo && count <= state.bornHi ? 1 : zero;
        } else {
          next[index] = count >= state.survLo && count <= state.survHi ? 1 : zero;
        }
      }
    }
    state.grid = next;
    state.next = grid;
  },
});
