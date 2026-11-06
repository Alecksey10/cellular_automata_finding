/**
 * Браузерный модуль эксперимента «циклические волны».
 *
 * Клетка переходит в следующую фазу, если рядом достаточно соседей этой фазы;
 * верхняя фаза либо сбрасывается в ноль, либо остаётся на месте. Правило
 * детерминированное, поэтому браузерный прогон совпадает с прогоном перебора.
 */

registerExperiment({
  id: "cyclic_waves",
  label: "циклические фазы",
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
      states: Number(values["grid2d.states"] || card.states || 3),
      threshold: Number(values["rule.cyclic.threshold"]),
      upperResets: Boolean(values["rule.cyclic.upper_resets"]),
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
    const states = state.states;
    const last = states - 1;
    const wrap = state.wrap;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const value = grid[y * width + x];
        let upcoming = value + 1;
        if (upcoming >= states) upcoming = 0;
        let count = 0;
        for (let k = 0; k < total; k += 1) {
          let yy = y + dy[k];
          let xx = x + dx[k];
          if (wrap) {
            yy = ((yy % height) + height) % height;
            xx = ((xx % width) + width) % width;
          } else if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
            continue;
          }
          if (grid[yy * width + xx] === upcoming) count += 1;
        }
        const index = y * width + x;
        if (value === last && state.upperResets) next[index] = state.zero;
        else if (count >= state.threshold) next[index] = upcoming;
        else next[index] = value;
      }
    }
    state.grid = next;
    state.next = grid;
  },
});
