/**
 * Браузерный модуль эксперимента «два кольца с гниением».
 *
 * Рождение смотрит на внутреннее кольцо, выживание — на всё окно; живая клетка
 * копит прочность, а верхняя медленно гниёт. Случайность гниения не берётся из
 * общего ГПСЧ: она детерминированно считается из координат, итерации и seed,
 * поэтому браузерный прогон совпадает с Python (включая int64-арифметику —
 * она сделана на BigInt).
 */

function rotRandom(y, x, iteration, seed) {
  let mixed = (BigInt(y) * 73856093n) ^ (BigInt(x) * 19349663n) ^
    (BigInt(iteration) * 83492791n) ^ (BigInt(seed % 100000) * 2654435n);
  mixed = (mixed * 1103515245n + 12345n) & 0x7fffffffn;
  mixed = (mixed * 1103515245n + 12345n) & 0x7fffffffn;
  return Number(mixed) / 2147483648;
}

registerExperiment({
  id: "rings_rot",
  label: "два кольца + гниение",
  create(card) {
    const values = card.values || {};
    const inner = innerNeighborhoodOffsets(values);
    const offsets = neighborhoodOffsets(values);
    return {
      width: card.width,
      height: card.height,
      grid: gridFromBase64(card.init, card.bits, card.width * card.height),
      next: new Uint8Array(card.width * card.height),
      outerDy: Int32Array.from(offsets.map((pair) => pair[0])),
      outerDx: Int32Array.from(offsets.map((pair) => pair[1])),
      innerDy: Int32Array.from(inner.map((pair) => pair[0])),
      innerDx: Int32Array.from(inner.map((pair) => pair[1])),
      wrap: neighborhoodWrap(values),
      zero: Number(values["grid2d.zero"] || 0),
      states: Number(values["grid2d.states"] || card.states || 3),
      bornInner: Number(values["rule.rot_aging.born_inner"]),
      bornTotal: Number(values["rule.rot_aging.born_total"]),
      survLo: Number(values["rule.rot_aging.surv_lo"]),
      survHi: Number(values["rule.rot_aging.surv_hi"]),
      rot: Number(values["rule.rot_aging.rot"]),
      seed: Number(card.seed || 0),
      iteration: 0,
    };
  },
  step(state) {
    const width = state.width;
    const height = state.height;
    const grid = state.grid;
    const next = state.next;
    const wrap = state.wrap;
    const zero = state.zero;
    const maxState = state.states - 1;
    const outerTotal = state.outerDy.length;
    const innerTotal = state.innerDy.length;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let inner = 0;
        let outer = 0;
        for (let k = 0; k < outerTotal; k += 1) {
          let yy = y + state.outerDy[k];
          let xx = x + state.outerDx[k];
          if (wrap) {
            yy = ((yy % height) + height) % height;
            xx = ((xx % width) + width) % width;
          } else if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
            continue;
          }
          if (grid[yy * width + xx] !== zero) outer += 1;
        }
        for (let k = 0; k < innerTotal; k += 1) {
          let yy = y + state.innerDy[k];
          let xx = x + state.innerDx[k];
          if (wrap) {
            yy = ((yy % height) + height) % height;
            xx = ((xx % width) + width) % width;
          } else if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
            continue;
          }
          if (grid[yy * width + xx] !== zero) inner += 1;
        }
        const index = y * width + x;
        const value = grid[index];
        if (value === zero) {
          next[index] = inner >= state.bornInner && outer >= state.bornTotal ? 1 : zero;
          continue;
        }
        let upcoming;
        if (outer >= state.survLo && outer <= state.survHi) {
          upcoming = value + 1;
          if (upcoming > maxState) upcoming = maxState;
        } else {
          upcoming = value - 1;
        }
        if (upcoming < zero) upcoming = zero;
        if (value === maxState && rotRandom(y, x, state.iteration, state.seed) < state.rot) {
          upcoming = zero;
        }
        next[index] = upcoming;
      }
    }
    state.iteration += 1;
    state.grid = next;
    state.next = grid;
  },
});
