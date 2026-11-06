/**
 * Браузерный модуль эксперимента «дуэль двух групп» (``expBeamFree``).
 *
 * Правило — прямой порт ``sweep/experiments/exp_beam_free/rule.py``:
 * внешнее кольцо 5×5 даёт силы A/B, внутреннее кольцо 3×3 решает подавление
 * и конверсию. Код механизма приходит из Python в ``card.params``, чтобы
 * порядок ``MECHANISMS`` не расходился между языками.
 */

const DUEL_MAJORITY = 0;
const DUEL_POWER = 1;
const DUEL_LEVEL = 2;
const DUEL_ANNIHIL = 3;
const DUEL_ASYMM = 4;
const DUEL_CONVERT = 5;
const DUEL_REINF = 6;
const DUEL_EXCLUDE = 7;
const DUEL_RING = 8;
const DUEL_ESCAL = 9;

registerExperiment({
  id: "expBeamFree",
  label: "дуэль двух групп",
  create(card) {
    const values = card.values || {};
    const params = card.params || {};
    const offsets = neighborhoodOffsets(values);
    const inner = innerNeighborhoodOffsets(values);
    return {
      width: card.width,
      height: card.height,
      grid: gridFromBase64(card.init, card.bits, card.width * card.height),
      next: new Uint8Array(card.width * card.height),
      dy: Int32Array.from(offsets.map((pair) => pair[0])),
      dx: Int32Array.from(offsets.map((pair) => pair[1])),
      innerDy: Int32Array.from(inner.map((pair) => pair[0])),
      innerDx: Int32Array.from(inner.map((pair) => pair[1])),
      wrap: neighborhoodWrap(values),
      zero: Number(values["grid2d.zero"] || 0),
      states: Number(values["grid2d.states"] || card.states || 9),
      split: Number(values["rule.duel.split"]),
      mechanism: Number(params.mechanism || 0),
      birthA: Number(values["rule.duel.birth_a"]),
      birthB: Number(values["rule.duel.birth_b"]),
      survALo: Number(values["rule.duel.surv_a_lo"]),
      survAHi: Number(values["rule.duel.surv_a_hi"]),
      survBLo: Number(values["rule.duel.surv_b_lo"]),
      survBHi: Number(values["rule.duel.surv_b_hi"]),
      suppressA: Number(values["rule.duel.suppress_a"]),
      suppressB: Number(values["rule.duel.suppress_b"]),
      convertA: Number(values["rule.duel.convert_a"]),
      convertB: Number(values["rule.duel.convert_b"]),
      boostA: Number(values["rule.duel.boost_a"]),
      boostB: Number(values["rule.duel.boost_b"]),
      blockA: Number(values["rule.duel.block_a"]),
      blockB: Number(values["rule.duel.block_b"]),
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
    const wrap = state.wrap;
    const zero = state.zero;
    const states = state.states;
    const split = state.split;
    const mechanism = state.mechanism;
    const bLevels = Math.max(1, states - 1 - split);

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let aIn = 0;
        let bIn = 0;
        let aOut = 0;
        let bOut = 0;
        let aPowerIn = 0;
        let bPowerIn = 0;
        let aPowerOut = 0;
        let bPowerOut = 0;
        let aMax = 0;
        let bMax = 0;

        for (let k = 0; k < total; k += 1) {
          let yy = y + dy[k];
          let xx = x + dx[k];
          if (wrap) {
            yy = ((yy % height) + height) % height;
            xx = ((xx % width) + width) % width;
          } else if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
            continue;
          }
          const value = grid[yy * width + xx];
          if (value === zero) continue;
          const inner = Math.abs(dy[k]) <= 1 && Math.abs(dx[k]) <= 1;
          if (value <= split) {
            if (inner) {
              aIn += 1;
              aPowerIn += value;
            } else {
              aOut += 1;
              aPowerOut += value;
            }
            if (value > aMax) aMax = value;
          } else {
            const level = value - split;
            if (inner) {
              bIn += 1;
              bPowerIn += level;
            } else {
              bOut += 1;
              bPowerOut += level;
            }
            if (level > bMax) bMax = level;
          }
        }

        const aTotal = aIn + aOut;
        const bTotal = bIn + bOut;
        const aPowerTotal = aPowerIn + aPowerOut;
        const bPowerTotal = bPowerIn + bPowerOut;
        const index = y * width + x;
        const cell = grid[index];
        const group = cell > zero ? (cell <= split ? 1 : 2) : 0;
        let value = zero;

        if (group === 1) {
          if (bIn >= state.suppressA) {
            if (state.convertA && bPowerIn >= aPowerIn) {
              let level = bIn;
              if (level < 1) level = 1;
              if (level > bLevels) level = bLevels;
              value = split + level;
            } else {
              value = zero;
            }
          } else if ((aOut >= state.survALo && aOut <= state.survAHi) ||
                     (state.boostA > 0 && aIn >= state.boostA)) {
            value = cell;
            if (state.boostA > 0 && aIn >= state.boostA && cell < split) value = cell + 1;
          } else {
            value = zero;
          }
        } else if (group === 2) {
          if (aIn >= state.suppressB) {
            if (state.convertB && aPowerIn >= bPowerIn) {
              let level = aIn;
              if (level < 1) level = 1;
              if (level > split) level = split;
              value = level;
            } else {
              value = zero;
            }
          } else if ((bOut >= state.survBLo && bOut <= state.survBHi) ||
                     (state.boostB > 0 && bIn >= state.boostB)) {
            value = cell;
            if (state.boostB > 0 && bIn >= state.boostB && cell < states - 1) value = cell + 1;
          } else {
            value = zero;
          }
        } else {
          const birthA = aIn >= state.birthA && bIn < state.blockA;
          const birthB = bIn >= state.birthB && aIn < state.blockB;
          if (birthA && !birthB) {
            value = 1;
          } else if (birthB && !birthA) {
            value = split + 1;
          } else if (birthA && birthB) {
            switch (mechanism) {
              case DUEL_MAJORITY:
                value = aTotal >= bTotal ? 1 : split + 1;
                break;
              case DUEL_POWER:
                value = aPowerTotal >= bPowerTotal ? 1 : split + 1;
                break;
              case DUEL_LEVEL:
                value = aMax >= bMax ? 1 : split + 1;
                break;
              case DUEL_ANNIHIL:
                value = aTotal >= state.suppressA && bTotal >= state.suppressB
                  ? zero
                  : (aTotal >= bTotal ? 1 : split + 1);
                break;
              case DUEL_ASYMM:
                value = aIn >= bIn ? 1 : split + 1;
                break;
              case DUEL_CONVERT:
                value = aPowerTotal >= bPowerTotal ? 1 : split + 1;
                break;
              case DUEL_REINF:
                value = aIn >= bIn ? 1 : split + 1;
                break;
              case DUEL_EXCLUDE:
                value = aIn >= bIn ? 1 : split + 1;
                break;
              case DUEL_RING:
                value = aIn >= bOut ? 1 : split + 1;
                break;
              case DUEL_ESCAL:
                value = aTotal >= bTotal ? 1 : split + 1;
                break;
              default:
                value = aTotal >= bTotal ? 1 : split + 1;
            }
          } else {
            value = zero;
          }
        }

        if (group === 1 && mechanism === DUEL_ANNIHIL &&
            bIn >= state.suppressA && aIn >= state.suppressB) value = zero;
        if (group === 2 && mechanism === DUEL_ANNIHIL &&
            aIn >= state.suppressB && bIn >= state.suppressA) value = zero;
        if (group === 1 && mechanism === DUEL_EXCLUDE && bIn >= state.blockA) value = zero;
        if (group === 2 && mechanism === DUEL_EXCLUDE && aIn >= state.blockB) value = zero;

        if (value < zero) value = zero;
        if (value >= states) value = states - 1;
        next[index] = value;
      }
    }
    state.grid = next;
    state.next = grid;
  },
});
