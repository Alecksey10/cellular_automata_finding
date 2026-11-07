/**
 * Браузерный модуль эксперимента «симбиоз» (``symbiosis``).
 *
 * Прямой порт ``sweep/experiments/symbiosis/kernel.py``: опора R20/R30
 * (круг или внешнее кольцо), запрет рождения вплотную к опоре, зависимые с
 * секторами круга R5, споры и импульсы. Номера секторов приходят из Python
 * в ``card.params.sector``, чтобы ``atan2`` не разошёлся между языками.
 * Случайность берётся не из общего ГПСЧ, а детерминированно из координат,
 * итерации и seed — как в Python, с int64-арифметикой на BigInt.
 */

const SYM_VOID = 0;
const SYM_ANCHOR = 1;
const SYM_MATRIX = 2;
const SYM_GRAZER = 3;
const SYM_WEAVER = 4;
const SYM_SPORE = 5;
const SYM_PULSE = 6;

const SYM_LOGIC_RADIUS = 5;
const SYM_SPACING_RADIUS = 2;
const SYM_SECTOR_MODES = ["spread", "ring", "gaps", "parity"];

/** Круг смещений, как ``disc_offsets`` в Python: целочисленная проверка d². */
function symbiosisDiscOffsets(radius, metric) {
  const offsets = [];
  for (let dy = -radius; dy <= radius; dy += 1) {
    for (let dx = -radius; dx <= radius; dx += 1) {
      if (!dy && !dx) continue;
      let inside;
      if (metric === "chebyshev") {
        inside = Math.max(Math.abs(dy), Math.abs(dx)) <= radius;
      } else if (metric === "manhattan") {
        inside = Math.abs(dy) + Math.abs(dx) <= radius;
      } else {
        inside = dy * dy + dx * dx <= radius * radius;
      }
      if (inside) offsets.push([dy, dx]);
    }
  }
  return offsets;
}

function symbiosisDistance(dy, dx, metric) {
  if (metric === "chebyshev") return Math.max(Math.abs(dy), Math.abs(dx));
  if (metric === "manhattan") return Math.abs(dy) + Math.abs(dx);
  return Math.hypot(dy, dx);
}

/** Внешнее кольцо круга R5: те же смещения, что ``logic_offsets("ring")``. */
function symbiosisLogicOffsets(mode, metric, radius) {
  const offsets = symbiosisDiscOffsets(radius, metric);
  if (mode !== "ring") return offsets;
  const cut = 0.72 * radius;
  return offsets.filter((pair) => {
    const distance = symbiosisDistance(pair[0], pair[1], metric);
    return distance > cut && distance <= radius;
  });
}

/** Запасной вариант, если в карточке нет готовых номеров секторов. */
function symbiosisSectorIds(offsets, sectors) {
  const tau = 2 * Math.PI;
  const turn = (0.5 / sectors) * tau;
  return offsets.map((pair) => {
    let angle = (Math.atan2(pair[0], pair[1]) + turn) % tau;
    if (angle < 0) angle += tau;
    const index = Math.floor((angle / tau) * sectors);
    return Math.min(sectors - 1, Math.max(0, index));
  });
}

/** Детерминированное число [0, 1): точная копия ``noise01`` (int64 на BigInt). */
function symbiosisNoise(y, x, iteration, seed) {
  let mixed = (BigInt(y) * 73856093n) ^ (BigInt(x) * 19349663n) ^
    (BigInt(iteration) * 83492791n) ^ (BigInt(seed % 100000) * 2654435n);
  mixed = (mixed * 1103515245n + 12345n) & 0x7fffffffn;
  mixed = (mixed * 1103515245n + 12345n) & 0x7fffffffn;
  return Number(mixed) / 2147483648;
}

/** Рисунок опоры по секторам: ``sector_ok`` из Python. */
function symbiosisSectorOk(counts, sectors, mode, need) {
  let nonempty = 0;
  let even = 0;
  let odd = 0;
  for (let index = 0; index < sectors; index += 1) {
    const value = counts[index];
    if (value > 0) nonempty += 1;
    if (index % 2 === 0) even += value;
    else odd += value;
  }
  if (mode === 0) return nonempty >= need; // spread
  if (mode === 1) return nonempty === sectors; // ring
  if (mode === 2) return nonempty >= 1 && nonempty < sectors; // gaps
  return even > 0 && odd > 0; // parity: маска чётных/нечётных долей
}

registerExperiment({
  id: "symbiosis",
  label: "опора R20/R30 + секторы R5",
  create(card) {
    const values = card.values || {};
    const params = card.params || {};
    const metric = String(values["rule.symbiosis.metric"] || "euclidean");
    const supportWindow = String(values["rule.symbiosis.support_window"] || "disc");
    const logicWindow = String(values["rule.symbiosis.logic_window"] || "disc");
    const anchorRadius = Number(values["rule.symbiosis.anchor_radius"] || 20);
    const matrixRadius = Number(values["rule.symbiosis.matrix_radius"] || 30);
    const sectors = Number(values["rule.symbiosis.sectors"] || 8);
    const modeName = String(values["rule.symbiosis.sector_mode"] || "spread");
    const mode = Math.max(0, SYM_SECTOR_MODES.indexOf(modeName));

    const support = symbiosisDiscOffsets(matrixRadius, metric);
    const cutAnchor = 0.72 * anchorRadius;
    const cutMatrix = 0.72 * matrixRadius;
    const inAnchor = new Uint8Array(support.length);
    const inMatrix = new Uint8Array(support.length);
    for (let k = 0; k < support.length; k += 1) {
      const distance = symbiosisDistance(support[k][0], support[k][1], metric);
      if (supportWindow === "ring") {
        inAnchor[k] = distance > cutAnchor && distance <= anchorRadius ? 1 : 0;
        inMatrix[k] = distance > cutMatrix && distance <= matrixRadius ? 1 : 0;
      } else {
        inAnchor[k] = distance <= anchorRadius ? 1 : 0;
        inMatrix[k] = distance <= matrixRadius ? 1 : 0;
      }
    }
    const logic = symbiosisLogicOffsets(logicWindow, metric, SYM_LOGIC_RADIUS);
    const sector = Array.isArray(params.sector) && params.sector.length === logic.length
      ? Int32Array.from(params.sector)
      : Int32Array.from(symbiosisSectorIds(logic, sectors));
    const spacing = symbiosisDiscOffsets(SYM_SPACING_RADIUS, "chebyshev");
    const width = Number(card.width);
    const height = Number(card.height);

    return {
      width: width,
      height: height,
      grid: gridFromBase64(card.init, card.bits, width * height),
      next: new Uint8Array(width * height),
      sdy: Int32Array.from(support.map((pair) => pair[0])),
      sdx: Int32Array.from(support.map((pair) => pair[1])),
      inAnchor: inAnchor,
      inMatrix: inMatrix,
      ldy: Int32Array.from(logic.map((pair) => pair[0])),
      ldx: Int32Array.from(logic.map((pair) => pair[1])),
      ndy: Int32Array.from(spacing.map((pair) => pair[0])),
      ndx: Int32Array.from(spacing.map((pair) => pair[1])),
      sector: sector,
      anchorSectors: new Int32Array(sectors),
      matrixSectors: new Int32Array(sectors),
      sectors: sectors,
      mode: mode,
      need: Number(values["rule.symbiosis.sector_need"] || 1),
      wrap: Boolean(values["rule.symbiosis.wrap"]),
      zero: Number(values["grid2d.zero"] || 0),
      nAnchorBirth: Number(values["rule.symbiosis.n_anchor_birth"] || 20),
      mAnchorBirth: Number(values["rule.symbiosis.m_anchor_birth"] || 1),
      nAnchorSurv: Number(values["rule.symbiosis.n_anchor_surv"] || 2),
      mAnchorSurv: Number(values["rule.symbiosis.m_anchor_surv"] || 0),
      nAnchorMax: Number(values["rule.symbiosis.n_anchor_max"] || 400),
      nMatrixBirth: Number(values["rule.symbiosis.n_matrix_birth"] || 30),
      mMatrixBirth: Number(values["rule.symbiosis.m_matrix_birth"] || 1),
      nMatrixSurv: Number(values["rule.symbiosis.n_matrix_surv"] || 2),
      mMatrixSurv: Number(values["rule.symbiosis.m_matrix_surv"] || 0),
      nMatrixMax: Number(values["rule.symbiosis.n_matrix_max"] || 540),
      grazerBirth: Number(values["rule.symbiosis.grazer_birth"] || 2),
      grazerSurv: Number(values["rule.symbiosis.grazer_surv"] || 1),
      grazerMax: Number(values["rule.symbiosis.grazer_max"] || 12),
      weaverBirth: Number(values["rule.symbiosis.weaver_birth"] || 2),
      weaverSurv: Number(values["rule.symbiosis.weaver_surv"] || 1),
      weaverMax: Number(values["rule.symbiosis.weaver_max"] || 12),
      pulseProb: Number(values["rule.symbiosis.pulse_prob"] || 0.04),
      sporeDecay: Number(values["rule.symbiosis.spore_decay"] || 0.01),
      seed: Number(card.seed || 0),
    };
  },
  step(state, rng, env) {
    const width = state.width;
    const height = state.height;
    const grid = state.grid;
    const next = state.next;
    const wrap = state.wrap;
    const zero = state.zero;
    const sdy = state.sdy;
    const sdx = state.sdx;
    const inAnchor = state.inAnchor;
    const inMatrix = state.inMatrix;
    const ldy = state.ldy;
    const ldx = state.ldx;
    const ndy = state.ndy;
    const ndx = state.ndx;
    const sector = state.sector;
    const anchorSectors = state.anchorSectors;
    const matrixSectors = state.matrixSectors;
    const sectors = state.sectors;
    const mode = state.mode;
    const need = state.need;
    const supportTotal = sdy.length;
    const logicTotal = ldy.length;
    const spacingTotal = ndy.length;
    const iteration = Number(env && env.iteration) || 0;
    const seed = state.seed;

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        // --- крупный круг: опора и её зависимые -----------------------------
        let anchors = 0;
        let matrices = 0;
        for (let k = 0; k < supportTotal; k += 1) {
          let yy = y + sdy[k];
          let xx = x + sdx[k];
          if (wrap) {
            yy = ((yy % height) + height) % height;
            xx = ((xx % width) + width) % width;
          } else if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
            continue;
          }
          const value = grid[yy * width + xx];
          if (inAnchor[k] && value === SYM_ANCHOR) anchors += 1;
          if (inMatrix[k] && value === SYM_MATRIX) matrices += 1;
        }

        // --- запрет вплотную: опора рождается только в промежутках ----------
        let nearAnchor = 0;
        let nearMatrix = 0;
        let nearGrazer = 0;
        let nearWeaver = 0;
        for (let k = 0; k < spacingTotal; k += 1) {
          let yy = y + ndy[k];
          let xx = x + ndx[k];
          if (wrap) {
            yy = ((yy % height) + height) % height;
            xx = ((xx % width) + width) % width;
          } else if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
            continue;
          }
          const value = grid[yy * width + xx];
          if (value === SYM_ANCHOR) nearAnchor += 1;
          else if (value === SYM_MATRIX) nearMatrix += 1;
          else if (value === SYM_GRAZER) nearGrazer += 1;
          else if (value === SYM_WEAVER) nearWeaver += 1;
        }

        // --- круг радиуса 5: зависимая логика и секторы ---------------------
        for (let index = 0; index < sectors; index += 1) {
          anchorSectors[index] = 0;
          matrixSectors[index] = 0;
        }
        let anchorsNear = 0;
        let matricesNear = 0;
        let grazers = 0;
        let weavers = 0;
        let pulses = 0;
        for (let k = 0; k < logicTotal; k += 1) {
          let yy = y + ldy[k];
          let xx = x + ldx[k];
          if (wrap) {
            yy = ((yy % height) + height) % height;
            xx = ((xx % width) + width) % width;
          } else if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
            continue;
          }
          const value = grid[yy * width + xx];
          const index = sector[k];
          if (value === SYM_ANCHOR) {
            anchorsNear += 1;
            anchorSectors[index] += 1;
          } else if (value === SYM_MATRIX) {
            matricesNear += 1;
            matrixSectors[index] += 1;
          } else if (value === SYM_GRAZER) {
            grazers += 1;
          } else if (value === SYM_WEAVER) {
            weavers += 1;
          } else if (value === SYM_PULSE) {
            pulses += 1;
          }
        }

        const index = y * width + x;
        const current = grid[index];
        const anchorShape = symbiosisSectorOk(anchorSectors, sectors, mode, need);
        const matrixShape = symbiosisSectorOk(matrixSectors, sectors, mode, need);
        const roll = symbiosisNoise(y, x, iteration, seed);
        let value = zero;

        if (current === SYM_VOID) {
          if (anchors >= state.nAnchorBirth && nearGrazer >= state.mAnchorBirth && nearAnchor === 0) {
            value = SYM_ANCHOR;
          } else if (matrices >= state.nMatrixBirth && nearWeaver >= state.mMatrixBirth && nearMatrix === 0) {
            value = SYM_MATRIX;
          } else if (anchorsNear >= state.grazerBirth && pulses >= 1 && anchorShape) {
            value = SYM_GRAZER;
          } else if (matricesNear >= state.weaverBirth && pulses >= 1 && matrixShape) {
            value = SYM_WEAVER;
          } else if ((grazers + weavers) >= 1 && (anchorsNear + matricesNear) >= 1 &&
                     roll < state.pulseProb) {
            value = SYM_PULSE;
          }
        } else if (current === SYM_ANCHOR) {
          if (anchors >= state.nAnchorSurv && nearGrazer >= state.mAnchorSurv &&
              anchors <= state.nAnchorMax) {
            value = SYM_ANCHOR;
          }
        } else if (current === SYM_MATRIX) {
          if (matrices >= state.nMatrixSurv && nearWeaver >= state.mMatrixSurv &&
              matrices <= state.nMatrixMax) {
            value = SYM_MATRIX;
          }
        } else if (current === SYM_GRAZER) {
          if (grazers > state.grazerMax) {
            value = SYM_VOID;
          } else if (anchorsNear >= state.grazerSurv && anchorsNear > 0 && anchorShape) {
            value = SYM_GRAZER;
          } else {
            value = SYM_SPORE;
          }
        } else if (current === SYM_WEAVER) {
          if (weavers > state.weaverMax) {
            value = SYM_VOID;
          } else if (matricesNear >= state.weaverSurv && matricesNear > 0 && matrixShape) {
            value = SYM_WEAVER;
          } else {
            value = SYM_SPORE;
          }
        } else if (current === SYM_SPORE) {
          if (anchorsNear >= 1) {
            value = SYM_GRAZER;
          } else if (matricesNear >= 1) {
            value = SYM_WEAVER;
          } else if (roll < state.sporeDecay) {
            value = SYM_VOID;
          } else {
            value = SYM_SPORE;
          }
        } else {
          value = SYM_VOID; // PULSE живёт один шаг
        }

        next[index] = value;
      }
    }
    state.grid = next;
    state.next = grid;
  },
});
