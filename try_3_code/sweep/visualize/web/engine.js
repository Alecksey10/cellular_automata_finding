/**
 * Низкоуровневое ядро браузерной визуализации.
 *
 * Здесь нет жизненного цикла карточки и реестра экспериментов — только чистые
 * помощники: упаковка сеток, ГПСЧ, палитра, окрестности и общие правила.
 * Реестр и запуск автомата живут в ``runtime.js``, интерфейс эксперимента
 * описан в ``sweep/visualize/web/README.md``.
 *
 * Модуль эксперимента (``sweep/experiments/<имя>/web.js``) получает готовые
 * карточки и helpers через общий scope собранного ``assets/rules.js``.
 */

// ---------------------------------------------------------------------------
// данные: сетки приходят упакованными битами/nibble и base64
// ---------------------------------------------------------------------------
function gridFromBase64(text, bits, count) {
  const out = new Uint8Array(count);
  if (!text) return out;
  const bin = atob(text);
  if (bits >= 8) {
    for (let i = 0; i < count; i += 1) out[i] = bin.charCodeAt(i) & 255;
    return out;
  }
  if (bits === 4) {
    for (let i = 0; i < count; i += 1) {
      const byte = bin.charCodeAt(i >> 1) & 255;
      out[i] = (i & 1) === 0 ? byte >> 4 : byte & 15;
    }
    return out;
  }
  for (let i = 0; i < count; i += 1) {
    out[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
  }
  return out;
}

/** Uint8Array -> base64 без огромной строки в apply(). */
function bytesToBase64(bytes) {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/**
 * Сетка -> base64 в том же формате, что ``sweep.visualize.build.pack_plane``.
 * Нужна, когда браузер меняет размер инициализации и должен пересобрать
 * карточку для ``module.create(card)``.
 */
function gridToBase64(grid, bits) {
  let nonZero = false;
  for (let i = 0; i < grid.length; i += 1) {
    if (grid[i]) {
      nonZero = true;
      break;
    }
  }
  if (!nonZero) return "";

  if (bits >= 8) return bytesToBase64(Uint8Array.from(grid));

  if (bits === 4) {
    const size = grid.length + (grid.length & 1);
    const out = new Uint8Array(size >> 1);
    for (let i = 0; i < grid.length; i += 2) {
      const hi = grid[i] & 15;
      const lo = i + 1 < grid.length ? grid[i + 1] & 15 : 0;
      out[i >> 1] = (hi << 4) | lo;
    }
    return bytesToBase64(out);
  }

  const out = new Uint8Array((grid.length + 7) >> 3);
  for (let i = 0; i < grid.length; i += 1) {
    if (grid[i]) out[i >> 3] |= 1 << (7 - (i & 7));
  }
  return bytesToBase64(out);
}

/**
 * Ближайший сосед: из базовой сетки делает сетку другого разрешения.
 * Так карточка может поменять «инициализирующий размер», не пересобирая
 * прогон на Python.
 */
function resizeGrid(grid, fromWidth, fromHeight, toWidth, toHeight) {
  if (fromWidth === toWidth && fromHeight === toHeight) return grid;
  const out = new Uint8Array(toWidth * toHeight);
  for (let y = 0; y < toHeight; y += 1) {
    const sy = Math.min(fromHeight - 1, Math.floor(((y + 0.5) * fromHeight) / toHeight));
    for (let x = 0; x < toWidth; x += 1) {
      const sx = Math.min(fromWidth - 1, Math.floor(((x + 0.5) * fromWidth) / toWidth));
      out[y * toWidth + x] = grid[sy * fromWidth + sx];
    }
  }
  return out;
}

function hashSeed(text) {
  let hash = 2166136261 >>> 0;
  const value = String(text || "");
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

/** mulberry32: маленький детерминированный ГПСЧ для стохастических правил. */
function makeRng(seed) {
  let state = (seed >>> 0) || 1;
  return function rng() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// отрисовка: сетка -> canvas, палитра берётся из визуализатора эксперимента
// ---------------------------------------------------------------------------
function parseColor(hex) {
  const text = String(hex || "#808080").replace("#", "");
  const value = parseInt(text.length === 3
    ? text.split("").map((c) => c + c).join("")
    : text, 16) || 0;
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return (255 << 24) | (b << 16) | (g << 8) | r; // little-endian RGBA
}

function paletteOf(card) {
  if (!card.__palette) {
    card.__palette = Uint32Array.from((card.colors || ["#101010"]).map(parseColor));
  }
  return card.__palette;
}

function paintGrid(card, canvas, grid, widthOverride, heightOverride) {
  const width = Number(widthOverride) || card.width;
  const height = Number(heightOverride) || card.height;
  const context = canvas.getContext("2d", { alpha: false });
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  let image = card.__image;
  if (!image || image.width !== width || image.height !== height) {
    image = context.createImageData(width, height);
    card.__image = image;
    card.__pixels = new Uint32Array(image.data.buffer);
  }
  const pixels = card.__pixels;
  const palette = paletteOf(card);
  const last = palette.length - 1;
  for (let i = 0; i < grid.length; i += 1) {
    const state = grid[i];
    pixels[i] = state < palette.length ? palette[state] : palette[last];
  }
  context.putImageData(image, 0, 0);
}

// ---------------------------------------------------------------------------
// окрестности и механизмы: те же смещения, что считает Python
// ---------------------------------------------------------------------------
function discOffsets(radius, metric) {
  const offsets = [];
  for (let dy = -radius; dy <= radius; dy += 1) {
    for (let dx = -radius; dx <= radius; dx += 1) {
      if (!dy && !dx) continue;
      let inside;
      if (metric === "chebyshev") inside = Math.max(Math.abs(dy), Math.abs(dx)) <= radius;
      else if (metric === "manhattan") inside = Math.abs(dy) + Math.abs(dx) <= radius;
      else inside = dy * dy + dx * dx <= radius * radius;
      if (inside) offsets.push([dy, dx]);
    }
  }
  return offsets;
}

function neighborhoodOffsets(values) {
  const id = values["select.neighborhood"];
  if (id === "neighborhood.moore") {
    return discOffsets(Number(values["neighborhood.moore.radius"]) || 1, "chebyshev");
  }
  if (id === "neighborhood.radius") {
    return discOffsets(Number(values["neighborhood.radius.radius"]) || 1,
      String(values["neighborhood.radius.metric"] || "euclidean"));
  }
  if (id === "neighborhood.square9") return discOffsets(4, "chebyshev");
  if (id === "neighborhood.square5") return discOffsets(2, "chebyshev");
  if (id === "neighborhood.duel_rings") return discOffsets(2, "chebyshev");
  if (id === "neighborhood.two_rings") return twoRingsOffsets(values).outer;
  return discOffsets(1, "chebyshev");
}

/** Два кольца: внутренний диск и внешнее кольцо без внутреннего. */
function twoRingsOffsets(values) {
  const innerRadius = Number(values["neighborhood.two_rings.inner_radius"]) || 1;
  const outerRadius = Number(values["neighborhood.two_rings.outer_radius"]) || 3;
  const metric = String(values["neighborhood.two_rings.metric"] || "euclidean");
  const inner = discOffsets(innerRadius, metric);
  const inside = new Set(inner.map((pair) => pair[0] + "," + pair[1]));
  const outer = discOffsets(outerRadius, metric).filter((pair) => !inside.has(pair[0] + "," + pair[1]));
  return { inner: inner, outer: outer };
}

/** Внутреннее кольцо для правил, которые считают два радиуса. */
function innerNeighborhoodOffsets(values) {
  const id = values["select.neighborhood"];
  if (id === "neighborhood.two_rings") return twoRingsOffsets(values).inner;
  if (id === "neighborhood.duel_rings") return discOffsets(1, "chebyshev");
  // у moore/radius/square внутреннее кольцо совпадает с окном
  return neighborhoodOffsets(values);
}

function neighborhoodWrap(values) {
  const id = values["select.neighborhood"];
  if (id === "neighborhood.moore") return Boolean(values["neighborhood.moore.wrap"]);
  if (id === "neighborhood.radius") return Boolean(values["neighborhood.radius.wrap"]);
  if (id === "neighborhood.square9") return Boolean(values["neighborhood.square9.wrap"]);
  if (id === "neighborhood.square5") return Boolean(values["neighborhood.square5.wrap"]);
  if (id === "neighborhood.duel_rings") return Boolean(values["neighborhood.duel_rings.wrap"]);
  if (id === "neighborhood.two_rings") return Boolean(values["neighborhood.two_rings.wrap"]);
  return false;
}

function mechanismOffsets(name) {
  const radius = 4;
  const cells = [];
  const ring = (lo, hi) => {
    for (let dy = -radius; dy <= radius; dy += 1) {
      for (let dx = -radius; dx <= radius; dx += 1) {
        const distance = Math.max(Math.abs(dy), Math.abs(dx));
        if (distance >= lo && distance <= hi) cells.push([dy, dx]);
      }
    }
  };
  switch (String(name)) {
    case "inner": ring(1, 1); break;
    case "middle": ring(2, 2); break;
    case "outer": ring(3, radius); break;
    case "rim": ring(radius, radius); break;
    case "cross":
      for (let k = -radius; k <= radius; k += 1) {
        if (k) {
          cells.push([k, 0]);
          cells.push([0, k]);
        }
      }
      break;
    case "diagonals":
      for (let k = -radius; k <= radius; k += 1) {
        if (k) {
          cells.push([k, k]);
          cells.push([k, -k]);
        }
      }
      break;
    case "corners":
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          if (Math.abs(dy) >= 2 && Math.abs(dx) >= 2) cells.push([dy, dx]);
        }
      }
      break;
    case "checker":
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          if ((dy + dx) % 2 === 0 && (dy || dx)) cells.push([dy, dx]);
        }
      }
      break;
    case "all": ring(1, radius); break;
    default: ring(1, radius);
  }
  return cells.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
}

/** Проценты от размера подмаски -> границы рождения и выживания. */
function percentCounts(size, percents) {
  const round = (value) => Math.floor(value + 0.5);
  const bornLo = Math.max(1, round((percents[0] * size) / 100));
  const bornHi = Math.max(bornLo, round((percents[1] * size) / 100));
  const survLo = Math.max(0, round((percents[2] * size) / 100));
  const survHi = Math.max(survLo, round((percents[3] * size) / 100));
  return [bornLo, bornHi, survLo, survHi];
}

// ---------------------------------------------------------------------------
// общее ядро «растущих стен»: maze и maze9 отличаются только подмаской
// ---------------------------------------------------------------------------
function makeMazeRule(options) {
  const id = options.id;
  const prefix = options.prefix;
  const mechanism = Boolean(options.mechanism);
  return {
    id: id,
    label: options.label || id,
    create(card) {
      const values = card.values || {};
      const offsets = mechanism
        ? mechanismOffsets(values[prefix + ".mechanism"])
        : neighborhoodOffsets(values);
      const percents = [
        Number(values[prefix + ".born_lo"]),
        Number(values[prefix + ".born_hi"]),
        Number(values[prefix + ".surv_lo"]),
        Number(values[prefix + ".surv_hi"]),
      ];
      const rule = mechanism ? percentCounts(offsets.length, percents) : percents;
      return {
        width: card.width,
        height: card.height,
        grid: gridFromBase64(card.init, card.bits, card.width * card.height),
        next: new Uint8Array(card.width * card.height),
        dy: Int32Array.from(offsets.map((pair) => pair[0])),
        dx: Int32Array.from(offsets.map((pair) => pair[1])),
        wrap: mechanism ? Boolean(values["neighborhood.square9.wrap"]) : neighborhoodWrap(values),
        rule: rule,
        rate: Number(values[prefix + ".rate"]),
        border: String(values[prefix + ".border"]) === "wall",
      };
    },
    step(state, rng) {
      const width = state.width;
      const height = state.height;
      const grid = state.grid;
      const next = state.next;
      const dy = state.dy;
      const dx = state.dx;
      const total = dy.length;
      const bornLo = state.rule[0];
      const bornHi = state.rule[1];
      const survLo = state.rule[2];
      const survHi = state.rule[3];
      const wrap = state.wrap;
      const rate = state.rate;
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          let count = 0;
          for (let k = 0; k < total; k += 1) {
            let yy = y + dy[k];
            let xx = x + dx[k];
            if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
              if (wrap) {
                yy = ((yy % height) + height) % height;
                xx = ((xx % width) + width) % width;
              } else {
                count += 1; // за краем сетки — стена
                continue;
              }
            }
            if (grid[yy * width + xx] !== 0) count += 1;
          }
          const value = grid[y * width + x];
          const index = y * width + x;
          if (value !== 0) {
            next[index] = count >= survLo && count <= survHi ? 1 : 0;
          } else {
            next[index] = count >= bornLo && count <= bornHi && rng() < rate ? 1 : 0;
          }
        }
      }
      if (state.border) {
        for (let x = 0; x < width; x += 1) {
          next[x] = 1;
          next[(height - 1) * width + x] = 1;
        }
        for (let y = 0; y < height; y += 1) {
          next[y * width] = 1;
          next[y * width + width - 1] = 1;
        }
      }
      state.grid = next;
      state.next = grid;
    },
  };
}

// ---------------------------------------------------------------------------
// общее ядро omega5: 8 состояний, окно 5×5, механизмы счёта и 20 реакций.
// Группы соседей и интервалы рождения/выживания/смерти считает Python
// (``web_parameters``), сюда они приходят готовыми в ``card.params``.
// ---------------------------------------------------------------------------
function omegaMetric(mechanism, g, total, c) {
  switch (mechanism) {
    case 0: case 1: case 2: case 9: case 10: case 17: case 18:
      return g[0];
    case 3:
      return Math.max(g[0], g[1], g[2], g[3], 0, 0, 0, 0);
    case 4:
      return Math.min(g[0], g[1], g[2], g[3], total, total, total, total);
    case 5:
      return Math.min(g[0] + g[3], g[1] + g[2]);
    case 6:
      return Math.max(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]);
    case 7:
      return Math.min(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]);
    case 8:
      return Math.min(g[0] + g[4], g[1] + g[5], g[2] + g[6], g[3] + g[7]);
    case 11:
      return Math.max(g[0], g[1]);
    case 12:
      return Math.min(g[0], g[1]);
    case 13: case 15:
      return Math.max(g[0], g[1], g[2], g[3], g[4]);
    case 14: case 16:
      return Math.min(g[0], g[1], g[2], g[3], g[4]);
    case 19: {
      let metric = 0;
      for (let state = 1; state <= 7; state += 1) if (c[state] > 0) metric += 1;
      return metric;
    }
    case 20:
      return Math.max(c[1], c[2], c[3], c[4], c[5], c[6], c[7], 0);
    case 21: {
      let metric = 0;
      for (let state = 1; state <= 7; state += 1) {
        if (c[state] > 0 && (metric === 0 || c[state] < metric)) metric = c[state];
      }
      return metric;
    }
    default:
      return g[0];
  }
}

function makeOmegaRule(id) {
  return {
    id: id,
    label: "omega5: счёт 5×5 + реакции",
    create(card) {
      const values = card.values || {};
      const params = card.params || {};
      const offsets = neighborhoodOffsets(values);
      return {
        width: card.width,
        height: card.height,
        grid: gridFromBase64(card.init, card.bits, card.width * card.height),
        next: new Uint8Array(card.width * card.height),
        dy: Int32Array.from(offsets.map((pair) => pair[0])),
        dx: Int32Array.from(offsets.map((pair) => pair[1])),
        groups: Int32Array.from(params.groups || offsets.map(() => 0)),
        wrap: neighborhoodWrap(values),
        zero: Number(values["grid2d.zero"] || 0),
        states: Number(values["grid2d.states"] || card.states || 8),
        birth: params.birth || [1, 0],
        survival: params.survival || [1, 0],
        death: params.death || [1, 0],
        birthState: Number(params.birthState || 1),
        trick: Number(params.trick || 0),
        mechanism: Number(params.mechanism || 0),
      };
    },
    step(state) {
      const width = state.width;
      const height = state.height;
      const grid = state.grid;
      const next = state.next;
      const dy = state.dy;
      const dx = state.dx;
      const groups = state.groups;
      const totalOffsets = dy.length;
      const wrap = state.wrap;
      const zero = state.zero;
      const states = state.states;
      const trick = state.trick;
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          let total = 0;
          const c = [0, 0, 0, 0, 0, 0, 0, 0];
          const g = [0, 0, 0, 0, 0, 0, 0, 0];
          for (let k = 0; k < totalOffsets; k += 1) {
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
            total += 1;
            if (value >= 1 && value <= 7) c[value] += 1;
            const bucket = groups[k];
            if (bucket >= 0 && bucket < 8) g[bucket] += 1;
          }

          const metric = omegaMetric(state.mechanism, g, total, c);
          const cell = grid[y * width + x];
          let value;
          if (cell === zero) {
            value = state.birth[0] <= metric && metric <= state.birth[1] ? state.birthState : zero;
          } else {
            value = cell;
            if (state.death[0] <= metric && metric <= state.death[1]) value = zero;
            else if (!(state.survival[0] <= metric && metric <= state.survival[1])) value = zero;
          }

          if (trick === 1) {
            if (cell === 6) value = 6;
          } else if (trick === 2) {
            if (cell === 4 && c[3] >= 1) value = 2;
            if (cell === 3 && c[4] >= 1) value = zero;
          } else if (trick === 3) {
            if (cell === 7) value = 7;
          } else if (trick === 4) {
            if (cell === 3 && c[2] >= 2) value = 4;
            if (cell === 4 && c[2] === 0) value = zero;
          } else if (trick === 5) {
            if (cell === zero && c[2] >= 1 && c[3] >= 1) value = 2;
          } else if (trick === 6) {
            if (cell === 6 && c[1] >= 1) value = zero;
          } else if (trick === 7) {
            if (cell === 3 && c[3] >= 3) value = 2;
            if (cell === 2 && c[2] >= 4) value = zero;
          } else if (trick === 8) {
            if (cell === zero && c[5] >= 2) value = 5;
          } else if (trick === 9) {
            if (cell === 5) value = 5;
          } else if (trick === 10) {
            if (cell === 5 && c[1] > c[5]) value = 1;
          } else if (trick === 11) {
            if (cell === zero && c[6] >= 3) value = 6;
          } else if (trick === 12) {
            if (cell === 7 && total >= 12) value = 1;
          } else if (trick === 13) {
            if (cell === 2 && c[2] >= 5) value = 2;
          } else if (trick === 14) {
            if (cell === 3) value = 2;
          } else if (trick === 15) {
            if (cell === 6 && c[1] >= 1) value = 2;
            if (cell === 1 && c[6] >= 1) value = zero;
          } else if (trick === 16) {
            if (cell === 7 && c[7] >= 2) value = 7;
          } else if (trick === 17) {
            if (cell === 3 && c[2] >= 1) value = 2;
            if (cell === 2 && c[3] >= 1) value = 3;
          } else if (trick === 18) {
            if (cell === 4) {
              if (c[3] >= 4) value = zero;
              else if (c[4] > 0) value = 4;
              else value = 3;
            }
          } else if (trick === 19) {
            if (cell === zero && c[1] >= 4) value = 6;
            if (cell === 6 && c[1] >= 2) value = 1;
          } else if (trick === 20) {
            if (cell === zero && c[2] >= 3) value = 1;
            if (cell === 1 && c[2] >= 2) value = 2;
          }

          if (value < zero) value = zero;
          if (value >= states) value = states - 1;
          next[y * width + x] = value;
        }
      }
      state.grid = next;
      state.next = grid;
    },
  };
}

