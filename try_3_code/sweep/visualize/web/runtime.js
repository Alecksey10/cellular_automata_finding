/**
 * Универсальный runtime браузерных экспериментов.
 *
 * Контракт эксперимента:
 *
 *     registerExperiment({
 *       id: "life",                 // совпадает с именем Python-эксперимента
 *       label: "B/S-пороги",        // подпись для интерфейса
 *       create(card, env) -> state, // построить состояние из карточки
 *       step(state, rng, env),      // один шаг; меняет state.grid
 *     });
 *
 * ``create`` получает карточку с полями ``width``, ``height``, ``init``,
 * ``bits``, ``states``, ``values`` и (если нужно) ``ants``/``turns``. Runtime
 * сам подменяет ``width``/``height``/``init`` при изменении размера, поэтому
 * модуль не должен кэшировать размеры вне состояния.
 *
 * Состояние — обычный объект. Обязательно только ``state.grid``:
 * плоский ``Uint8Array`` длиной ``width * height``. Модуль может завести
 * ``state.next`` и другие поля; ``step`` обычно меняет ``grid``/``next``
 * местами.
 */

const EXPERIMENTS = Object.create(null);
const MIN_SIZE = 8;
const MAX_SIZE = 512;
const MAX_ITERATIONS = 1_000_000_000;

function registerExperiment(module) {
  if (!module || !module.id) throw new Error("у web-модуля нет id");
  if (typeof module.create !== "function" || typeof module.step !== "function") {
    throw new Error("web-модуль " + module.id + ": нужны create() и step()");
  }
  if (EXPERIMENTS[module.id]) throw new Error("web-модуль " + module.id + " уже зарегистрирован");
  EXPERIMENTS[module.id] = module;
  return module;
}

function clamp(value, lo, hi, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(lo, Math.min(hi, Math.round(number)));
}

function clampSpeed(value, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return fallback;
  return Math.max(0.1, Math.min(1000, number));
}

/** Копия карточки под новый размер: init, ширина/высота и позиции муравьёв. */
function resizeCard(card, width, height) {
  const fromWidth = Number(card.width);
  const fromHeight = Number(card.height);
  if (width === fromWidth && height === fromHeight) return card;

  const source = gridFromBase64(card.init, card.bits, fromWidth * fromHeight);
  const grid = resizeGrid(source, fromWidth, fromHeight, width, height);
  const resized = Object.assign({}, card, {
    width: width,
    height: height,
    init: gridToBase64(grid, card.bits),
  });
  // кэши canvas нельзя делить между карточками разного размера
  delete resized.__image;
  delete resized.__pixels;
  delete resized.__palette;

  if (Array.isArray(card.ants)) {
    const ratioX = width / fromWidth;
    const ratioY = height / fromHeight;
    resized.ants = card.ants.map((ant) => [
      Math.max(0, Math.min(height - 1, Math.round(ant[0] * ratioY))),
      Math.max(0, Math.min(width - 1, Math.round(ant[1] * ratioX))),
      ant[2],
    ]);
  }
  return resized;
}

/**
 * Один автомат: инициализация, размер, цель по итерациям, скорость и цикл.
 * Живёт независимо от DOM, поэтому им можно пользоваться и в своих
 * экспериментах/тестах, не заходя в галерею.
 */
class Runner {
  constructor(record, module, options) {
    options = options || {};
    this.record = record;
    this.module = module || EXPERIMENTS[record.experiment];
    if (!this.module) throw new Error("нет web-модуля для " + record.experiment);

    this.nativeWidth = Number(record.width) || 1;
    this.nativeHeight = Number(record.height) || 1;
    this.width = clamp(options.width, MIN_SIZE, MAX_SIZE, this.nativeWidth);
    this.height = clamp(options.height, MIN_SIZE, MAX_SIZE, this.nativeHeight);
    this.iterations = clamp(options.iterations, 1, MAX_ITERATIONS, Math.max(1, Number(record.iterations) || 1));
    this.speed = clampSpeed(options.speed, 10);
    this.loop = Boolean(options.loop);
    this.iteration = 0;
    this.state = null;
    this.card = null;
    this.rng = null;
    this.reset();
  }

  /** Размер по умолчанию: ширина из карточки, высота — с сохранением пропорций. */
  static sizeFor(record, width) {
    const nativeWidth = Number(record.width) || 1;
    const nativeHeight = Number(record.height) || 1;
    const target = clamp(width, MIN_SIZE, MAX_SIZE, nativeWidth);
    const height = Math.max(1, Math.round((target * nativeHeight) / nativeWidth));
    return [target, height];
  }

  reset() {
    this.card = resizeCard(this.record, this.width, this.height);
    this.rng = makeRng(hashSeed(this.record.digest + ":" + this.width + "x" + this.height));
    this.state = this.module.create(this.card, { rng: this.rng, size: this.width });
    this.iteration = 0;
    return this;
  }

  setSize(width) {
    const size = Runner.sizeFor(this.record, width);
    if (size[0] === this.width && size[1] === this.height) return false;
    this.width = size[0];
    this.height = size[1];
    this.reset();
    return true;
  }

  setIterations(value) {
    this.iterations = clamp(value, 1, MAX_ITERATIONS, this.iterations);
  }

  setSpeed(value) {
    this.speed = clampSpeed(value, this.speed);
  }

  get done() {
    return this.iteration >= this.iterations;
  }

  /**
   * Сделать до ``steps`` шагов. При цикле достижение цели не останавливает
   * автомат: состояние начинается заново, а счётчик итераций обнуляется.
   */
  advance(steps) {
    let budget = Math.max(0, Math.floor(Number(steps) || 0));
    let made = 0;
    while (budget > 0) {
      if (this.done) {
        if (!this.loop) break;
        this.reset();
      }
      this.module.step(this.state, this.rng, { iteration: this.iteration, size: this.width });
      this.iteration += 1;
      budget -= 1;
      made += 1;
    }
    return made;
  }
}

function createRunner(record, options) {
  const module = EXPERIMENTS[record.experiment];
  if (!module) return null;
  return new Runner(record, module, options);
}

function experimentIds() {
  return Object.keys(EXPERIMENTS).sort();
}

/** Итоговый кадр прогона (нативное разрешение карточки). */
function finalGrid(card) {
  return gridFromBase64(card.final, card.bits, card.width * card.height);
}
