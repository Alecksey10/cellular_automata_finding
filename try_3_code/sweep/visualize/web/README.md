# Браузерный runtime и модули экспериментов

Статическая галерея (`python -m sweep export`) не запускает Python в браузере.
Сборка склеивает четыре слоя в два файла:

| файл | что внутри | когда трогать |
|---|---|---|
| `engine.js` | чистые помощники: base64 ↔ сетка, ГПСЧ, палитра, окрестности, общие правила `maze`/`omega5` | если нужен новый общий примитив |
| `runtime.js` | реестр экспериментов, `Runner` (размер, итерации, скорость, цикл) | если меняется жизненный цикл карточки |
| `gallery.js` | DOM: карточки, страницы, кнопки, модальное окно | если меняется интерфейс галереи |
| `page.html` | разметка и стили страницы | если нужен новый элемент управления |

`build.py` собирает `engine.js + runtime.js + sweep/experiments/<имя>/web.js`
в `site/assets/rules.js`. Модули экспериментов идут последними, чтобы
`registerExperiment` был уже определён.

## Контракт эксперимента

Рядом с `sweep/experiments/<имя>/__init__.py` положите `web.js`:

```js
registerExperiment({
  id: "my_rule",                    // = имя Python-эксперимента
  label: "моё правило",             // подпись
  create(card, env) {               // построить состояние
    return {
      width: card.width,
      height: card.height,
      grid: gridFromBase64(card.init, card.bits, card.width * card.height),
      next: new Uint8Array(card.width * card.height),
      zero: Number(card.values["grid2d.zero"] || 0),
    };
  },
  step(state, rng, env) {           // ровно одна итерация
    // ... state.grid -> state.next ...
    const grid = state.grid;
    state.grid = state.next;
    state.next = grid;
  },
});
```

Обязательны только `id`, `create`, `step`. Всё остальное — необязательные
поля и helpers.

### Что получает `create(card, env)`

* `card.width`, `card.height` — текущее разрешение (может отличаться от
  исходного, если пользователь поменял «размер» в галерее);
* `card.init` — упакованная base64 инициализирующая сетка;
* `card.bits` — 1, 4 или 8 бит на клетку;
* `card.states`, `card.values` — метаданные и скалярный конфиг из Python;
* `card.ants`, `card.turns` — дополнительные поля, которые кладёт `build.py`
  для конкретного эксперимента (например, Ленгтона);
* `env.rng` — детерминированный mulberry32, уже засеянный от `digest`,
  размера и seed;
* `env.size` — ширина текущей сетки.

Runtime сам пересобирает `card.init`, `width` и `height` при изменении
размера, поэтому не кэшируйте их в замыкании модуля. Всё, что зависит от
размера, держите в `state`.

### Что получает `step(state, rng, env)`

* `state` — объект, который вернул `create`;
* `rng` — тот же генератор (для стохастических правил);
* `env.iteration` — номер текущего шага (с нуля);
* `env.size` — ширина сетки.

`step` обязан обновить `state.grid`. Если удобно считать в буфер, заведите
`state.next` и меняйте ссылки местами — как в существующих модулях.

## Общие helpers

Доступны из общего scope `rules.js` (и через `window.SweepRules`):

* `gridFromBase64(text, bits, count)` — распаковать сетку;
* `gridToBase64(grid, bits)` — упаковать обратно;
* `resizeGrid(grid, fromW, fromH, toW, toH)` — ближайший сосед;
* `resizeCard(card, width, height)` — карточка с новым размером;
* `makeRng(seed)`, `hashSeed(text)`;
* `paintGrid(card, canvas, grid, width, height)` — нарисовать;
* `neighborhoodOffsets(values)`, `innerNeighborhoodOffsets(values)`,
  `neighborhoodWrap(values)` — окрестности из конфига;
* `mechanismOffsets(name)`, `percentCounts(size, percents)` — окна `maze9`;
* `makeMazeRule({id, label, prefix, mechanism})` — общее ядро лабиринтов;
* `makeOmegaRule(id)` — общее ядро `omega5`/`omega5_dense`.

Если помощник нужен нескольким экспериментам — добавляйте его в `engine.js`,
а не копируйте в `web.js`. Если помощник относится к одному эксперименту —
он может жить прямо в его `web.js` (как `rotRandom` у `rings_rot`).

## Универсальные контролы

`Runner` не зависит от DOM и даёт галерее одинаковые действия для любого
эксперимента:

```js
const runner = SweepRules.createRunner(record, {
  iterations: record.iterations, // цель обзора
  speed: 10,                     // итераций в секунду, можно 1
  loop: false,                   // начинать заново после цели
});

runner.reset();                  // вернуться к инициализации
runner.setSize(128);             // поменять инициализирующий размер
runner.setIterations(500);       // сколько шагов считать обзором
runner.setSpeed(1);              // 1 итерация в секунду
runner.loop = true;              // включить цикл
runner.advance(17);              // сделать до 17 шагов
runner.iteration;                // текущий шаг
runner.state.grid;               // текущая сетка
runner.width, runner.height;     // текущее разрешение
```

Галерея использует `runner.speed` как «итер/с», а не как FPS отрисовки:
при скорости 1 шаг происходит примерно раз в секунду, даже если браузер
рисует 60 кадров/с. При включённом цикле достижение цели не останавливает
автомат, а начинает состояние заново.

## Как добавить эксперимент

1. Опишите Python-эксперимент в `sweep/experiments/<имя>/` и
   зарегистрируйте `build` в `sweep/experiments/__init__.py`.
2. Добавьте визуализатор (`state_color`, `legend`, `rule_info`) рядом с
   экспериментом.
3. Напишите `web.js` по контракту выше.
4. Проверьте сборку:

   ```bash
   python -m sweep export --run-dir runs/<имя> --out /tmp/site
   ```

   Сборка печатает пропущенные эксперименты (у них нет `web.js`) и
   проверяет `rules.js`/`gallery.js` через `node --check`, если node есть.
5. Откройте `/tmp/site/index.html` или `python -m http.server` в каталоге.
6. Добавьте тест в `tests/test_sweep.py`: `module_path("<имя>")` не `None`,
   а `build_site` кладёт карточку.

## Необязательные Python-хуки эксперимента

Чтобы новый автомат не требовал правок в `build.py`, `replay.py` или
`server.py`, эксперимент может определить в своём пакете:

```python
def web_parameters(values):          # параметры для browser-модуля
    ...

def web_card_details(values, seed=0):  # дополнительные поля карточки
    ...

def web_meta_parameters(record):       # регуляторы галереи
    ...
```

- `web_parameters` → `card.params` (например, группы `omega5` или код механизма).
- `web_card_details` → сливается в карточку (например, `turns`/`ants` у Ленгтона).
- `web_meta_parameters` → список мета-параметров вместо стандартного набора
  `phase.*` + `размер` + `seed`.

Если каталог пакета не совпадает с именем эксперимента, задайте
`web_dir` в конструкторе:

```python
Experiment("expBeamFree", ..., web_dir="exp_beam_free")
```

## Частые ошибки

* **Модуль молча не появился в сборке.** Проверьте имя файла:
  `sweep/experiments/<id>/web.js`, где `<id>` совпадает с
  `record.config.experiment`.
* **После смены размера картинка ломается.** `create` запомнил `card.width`
  в замыкании. Читайте размер из `card`/`state` каждого вызова.
* **Стохастический автомат не повторяется.** Используйте `rng`, переданный
  в `create`/`step`, а не `Math.random()`.
* **Детерминированный автомат не совпадает с Python.** Проверьте порядок
  обхода (`y`, затем `x`), границы (`wrap`) и `bits` упаковки сетки.
* **Карточка «живая», но не двигается.** Проверьте, что `step` меняет
  `state.grid`, а не только локальную переменную.
