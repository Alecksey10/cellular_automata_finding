# try_2_code — движок перебора клеточных автоматов

Этот каталог — рабочая реализация идеи «сначала собрать параметры, потом
исполнять»:

* **планирование**: эксперимент объявляет оси перебора, зависимости между ними,
  выбор модулей и ограничения; код автомата на этом этапе не исполняется;
* **исполнение**: из плана лениво порождаются конфигурации; каждая
  фиксируется в `Fixed`, по ней собирается автомат и прогоняется тест;
* **модульность**: данные, инициализация, окрестность, правило и фильтры —
  отдельные плагины, которые сами объявляют свои параметры и требования;
* **ускорение**: numba-ядра компилируются один раз и принимают параметры
  данными (смещения, пороги), а тесты идут последовательно, потоками или
  процессами;
* **воспроизводимость**: конфиг — канонический JSON с дайджестом, рядом
  хранится seed, поэтому любой тест можно повторить и отрисовать.

Запуск (из этого каталога, интерпретатор — `../.venv/bin/python`):

```bash
python -m casweep modules          # что вообще есть
python -m casweep space --limit 500000 --preview 3
python -m casweep run --random --until 5 --seed 7 --workers 4 --mode process --out runs/rot
python -m casweep view --run-dir runs/rot --index 0
python -m casweep compare --run-dir runs/rot --limit 6 --steps 40
python -m casweep replay --digest <дайджест> --run-dir runs/rot --trace

# то же через скрипт конкретного эксперимента (без файлов)
python -m experiments.two_rings_rot.view --run-dir runs/rot --index 0

# сетки 3×3 и 5×5, циклические волны, перебор до N успешных
python -m casweep run --experiment experiments.cyclic5 --random --until 5 \
    --seed 9 --workers 4 --mode process --out runs/waves

# базовый демо-эксперимент (пороговые правила и игра «Жизнь»)
python -m casweep space --experiment experiments.life2d:build_experiment
```

## Что изменено по сравнению с примером из постановки

В исходном примере код исполняется и в planning-фазе: `width.get()`,
`neighborhood_method.get().getMaxCount(...)`, `if` по значениям. Чтобы это
работало, пришлось бы строить символьные выражения и символьные ветвления —
хрупко и не даёт надёжного числа вариаций.

Здесь planning-фаза — это **явное дерево осей**. Императивный код остаётся, но
исполняется уже с зафиксированными значениями:

| В примере | В движке |
|---|---|
| `Generator.Value(2)` | `namespace.value("zero_state", 0)` |
| `Generator.Range(2, 10, 1)` | `namespace.range("states", 2, 10, 1)` |
| `Generator.Range(width.get(), 128, 32)` | `namespace.range("height", width_handle, 129, 32)` |
| `Generator.Chooser([...])` | `space.select(Role.X, [ModuleA(), ModuleB()])` |
| `Generator.ChooseDepend(...)` | параметры каждого модуля объявляются в его ветке |
| `born_threshold_greater = Generator.Range(1, maxCount)` | `namespace.range("born_above", 1, peer.derived("max_count"))` |
| `Processor.STOP_ITERATION(...)` | `StepOutcome(stop=...)` / `FilterVerdict(ok=False, ...)` |
| «число вариаций» | `space.count()` / `space.estimate()` |

`Generator.get()` заменён на `Handle` — ссылку на значение. Арифметика над
`Handle` (`born + 1`) поддерживается только для сдвига границ, а не для
произвольных выражений: этого достаточно для зависимых диапазонов и не
превращает движок в интерпретатор.

## Карта файлов

```
casweep/
  handles.py      Handle/Shift: ленивые ссылки и разворачивание
  fixed.py        Fixed: зафиксированный конфиг, canonical JSON, digest
  interfaces.py   Capability/Requirement: типизированные возможности и требования
  space.py        Axis/Namespace/Space/Selection: оси, ветки модулей, ограничения, обход
  module.py       Module, Role, BuildView, FilterBinding
  assembly.py     проверка совместимости и сборка рабочих объектов
  experiment.py   Experiment/Phase/FilterUse: модули, стадии, перебор конфигов
  trial.py        прогон одного теста: инициализация, итерации, фильтры, остановки
  parallel.py     serial/thread/process планировщик, прерывание
  runner.py       перебор + отсев несовместимых + прогресс + сводка
  storage.py      successes.jsonl / trials.jsonl (--save-all) / summary.json
  cli.py          CLI
modules/          предметные плагины: 2D-сетка, рандомизаторы, окрестности
                  (3×3, 5×5, два кольца, радиус/кольцо/квадрат), правила
                  (B/S, жизнь, голосование, чётность, цикл, рефрактерность,
                  старение), фильтры
experiments/      пакеты экспериментов: <name>/__init__.py + <name>/view.py
visualize/        общий стандарт: live (терминал), standard (кадры/палитра),
                  viewer/gallery (экспорт по желанию)
tests/            unittest-тесты (stdlib, без pytest)
```

## Структура экспериментов

Каждый эксперимент — пакет: сборка в `__init__.py`, живой просмотр в `view.py`.
Импорт-пути не изменились (`experiments.two_rings_rot:build_experiment`).

```
experiments/
  life2d/          __init__.py  view.py
  two_rings/       __init__.py  view.py
  two_rings_rot/   __init__.py  view.py
  moore3/          __init__.py  view.py
  multi3/          __init__.py  view.py
  cyclic5/         __init__.py  view.py
```

`view.py` — тонкая обёртка над общим стандартом; палитра выбирается под
эксперимент (`age` для старения, `hue` для фаз):

```bash
python -m experiments.two_rings_rot.view --run-dir runs/rot --index 0
python -m experiments.cyclic5.view --random --steps 80
python -m experiments.life2d.view --first --steps 40 --fps 12
```

## Пример эксперимента

```python
from casweep.experiment import Experiment
from casweep.module import Role
from modules import (DiscreteGrid2D, DensityFilter, ThresholdRule,
                     SquareNeighborhood, LifeRule, UniformRandomizer)

def build_experiment() -> Experiment:
    experiment = Experiment("life2d")

    experiment.select(Role.DATA, [DiscreteGrid2D()])
    experiment.select(Role.RANDOMIZER, [UniformRandomizer()])
    experiment.select(Role.NEIGHBORHOOD, [SquareNeighborhood()])
    experiment.select(Role.RULE, [ThresholdRule(), LifeRule()])

    density = experiment.filter(DensityFilter())
    squares = experiment.filter(MonochromeSquaresFilter())

    warmup = experiment.space.range("stage.warmup", 0, 10, 2)
    observe = experiment.space.range("stage.observe", 10, 30, 10)
    experiment.phase("warmup", warmup, filters=[(density, "end")])
    experiment.phase("observe", observe, filters=[(density, "each"), (squares, "each")])
    return experiment
```

Порядок `select` важен: правило объявляет свои пороги, глядя на
`max_count` окрестности. Поэтому окрестность выбирается раньше правила.

`space.range` — полуоткрытый диапазон, как `range` в Python. Границы (и шаг)
могут быть `Handle`: `namespace.range("height", width, 129, 32)` даёт для
каждой ширины свой набор высот.

## Способы перебора

Пространство умеет отдавать конфиги несколькими способами, без изменения
описания осей:

| Способ | Что делает | Когда нужен |
|---|---|---|
| `product` | полный декартов порядок | точный обход, отладка |
| `shuffle` | тот же набор, значения осей перемешаны (`seed`) | ранние прогоны не застревают в первых значениях |
| `diagonal` | на каждом раунде каждая ось берёт следующее значение; не больше длины самого длинного домена | огромное пространство, нужен покрытый срез, а не миллионы конфигов |
| `random_bindings` / `--random` | случайные валидные конфиги, с ограничениями | поиск по пространству, где полный обход невозможен |

`diagonal` уважает ограничения: если «диагональное» значение запрещено, ось
берёт ближайшее допустимое, поэтому парные `r1`/`r2` с `r2 > r1` не вырождаются
в пустую выборку. Повторы после сжатия доменов отбрасываются по дайджесту.

```python
from casweep.space import SWEEP_STRATEGIES

experiment.count()                                  # полный размер
# experiment.count(strategy="diagonal")             # размер среза
experiment.count(strategy="diagonal", feasible=True)  # только совместимые

experiment.space.axis("data.states")                # ось по ключу
experiment.space.keys()                             # все ключи, включая ветки
experiment.space.values("data.states")              # домен при пустой привязке
experiment.space.dimension                          # число осей верхнего уровня
experiment.space.sample(100, seed=3, strategy="diagonal")  # Reservoir по срезу
experiment.space.random_bindings(100, seed=3, unique=True)
print(SWEEP_STRATEGIES)
```

В CLI те же режимы включаются флагами `--order product|shuffle|diagonal` и
`--stride N` (каждый N-й лист уже после отбора), а `casweep space` печатает
точный размер полного перебора и размер диагонального среза.

`--random` — отдельный источник: он не обходит дерево, а многократно строит
случайную валидную привязку с проверкой ограничений.

## Сетки 3×3 и 5×5: симметрия и циклические волны

Три готовых сценария (плюс прежние `life2d` и `two_rings`):

* [experiments/moore3](experiments/moore3/__init__.py) — 3×3 (8 соседей), два
  состояния. Правила: `rule.totalistic` (полосы рождения/выживания),
  `rule.life` (B3/S23), `rule.vote` (симметричное голосование),
  `rule.parity` (чётность; контрольный пример — должен отбраковываться).
  Фильтры: заполнение, активность, период ≤ 1, симметрия, кластеры.
* [experiments/multi3](experiments/multi3/__init__.py) — 3×3, 3…9 состояний:
  голосование и `rule.cyclic`; отбираются волны фаз (каждая фаза
  присутствует, энтропия не проваливается), активность и период.
* [experiments/cyclic5](experiments/cyclic5/__init__.py) — 5×5 (24 соседа),
  3…9 состояний (рефрактерному правилу нужно ≥ 4): `rule.cyclic` («твердеет,
  затем умирает») и `rule.refractory` (возбуждение → рефрактерность →
  смерть), те же фильтры плюс кластеры.

Все новые правила тоталистические: решение зависит только от числа соседей в
нужной фазе, поэтому они симметричны относительно поворотов и отражений. Фильтр
`filter.symmetry` проверяет уже результат динамики: IoU маски с лучшим её
поворотом/отражением (у блока 1.0, у одинокой клетки 0.0 — поэтому IoU, а не
доля совпавших клеток).

| Правило | Состояния | Параметры | Смысл |
|---|---|---|---|
| `rule.totalistic` | 2 | `born_lo..born_hi`, `survive_lo..survive_hi` | B/S-полосы |
| `rule.vote` | ≥2 | `quorum` | большинство соседей |
| `rule.parity` | 2 | — | чётность числа живых |
| `rule.cyclic` | ≥3 | `threshold`, `death` (`next`/`always`) | фазы цикла, верхняя фаза умирает |
| `rule.refractory` | ≥4 | `threshold` | возбуждение, рефрактерность, смерть |

Фильтры (пороги фиксированы экспериментом, а не перебираются — иначе один и тот
же автомат проходит случайно):

* `filter.fill_band` — заполнение в полосе;
* `filter.activity` — доля изменившихся клеток в полосе (застыл/мелькает);
* `filter.period` — цикл маски не длиннее `max_period` (окно истории);
* `filter.symmetry` — IoU с лучшим поворотом/отражением;
* `filter.entropy` — нормированная энтропия распределения фаз;
* `filter.phases` — каждая фаза цикла занимает хотя бы `min_share`;
* `filter.clusters` — число 8-связных компонент и средний размер;
* прежние `filter.clouds`, `filter.stability`, `filter.density`, `filter.squares`.

```bash
python -m casweep space --experiment experiments.moore3 --limit 50000
python -m casweep run --experiment experiments.cyclic5 --random --until 5 \
    --seed 9 --workers 4 --mode process --out runs/waves
python -m casweep run --experiment experiments.moore3 --random --limit 500 \
    --max-iterations 60 --workers 4 --mode process --out runs/moore3
```

## Автомат с двумя кольцами и отбор облаков

[experiments/two_rings](experiments/two_rings/__init__.py) — сцена, для которой
добавлены `neighborhood.two_rings`, `rule.aging_fortress`, `filter.clouds` и
`filter.stability`.

Перебор: `r1` от 1 до 5, `r2` от 1 до 7 (ограничение `r2 > r1`), граница
`wrap`/`clip`, число состояний 3..6, размер сетки (ширина ≥ 32),
инициализация, пороги рождения по внутреннему кольцу и по сумме колец, доля
переполнения и `fortify`.

Правило: состояние живой клетки — возраст. Клетка взрослеет, если сумма
соседей попадает в полосу
`[born_total - fortify*(age-1), crowd_limit + fortify*(age-1)]`; иначе она
теряет один уровень возраста и умирает только с последнего. Мёртвая клетка
рождается, если внутреннее кольцо даёт `born_inner` соседей и сумма —
`born_total`. Так старые клетки переносят и голод, и плотность, и не гибнут
от одной неудачной итерации.

Отбор:

* `filter.clouds` после ровно 5 итераций (и затем каждую итерацию) требует
  заполнение 5–85%, среднюю компоненту ≥ 6, долю клеток с ≥ 3 живыми соседями
  ≥ 0.6 и среднюю локальную плотность ≥ 0.4;
* `filter.stability` сравнивает маски «живая/мёртвая» (старение — не
  мелькание): изменение за итерацию ≤ 0.2 и отсутствие периода 2.

Пороги критериев фиксируются в эксперименте (`CloudFilter(min_fill=..., ...)`),
а не перебираются: это часть определения «желаемого» автомата, иначе один и тот
же автомат проходит или не проходит в зависимости от случайного порога.

```bash
python -m casweep run --random --limit 300 --seed 7 --workers 4 \
    --mode process --out runs/two_rings
```

На 300 случайных конфигах (seed 7) критерии проходят единицы: пространство
около 1,9·10⁸, а фильтры строгие. Прошедшие конфиги лежат в `successes.jsonl`,
`view` по дайджесту покажет облако. Для более широкого поиска увеличьте
`--limit` или ослабьте критерии в конструкторе фильтров.

## Два кольца с частичным гниением

[experiments/two_rings_rot](experiments/two_rings_rot/__init__.py) — отдельный
сценарий под цель «стабильные автоматы, где старое ядро медленно гниёт».
Это эксперимент по умолчанию для CLI.

Правило [`rule.rot_aging`](modules/rules/rot.py): состояние живой клетки —
прочность `1..states-1`. Каждый кадр:

* мёртвая клетка рождается в `1`, если внутреннее кольцо даёт не меньше
  `born_inner` соседей, а сумма колец — `born_total`;
* живая клетка взрослеет на единицу, если сумма колец лежит в
  `[survive_lo, survive_hi]`, иначе теряет один уровень (умирает только с
  первого);
* клетка, уже стоящая в последнем состоянии, с вероятностью `rot`
  превращается в `zero_state` — гниение идёт по клеткам, а не всем облаком.

Случайность берётся из `np.random.default_rng(seed)` теста: правило получает
генератор через `bind_rng`, поэтому один и тот же seed воспроизводит
траекторию, а numba-ядро остаётся `nogil` и работает в потоках/процессах.

Перебор:

| Ось | Значения |
|---|---|
| `r1` | 1..5 |
| `r2` | 1..7, только `r2 > r1` |
| `states` | 3..6 |
| `born_inner` | 1..`max_inner` (у r1=1 это 4, у r1=5 — 80) |
| `born_total`, `survive_lo`, `survive_hi` | 1..`max_count`, `born_total ≤ survive_lo ≤ survive_hi` |
| `rot` | 0.02, 0.05, 0.1, 0.2, 0.35 (ноль исключён) |
| сетка, граница, инициализация | штатные оси модулей; ширина ≥ 32 |

Стадии:

1. **warmup** 3..5 итераций — шум собирается в кластеры; ничего не
   проверяется, фильтры только запоминают последний кадр;
2. **observe** 10, 20 или 30 итераций, на каждом кадре:
   * `filter.stability` — маска живых меняется не более чем на 20 % и нет
     периода 2;
   * `filter.zero_balance` — доля `zero_state` меняется не более чем на
     8 процентных пунктов за кадр (явное требование);
   * `filter.rot_structure` — последнее состояние занимает 1..50 % сетки;
   * в конце — заполнение 5..95 %.

Пороги перебираются по всему кольцу, поэтому у больших радиусов
(`max_count` до 148) пространство огромно; для него и нужен `--random`.
Полный обход с `--limit` тоже работает, но первые конфиги будут с малыми
порогами.

```bash
# 5 стабильных автоматов с гниением (обычно ~0,3 % случайных конфигов)
python -m casweep run --random --until 5 --seed 7 --workers 4 \
    --mode process --out runs/rot

# живой просмотр одного автомата в терминале (без файлов)
python -m casweep view --digest <дайджест> --run-dir runs/rot

# несколько найденных автоматов рядом
python -m casweep compare --run-dir runs/rot --limit 6 --steps 40

# то же из пакета эксперимента
python -m experiments.two_rings_rot.view --digest <дайджест> --run-dir runs/rot
```

`view`/`compare` не пишут ничего на диск: конфига и seed из
`successes.jsonl` достаточно, чтобы повторить траекторию. PNG/HTML — только
явный экспорт (`render`/`gallery`, см. раздел визуализации).

## Как добавить модуль

Любой модуль — наследник `casweep.module.Module`; регистрации нет, достаточно
передать экземпляр в `experiment.select(...)`.

### Окрестность

```python
from casweep.module import BuildView
from casweep.space import DeclareView, Namespace
from modules.neighborhoods.base import NeighborhoodModule, offsets_from_predicate

class MyNeighborhood(NeighborhoodModule):
    id = "neighborhood.my"

    def declare(self, namespace: Namespace, view: DeclareView) -> None:
        super().declare(namespace, view)          # boundary: wrap/clip
        namespace.range("radius", 1, 4, 1)

    def offsets(self, fixed):
        radius = int(fixed["neighborhood.radius"])
        return offsets_from_predicate(radius, "radius", radius, radius)
```

`offset`-ы — это данные: одно numba-ядро правила работает с любой формой и
радиусом без перекомпиляции. `max_count` и `extent` модуль получает
автоматически (они объявлены в `derived` базового класса).

### Правило

```python
class MyRule(Module):
    id = "rule.my"
    role = Role.RULE
    requires = (
        Requirement(CapabilityKind.TWO_D),
        Requirement(CapabilityKind.MULTI_STATE, (at_least("states", 2),)),
        Requirement(CapabilityKind.NEIGHBORHOOD_COUNT, (at_least("max_count", 1),)),
    )

    def declare(self, namespace, view):
        max_count = view.peer(Role.NEIGHBORHOOD).derived("max_count")
        namespace.range("born_above", 1, max_count)

    def build(self, fixed: Fixed, view: BuildView):
        data = view.object(Role.DATA)
        neighborhood = view.object(Role.NEIGHBORHOOD)
        return MyRuntime(neighborhood.di, neighborhood.dj, ...)

    # можно добавить precompile(), чтобы прогреть njit-ядро до пула
```

`MyRuntime.step(src, dst)` делает одну итерацию и может вернуть
`StepOutcome(stop="причина")` — движок завершит тест со статусом `stopped`.
Метрики итерации складываются в `StepOutcome(metrics={...})`.

Готовое правило `rule.thresholds` повторяет псевдокод из постановки
буквально, включая ветку `else` (состояние увеличивается и у мёртвой клетки).
Это не ошибка: если нужна другая семантика, напишите свой модуль правила,
ядро при этом не меняется.

### Фильтр

```python
class MyFilter(Module):
    id = "filter.my"
    role = Role.FILTER
    requires = (Requirement(CapabilityKind.ZERO_STATE),)

    def declare(self, namespace, view):
        namespace.range("threshold", 0.1, 0.9, 0.1)

    def build(self, fixed, view):
        return MyRuntime(float(view.value("threshold")))

    # MyRuntime.check(grid) -> FilterVerdict(ok, reason, metrics)
```

Фильтр подключается к стадии в режиме `"each"` (каждую итерацию) или `"end"`
(после стадии). Именно фильтры реализуют «досрочно отбросить скучный автомат».

### Данные

Модуль данных объявляет геометрию и число состояний и возвращает фабрику с
`create()`/`like()`. Он же описывает `capabilities(fixed)` — динамические
возможности вроде `MULTI_STATE(states=5)`. 1D/3D добавляются как отдельный
модуль данных и совместимые с ним окрестности/правила.

## Совместимость и отсев конфигов

Модуль объявляет `requires` — типизированные требования, данные — `capabilities`.
Проверка выполняется до прогона:

```
[OK  ] rule         rule.thresholds  требует neighborhood.count[max_count >= 1]  <- neighborhood.count(max_count=8)
[FAIL] rule         rule.life        требует states.multi[states == 2]            <- states.multi(states=3)
```

Несовместимые конфиги не исполняются: `iter_configs(feasible=True)` (по
умолчанию) отсекает их при обходе, а раннер всё равно подстрахован и запишет
`infeasible`, если обход отключён (`--all-configs`).

Дополнительно можно сузить перебор вручную:

```python
experiment.space.constrain(
    lambda values: values["data.width"] * values["data.height"] <= 128 * 64,
    ["data.width", "data.height"],
    name="cost_guard",
)
```

Ограничение срабатывает сразу, как только все его ключи получили значения,
поэтому отсекает ветки дерева, а не только готовые конфиги.

## Производительность

* Numba: ядра (`modules/rules/kernels.py`) параметризованы массивами смещений и
  скалярами; `cache=True` переиспользует машинный код, `nogil=True` позволяет
  режим потоков.
* Пул процессов использует `fork`: эксперимент с лямбдами наследуется без
  pickle, скомпилированные ядра тоже. Перед пулом вызывается `preflight`
  (сборка + прогрев ядра), чтобы не компилировать в каждом процессе.
* Подпись совместимости (модули + их возможности) кэшируется, поэтому обход
  сотен тысяч конфигов не пересчитывает отчёт на каждом листе.
* Число вариаций: `space.count(limit=...)` — точный подсчёт с ограничениями
  (для независимых «хвостов» используется произведение размеров доменов);
  `space.estimate()` — быстрая грубая оценка без ограничений.

Пространства экспериментов всё равно велики (у `life2d` ~2,2·10⁸ «сырых»
комбинаций, у `two_rings` ~1,9·10⁸): пороги правил зависят от окрестности и
друг от друга. Это нормальный режим: работают `--limit`, `--random` и
`--skip`. На 4 процессах 300 тестов 16×16 проходят примерно за секунду.
Несовместимые ветки отсекаются по частичной привязке, как только выбор
модулей их исключает.

## Воспроизведение и визуализация

Конфиг — словарь `ключ -> значение`; `Fixed.canonical()` — стабильный JSON,
`Fixed.digest()` — короткий хэш. По умолчанию сохраняются только успешные
конфиги (`successes.jsonl`): на большом переборе полный лог отклонённых
тестов занимает гигабайты и не нужен. `--save-all` включает полный
`trials.jsonl`; `summary.json` с счётчиками и причинами пишется всегда.
`--resume` читает уже сохранённые успехи и не повторяет их.

```bash
python -m casweep replay --digest abc123 --run-dir runs/demo --trace
```

При поиске по дайджесту seed берётся из сохранённого теста, поэтому траектория
воспроизводится; `--seed` позволяет намеренно её изменить.

Визуализация по умолчанию — **живая, в терминале, без файлов**. Общий код —
[visualize/live.py](visualize/live.py); палитра и кадры — в
[visualize/standard.py](visualize/standard.py). Любая сетка с `.array`,
`.states`, `.zero_state` рисуется одинаково:

* `scheme="age"` — монотонная шкала «молодая → старая» (старение);
* `scheme="hue"` — фазы по кругу тонов (циклические правила);
* цветные полублоки в терминале, при `--no-color` — ASCII;
* при перенаправлении вывода цвет и перерисовка отключаются сами.

Один автомат:

```bash
python -m casweep view --run-dir runs/rot --index 0
python -m casweep view --config '<json>' --steps 60 --fps 10
python -m experiments.two_rings_rot.view --digest abc123 --run-dir runs/rot
```

Несколько автоматов рядом (кадр за кадром):

```bash
python -m casweep compare --run-dir runs/rot --limit 6 --steps 40
python -m casweep compare --random 6 --steps 30 --scheme hue
```

`view`/`compare` ничего не пишут на диск: конфиг + seed воспроизводят
траекторию из `successes.jsonl`, поэтому картинки не нужно хранить. У каждого
пакета `experiments/<name>/` есть `view.py` — тонкая обёртка над тем же
стандартом (`python -m experiments.multi3.view`, и т.д.).

Если `--experiment` не указан, команды с `--run-dir` (`view`, `compare`,
`render`, `replay`, `gallery`) определяют эксперимент по сохранённому конфигу:
по выбранным модулям окрестности/правила и набору фильтров. Поэтому старые
прогоны, например `runs/two_rings`, открываются без ручного выбора фабрики:

```bash
python -m casweep compare --run-dir runs/two_rings --limit 4
python -m casweep view --run-dir runs/two_rings --index 0
```

## Браузерная галерея: много автоматов сразу

Для просмотра без терминала есть `browse`. Он собирает одну страницу с
карточками: у каждой своя анимация, слайдер кадров и подпись конфига.

```bash
python -m casweep browse --run-dir runs/two_rings --limit 9 --steps 30 --no-open
python -m casweep browse --random 6 --steps 40 --scheme auto
```

По умолчанию файл кладётся **рядом с кодом эксперимента**:

```
experiments/two_rings/_viewer/index.html
```

Это самодостаточный HTML: кадры встроены в страницу, сервер и Python при
открытии не нужны. `--out` переопределяет каталог, `--no-open` отключает
автозапуск браузера. Старые прогоны определяются по конфигу, как и в
`view`/`compare`.

Перед отрисовкой `browse` проверяет, объявил ли модуль свой визуализатор:

```python
from visualize.protocol import Standard2DVisualizer

class MyRule(Module):
    visualizer = Standard2DVisualizer("hue", name="my-phases")
```

Приоритет: правило → окрестность → данные; если ничего не объявлено,
используется общий 2D-стандарт. `--scheme auto` берёт палитру из
визуализатора, `age`/`hue` принудительно её переопределяют. Посмотреть, что
именно нашлось, можно в `casweep modules` — там появилась строка
`визуализатор: ...`.

Визуализатор может переопределить не только `scheme`, но и `render_frame`, если
модулю нужна своя отрисовка. Общий код галереи при этом не меняется.

Если картинка всё-таки нужна для отчёта, есть явный экспорт. Его файлы тоже
по умолчанию лежат рядом с экспериментом (`experiments/<name>/_viewer/...`):

```bash
python -m casweep render --digest abc123 --run-dir runs/rot
python -m casweep gallery --run-dir runs/rot --limit 12
```

`render` пишет `final.png`, кадры, `strip.png`, `sheet.png` и `viewer.html`;
`gallery` собирает `index.html`. Это только по запросу; обычный прогон хранит
лишь `successes.jsonl` и `summary.json`. Каталог `_viewer` можно удалить — он
целиком генерируется заново.

## CLI

```
python -m casweep modules      # роли, модули, параметры, требования
python -m casweep space        # оси, оценка/точное число, первые конфиги
python -m casweep run          # прогон: --limit --skip --workers --mode
                               #        --out --resume --random --all-configs
                               #        --until --max-iterations --save-all
                               #        --stop-file --order --stride
python -m casweep replay       # один конфиг + отчёт совместимости + трасса
python -m casweep view         # живой автомат в терминале (без файлов)
python -m casweep compare      # несколько автоматов рядом (без файлов)
python -m casweep browse       # браузерная галерея: много автоматов с анимацией
python -m casweep render       # экспорт: кадры, лента, лист, viewer.html
python -m casweep gallery      # экспорт: index.html по успехам прогона
```

Режимы прогона:

* `--max-iterations N` — общий бюджет итераций на тест: стадии урезаются по
  остатку, фильтры конца стадии всё равно проверяются (в метриках появляется
  `capped`);
* `--until N` — перебор идёт, пока не наберётся N прошедших автоматов; вместе с
  `--random` это поиск по огромному пространству без `--limit`;
* `--stop-file PATH` — остановка из другого терминала (`touch PATH`): новые
  задачи перестают браться, успешные результаты остаются;
* `--save-all` — полный лог вместе с отказами.

`Ctrl+C` во время прогона останавливает планировщик, сохраняет уже полученные
успехи и пишет `summary.json` с `"interrupted": true`.

## Тесты

```bash
python -m unittest discover -s tests -v
```

Покрыто: зависимые диапазоны, ветки модулей и производные значения,
ограничения и подсчёт, диагональный срез и его совпадение с подсчётом,
включительные границы диапазонов (в том числе пустые и обратные),
канонический JSON/дайджест, динамическая совместимость
(`life` только при двух состояниях), детерминизм прогона, ранняя отбраковка
фильтрами, процессный пул и возобновление, два кольца и старение, критерии
облаков; новые правила (тоталистическое, голосование, цикл, рефрактерность),
фильтры (симметрия, период, активность, энтропия, фазы, кластеры, заполнение,
дрейф доли мёртвых, структура последнего состояния), режимы `--until`,
`--max-iterations`, `--stop-file`, политика хранения (только успехи /
`--save-all`), а также правило гниения, новый эксперимент, живой терминальный
просмотр `view`/`compare` без записи файлов, браузерную галерею `browse`,
определение визуализатора по модулю и фолбэк на общий стандарт, пакеты
экспериментов с `view.py` и явный экспорт PNG/HTML.

## Ограничения и направления развития

* Пока реализован только 2D-сценарий; 1D/3D — новые модули данных/окрестностей.
* `max_count` — теоретический максимум соседей: у краёв при `clip`/`wrap`
  фактических соседей меньше. Если это важно правилу, его нужно учитывать в
  собственной реализации `step`.
* Подсчёт числа совместимых конфигов без `limit` может быть очень дорогим —
  используйте `--limit` и `space.estimate()`.
* Нет трассировки произвольного императивного кода с `.get()`: планирование
  декларативное (см. раздел выше). Это осознанный компромисс ради надёжного
  числа вариаций и статической проверки совместимости.
* Нет автоматического отсева по «влезает ли окрестность в сетку»: это
  выражается ограничением (`space.constrain`) или фильтром.
