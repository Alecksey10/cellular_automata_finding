# try_3_code — движок перебора клеточных автоматов

Рабочая реализация требований из `try_3_plans`: описание эксперимента отдельно
от исполнения, явные интерфейсы модулей, numba-ядра, параллельный перебор,
сохранение успешных конфигов вместе с настройками и визуализация кадров.

## Установка

Нужны Python ≥ 3.11, `numpy` и `numba`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Пакет не требует установки: все команды запускаются из корня `try_3_code`.

```bash
cd try_3_code
../.venv/bin/python -m sweep --help
```

## Быстрый старт

```bash
# карта осей и размер пространства
python -m sweep space -e rings_rot --count --limit 5000

# модули, требования и фазы эксперимента
python -m sweep modules -e rings_rot

# случайный перебор 1000 конфигов в 4 процесса
python -m sweep run -e rings_rot --strategy random --limit 1000 \
    --mode process --workers 4 --seed 1 --out runs/rings

# остановиться, как только найдено 200 успехов
python -m sweep run -e life --strategy random --until 200 \
    --mode process --workers 8 --out runs/life

# продолжить прерванный прогон (готовые конфиги пропускаются)
python -m sweep run -e life --strategy random --until 200 \
    --mode process --workers 8 --out runs/life --resume

# посмотреть успехи
python -m sweep show --run-dir runs/life

# открыть галерею прогона в браузере: карточки всех автоматов, кадры
# собираются из конфига на лету, длительности фаз можно менять и пересобирать
python -m sweep view --run-dir runs/life --port 8000

# по желанию — экспорт PNG/HTML (по умолчанию в ./viz, каталог прогона не трогается)
python -m sweep viz --run-dir runs/life --index 0 --out viz
python -m sweep live --run-dir runs/life --index 0 --fps 8
```

`Ctrl+C` прерывает прогон: работники завершают текущий тест, уже записанные
результаты остаются на диске, повторный запуск с `--resume` продолжит с места
остановки.

## Слои

| Слой | Модуль | Ответственность |
|---|---|---|
| описание | `sweep/experiment.py` | оси, выбранные модули, фазы, фильтры |
| перебор | `sweep/space.py`, `domains.py`, `handles.py` | зависимые оси, ограничения, стратегии |
| интерфейсы | `sweep/interfaces.py`, `module.py` | `Capability` / `Requirement`, роли модулей |
| сборка | `sweep/assembly.py` | проверка совместимости и построение рантайма |
| исполнение | `sweep/runtime.py` | один тест: инициализация, фазы, метрики, фильтры |
| параллельность | `sweep/parallel.py` | serial / thread / process, отмена, прогресс |
| хранение | `sweep/storage.py` | `successes.jsonl`, `trials.jsonl`, `summary.json` |
| визуализация | `sweep/visualize/` | общий интерфейс, PNG/HTML, терминал |
| эксперименты | `sweep/experiments/` | `life`, `rings_rot`, `cyclic_waves` |

Ключевые правила:

* конфиг — только скаляры и `seed`; массивы и numba-объекты создаются заново;
* прогон хранит только конфиги и метрики: кадры и картинки не сохраняются, их
  пересобирает `sweep/replay.py` из того же конфига; kernel компилируется один
  раз на эксперимент, а не на конфиг;
* несовместимые пары «данные × правило» отсекаются до итераций;
* итерация пишется прямо в файле эксперимента и передаётся в
  `experiment.iterate(...)`;
* визуализатор эксперимента — отдельный класс за общим интерфейсом.

## Файлы прогона

```
runs/<name>/
  trials.jsonl      каждый исполненный конфиг (компактно)
  successes.jsonl   только успешные конфиги
  summary.json      счётчики и параметры запуска
```

Других файлов прогон не создаёт. Запись в `successes.jsonl` — одновременно
результат отбора и рецепт автомата: по `config` и `seed` его собирают заново
(`sweep/replay.py`), поэтому `view`, `viz` и `live` показывают свежее
исполнение, а не сохранённые картинки.

## Свой эксперимент

```python
from numba import njit
from sweep.experiment import Experiment
from sweep.interfaces import CapabilityKind, Requirement, eq
from sweep.module import Role
from sweep.modules.data import Grid2D
from sweep.modules.filters import DensityFilter
from sweep.modules.neighborhoods import Moore
from sweep.modules.randomizers import UniformFill


@njit
def step(src, dst, zero, born):
    h, w = src.shape
    for y in range(h):
        for x in range(w):
            n = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    n += src[(y + dy) % h, (x + dx) % w] != zero
            alive = src[y, x] != zero
            dst[y, x] = 1 if (n == born if not alive else 2 <= n <= 3) else zero


def build():
    e = Experiment("my_rule", "пример", stop_on_extinct=True)
    e.select(Role.DATA, Grid2D())
    e.select(Role.RANDOMIZER, UniformFill())
    e.select(Role.NEIGHBORHOOD, Moore())
    e.iterate(
        step,
        module_id="rule.my_rule",
        requires=(
            Requirement(CapabilityKind.GRID_2D),
            Requirement(CapabilityKind.ZERO_STATE),
            Requirement(CapabilityKind.NEIGHBORHOOD),
        ),
        declare=lambda ns, view: ns.range("born", 2, 5),
        bind=lambda fixed, view: (fixed["grid2d.zero"], fixed["rule.my_rule.born"]),
    )
    e.constrain(
        lambda v: v["grid2d.width"] * v["grid2d.height"] <= 96 * 96,
        ["grid2d.width", "grid2d.height"],
    )
    e.phase("warmup", e.range("phase.warmup", 0, 7, 3),
            filters=[DensityFilter(min_fill=0.01, max_fill=0.9)])
    e.phase("observe", e.range("phase.observe", 10, 31, 10),
            filters=[DensityFilter(min_fill=0.01, max_fill=0.9)],
            frequency="each")
    return e
```

Зарегистрируйте `build` в `sweep/experiments/__init__.py` и добавьте
визуализатор в `sweep/experiments/<name>/visualizer.py`.

## Примеры из постановки

Файлы в `examples/` — исходные планы и наброски DSL, а не исполняемые модули
этого пакета. Движок реализует те же идеи (`Generator` → оси, `Processor` →
фазы/фильтры, `Modules` → роли), но с явным описанием в `sweep`.
