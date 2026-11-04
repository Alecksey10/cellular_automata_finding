Я бы предложил спроектировать это как движок экспериментов с декларативным описанием конфигураций и императивным исполнением теста.

Причём сейчас важнее не придумать идеальный DSL, а определить границы между основными типами. Если эти границы будут удачными, ты сможешь менять перебор, добавлять новые автоматы, подключать Numba, новые фильтры и визуализаторы без переписывания ядра.

Я вижу здесь четыре главные сущности:

1. Experiment — что мы исследуем.
2. Config — с какими параметрами выполняется конкретный эксперимент.
3. Execution — как выполняется эксперимент.
4. Result — что получилось и что мы сохранили.

И отдельно:

- Search Engine — создаёт конфигурации.
- Runtime — исполняет тест.
- Modules — подключаемые алгоритмы.
- Data — типизированные входы, выходы, события и метрики.
- Storage — сохраняет результаты.
- Visualization — читает сохранённые результаты.

Далее я опишу минимальную архитектуру, которую можно постепенно расширять.

# 1. Общая архитектура


У каждой части должна быть своя ответственность.

| Подсистема     | За что отвечает                       |
| -------------- | ------------------------------------- |
| Experiment DSL | Пользовательское описание теста       |
| Compiler       | Анализирует описание, создаёт IR      |
| Search Engine  | Генерирует конфигурации               |
| Scheduler      | Распределяет задачи по исполнителям   |
| Runtime        | Выполняет одну конфигурацию           |
| Test           | Сам алгоритм исследования             |
| Data System    | Типы входов, выходов, метрик, событий |
| Filter System  | Решает, продолжать ли выполнение      |
| Storage        | Сохраняет конфиги и результаты        |
| Visualization  | Отображает результаты                 |

Очень важно: Search Engine не должен знать, что именно исследуется. Он должен уметь перебирать параметры хоть клеточного автомата, хоть математической функции, хоть физической симуляции.

# 2. Основные типы

Я бы разделил типы на три слоя.

Слой A — описание

`Parameter`, `Domain`, `Reference`, `Experiment`, `TestDefinition`

Слой B — исполнение

`Config`, `ExecutionContext`, `TestRuntime`, `State`, `Event`

Слой C — результаты

`Result`, `Metric`, `Artifact`, `Trace`, `Termination`

## 2.1. Parameter — описание параметра

Параметр — это не значение, а описание того, как значение должно быть получено.

```
@dataclass(frozen=True)class Parameter:    id: ParameterId    domain: Domain    default: Any = None    metadata: Metadata = Metadata()
```

Основные виды:

```
ValueDomain
RangeDomain
ChoiceDomain
DependentDomain
DerivedDomain
```

Примеры:

```
Generator.Value(5)
Generator.Range(1, 10, 1)
Generator.Chooser([1, 2, 3])
Generator.Range(x, y, 1)
```

Но я бы не создавал отдельные классы параметров для каждого случая. Лучше иметь общий `Parameter`, а спецификацию значения вынести в `Domain`.

Так проще расширять систему.

## 2.2. Domain — пространство возможных значений

```
class Domain(Protocol):
    def iter_values(self, context) -> Iterator[Any]:
        ...
    def resolve(self, context) -> Any:
        ...
    def dependencies(self) -> set[ParameterId]:
        ...
```

Пример:

```
@dataclass(frozen=True)
class RangeDomain:
    start: Expr
    stop: Expr
    step: Expr
```

`Expr` — выражение, которое может зависеть от других параметров.

Например:

```
z = Generator.Range(x, y, 1)
```

Тогда `start` ссылается на `x`, `stop` — на `y`.

Я бы сделал следующие типы выражений:

```
LiteralExpr
ParameterRefExpr
BinaryExpr
FunctionExpr
```

Это позволит писать:

```
Generator.Range(    x + 1,    y * 2,    1,)
```

без необходимости создавать специальные генераторы для каждой арифметической операции.

## 2.3. Config — зафиксированная конфигурация

Это один из самых важных типов.

```
@dataclass(frozen=True)
class Config:
    id: ConfigId
    values: Mapping[ParameterId, Any]
    seed: int
```

Например:

```
Config(
    id="config_000124",
    values={
        "width": 5,
        "height": 5,
        "states_count": 4,
        "neighborhood": "ring_3x3",
        "birth_min": 2,
        "birth_max": 3,
        "survival_min": 1,
        "survival_max": 3,
    },
    seed=12345,
)
```

После создания `Config` значения больше не должны меняться.

Это принципиально для:

- Повторяемости.
- Кеширования.
- Параллельного исполнения.
- Восстановления после сбоя.
- Сравнения результатов.

Важный момент: в конфигурации должны находиться параметры эксперимента, но не вычисленные массивы, не функции и не объекты Numba.

Начальные данные должны генерироваться уже на этапе исполнения с использованием `Config.seed`.

# 3. Experiment и TestDefinition

Здесь я бы разделил само описание эксперимента и тест, который выполняется для каждой конфигурации.

```
@dataclass
class Experiment:
    id: ExperimentId
    parameters: list[Parameter]
    test: TestDefinition
    search: SearchDefinition
    metadata: Metadata
```

`TestDefinition` описывает вычисление:

```
@dataclass
class TestDefinition:
    inputs: list[InputSpec]
    outputs: list[OutputSpec]
    stages: list[Stage]
    events: list[EventSpec]
```

Но сам `TestDefinition` необязательно должен быть полностью декларативным.

Можно сделать так:

```
class Test(Protocol):

    def initialize(self, context) -> State:
        ...

    def execute(self, context, state) -> TestOutcome:
        ...
```

И пользователь пишет:

```
class MyAutomatonTest:

    def initialize(self, context):
        ...

    def execute(self, context, state):
        ...
```

Это уже достаточно для первой версии.

Если потом понадобится более сложная компиляция пользовательского кода, этот класс можно превратить в IR.

# 4. Типы данных теста

Ты отдельно упомянул входы, выходы, события. Я бы сделал это центральной частью системы, но не пытался объединять всё в один универсальный тип.

Есть четыре независимые категории:

Input

Входные данные

То, что тест получает при запуске.

Параметры, начальные массивы, внешние файлы, seed.

State

Состояние

То, что изменяется в ходе вычисления.

Текущий массив автомата, буфер, состояние алгоритма.

Event

События

То, что произошло во время выполнения.

Рождение клетки, смерть, смена стадии, остановка, ошибка.

Output

Результаты

То, что тест выдал в конце или по ходу исполнения.

Метрики, итоговое состояние, временные ряды, сохранённые кадры.

## 4.1. InputSpec

Описывает вход теста.

```
@dataclass(frozen=True)
class InputSpec:
    name: str
    type: DataType
    source: InputSource
    required: bool = True
```

Типы источников:

```
ParameterInput
GeneratedInput
FileInput
ExternalInput
```

Например:

```
InputSpec(
    name="initial_data",
    type=MatrixType(dtype="uint8", dimensions=2),
    source=GeneratedInput(
        generator=Modules.Generators.Discrete.Matrix2D
    ),
)
```

Это не означает, что движок должен знать, как генерировать матрицу. Он просто вызывает соответствующий модуль.

## 4.2. State

Я бы сделал его простым контейнером.

```
@dataclassclass State:    values: dict[str, Any]
```

Пример:

```
state = State(    values={        "current_data": current_data,        "buffer_data": buffer_data,        "iteration": 0,    })
```

Но не стоит заставлять все состояния быть словарями.

Для Numba лучше иметь конкретные структуры:

```
@numba.experimental.jitclass(...)class AutomatonState:    ...
```

Или просто tuple / массивы.

Внешний Runtime может работать с абстрактным `State`, а вычислительное ядро — с конкретным представлением.

Это позволит не платить за абстракции внутри каждого клеточного цикла.

## 4.3. OutputSpec

Описывает выходы теста.

```
@dataclass(frozen=True)class OutputSpec:    name: str    type: DataType    persistence: PersistencePolicy
```

Примеры:

```
OutputSpec(    name="final_data",    type=MatrixType(dtype="uint8", dimensions=2),    persistence=PersistencePolicy.ON_SUCCESS,)
```

```
OutputSpec(    name="density_trace",    type=SeriesType(dtype="float64"),    persistence=PersistencePolicy.ON_SUCCESS,)
```

Я бы сразу предусмотрел политику сохранения:

```
NEVERON_SUCCESSON_FAILUREALWAYSON_EVENT
```

Это важно, потому что если у тебя миллион конфигураций, сохранять все 100 итераций каждого автомата может оказаться неразумно.

# 5. События

События — отдельный канал данных.

Они не должны смешиваться с метриками.

Метрика говорит:

> Плотность мёртвых клеток равна 0.6.

Событие говорит:

> На итерации 18 автомат был остановлен фильтром плотности.

Я бы сделал базовый тип:

```
@dataclass(frozen=True)class Event:    name: str    timestamp: float    iteration: int | None    payload: Mapping[str, Any]
```

Но важнее не сам класс, а интерфейс публикации.

```
class EventEmitter(Protocol):    def emit(self, event: Event) -> None:        ...
```

Например:

```
context.events.emit(
    Event(
        name="cell_born",
        timestamp=context.time,
        iteration=context.iteration,
        payload={
            "x": x,
            "y": y,
            "neighbors": neighbors_count,
        },
    )
)
```

Я бы разделил события на несколько типов:

| Тип       | Пример                             |
| --------- | ---------------------------------- |
| Lifecycle | `test_started`, `test_finished`    |
| Stage     | `stage_started`, `stage_finished`  |
| Iteration | `iteration_completed`              |
| Domain    | `cell_born`, `cell_died`           |
| Filter    | `filter_passed`, `filter_failed`   |
| Execution | `kernel_compiled`, `worker_failed` |
| Storage   | `artifact_saved`                   |

При этом не стоит отправлять `cell_born` на каждую клетку в каждом шаге через обычный Python EventEmitter.

Это создаст огромные накладные расходы.

Вместо этого можно делать агрегирование внутри Numba:

```
event_counts = {    "births": 0,    "deaths": 0,    "survivals": 0,}
```

А после итерации передавать агрегированные события Runtime.

# 6. Метрики и данные фильтрации

Для твоей системы я бы сделал три типа:

- `Metric` — одно значение.
- `MetricSeries` — последовательность значений.
- `MetricSnapshot` — набор метрик на определённый момент.

```
@dataclassclass Metric:    name: str    value: float | int | bool    unit: str | None = None
```

```
@dataclassclass MetricSnapshot:    iteration: int    stage: str    values: dict[str, float | int | bool]
```

Например:

```
MetricSnapshot(    iteration=20,    stage="main",    values={        "zero_density": 0.42,        "alive_density": 0.58,        "mean_survival_strength": 2.8,        "birth_count": 4,        "death_count": 3,    },)
```

Теперь фильтр может работать с этим объектом.

```
class Filter(Protocol):    def evaluate(        self,        metrics: MetricSnapshot,        context: FilterContext,    ) -> FilterResult:        ...
```

```
@dataclassclass FilterResult:    passed: bool    reason: str | None = None    metrics: dict[str, Any] | None = None
```

Например:

```
class ZeroDensityFilter:    def __init__(self, max_density):        self.max_density = max_density    def evaluate(self, metrics, context):        density = metrics.values["zero_density"]        if density > self.max_density:            return FilterResult(                passed=False,                reason="ZERO_DENSITY_EXCEEDED",            )        return FilterResult(passed=True)
```

Для фильтрации между итерациями можно добавить отдельную политику:

```
class FilterPolicy:    frequency: Literal[        "every_iteration",        "stage_end",        "test_end",        "event",    ]
```

Тогда движок знает, когда проверять фильтр, а сам фильтр не знает о стадиях исполнения.

# 7. Runtime и жизненный цикл теста

Здесь я бы ввёл всего несколько основных типов.

## ExecutionContext

Контекст — всё, что доступно тесту во время исполнения.

```
@dataclassclass ExecutionContext:    config: Config    inputs: InputData    metrics: MetricStore    events: EventEmitter    artifacts: ArtifactStore    cancellation: CancellationToken    execution_id: ExecutionId
```

Контекст — не глобальная переменная.

Он создаётся на одну задачу и передаётся в тест.

Например:

```
def initialize(context):    ...
```

```
def iterate(context, state):    ...
```

```
def finalize(context, state):    ...
```

Это позволит:

- Получать параметры.
- Публиковать метрики.
- Отправлять события.
- Сохранять артефакты.
- Останавливать тест.
- Обрабатывать отмену.

## TestOutcome

Тест должен возвращать понятный итог.

```
@dataclassclass TestOutcome:    status: TestStatus    metrics: MetricSnapshot    artifacts: list[ArtifactRef]    reason: str | None = None
```

Статусы:

```
class TestStatus(Enum):    SUCCESS = "success"    REJECTED = "rejected"    FAILED = "failed"    CANCELLED = "cancelled"    TIMEOUT = "timeout"
```

Здесь я бы не использовал `Processor.STOP_ITERATION` как единственный механизм управления.

Лучше:

```
context.stop(    reason="FAILED_MAIN_ZERO_DENSITY")
```

Либо:

```
return TestOutcome(    status=TestStatus.REJECTED,    reason="FAILED_MAIN_ZERO_DENSITY",)
```

А `Processor.STOP_ITERATION` можно оставить как удобный синтаксический сахар.

# 8. Стадии теста

С твоими тремя стадиями я бы сделал отдельную сущность `Stage`.

```
@dataclassclass Stage:    name: str    iterations: Expr    filter_policy: FilterPolicy    filters: list[Filter]
```

Например:

```
warmup = Stage(    name="warmup",    iterations=warmup_iterations,    filter_policy=FilterPolicy(        frequency="stage_end"    ),    filters=[        zero_density_filter,    ],)
```

```
main = Stage(    name="main",    iterations=main_iterations,    filter_policy=FilterPolicy(        frequency="every_iteration"    ),    filters=[        zero_density_filter,        alive_density_filter,    ],)
```

```
final = Stage(    name="final",    iterations=final_iterations,    filter_policy=FilterPolicy(        frequency="stage_end"    ),    filters=[        final_density_filter,    ],)
```

Тогда общий интерфейс:

```
class StageExecutor:    def execute(        self,        stage: Stage,        context: ExecutionContext,        state: State,    ) -> StageOutcome:        ...
```

`StageExecutor` должен знать:

- Сколько итераций выполнить.
- Когда вычислить метрики.
- Когда запустить фильтры.
- Когда завершить стадию.
- Какую стадию запускать дальше.

Но он не должен знать, что такое клетка или соседство.

# 9. Основной движок перебора

Вот здесь я бы постарался сохранить минимальное количество абстракций.

Мне кажется, тебе достаточно четырёх типов:

```
ExperimentCompilerSearchEngineSchedulerWorker
```

## SearchEngine

```
class SearchEngine(Protocol):    def generate(        self,        experiment: ExperimentIR,    ) -> Iterator[Config]:        ...
```

Реализации:

```
GridSearchRandomSearchAdaptiveSearch
```

Для первой версии хватит `GridSearch`.

## Scheduler

```
class Scheduler(Protocol):    def submit(self, config: Config) -> TaskHandle:        ...    def get_results(self) -> Iterator[TaskResult]:        ...
```

Scheduler распределяет задачи между процессами.

Он не знает, как устроен тест.

## Worker

```
class Worker:    def execute(        self,        experiment: CompiledExperiment,        config: Config,    ) -> TaskResult:        ...
```

Внутри:

1. Создать `ExecutionContext`.
2. Подготовить входы.
3. Создать состояние.
4. Выполнить стадии.
5. Получить `TestOutcome`.
6. Сохранить результат.
7. Вернуть `TaskResult`.

## TaskResult

```
@dataclassclass TaskResult:    config_id: ConfigId    status: TestStatus    outcome: TestOutcome | None    error: ErrorInfo | None    duration: float
```

Именно `TaskResult` я бы использовал как основной объект для планировщика и интерфейса пользователя.

# 10. Модули клеточного автомата

Я бы не делал отдельный интерфейс для каждого вида соседства.

Лучше использовать общие интерфейсы.

## Neighborhood

```
class Neighborhood(Protocol):    def max_count(self, params) -> int:        ...    def count(        self,        data,        x: int,        y: int,        params,    ) -> int:        ...
```

Для разных реализаций:

```
Ring3x3Ring5x5Radius5x5Quadrants2DSymmetricMask2D
```

Но нужно решить, что конкретно означает `count`.

Я бы рекомендовал:

```
count_state(data, x, y, state)
```

И отдельно:

```
sum_state(data, x, y)
```

Тогда можно будет вычислять:

- Количество соседей состояния 0.
- Количество соседей состояния 1.
- Количество соседей состояния 2.
- Сумму состояний соседей.
- Количество соседей, имеющих состояние больше заданного.

Это намного гибче, чем один универсальный метод.

## Rule

Правило должно быть отдельной сущностью:

```
class Rule(Protocol):    def apply(        self,        current_state,        neighborhood_data,        params,    ) -> int:        ...
```

Например:

```
class SurvivalStrengthRule:    def apply(        self,        current_state,        neighbors_count,        params,    ):        if current_state == 0:            if params.birth_min <= neighbors_count <= params.birth_max:                return 1            return 0        if params.survival_min <= neighbors_count <= params.survival_max:            return min(                current_state + 1,                params.max_survival_strength,            )        return 0
```

При этом `Rule` не должен знать о том, как вычисляется количество соседей.

И `Neighborhood` не должен знать, что клетка может рождаться, выживать или умирать.

Это два независимых модуля.

# 11. Numba Backend

Я бы предусмотрел отдельный интерфейс для компиляции.

```
class KernelBackend(Protocol):    def compile(        self,        kernel: KernelDefinition,        specialization: Specialization,    ) -> CompiledKernel:        ...    def execute(        self,        kernel: CompiledKernel,        inputs: tuple,    ) -> Any:        ...
```

И реализации:

```
PythonBackendNumbaBackend
```

Позже:

```
CythonBackendCppBackend
```

Но не стоит пытаться сделать Numba полностью универсальным.

Например, `Neighborhood` — Python-класс с динамическим выбором метода — не обязательно сможет работать внутри `numba.njit`.

Я бы сделал два уровня интерфейса:

- Python-интерфейс модуля.
- Компилируемое представление модуля.

Например:

```
Modules.Neighborhood.Ring3x3
```

Может иметь:

```
class Ring3x3:    def build_kernel(self, params):        ...
```

И возвращать функцию, которую Numba может скомпилировать.

Тогда движок не должен разбираться в деталях реализации каждого модуля.

# 12. Сохранение данных

Я бы сразу разделил сохранение на два понятия.

## Artifact

Это большой объект, который можно сохранить отдельно:

- Матрица автомата.
- Последовательность кадров.
- Видеозапись.
- Файл с конфигурацией.
- График.
- Снимок состояния.
- Лог выполнения.

```
@dataclassclass Artifact:    id: ArtifactId    name: str    type: ArtifactType    uri: str    metadata: dict
```

`uri` может ссылаться на файл, запись в базе, memory-mapped массив и т. д.

Например:

```
Artifact(    id="artifact_123",    name="final_state",    type=ArtifactType.NDARRAY,    uri="results/config_123/final.npy",    metadata={        "shape": [5, 5],        "dtype": "uint8",    },)
```

## Result

А `Result` — это компактная запись о выполнении.

```
@dataclassclass Result:    config: Config    status: TestStatus    metrics: MetricSnapshot    artifacts: list[ArtifactRef]    termination: Termination
```

Таким образом, интерфейс пользователя может показывать 10 000 результатов, не загружая в память 10 000 массивов.

Это очень важное решение для масштабирования.

# 13. Типы, которые я бы действительно реализовал первыми

Всё описанное выше — не значит, что нужно реализовывать 50 классов сразу.

Я бы разделил их на обязательные и отложенные.

MVP

Минимальное ядро

Достаточно для первых рабочих экспериментов.

- `Parameter`
- `Domain`
- `Config`
- `Experiment`
- `Test`
- `ExecutionContext`
- `TestOutcome`
- `MetricSnapshot`
- `Filter`
- `Stage`
- `SearchEngine`
- `Worker`
- `Result`
- `Artifact`

Следующий этап

Расширение

- `ExperimentIR`
- `Compiler`
- `Scheduler`
- `KernelBackend`
- `EventEmitter`
- `ArtifactStore`
- `ResultStore`
- `CancellationToken`
- `Specialization`
- `SearchStrategy`

Позже

Сложные возможности

- Adaptive Search
- Кеширование компиляции
- Динамическое pruning
- Распределённое исполнение
- Восстановление после сбоя
- Граф зависимостей вычислений
- Автоматическая кодогенерация
- Плагины внешних backend

# 14. Три концепции, которые позволят не перемудрить

Здесь, пожалуй, самое важное.

## Концепция A. Experiment как обычная функция

Самый простой вариант:

```
def experiment(context):    ...
```

А движок только:

- Создаёт конфигурацию.
- Передаёт её в `context`.
- Вызывает функцию.
- Сохраняет результат.

Преимущества:

- Просто реализовать.
- Максимальная свобода.
- Можно писать обычный Python.
- Нет необходимости создавать сложный DSL.

Недостаток:

- Труднее анализировать программу до исполнения.
- Труднее заранее считать число вариантов.
- Нельзя гарантировать, что весь код компилируется в Numba.

Я бы использовал этот вариант как первую рабочую версию.

## Концепция B. Experiment как декларативное описание

Пользователь описывает:

```
Experiment(    parameters=...,    stages=...,    filters=...,)
```

Движок самостоятельно строит план исполнения.

Преимущества:

- Хорошо анализируется.
- Можно оптимизировать.
- Можно автоматически генерировать код.
- Можно заранее проверять зависимости.

Недостатки:

- Меньше свободы.
- DSL становится сложнее.
- Нужно реализовывать интерпретатор или компилятор.

Это хороший вариант для более зрелого проекта.

## Концепция C. Гибридная — моя рекомендация

Параметры и структура эксперимента декларативные, а вычислительная логика — обычные функции.

Например:

```
def main():    radius = Generator.Range(1, 5, 1)    states = Generator.Range(2, 8, 1)    @numba_kernel    def iterate(data, radius, states):        ...    Processor.Run(        initialize=initialize,        iterate=iterate,        stages=[            Stage(...),            Stage(...),            Stage(...),        ],    )
```

Преимущества:

- Перебор параметров можно анализировать.
- Сложную логику можно писать обычным Python.
- Numba-компиляция остаётся отдельной задачей.
- Можно постепенно переходить к полноценному IR.
- Модули не привязаны к реализации движка.

Именно этот вариант я бы выбрал.

# 15. Как бы я свёл всё это к единому API

Предлагаю вот такой вариант интерфейса, который будет достаточно простым для написания, но не ограничит дальнейшее развитие.

```
from engine import Generator, Experiment, Processor
from engine import Stage, Filter, Modules, Numba


def main():

    # =============================================
    # PARAMETERS
    # =============================================

    width = Generator.Value(5)
    height = Generator.Value(5)

    states = Generator.Range(2, 8, 1)

    neighborhood = Generator.Chooser([
        Modules.Neighborhood.Ring3x3,
        Modules.Neighborhood.Ring5x5,
        Modules.Neighborhood.Radius5x5,
        Modules.Neighborhood.Quadrants2D,
        Modules.Neighborhood.SymmetricMask2D,
    ])

    birth_min = Generator.Range(1, 8, 1)
    birth_max = Generator.Range(birth_min, 8, 1)

    survival_min = Generator.Range(1, 8, 1)
    survival_max = Generator.Range(survival_min, 8, 1)

    max_strength = Generator.Range(2, 8, 1)


    # =============================================
    # INPUTS
    # =============================================

    initial_data = Generator.Data(
        Modules.Generators.Discrete.Matrix2D,
        shape=(height, width),
        states=states,
    )


    # =============================================
    # KERNELS
    # =============================================

    @Numba.Kernel
    def iterate(data, buffer, neighborhood, rule):
        ...


    @Numba.Kernel
    def calculate_metrics(data):
        ...


    # =============================================
    # FILTERS
    # =============================================

    @Filter
    def density_filter(metrics, config):

        return metrics["zero_density"] <= config.max_density


    # =============================================
    # STAGES
    # =============================================

    warmup = Stage(
        name="warmup",
        iterations=Generator.Range(1, 20, 1),
        filters=[density_filter],
        filter_frequency="stage_end",
    )

    main_stage = Stage(
        name="main",
        iterations=Generator.Range(10, 100, 1),
        filters=[density_filter],
        filter_frequency="every_iteration",
    )

    final_stage = Stage(
        name="final",
        iterations=Generator.Range(10, 100, 1),
        filters=[density_filter],
        filter_frequency="stage_end",
    )


    # =============================================
    # EXPERIMENT
    # =============================================

    experiment = Experiment(

        inputs={
            "initial_data": initial_data,
        },

        kernels={
            "iterate": iterate,
            "metrics": calculate_metrics,
        },

        stages=[
            warmup,
            main_stage,
            final_stage,
        ],

        outputs={
            "final_state": "ndarray",
            "metrics": "metric_snapshot",
        },

        search=Processor.GridSearch(),

        execution=Processor.Execution(
            backend="multiprocessing",
            workers=8,
        ),

        storage=Processor.Storage(
            save_configs=True,
            save_metrics=True,
            save_artifacts=True,
            save_only_success=True,
        ),
    )


    # =============================================
    # RUN
    # =============================================

    Processor.Run(experiment)
```

## Моё итоговое мнение

Я бы держался следующих принципов:

1. `Generator` только описывает пространство значений.
2. `Config` содержит уже зафиксированные значения.
3. `Experiment` описывает, что нужно выполнить.
4. `Test` реализует конкретную исследуемую логику.
5. `Runtime` выполняет одну конфигурацию.
6. `SearchEngine` выдаёт конфигурации.
7. `Scheduler` распределяет их.
8. `Metrics`, `Events`, `Artifacts` — отдельные системы данных.
9. `Modules` подключаются через интерфейсы.
10. Numba-компиляция не должна зависеть от переборщика.

И самое главное — я бы не пытался сейчас реализовать универсальный компилятор Python в Numba.

Сначала нужно добиться того, чтобы одна конфигурация корректно выполнялась, возвращала метрики и сохраняла результат. Затем — чтобы конфигурации перебирались. Затем — чтобы это работало параллельно.

Полноценный DSL, IR и автоматическая генерация кода появятся естественным образом, когда станет понятно, какие конструкции действительно нужно анализировать.

Для старта я бы реализовал ровно три главных интерфейса:

```
class SearchEngine:    def generate(self, experiment) -> Iterator[Config]:        ...class Test:    def initialize(self, context):        ...    def execute(self, context, state):        ...    def finalize(self, context, state):        ...class ResultStore:    def save(self, result):        ...
```

Вокруг них уже можно постепенно построить всю остальную систему.