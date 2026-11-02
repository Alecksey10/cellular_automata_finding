Каждый тест в отдельном потоке/проссе, в общем, параллельно
## Пример


процесс генерации конфига
```
get_automaton_params_changer:
    module:
        modules.physics.burning_forest
        type:


```

Пример сгенерированного конфига
```
# иницализация автомата перед тестами
build_automaton:
    module: modules.physics.burning_forest #каждый автоматон может быть инициализирован из dict
    params: config.automata_params


stage 1:
    iteration:
        count: 1
        logic: # вызываемая функция
            module:
                modules.physics.burning_forest
                type: iterator
    filter:
        condition:
            type: signal
            signals_storage: signal_stop()
    filter:
        condition:
            type:
            shared_logic: signal_stop()
filter:
    condition:
        signals_storage: signal_stop()

stage n: ... (конфиг для конфигурирования также можно писать, чтобы менялось число итераций в тестах)


```

Сам автомат опеределяется как 

```
# Интерфейсы данных
States.HasZeroState
States.MultiState:
    states: n
States.Discrete
Topology.2D


# Итого передаваемые данные по существу
Data:
    raw_data: ...
    interfaces:
        {
            States.HasZeroState,
            States.MultiState:
                states: n (пустота, лес, огонь, пепел)
            Topology.2D
        }


# какой-нибудь модуль обработки данных
2D.Neighborhood.Count.RadiusStrategy:
    ...
    process(data):
        isInstance(data, States.HasZeroState) #не суммирует только нулевые состояния
        isInstance(data, Topology.2D) #работает только на двухмерной сетке
        и т.д., например, States.Discrete
        ...
    interfaces():
        2D, discrete_states
2D.Neighborhood.Count.RingStrategy:
    ...
    process(data):
        process(data):
        isInstance(data, States.HasZeroState) #не суммирует только нулевые состояния
        isInstance(data, Topology.2D) #работает только на двухмерной сетке
        и т.д., например, States.Discrete
        ...
    interfaces():
        2D, discrete_states
2D.Neighborhood.Count.RingStrategy:
    ...
    process(data):
        process(data):
        isInstance(data, States.HasZeroState) #не суммирует только нулевые состояния
        isInstance(data, Topology.2D) #работает только на двухмерной сетке
        и т.д., например, States.Discrete
        ...
    interfaces():
        2D, discrete_states


# модуль автомата
class SomeAutomaton:
    1) Метод определения числа соседей клетки.
        Любой передаваемый метод - 
    2) Методы для перебора параметров автомата можно писать в отдельных файлах и подгружать при построении конфига. 
```



## Как можно спроектировать систему

* Генератор конфигов
    * генерируется конфиг:
        * умеет перебирать значения (кучу значений), опционально и т.д
        * умеет перебирать модули, интерфейсы и т.д.


* Исполнитель конфигов
    * Для упрощения можно спрокетировать систему, ожидающую некоторый набор параметров, 




