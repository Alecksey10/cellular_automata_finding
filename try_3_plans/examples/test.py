Я хочу разработать некоторую систему для перебора клеточных автоматов, но для начала необходимо продумать движок для перебора параметров, который идёт поверх реализуемой программы. 
Программа пишется и на первый прогон не исполняется, а собирает параметры, возможно, она будет местами не такая, как снизу ,возможно, будет перекомпиироваться в промежуточные файлы, но как мне реализовать подобное?
- генерируется переборщик конфигов (только желаемые конфиги сохраняются)
- желательна модульная организация, при которой я в отдельном месте напишу визуализатор автоматов
- должна быть возможность использовать компилирующие блоки numba, многопоточность/процессорность для различных конфигураций.


def main():
    # пишим 2д реализацию.

    dimension_count = Generator.Value(2)
    states_count = Generator.Range(2, 10, 1)
    death_state = Generator.Value(0)
    born_state = Generator.Value(1)
    width = Generator.Range(32, 128, 32)
    height = Generator.Range(width.get(), 128, 32)

    # допустим, подобный генератор принимает способ генерации, маску или что-нибудь такое.
    getGeneratedData:SomeDataClass = Generator.Chooser([Modules.Generators.Discrete.2D.GenerateNStatedMatrix])
    generateStrategies = Generator.Chooser([Modules.Randomizer.PlainRandom, Modules.Randomizer.Normal, Modules.Randomizer.etc])
    def initialize_iteration():
            strategy = generateStrategies.choose()
            initial_data = getGeneratedData.create(states_count.get())
            ... # какой-нибудь код для инициализации
            ... # например, strategy.fill(dimension=dimension_count, array = initial_data, min=0, max=states_count.get())
            return initial_data
        current_data = initialize_iteration()
        buffer_data = getGeneratedData.create(states_count.get())


        neighborhood_strategy = Generator.Choose(['ring', 'square', 'circle'])
        neighborhood_method = Generator.ChooseDepend(neighborhood_strategy, [('ring', Modules.Filters.Neighborhood.2D.Ring),
        ('square', Modules.Filters.Neighborhood.2D.Sqare),
        ('circle', Modules.Filters.Neighborhood.2D.Radius)])
        neighborhood_params = Generator.ChooseDepend(neighborhood_strategy, [('ring', {'radius1':Generator.Range(1, 5,1), 'radius2':Generator.Range(1, 5,1)},),
        ('square', {'size':Generator.Range(1, 5,1)}),
        ('circle', {'radius':Generator.Range(1, 5,1)})])
        # все данные neighborhood_strategy классы должны поддерживать интерфейс - getMaxCount
        born_threshold_greater = Generator.Range(1, neighborhood_method.get().getMaxCount(*neighborhood_params.get()))
        death_threshold_1_less = Generator.Range(1, born_threshold.get())
        death_threshold_2_greater = Generator.Range(born_threshold.get()+1, neighborhood_method.get().getMaxCount(*neighborhood_params.get())+1)
        # А иначе пусть выживают, например. 

    def iterate_data(): 
            # какая-нибудь numba функция пусть генерируется/компилируется
            for i in range(0, height.get()):
                for j in range(0, width.get()):
                    neighbors_count = (neighborhood_method.get()).run(data = current_data, *neighborhood_params)
                    if(neighbors_count>death_threshold_2_greater)
                        buffer_data[i, j] = death_state.get()
                    elif(neighbors_count<death_threshold_1_less)
                        buffer_data[i, j] = death_state.get()
                    elif(neighbors_count>born_threshold_greater)
                        buffer_data[i, j] = born_state.get()
                    else:
                        buffer_data[i, j] = min(buffer_data[i, j]+1, states_count)
            swap(buffer_data, current_data)
            return data

        warm_up_iterations_count = Generator(Range(0, 10))
        #прогревание
        for i in range(0, warm_up_iterations_count):
            # ТО, что пойдёт на оптимизацию
            iterate_data()
        # здесь пусть будут методы фильтрации
        filter_params = Generator.Range(0.4, 0.7, 0.04)
        if(neighborhood_method.get().getCountOf(death_state.get())/
            neighborhood_method.get().getMaxCount(*neighborhood_params.get()) < filter_params.get()):
            Processor.STOP_ITERATION('FAILED WARM UP')

        watch_iterations_count = Generator.Value(10)

        for i in range(0, watch_iterations_count.count()):
            iterate_data()
            if(neighborhood_method.get().getCountOf(death_state.get())/
                neighborhood_method.get().getMaxCount(*neighborhood_params.get()) < filter_params.get()):
                Processor.STOP_ITERATION('FAILED ITERATED')
            # какой-нибудь фильтр, чтобы в данных было более 20 квадратиков определённых типов. Просто такой фильтр для демонстрации гибкости
            if(SomeModule.Filters.CountOfSquares.2D(data = current_data, types = [death_state]) > 20):
                Processor.STOP_ITERATION('FAILED ITERATED')
        Processor.STOP_ITERATION('SUCCESS')

То есть из данного кода вычисляется сетка перебора и число вариаций.
А затем, из какого-нибудь организатора вычислений происходит вычисление данного кода с компиляцией возможных моментов. 
На момент прогона все значения зафиксированы тестом. 


Можно поступить следующим образом:
* Программа компилируется в какую-нибудь низкоуровневую (то есть мета-программирование)
* Вместо этих генераторов подставляются значения по ссылкам
* после исполнения теста запускается новый тест с другими значениями

Например, простой пример:

## Пример 1
def main():
    x1 = Generator.Range(2, 100, 1)
    y1 = Generator.Range(2, 100, 1)
    v1 = Generator.Chooser([1,2,3,4,5,10,1,5])
    z1 = Generator.Range(x1, y1, 1)
    
    @numba.njit
    def someLogic(x1, y1, v1, z1):
        for i in range(0, dimension_count):
            if((x1**2+1255611+y1+z1^2)%44+v1==22):
                return 'success'

    someLogic(x1.get(), y1.get(),v1.get(), z1.get())
    
    # То есть вариантов перебора 99*99*8*(~50) - много, но отсюда сгенерируется программа, которая сможет компилироваться в numba (допустим, мне потребуется так написать интерфейс)
    ----
    То есть будет нечто подобное
    def main():
        # чётко назначены значения
        x1 = value1
        y1 = value2
        v1 = value3
        z1 = value4

        @numba.njit
        def someLogic(x1, y1, v1, z1):
            ...

        someLogic(x1.get(), y1.get(),v1.get(), z1.get())




from automata import Generator, Processor, Modules


def main():

    dimension = Generator.Value(2)

    states_count = Generator.Range(2, 10, 1)

    death_state = Generator.Value(0)
    born_state = Generator.Value(1)

    width = Generator.Range(32, 128, 32)
    height = Generator.Range(width, 128, 32)

    initial_data = Generator.Data(
        Modules.Generators.Discrete.Matrix2D,
        shape=(height, width),
        states=states_count,
    )

    randomizer = Generator.Chooser([
        Modules.Randomizer.PlainRandom,
        Modules.Randomizer.Normal,
    ])

    neighborhood_type = Generator.Chooser([
        "ring",
        "square",
        "circle",
    ])

    neighborhood = Generator.ChooseDepend(
        neighborhood_type,
        {
            "ring": Modules.Neighborhood.Ring,
            "square": Modules.Neighborhood.Square,
            "circle": Modules.Neighborhood.Circle,
        }
    )

    neighborhood_params = Generator.ChooseDepend(
        neighborhood_type,
        {
            "ring": {
                "radius": Generator.Range(1, 5, 1),
            },

            "square": {
                "size": Generator.Range(1, 5, 1),
            },

            "circle": {
                "radius": Generator.Range(1, 5, 1),
            },
        }
    )

    max_neighbors = neighborhood.getMaxCount(
        **neighborhood_params
    )

    born_threshold = Generator.Range(
        1,
        max_neighbors,
        1,
    )

    death_threshold = Generator.Range(
        1,
        born_threshold,
        1,
    )

    @Processor.Kernel
    def iterate(
        current_data,
        buffer_data,
        neighborhood,
        neighborhood_params,
        born_threshold,
        death_threshold,
        states_count,
        death_state,
        born_state,
    ):

        for i in range(current_data.shape[0]):
            for j in range(current_data.shape[1]):

                neighbors = neighborhood.run(
                    current_data,
                    i,
                    j,
                    **neighborhood_params,
                )

                if neighbors < death_threshold:
                    buffer_data[i, j] = death_state

                elif neighbors > born_threshold:
                    buffer_data[i, j] = born_state

                else:
                    buffer_data[i, j] = min(
                        current_data[i, j] + 1,
                        states_count - 1,
                    )

        return buffer_data

    Processor.Run(
        initialize=initial_data,
        iterate=iterate,
        iterations=100,
        save_if=lambda result: result.metrics["alive_ratio"] > 0.2,
    )
    