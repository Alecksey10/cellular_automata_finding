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