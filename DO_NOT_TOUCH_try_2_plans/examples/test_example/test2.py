from automata import Generator, Processor, Modules
from automata import numba_kernel

# ============================================================
# EXPERIMENT: 5x5 CELLULAR AUTOMATON
# ============================================================
def main():
    # --------------------------------------------------------
    # 1. ПАРАМЕТРЫ АВТОМАТА
    # --------------------------------------------------------
    dimension = Generator.Value(2)
    width = Generator.Value(64)
    height = Generator.Value(64)
    states_count = Generator.Range(2, 8, 1)
    zero_state = Generator.Value(0)
    birth_state = Generator.Value(1)
    # Насколько крепко должна выживать клетка.
    # После каждого успешного выживания её состояние растёт.
    max_survival_strength = states_count

    # --------------------------------------------------------
    # 2. МЕТОДЫ ВЫЧИСЛЕНИЯ СОСЕДЕЙ
    # --------------------------------------------------------
    neighborhood_type = Generator.Chooser([
        "quadrants",
        "ring_3x3",
        "ring_5x5",
        "radius_5x5",
        "symmetric_random_mask",
    ])

    # Каждый метод реализует интерфейс:
    #
    #     getMaxCount()
    #     count(data, x, y)
    #
    # Дополнительно:
    #
    #     getCountOf(data, state)
    #     getCountOfNeighborhood(data, state, x, y)
    #
    # Конкретная реализация выбирается движком.
    neighborhood_method = Generator.ChooseDepend(
        neighborhood_type,
        {
            "quadrants":
                Modules.Neighborhood.Count.Quadrants2D,
            "ring_3x3":
                Modules.Neighborhood.Count.Ring3x3,
            "ring_5x5":
                Modules.Neighborhood.Count.Ring5x5,
            "radius_5x5":
                Modules.Neighborhood.Count.Radius5x5,
            "symmetric_random_mask":
                Modules.Neighborhood.Count.SymmetricRandomMask2D,
        }
    )

    # --------------------------------------------------------
    # 3. ПАРАМЕТРЫ МЕТОДОВ СОСЕДСТВА
    # --------------------------------------------------------
    neighborhood_params = Generator.ChooseDepend(
        neighborhood_type,
        {
            "quadrants": {
                "include_center": Generator.Value(False),
            },
            "ring_3x3": {
                "include_center": Generator.Value(False),
            },
            "ring_5x5": {
                "include_center": Generator.Value(False),
            },
            "radius_5x5": {
                "radius": Generator.Range(1, 3, 1),
                "metric": Generator.Chooser([
                    "euclidean",
                    "manhattan",
                    "chebyshev",
                ]),
            },
            "symmetric_random_mask": {
                "mask": Generator.Mask(
                    shape=(5, 5),
                    symmetric=True,
                    center=False,
                    probability=Generator.Range(
                        0.1,
                        0.9,
                        0.1,
                    ),
                ),
            },
        }
    )

    # --------------------------------------------------------
    # 4. ПРАВИЛА РОЖДЕНИЯ, СМЕРТИ И ВЫЖИВАНИЯ
    # --------------------------------------------------------
    max_neighbors = neighborhood_method.getMaxCount(
        **neighborhood_params
    )

    # Если клетка мертва, она рождается при таком количестве
    # соседей.
    birth_min = Generator.Range(
        1,
        max_neighbors,
        1,
    )
    birth_max = Generator.Range(
        birth_min,
        max_neighbors,
        1,
    )

    # Если клетка жива, то при данном количестве соседей
    # она может выжить.
    survival_min = Generator.Range(
        1,
        max_neighbors,
        1,
    )
    survival_max = Generator.Range(
        survival_min,
        max_neighbors,
        1,
    )

    # --------------------------------------------------------
    # 5. НАСТРОЙКА СИЛЫ ВЫЖИВАНИЯ
    # --------------------------------------------------------
    # При каждом успешном выживании:
    #
    #     state += 1
    #
    # Но состояние не может превысить max_survival_strength.
    survival_increment = Generator.Value(1)

    # --------------------------------------------------------
    # 6. ИНИЦИАЛИЗАЦИЯ
    # --------------------------------------------------------
    initial_density = Generator.Range(
        0.1,
        0.9,
        0.1,
    )
    initial_generator = Generator.Chooser([
        Modules.Generators.Discrete.Matrix2D.Random,
        Modules.Generators.Discrete.Matrix2D.RandomClusters,
        Modules.Generators.Discrete.Matrix2D.RandomSymmetric,
    ])

    # --------------------------------------------------------
    # 7. ПАРАМЕТРЫ ТРЁХ СТАДИЙ
    # --------------------------------------------------------
    warmup_iterations = Generator.Range(
        2,
        5,
        1,
    )
    main_iterations = Generator.Range(
        1,
        10,
        1,
    )
    final_iterations = Generator.Range(
        1,
        3,
        1,
    )

    # --------------------------------------------------------
    # 8. ФИЛЬТРЫ
    # --------------------------------------------------------
    # Вся фильтрация читает метрики из общего объекта данных.
    #
    # Метрики рассчитываются на стороне автомата,
    # а фильтры только читают их.
    filter_data = Processor.Data({
        "zero_density": float,
        "alive_density": float,
        "mean_survival_strength": float,
        "birth_count": int,
        "death_count": int,
        "survival_count": int,
        "iteration": int,
    })

    # Допустимая плотность мёртвых клеток.
    max_zero_density = Generator.Range(
        0.4,
        0.6,
        0.05,
    )

    # Минимальная плотность живых клеток.
    min_alive_density = Generator.Range(
        0.35,
        0.65,
        0.05,
    )

    # --------------------------------------------------------
    # 9. NUMBA: ОСНОВНОЕ ПРАВИЛО
    # --------------------------------------------------------
    @numba_kernel
    def apply_rule(
        current_data,
        buffer_data,
        neighborhood_method,
        neighborhood_params,
        zero_state,
        birth_state,
        birth_min,
        birth_max,
        survival_min,
        survival_max,
        max_survival_strength,
        survival_increment,
    ):
        for y in range(current_data.shape[0]):
            for x in range(current_data.shape[1]):
                # Подсчёт соседей.
                #
                # Метод neighborhood_method будет выбран
                # и скомпилирован для данной конфигурации.
                neighbors_count = neighborhood_method.count(
                    current_data,
                    x,
                    y,
                    **neighborhood_params,
                )

                current_state = current_data[y, x]

                # --------------------------------------------
                # РОЖДЕНИЕ
                # --------------------------------------------
                if current_state == zero_state:
                    if (
                        birth_min
                        <= neighbors_count
                        <= birth_max
                    ):
                        buffer_data[y, x] = birth_state
                    else:
                        buffer_data[y, x] = zero_state

                # --------------------------------------------
                # ВЫЖИВАНИЕ
                # --------------------------------------------
                else:
                    if (
                        survival_min
                        <= neighbors_count
                        <= survival_max
                    ):
                        buffer_data[y, x] = min(
                            current_state + survival_increment,
                            max_survival_strength,
                        )
                    else:
                        buffer_data[y, x] = zero_state
        return buffer_data

    # --------------------------------------------------------
    # 10. МЕТРИКИ
    # --------------------------------------------------------
    @numba_kernel
    def calculate_metrics(
        data,
        zero_state,
        iteration,
    ):
        total_cells = data.size
        zero_count = 0
        alive_count = 0
        survival_strength_sum = 0
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                state = data[y, x]
                if state == zero_state:
                    zero_count += 1
                else:
                    alive_count += 1
                    survival_strength_sum += state

        zero_density = zero_count / total_cells
        alive_density = alive_count / total_cells
        if alive_count > 0:
            mean_survival_strength = (
                survival_strength_sum / alive_count
            )
        else:
            mean_survival_strength = 0.0

        return {
            "zero_density": zero_density,
            "alive_density": alive_density,
            "mean_survival_strength": mean_survival_strength,
            "iteration": iteration,
        }

    # --------------------------------------------------------
    # 11. ФИЛЬТРЫ
    # --------------------------------------------------------
    @Processor.Filter
    def filter_zero_density(metrics):
        return (
            metrics["zero_density"]
            <= max_zero_density
        )

    @Processor.Filter
    def filter_alive_density(metrics):
        return (
            metrics["alive_density"]
            >= min_alive_density
        )

    @Processor.Filter
    def filter_survival_strength(metrics):
        return (
            metrics["mean_survival_strength"]
            >= 1.0
        )

    # --------------------------------------------------------
    # 12. ЭКСПЕРИМЕНТ
    # --------------------------------------------------------
    @Processor.Experiment
    def run_experiment(context):
        # ================================================
        # INITIALIZATION
        # ================================================
        current_data = initial_generator.create(
            shape=(height, width),
            states=states_count,
            density=initial_density,
        )
        buffer_data = current_data.copy()

        # ================================================
        # STAGE 1: WARMUP
        # ================================================
        for iteration in range(warmup_iterations):
            buffer_data = apply_rule(
                current_data,
                buffer_data,
                neighborhood_method,
                neighborhood_params,
                zero_state,
                birth_state,
                birth_min,
                birth_max,
                survival_min,
                survival_max,
                max_survival_strength,
                survival_increment,
            )
            current_data, buffer_data = (
                buffer_data,
                current_data,
            )

        # Фильтрация только в конце прогрева.
        metrics = calculate_metrics(
            current_data,
            zero_state,
            warmup_iterations,
        )
        filter_data.update(metrics)
        if not filter_zero_density(filter_data):
            Processor.STOP_ITERATION(
                "FAILED_WARMUP_ZERO_DENSITY"
            )

        # ================================================
        # STAGE 2: MAIN ITERATION
        # ================================================
        for iteration in range(main_iterations):
            buffer_data = apply_rule(
                current_data,
                buffer_data,
                neighborhood_method,
                neighborhood_params,
                zero_state,
                birth_state,
                birth_min,
                birth_max,
                survival_min,
                survival_max,
                max_survival_strength,
                survival_increment,
            )
            current_data, buffer_data = (
                buffer_data,
                current_data,
            )

            # Фильтрация на каждой итерации.
            metrics = calculate_metrics(
                current_data,
                zero_state,
                iteration,
            )
            filter_data.update(metrics)

            if not filter_zero_density(filter_data):
                Processor.STOP_ITERATION(
                    "FAILED_MAIN_ZERO_DENSITY"
                )

            if not filter_alive_density(filter_data):
                Processor.STOP_ITERATION(
                    "FAILED_MAIN_ALIVE_DENSITY"
                )

            if not filter_survival_strength(filter_data):
                Processor.STOP_ITERATION(
                    "FAILED_MAIN_SURVIVAL_STRENGTH"
                )

        # ================================================
        # STAGE 3: FINAL ITERATIONS
        # ================================================
        for iteration in range(final_iterations):
            buffer_data = apply_rule(
                current_data,
                buffer_data,
                neighborhood_method,
                neighborhood_params,
                zero_state,
                birth_state,
                birth_min,
                birth_max,
                survival_min,
                survival_max,
                max_survival_strength,
                survival_increment,
            )
            current_data, buffer_data = (
                buffer_data,
                current_data,
            )

        # Фильтрация только в конце последней стадии.
        metrics = calculate_metrics(
            current_data,
            zero_state,
            final_iterations,
        )
        filter_data.update(metrics)

        if not filter_zero_density(filter_data):
            Processor.STOP_ITERATION(
                "FAILED_FINAL_ZERO_DENSITY"
            )

        if not filter_alive_density(filter_data):
            Processor.STOP_ITERATION(
                "FAILED_FINAL_ALIVE_DENSITY"
            )

        # ================================================
        # SUCCESS
        # ================================================
        Processor.SUCCESS(
            data=current_data,
            metrics=filter_data,
            artifacts={
                "initial_data": initial_data,
                "final_data": current_data,
            },
        )

    # --------------------------------------------------------
    # 13. ЗАПУСК ПЕРЕБОРА
    # --------------------------------------------------------
    Processor.Run(
        run_experiment,
        workers=8,
        backend="multiprocessing",
        save_only_success=True,
        save_config=True,
        save_metrics=True,
        save_artifacts=True,
        numba={
            "parallel": False,
            "cache": True,
        },
    )
