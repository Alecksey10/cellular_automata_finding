class States:
    class HasZeroState:
        """Нулевое состояние имеет специальный смысл."""
        state: ...
        pass

    class MultiState:
        """Данные содержат конечное число состояний."""
        pass

    class Discrete:
        """Состояния дискретны."""
        pass