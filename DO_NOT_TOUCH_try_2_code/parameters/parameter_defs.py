# Parameter definitions for cellular automata experiments

# Width ranges (step 32 от 64)
WIDTHS = [64, 96, 128]

# Height ranges (от width до 129, step 32)
HEIGHTS_MAP = {
    64: [64, 96, 128],
    96: [96, 128],
    128: [128],
}

# Number of states
STATES_LIST = [3, 4, 5, 6]

# Rule rot probability
ROT_VALUES = [1, 2, 3]

# Stage warmup iterations
WARMUP_VALUES = [3, 4, 5]

# Stage observe iterations
OBSERVE_VALUES = [10, 20, 30]
