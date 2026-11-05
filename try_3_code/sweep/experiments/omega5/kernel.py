"""Numba-ядро правила ``omega5``.

Ядро получает плоские массивы смещений 5×5 и номер группы для каждого
смещения. Группы нужны механизмам счёта: кольца, четверти, направления,
строки/столбцы, шахматная маска и т.п. Состояния считаются отдельно, чтобы
«хитрые» реакции могли смотреть на количество конкретного состояния.
"""

from __future__ import annotations

from numba import njit

from ...kernels import JIT_KWARGS

# Коды механизмов должны совпадать с rule.MECHANISMS.
M_FULL = 0
M_INNER = 1
M_OUTER = 2
M_QUARTERS_MAX = 3
M_QUARTERS_MIN = 4
M_QUARTERS_OPPOSITE_MIN = 5
M_SECTORS_MAX = 6
M_SECTORS_MIN = 7
M_SECTORS_OPPOSITE_MIN = 8
M_AXIS = 9
M_DIAGONAL = 10
M_CHESS_MAX = 11
M_CHESS_MIN = 12
M_ROWS_MAX = 13
M_ROWS_MIN = 14
M_COLS_MAX = 15
M_COLS_MIN = 16
M_CORNERS = 17
M_EDGE_MID = 18
M_DISTINCT_STATES = 19
M_DOMINANT_STATE = 20
M_MINORITY_STATE = 21


@njit(inline="always")
def _max8(a, b, c, d, e, f, g, h):
    value = a
    if b > value:
        value = b
    if c > value:
        value = c
    if d > value:
        value = d
    if e > value:
        value = e
    if f > value:
        value = f
    if g > value:
        value = g
    if h > value:
        value = h
    return value


@njit(inline="always")
def _min8(a, b, c, d, e, f, g, h):
    value = a
    if b < value:
        value = b
    if c < value:
        value = c
    if d < value:
        value = d
    if e < value:
        value = e
    if f < value:
        value = f
    if g < value:
        value = g
    if h < value:
        value = h
    return value


@njit(inline="always")
def _max5(a, b, c, d, e):
    value = a
    if b > value:
        value = b
    if c > value:
        value = c
    if d > value:
        value = d
    if e > value:
        value = e
    return value


@njit(inline="always")
def _min5(a, b, c, d, e):
    value = a
    if b < value:
        value = b
    if c < value:
        value = c
    if d < value:
        value = d
    if e < value:
        value = e
    return value


@njit(**JIT_KWARGS)
def step_omega(
    src,
    dst,
    dy,
    dx,
    group,
    wrap,
    zero,
    states,
    birth_lo,
    birth_hi,
    birth_state,
    surv_lo,
    surv_hi,
    death_lo,
    death_hi,
    trick,
    mechanism,
):
    """Один шаг: счёт соседей, базовые категории, затем особая реакция."""

    height, width = src.shape
    total_offsets = dy.size
    for y in range(height):
        for x in range(width):
            total = 0
            c1 = 0
            c2 = 0
            c3 = 0
            c4 = 0
            c5 = 0
            c6 = 0
            c7 = 0
            g0 = 0
            g1 = 0
            g2 = 0
            g3 = 0
            g4 = 0
            g5 = 0
            g6 = 0
            g7 = 0
            for k in range(total_offsets):
                yy = y + dy[k]
                xx = x + dx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                value = src[yy, xx]
                if value == zero:
                    continue
                total += 1
                if value == 1:
                    c1 += 1
                elif value == 2:
                    c2 += 1
                elif value == 3:
                    c3 += 1
                elif value == 4:
                    c4 += 1
                elif value == 5:
                    c5 += 1
                elif value == 6:
                    c6 += 1
                elif value == 7:
                    c7 += 1
                bucket = group[k]
                if bucket == 0:
                    g0 += 1
                elif bucket == 1:
                    g1 += 1
                elif bucket == 2:
                    g2 += 1
                elif bucket == 3:
                    g3 += 1
                elif bucket == 4:
                    g4 += 1
                elif bucket == 5:
                    g5 += 1
                elif bucket == 6:
                    g6 += 1
                elif bucket == 7:
                    g7 += 1

            # Один скалярный счётчик, выбранный механизмом.
            metric = 0
            if mechanism == M_FULL or mechanism == M_INNER or mechanism == M_OUTER:
                metric = g0
            elif mechanism == M_QUARTERS_MAX:
                metric = _max8(g0, g1, g2, g3, 0, 0, 0, 0)
            elif mechanism == M_QUARTERS_MIN:
                metric = _min8(g0, g1, g2, g3, total, total, total, total)
            elif mechanism == M_QUARTERS_OPPOSITE_MIN:
                left = g0 + g3
                right = g1 + g2
                metric = left if left < right else right
            elif mechanism == M_SECTORS_MAX:
                metric = _max8(g0, g1, g2, g3, g4, g5, g6, g7)
            elif mechanism == M_SECTORS_MIN:
                metric = _min8(g0, g1, g2, g3, g4, g5, g6, g7)
            elif mechanism == M_SECTORS_OPPOSITE_MIN:
                metric = g0 + g4
                if g1 + g5 < metric:
                    metric = g1 + g5
                if g2 + g6 < metric:
                    metric = g2 + g6
                if g3 + g7 < metric:
                    metric = g3 + g7
            elif mechanism == M_AXIS or mechanism == M_DIAGONAL:
                metric = g0
            elif mechanism == M_CHESS_MAX:
                metric = g0 if g0 > g1 else g1
            elif mechanism == M_CHESS_MIN:
                metric = g0 if g0 < g1 else g1
            elif mechanism == M_ROWS_MAX:
                metric = _max5(g0, g1, g2, g3, g4)
            elif mechanism == M_ROWS_MIN:
                metric = _min5(g0, g1, g2, g3, g4)
            elif mechanism == M_COLS_MAX or mechanism == M_COLS_MIN:
                # столбцы и строки используют одну и ту же раскладку групп:
                # для cols группа = dx + 2, для rows — dy + 2.
                if mechanism == M_COLS_MAX:
                    metric = _max5(g0, g1, g2, g3, g4)
                else:
                    metric = _min5(g0, g1, g2, g3, g4)
            elif mechanism == M_CORNERS or mechanism == M_EDGE_MID:
                metric = g0
            elif mechanism == M_DISTINCT_STATES:
                if c1 > 0:
                    metric += 1
                if c2 > 0:
                    metric += 1
                if c3 > 0:
                    metric += 1
                if c4 > 0:
                    metric += 1
                if c5 > 0:
                    metric += 1
                if c6 > 0:
                    metric += 1
                if c7 > 0:
                    metric += 1
            elif mechanism == M_DOMINANT_STATE:
                metric = _max8(c1, c2, c3, c4, c5, c6, c7, 0)
            elif mechanism == M_MINORITY_STATE:
                metric = 0
                if c1 > 0:
                    metric = c1
                if c2 > 0 and (metric == 0 or c2 < metric):
                    metric = c2
                if c3 > 0 and (metric == 0 or c3 < metric):
                    metric = c3
                if c4 > 0 and (metric == 0 or c4 < metric):
                    metric = c4
                if c5 > 0 and (metric == 0 or c5 < metric):
                    metric = c5
                if c6 > 0 and (metric == 0 or c6 < metric):
                    metric = c6
                if c7 > 0 and (metric == 0 or c7 < metric):
                    metric = c7

            state = src[y, x]
            if state == zero:
                nxt = zero
                if birth_lo <= metric <= birth_hi:
                    nxt = birth_state
            else:
                nxt = state
                if death_lo <= metric <= death_hi:
                    nxt = zero
                elif not (surv_lo <= metric <= surv_hi):
                    nxt = zero

            # --- около двадцати «хитрых» реакций -----------------------
            if trick == 1:  # stable6
                if state == 6:
                    nxt = 6
            elif trick == 2:  # predator4
                if state == 4 and c3 >= 1:
                    nxt = 2
                if state == 3 and c4 >= 1:
                    nxt = zero
            elif trick == 3:  # immortal7
                if state == 7:
                    nxt = 7
            elif trick == 4:  # food_chain
                if state == 3 and c2 >= 2:
                    nxt = 4
                if state == 4 and c2 == 0:
                    nxt = zero
            elif trick == 5:  # phoenix2
                if state == zero and c2 >= 1 and c3 >= 1:
                    nxt = 2
            elif trick == 6:  # anti6
                if state == 6 and c1 >= 1:
                    nxt = zero
            elif trick == 7:  # mirror3
                if state == 3 and c3 >= 3:
                    nxt = 2
                if state == 2 and c2 >= 4:
                    nxt = zero
            elif trick == 8:  # spread5
                if state == zero and c5 >= 2:
                    nxt = 5
            elif trick == 9:  # immune5
                if state == 5:
                    nxt = 5
            elif trick == 10:  # convert5
                if state == 5 and c1 > c5:
                    nxt = 1
            elif trick == 11:  # seed6
                if state == zero and c6 >= 3:
                    nxt = 6
            elif trick == 12:  # bomb7
                if state == 7 and total >= 12:
                    nxt = 1
            elif trick == 13:  # hide2
                if state == 2 and c2 >= 5:
                    nxt = 2
            elif trick == 14:  # chain3
                if state == 3:
                    nxt = 2
            elif trick == 15:  # eat6
                if state == 6 and c1 >= 1:
                    nxt = 2
                if state == 1 and c6 >= 1:
                    nxt = zero
            elif trick == 16:  # quarantine7
                if state == 7 and c7 >= 2:
                    nxt = 7
            elif trick == 17:  # fuse23
                if state == 3 and c2 >= 1:
                    nxt = 2
                if state == 2 and c3 >= 1:
                    nxt = 3
            elif trick == 18:  # decay4
                if state == 4:
                    if c3 >= 4:
                        nxt = zero
                    elif c4 > 0:
                        nxt = 4
                    else:
                        nxt = 3
            elif trick == 19:  # burst1
                if state == zero and c1 >= 4:
                    nxt = 6
                if state == 6 and c1 >= 2:
                    nxt = 1
            elif trick == 20:  # swarm2
                if state == zero and c2 >= 3:
                    nxt = 1
                if state == 1 and c2 >= 2:
                    nxt = 2

            if nxt < zero:
                nxt = zero
            if nxt >= states:
                nxt = states - 1
            dst[y, x] = nxt


__all__ = ["step_omega"]
