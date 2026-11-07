"""Numba-ядро автомата «симбиоз».

Правила (все пороги — оси конфига):

* **ANCHOR** рождается в пустоте, если в круге радиуса 20 уже есть ``nA``
  опорных клеток, в радиусе 2 есть ``mA`` зависимых GRAZER — тех, что без
  опоры не живут, — и нет опоры вплотную (иначе колония заливает поле
  плитой). **MATRIX** устроен так же, но с радиусом 30 и WEAVER.
* Опоры выживают, пока в их круге достаточно подобных и зависимых, но
  погибают от переполнения: сплошная плита рассыпается, остаются кольца.
* **GRAZER**/**WEAVER** рождаются только рядом с опорой (круг радиуса 5) и
  только если опора разложена по секторам круга согласно выбранному режиму:
  ``spread`` — минимум занятых секторов, ``ring`` — все секторы, ``gaps`` —
  есть хотя бы один пустой сектор, ``parity`` — маска чётных/нечётных долей.
  Им нужен ещё хотя бы один PULSE — сигнал, который испускают сами зависимые.
* Зависимая клетка без опоры становится **SPORE** и ждёт; спора просыпается
  в GRAZER/WEAVER, когда опора вернулась, и медленно распадается без неё.
* **PULSE** живёт один шаг и нужен как «метаболит»: без импульсов зависимые
  не размножаются, поэтому колония не может размножаться вечно.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ...kernels import JIT_KWARGS


@njit(inline="always")
def noise01(y, x, iteration, seed):
    """Детерминированное число [0, 1) из координат, итерации и seed."""

    mixed = (y * 73856093) ^ (x * 19349663) ^ (iteration * 83492791) ^ ((seed % 100000) * 2654435)
    mixed = (mixed * 1103515245 + 12345) & 0x7FFFFFFF
    mixed = (mixed * 1103515245 + 12345) & 0x7FFFFFFF
    return mixed / 2147483648.0


@njit(inline="always")
def sector_ok(counts, sectors, mode, need):
    """Проверить рисунок опоры по секторам круга радиуса 5."""

    nonempty = 0
    even = 0
    odd = 0
    for index in range(sectors):
        value = counts[index]
        if value > 0:
            nonempty += 1
        if index % 2 == 0:
            even += value
        else:
            odd += value
    if mode == 0:  # spread
        return nonempty >= need
    if mode == 1:  # ring
        return nonempty == sectors
    if mode == 2:  # gaps
        return 1 <= nonempty < sectors
    return even > 0 and odd > 0  # parity (маска чётных/нечётных секторов)


@njit(**JIT_KWARGS)
def step_symbiosis(
    src,
    dst,
    iteration,
    seed,
    sdy,
    sdx,
    in_anchor,
    in_matrix,
    ldy,
    ldx,
    ndy,
    ndx,
    sector,
    wrap,
    zero,
    n_anchor_birth,
    m_anchor_birth,
    n_anchor_surv,
    m_anchor_surv,
    n_anchor_max,
    n_matrix_birth,
    m_matrix_birth,
    n_matrix_surv,
    m_matrix_surv,
    n_matrix_max,
    grazer_birth,
    grazer_surv,
    grazer_max,
    weaver_birth,
    weaver_surv,
    weaver_max,
    sectors,
    sector_mode,
    sector_need,
    pulse_prob,
    spore_decay,
):
    """Одна итерация симбиоза: 7 состояний, крупная опора и секторы радиуса 5."""

    height, width = src.shape
    support_total = sdy.size
    logic_total = ldy.size
    spacing_total = ndy.size
    span = 8
    anchor_sectors = np.zeros(span, dtype=np.int32)
    matrix_sectors = np.zeros(span, dtype=np.int32)

    for y in range(height):
        for x in range(width):
            # --- крупный круг: опора и её зависимые -----------------------
            anchors = 0
            matrices = 0
            for k in range(support_total):
                yy = y + sdy[k]
                xx = x + sdx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                value = src[yy, xx]
                if in_anchor[k] and value == 1:
                    anchors += 1
                if in_matrix[k] and value == 2:
                    matrices += 1

            # --- запрет вплотную: опора рождается только в промежутках ----
            near_anchor = 0
            near_matrix = 0
            near_grazer = 0
            near_weaver = 0
            for k in range(spacing_total):
                yy = y + ndy[k]
                xx = x + ndx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                value = src[yy, xx]
                if value == 1:
                    near_anchor += 1
                elif value == 2:
                    near_matrix += 1
                elif value == 3:
                    near_grazer += 1
                elif value == 4:
                    near_weaver += 1

            # --- круг радиуса 5: зависимая логика и секторы ----------------
            for index in range(sectors):
                anchor_sectors[index] = 0
                matrix_sectors[index] = 0
            anchors_near = 0
            matrices_near = 0
            grazers = 0
            weavers = 0
            pulses = 0
            for k in range(logic_total):
                yy = y + ldy[k]
                xx = x + ldx[k]
                if wrap:
                    yy = yy % height
                    xx = xx % width
                elif yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                value = src[yy, xx]
                index = sector[k]
                if value == 1:
                    anchors_near += 1
                    anchor_sectors[index] += 1
                elif value == 2:
                    matrices_near += 1
                    matrix_sectors[index] += 1
                elif value == 3:
                    grazers += 1
                elif value == 4:
                    weavers += 1
                elif value == 6:
                    pulses += 1

            state = src[y, x]
            anchor_shape = sector_ok(anchor_sectors, sectors, sector_mode, sector_need)
            matrix_shape = sector_ok(matrix_sectors, sectors, sector_mode, sector_need)
            roll = noise01(y, x, iteration, seed)

            if state == zero:
                if anchors >= n_anchor_birth and near_grazer >= m_anchor_birth and near_anchor == 0:
                    dst[y, x] = 1
                elif matrices >= n_matrix_birth and near_weaver >= m_matrix_birth and near_matrix == 0:
                    dst[y, x] = 2
                elif (
                    anchors_near >= grazer_birth
                    and pulses >= 1
                    and anchor_shape
                ):
                    dst[y, x] = 3
                elif (
                    matrices_near >= weaver_birth
                    and pulses >= 1
                    and matrix_shape
                ):
                    dst[y, x] = 4
                elif (grazers + weavers) >= 1 and (anchors_near + matrices_near) >= 1 and roll < pulse_prob:
                    dst[y, x] = 6
                else:
                    dst[y, x] = zero
            elif state == 1:
                if (
                    anchors >= n_anchor_surv
                    and near_grazer >= m_anchor_surv
                    and anchors <= n_anchor_max
                ):
                    dst[y, x] = 1
                else:
                    dst[y, x] = zero
            elif state == 2:
                if (
                    matrices >= n_matrix_surv
                    and near_weaver >= m_matrix_surv
                    and matrices <= n_matrix_max
                ):
                    dst[y, x] = 2
                else:
                    dst[y, x] = zero
            elif state == 3:
                if grazers > grazer_max:
                    dst[y, x] = zero
                elif anchors_near >= grazer_surv and anchors_near > 0 and anchor_shape:
                    dst[y, x] = 3
                else:
                    dst[y, x] = 5
            elif state == 4:
                if weavers > weaver_max:
                    dst[y, x] = zero
                elif matrices_near >= weaver_surv and matrices_near > 0 and matrix_shape:
                    dst[y, x] = 4
                else:
                    dst[y, x] = 5
            elif state == 5:
                if anchors_near >= 1:
                    dst[y, x] = 3
                elif matrices_near >= 1:
                    dst[y, x] = 4
                elif roll < spore_decay:
                    dst[y, x] = zero
                else:
                    dst[y, x] = 5
            else:  # PULSE живёт один шаг
                dst[y, x] = zero


__all__ = ["noise01", "sector_ok", "step_symbiosis"]
