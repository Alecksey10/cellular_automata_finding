"""Numba-ядра правил.

Ядро параметризовано данными (смещения окрестности, пороги, состояния), поэтому
компилируется один раз на всю серию конфигов. ``nogil=True`` позволяет запускать
тесты потоками, ``cache=True`` переиспользует машинный код между запусками.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def threshold_step(
    src,
    dst,
    di,
    dj,
    born_above,
    death_below,
    death_above,
    zero_state,
    birth_state,
    last_state,
    wrap,
):
    """Пороговое правило из постановки: рождение/смерть по числу соседей.

    Правило повторяет псевдокод буквально, включая ветку ``else``: в ней
    увеличивается состояние клетки (в том числе мёртвой), пока не достигнет
    ``last_state``. Если нужна классическая семантика «стареют только живые»,
    добавьте отдельный модуль правила — ядро менять не нужно.
    """

    height, width = src.shape
    count_cells = di.shape[0]
    for i in range(height):
        for j in range(width):
            neighbours = 0
            for k in range(count_cells):
                ni = i + di[k]
                nj = j + dj[k]
                if wrap:
                    ni = ni % height
                    nj = nj % width
                else:
                    if ni < 0:
                        ni = 0
                    elif ni >= height:
                        ni = height - 1
                    if nj < 0:
                        nj = 0
                    elif nj >= width:
                        nj = width - 1
                if src[ni, nj] != zero_state:
                    neighbours += 1
            previous = src[i, j]
            if neighbours > death_above or neighbours < death_below:
                dst[i, j] = zero_state
            elif neighbours > born_above:
                dst[i, j] = birth_state
            else:
                aged = previous + 1
                dst[i, j] = last_state if aged > last_state else aged


@njit(cache=True, nogil=True)
def aging_step(
    src,
    dst,
    inner_di,
    inner_dj,
    outer_di,
    outer_dj,
    born_inner,
    born_total,
    crowd_limit,
    fortify,
    max_age,
    zero_state,
    wrap,
):
    """Возраст клетки = сила: старые клетки переносят более широкую полосу соседей."""

    height, width = src.shape
    inner_count = inner_di.shape[0]
    outer_count = outer_di.shape[0]
    for i in range(height):
        for j in range(width):
            inner = 0
            for k in range(inner_count):
                ni = i + inner_di[k]
                nj = j + inner_dj[k]
                if wrap:
                    ni = ni % height
                    nj = nj % width
                else:
                    if ni < 0:
                        ni = 0
                    elif ni >= height:
                        ni = height - 1
                    if nj < 0:
                        nj = 0
                    elif nj >= width:
                        nj = width - 1
                if src[ni, nj] != zero_state:
                    inner += 1

            outer = 0
            for k in range(outer_count):
                ni = i + outer_di[k]
                nj = j + outer_dj[k]
                if wrap:
                    ni = ni % height
                    nj = nj % width
                else:
                    if ni < 0:
                        ni = 0
                    elif ni >= height:
                        ni = height - 1
                    if nj < 0:
                        nj = 0
                    elif nj >= width:
                        nj = width - 1
                if src[ni, nj] != zero_state:
                    outer += 1

            total = inner + outer
            age = src[i, j]
            if age == 0:
                if inner >= born_inner and total >= born_total:
                    dst[i, j] = 1
                else:
                    dst[i, j] = 0
            else:
                strength = fortify * (age - 1)
                need = born_total - strength
                if need < 0:
                    need = 0
                limit = crowd_limit + strength
                if total >= need and total <= limit:
                    aged = age + 1
                    dst[i, j] = max_age if aged > max_age else aged
                else:
                    # Старая клетка не умирает сразу: она теряет один уровень
                    # возраста и погибает только с последнего. Это и есть
                    # «крепче»: возраст — буфер прочности.
                    dst[i, j] = age - 1


@njit(cache=True, nogil=True)
def life_step(src, dst, di, dj, zero_state, birth_state, wrap):
    """Классическая игра «Жизнь» (B3/S23) для двух состояний."""

    height, width = src.shape
    count_cells = di.shape[0]
    for i in range(height):
        for j in range(width):
            neighbours = 0
            for k in range(count_cells):
                ni = i + di[k]
                nj = j + dj[k]
                if wrap:
                    ni = ni % height
                    nj = nj % width
                else:
                    if ni < 0:
                        ni = 0
                    elif ni >= height:
                        ni = height - 1
                    if nj < 0:
                        nj = 0
                    elif nj >= width:
                        nj = width - 1
                if src[ni, nj] != zero_state:
                    neighbours += 1
            alive = src[i, j] != zero_state
            if alive:
                dst[i, j] = birth_state if neighbours == 2 or neighbours == 3 else zero_state
            else:
                dst[i, j] = birth_state if neighbours == 3 else zero_state


@njit(cache=True, inline="always")
def _neighbour(src, i, j, di, dj, k, wrap):
    """Сосед с учётом границы; возвращает значение клетки."""

    height, width = src.shape
    ni = i + di[k]
    nj = j + dj[k]
    if wrap:
        ni = ni % height
        nj = nj % width
    else:
        if ni < 0:
            ni = 0
        elif ni >= height:
            ni = height - 1
        if nj < 0:
            nj = 0
        elif nj >= width:
            nj = width - 1
    return src[ni, nj]


@njit(cache=True, nogil=True)
def totalistic_step(
    src,
    dst,
    di,
    dj,
    born_lo,
    born_hi,
    survive_lo,
    survive_hi,
    zero_state,
    birth_state,
    wrap,
):
    """Тоталистическое правило B/S: решение зависит только от числа соседей.

    ``born_lo..born_hi`` — число соседей, при котором рождается мёртвая клетка;
    ``survive_lo..survive_hi`` — при котором выживает живая. Правило инвариантно
    относительно любой перестановки соседей, поэтому автоматически симметрично
    относительно поворотов и отражений (для окрестности, которая сама
    симметрична).
    """

    height, width = src.shape
    count_cells = di.shape[0]
    for i in range(height):
        for j in range(width):
            neighbours = 0
            for k in range(count_cells):
                if _neighbour(src, i, j, di, dj, k, wrap) != zero_state:
                    neighbours += 1
            if src[i, j] != zero_state:
                if survive_lo <= neighbours <= survive_hi:
                    dst[i, j] = src[i, j]
                else:
                    dst[i, j] = zero_state
            else:
                if born_lo <= neighbours <= born_hi:
                    dst[i, j] = birth_state
                else:
                    dst[i, j] = zero_state


@njit(cache=True, nogil=True)
def vote_step(src, dst, di, dj, quorum, states, zero_state, wrap):
    """Симметричное голосование: клетка становится самым частым состоянием соседей.

    Голос учитывает и пустое состояние, поэтому плотные области «схлопываются»
    в домены, а не заливаются. Ничья решается в пользу текущего состояния.
    Голосование не зависит от порядка соседей — это и есть требование симметрии.
    """

    height, width = src.shape
    count_cells = di.shape[0]
    counts = np.zeros(states, dtype=np.int32)
    for i in range(height):
        for j in range(width):
            for s in range(states):
                counts[s] = 0
            for k in range(count_cells):
                counts[_neighbour(src, i, j, di, dj, k, wrap)] += 1
            best = src[i, j]
            best_count = counts[best]
            for s in range(states):
                if counts[s] > best_count:
                    best = s
                    best_count = counts[s]
            if best_count >= quorum:
                dst[i, j] = best
            else:
                dst[i, j] = src[i, j]


@njit(cache=True, nogil=True)
def parity_step(src, dst, di, dj, zero_state, birth_state, wrap):
    """Чётность: клетка жива, если число живых соседей нечётно.

    Крайне симметричное и крайне нестабильное правило — полезно как контрольный
    пример для фильтров: почти всегда отбраковывается как мелькание.
    """

    height, width = src.shape
    count_cells = di.shape[0]
    for i in range(height):
        for j in range(width):
            neighbours = 0
            for k in range(count_cells):
                if _neighbour(src, i, j, di, dj, k, wrap) != zero_state:
                    neighbours += 1
            dst[i, j] = birth_state if neighbours % 2 == 1 else zero_state


@njit(cache=True, nogil=True)
def cyclic_step(src, dst, di, dj, threshold, states, zero_state, always_die, wrap):
    """Циклическое правило: клетка «твердеет», затем погибает.

    Состояния ``0..states-1`` — фазы цикла. Клетка переходит в следующую фазу
    (``states-1 -> 0`` — смерть), если у неё не меньше ``threshold`` соседей уже
    в этой следующей фазе. Так волна фазы бежит по сетке.

    ``always_die``: самая твёрдая фаза погибает безусловно, а не ждёт пустых
    соседей. Это даёт затухающие волны вместо вечного цикла.
    """

    height, width = src.shape
    count_cells = di.shape[0]
    last = states - 1
    for i in range(height):
        for j in range(width):
            current = src[i, j]
            following = current + 1
            if following > last:
                following = zero_state
            if current == last and always_die:
                dst[i, j] = zero_state
                continue
            matches = 0
            for k in range(count_cells):
                if _neighbour(src, i, j, di, dj, k, wrap) == following:
                    matches += 1
            if matches >= threshold:
                dst[i, j] = following
            else:
                dst[i, j] = current


@njit(cache=True, nogil=True)
def rot_aging_step(
    src,
    dst,
    noise,
    inner_di,
    inner_dj,
    outer_di,
    outer_dj,
    born_inner,
    born_total,
    survive_lo,
    survive_hi,
    last_state,
    rot,
    zero_state,
    wrap,
):
    """Старение с частичным гниением: выжила — твёрже, не выжила — на шаг вниз.

    * мёртвая клетка рождается в состояние ``1``, если внутреннее кольцо даёт
      не меньше ``born_inner`` соседей, а сумма колец — ``born_total``;
    * живая клетка взрослеет на единицу, пока не достигнет ``last_state``,
      если сумма колец лежит в ``[survive_lo, survive_hi]``;
    * иначе она теряет один уровень и умирает только с первого;
    * клетка, уже стоящая в ``last_state``, с вероятностью ``rot`` превращается
      в ``zero_state`` — поэтому старые облака не исчезают сразу, а гниют
      по клеткам, и доля мёртвых меняется плавно.

    ``noise`` — массив тех же размеров с равномерными числами ``[0, 1)``;
    его заполняет рантайм правила перед вызовом, чтобы случайность была
    воспроизводимой и не зависела от глобального состояния numba.
    """

    height, width = src.shape
    inner_count = inner_di.shape[0]
    outer_count = outer_di.shape[0]
    for i in range(height):
        for j in range(width):
            inner = 0
            for k in range(inner_count):
                if _neighbour(src, i, j, inner_di, inner_dj, k, wrap) != zero_state:
                    inner += 1

            outer = 0
            for k in range(outer_count):
                if _neighbour(src, i, j, outer_di, outer_dj, k, wrap) != zero_state:
                    outer += 1

            total = inner + outer
            age = src[i, j]
            if age == zero_state:
                if inner >= born_inner and total >= born_total:
                    dst[i, j] = 1
                else:
                    dst[i, j] = zero_state
            elif survive_lo <= total <= survive_hi:
                aged = age + 1
                if aged > last_state:
                    aged = last_state
                if age == last_state and noise[i, j] < rot:
                    dst[i, j] = zero_state
                else:
                    dst[i, j] = aged
            else:
                dst[i, j] = age - 1


@njit(cache=True, nogil=True)
def refractory_step(src, dst, di, dj, threshold, last_state, zero_state, wrap):
    """Возбудимая среда с рефрактерностью (по мотивам Greenberg–Hastings).

    * ``0`` — возбудимая клетка: возбуждается, если рядом не меньше ``threshold``
      возбуждённых соседей (``1``);
    * ``1`` — возбуждённая: сразу переходит в рефрактерную фазу ``2``;
    * ``2..last_state-1`` — отвердевает: каждая итерация повышает фазу;
    * ``last_state`` — последняя рефрактерная фаза, затем клетка «умирает» в ``0``.
    """

    height, width = src.shape
    count_cells = di.shape[0]
    for i in range(height):
        for j in range(width):
            current = src[i, j]
            if current == zero_state:
                excited = 0
                for k in range(count_cells):
                    if _neighbour(src, i, j, di, dj, k, wrap) == 1:
                        excited += 1
                dst[i, j] = 1 if excited >= threshold else zero_state
            elif current == 1:
                dst[i, j] = 2
            elif current < last_state:
                dst[i, j] = current + 1
            else:
                dst[i, j] = zero_state
