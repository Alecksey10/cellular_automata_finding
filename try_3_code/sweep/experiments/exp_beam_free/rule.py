"""Правило ``expBeamFree``: дуэль двух групп в двух кольцах."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...experiment import Experiment
from ...interfaces import CapabilityKind, Requirement
from ...kernels import JIT_KWARGS
from ...module import Role
from ...neighbors import DuelRings
from numba import njit

MECHANISMS = [
    "majority_count",
    "power_dominance",
    "level_dominance",
    "mutual_annihilation",
    "asymmetric_suppression",
    "conversion",
    "reinforcement",
    "exclusion",
    "ring_priority",
    "escalation",
]

M_MAJORITY = 0
M_POWER = 1
M_LEVEL = 2
M_ANNIHIL = 3
M_ASYMM = 4
M_CONVERT = 5
M_REINF = 6
M_EXCLUDE = 7
M_RING = 8
M_ESCAL = 9


def _group(state: int, split: int) -> int:
    if state <= 0:
        return 0
    if state <= split:
        return 1
    return 2


@njit(**JIT_KWARGS)
def step_duel(
    src,
    dst,
    dy,
    dx,
    inner_dy,
    inner_dx,
    wrap,
    zero,
    states,
    split,
    mechanism,
    birth_a,
    birth_b,
    surv_a_lo,
    surv_a_hi,
    surv_b_lo,
    surv_b_hi,
    suppress_a,
    suppress_b,
    convert_a,
    convert_b,
    boost_a,
    boost_b,
    block_a,
    block_b,
):
    height, width = src.shape
    total = dy.size
    inner_total = inner_dy.size
    b_levels = states - 1 - split
    if b_levels < 1:
        b_levels = 1

    for y in range(height):
        for x in range(width):
            a_in = 0
            b_in = 0
            a_out = 0
            b_out = 0
            a_power_in = 0
            b_power_in = 0
            a_power_out = 0
            b_power_out = 0
            a_max = 0
            b_max = 0

            for k in range(total):
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
                inner = abs(dy[k]) <= 1 and abs(dx[k]) <= 1 and not (dy[k] == 0 and dx[k] == 0)
                if value <= split:
                    if inner:
                        a_in += 1
                        a_power_in += value
                    else:
                        a_out += 1
                        a_power_out += value
                    if value > a_max:
                        a_max = value
                else:
                    if inner:
                        b_in += 1
                        b_power_in += value - split
                    else:
                        b_out += 1
                        b_power_out += value - split
                    if value - split > b_max:
                        b_max = value - split

            a_total = a_in + a_out
            b_total = b_in + b_out
            a_power_total = a_power_in + a_power_out
            b_power_total = b_power_in + b_power_out

            state = src[y, x]
            group = 0
            if state > zero:
                group = 1 if state <= split else 2

            nxt = zero

            if group == 1:
                if b_in >= suppress_a:
                    if convert_a and b_power_in >= a_power_in:
                        level = b_in
                        if level < 1:
                            level = 1
                        if level > b_levels:
                            level = b_levels
                        nxt = split + level
                    else:
                        nxt = zero
                elif (a_out >= surv_a_lo and a_out <= surv_a_hi) or (boost_a > 0 and a_in >= boost_a):
                    nxt = state
                    if boost_a > 0 and a_in >= boost_a and state < split:
                        nxt = state + 1
                else:
                    nxt = zero

            elif group == 2:
                level = state - split
                if a_in >= suppress_b:
                    if convert_b and a_power_in >= b_power_in:
                        lvl = a_in
                        if lvl < 1:
                            lvl = 1
                        if lvl > split:
                            lvl = split
                        nxt = lvl
                    else:
                        nxt = zero
                elif (b_out >= surv_b_lo and b_out <= surv_b_hi) or (boost_b > 0 and b_in >= boost_b):
                    nxt = state
                    if boost_b > 0 and b_in >= boost_b and state < states - 1:
                        nxt = state + 1
                else:
                    nxt = zero

            else:
                birthA = a_in >= birth_a and b_in < block_a
                birthB = b_in >= birth_b and a_in < block_b

                if birthA and not birthB:
                    nxt = 1
                elif birthB and not birthA:
                    nxt = split + 1
                elif birthA and birthB:
                    if mechanism == M_MAJORITY:
                        nxt = 1 if a_total >= b_total else split + 1
                    elif mechanism == M_POWER:
                        nxt = 1 if a_power_total >= b_power_total else split + 1
                    elif mechanism == M_LEVEL:
                        nxt = 1 if a_max >= b_max else split + 1
                    elif mechanism == M_ANNIHIL:
                        if a_total >= suppress_a and b_total >= suppress_b:
                            nxt = zero
                        else:
                            nxt = 1 if a_total >= b_total else split + 1
                    elif mechanism == M_ASYMM:
                        nxt = 1 if a_in >= b_in else split + 1
                    elif mechanism == M_CONVERT:
                        nxt = 1 if a_power_total >= b_power_total else split + 1
                    elif mechanism == M_REINF:
                        nxt = 1 if a_in >= b_in else split + 1
                    elif mechanism == M_EXCLUDE:
                        nxt = 1 if a_in >= b_in else split + 1
                    elif mechanism == M_RING:
                        nxt = 1 if a_in >= b_out else split + 1
                    elif mechanism == M_ESCAL:
                        if a_total >= b_total:
                            nxt = 1
                        else:
                            nxt = split + 1
                    else:
                        nxt = 1 if a_total >= b_total else split + 1
                else:
                    nxt = zero

            # подавление уже существующих клеток механизмами
            if group == 1 and mechanism == M_ANNIHIL and b_in >= suppress_a and a_in >= suppress_b:
                nxt = zero
            if group == 2 and mechanism == M_ANNIHIL and a_in >= suppress_b and b_in >= suppress_a:
                nxt = zero
            if group == 1 and mechanism == M_EXCLUDE and b_in >= block_a:
                nxt = zero
            if group == 2 and mechanism == M_EXCLUDE and a_in >= block_b:
                nxt = zero

            if nxt < zero:
                nxt = zero
            if nxt >= states:
                nxt = states - 1
            dst[y, x] = nxt


def declare_rule(ns, view) -> None:
    states = view.peer(Role.DATA).derived("states")
    inner_max = view.peer(Role.NEIGHBORHOOD).derived("inner_max_count")
    outer_max = view.peer(Role.NEIGHBORHOOD).derived("max_count")
    ns.range("split", 5, states - 1)
    ns.choice("mechanism", MECHANISMS)
    ns.range("birth_a", 1, inner_max + 1)
    ns.range("birth_b", 1, inner_max + 1)
    ns.range("surv_a_lo", 0, outer_max + 1)
    ns.range("surv_a_hi", ns.handle("surv_a_lo"), outer_max + 1)
    ns.range("surv_b_lo", 0, outer_max + 1)
    ns.range("surv_b_hi", ns.handle("surv_b_lo"), outer_max + 1)
    ns.range("suppress_a", 0, inner_max + 1)
    ns.range("suppress_b", 0, inner_max + 1)
    ns.choice("convert_a", [0, 1])
    ns.choice("convert_b", [0, 1])
    ns.range("boost_a", 0, inner_max + 1)
    ns.range("boost_b", 0, inner_max + 1)
    ns.range("block_a", 0, inner_max + 1)
    ns.range("block_b", 0, inner_max + 1)


def bind_rule(fixed: Mapping[str, Any], view) -> tuple:
    spec = view.neighborhood()
    mechanism = MECHANISMS.index(str(fixed["rule.duel.mechanism"]))
    return (
        spec.dy,
        spec.dx,
        spec.inner_dy,
        spec.inner_dx,
        spec.wrap_flag,
        int(fixed["grid2d.zero"]),
        int(fixed["grid2d.states"]),
        int(fixed["rule.duel.split"]),
        mechanism,
        int(fixed["rule.duel.birth_a"]),
        int(fixed["rule.duel.birth_b"]),
        int(fixed["rule.duel.surv_a_lo"]),
        int(fixed["rule.duel.surv_a_hi"]),
        int(fixed["rule.duel.surv_b_lo"]),
        int(fixed["rule.duel.surv_b_hi"]),
        int(fixed["rule.duel.suppress_a"]),
        int(fixed["rule.duel.suppress_b"]),
        int(fixed["rule.duel.convert_a"]),
        int(fixed["rule.duel.convert_b"]),
        int(fixed["rule.duel.boost_a"]),
        int(fixed["rule.duel.boost_b"]),
        int(fixed["rule.duel.block_a"]),
        int(fixed["rule.duel.block_b"]),
    )


__all__ = ["MECHANISMS", "bind_rule", "declare_rule", "step_duel"]
