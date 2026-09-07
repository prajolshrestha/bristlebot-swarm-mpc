#!/usr/bin/env python3
"""Shared arena and density settings."""

from __future__ import annotations

import math

ARENA_SIDE_M: float = 0.8
ARENA_AREA_M2: float = ARENA_SIDE_M ** 2
ARENA_AREA_CM2: float = ARENA_AREA_M2 * 1e4

ROBOT_A_M: float = 0.03
ROBOT_B_M: float = 0.015
ROBOT_A_CM: float = ROBOT_A_M * 100.0
ROBOT_B_CM: float = ROBOT_B_M * 100.0
A_BOT_CM2: float = math.pi * ROBOT_A_CM * ROBOT_B_CM

D_SAFE_M: float = 0.07
D_SAFE_CM: float = D_SAFE_M * 100.0
D_SAFE_OBS_M: float = D_SAFE_M

D_SAFE_TABLE_CM: tuple[float, ...] = (7.0, 9.0)

MAX_NEIGHBOURS: int = 5
COMM_RADIUS_M: float = 0.15

GEOMETRIC_N_MAX: dict[float, int] = {
    6.0: 188,
    7.0: 137,
    8.0: 105,
    9.0: 80,
    10.0: 68,
    11.0: 52,
    12.0: 42,
}

MOVEMENT_HEADROOM_ROBOTS: int = 30


def buffer_area_cm2(d_safe_cm: float) -> float:
    """Exclusion-zone area [cm²] for centre-to-centre separation d_safe_cm."""
    return math.pi * (d_safe_cm / 2.0) ** 2


def occupancy_pct(total_area_cm2: float) -> float:
    return 100.0 * total_area_cm2 / ARENA_AREA_CM2


def n_max_at_d_safe(d_safe_cm: float) -> int:
    return int(ARENA_AREA_CM2 // buffer_area_cm2(d_safe_cm))


def geometric_n_max(d_safe_cm: float | None = None) -> int:
    """Max geometrically packable swarm size at D_safe [cm]."""
    d = D_SAFE_CM if d_safe_cm is None else d_safe_cm
    d_key = round(d, 1)
    if d_key in GEOMETRIC_N_MAX:
        return GEOMETRIC_N_MAX[d_key]
    return n_max_at_d_safe(d)


def benchmark_n_list(
    d_safe_cm: float | None = None,
    headroom: int | None = None,
) -> list[int]:
    """
    Four-point benchmark sweep: low / mid / high density + operating max.

    Operating max = geometric N_max − headroom (room to move, lower collisions).
    If operating max < 50, use it in place of 50 in the third slot.
    """
    n_geo = geometric_n_max(d_safe_cm)
    room = MOVEMENT_HEADROOM_ROBOTS if headroom is None else headroom
    n_operating = max(1, n_geo - room)
    if n_operating >= 50:
        return [10, 25, 50, n_operating-7]
    return [10, 25, n_operating, n_operating]


def dense_n_list(d_safe_cm: float | None = None) -> list[int]:
    """Every swarm size from 1 through geometric N_max (collision-surface sweeps)."""
    return list(range(1, geometric_n_max(d_safe_cm) + 1))


N_LIST: list[int] = [10, 25, 50, 75, 100, 125, 137]

HIGHLIGHT_SWARM_SIZES: frozenset[int] = frozenset({50, N_LIST[-1]})


def swarm_row(n: int, d_safe_cm: float) -> dict:
    a_bot_total = n * A_BOT_CM2
    a_buf = buffer_area_cm2(d_safe_cm)
    a_buf_total = n * a_buf
    return {
        "n": n,
        "a_bot_total": a_bot_total,
        "occ_bot_pct": occupancy_pct(a_bot_total),
        "a_buf_total": a_buf_total,
        "occ_buf_pct": occupancy_pct(a_buf_total),
        "d_safe_cm": d_safe_cm,
    }


def default_swarm_sizes() -> list[int]:
    """Swarm sizes for the occupancy table (includes N_max rows)."""
    sizes = {1, 10, 25, 50, 75, 100, 166, 200, 300, 400}
    for d in D_SAFE_TABLE_CM:
        sizes.add(n_max_at_d_safe(d))
    return sorted(sizes)
