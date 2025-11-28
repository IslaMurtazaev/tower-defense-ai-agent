# astar_controller.py
"""
A* Soldier Deployment Controller for Winterfell Defense
--------------------------------------------------------
This module is responsible for computing when soldiers
SHOULD be deployed (instead of blindly spawning).

It uses A* pathfinding to check if existing soldiers can
intercept incoming enemies. Only deploys new soldiers if
existing ones cannot reach enemies in time.

The main file can import:

    from astar_controller import should_deploy_soldiers

and call:

    if should_deploy_soldiers(
        enemy_positions,
        soldier_positions,
        BASE_POS,
        enemy_speed,
        soldier_speed,
    ):
        deploy_soldier()

This reduces soldier spam and economizes resources by only
deploying when necessary.
"""

import heapq
import math


# =========================================================
# Grid/A* Configuration
# =========================================================

GRID_SIZE = 40               # coarse grid cell size
INTERCEPT_TIME_MARGIN = 0.9  # soldier must arrive slightly before the enemy
MAX_ENEMY_DISTANCE = 1000    # ignore enemies beyond this distance from base


# =========================================================
# Internal helpers for grid-based A*
# =========================================================

def heuristic(a, b):
    """Euclidean distance heuristic"""
    return math.hypot(a[0]-b[0], a[1]-b[1])


def neighbors(node, grid_w, grid_h):
    """Yield neighboring nodes within the grid bounds."""
    x, y = node
    steps = [
        (1,0), (-1,0), (0,1), (0,-1),
        (1,1), (1,-1), (-1,1), (-1,-1)
    ]
    for dx, dy in steps:
        nx, ny = x+dx, y+dy
        if 0 <= nx < grid_w and 0 <= ny < grid_h:
            yield (nx, ny)


def astar(start, goal, grid_w, grid_h):
    """
    Standard A* on an empty walkable grid.
    We don’t need walls because paths are already enforced in the main engine.
    Here the grid is used only to estimate time-to-impact.
    """

    open_set = []
    heapq.heappush(open_set, (0, start))

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return g_score[current]

        for nb in neighbors(current, grid_w, grid_h):
            tentative = g_score[current] + heuristic(current, nb)
            if tentative < g_score.get(nb, float('inf')):
                g_score[nb] = tentative
                f_score[nb] = tentative + heuristic(nb, goal)
                heapq.heappush(open_set, (f_score[nb], nb))

    return float('inf')   # no path (should not happen)


# =========================================================
# Helper utilities
# =========================================================

def _grid_dimensions():
    """Return grid width/height based on battlefield size."""
    return 1280 // GRID_SIZE, 800 // GRID_SIZE


def _to_node(pos):
    """Convert a world position to grid coordinates."""
    return int(pos[0] // GRID_SIZE), int(pos[1] // GRID_SIZE)


def _node_in_bounds(node, grid_w, grid_h):
    """Check whether a node lies within the grid."""
    x, y = node
    return 0 <= x < grid_w and 0 <= y < grid_h


def _position_in_bounds(pos):
    """Ensure the entity is within the gameplay view."""
    x, y = pos
    return 0 <= x <= 1280 and 0 <= y <= 800


def _travel_time(start_node, end_node, grid_w, grid_h, speed):
    """Return the travel time between two nodes, or None if unreachable."""
    distance = astar(start_node, end_node, grid_w, grid_h)
    if distance == float('inf'):
        return None
    return distance * GRID_SIZE / speed


def _build_context(base_pos, enemy_speed, soldier_speed):
    """Create a reusable context dictionary with grid details."""
    grid_w, grid_h = _grid_dimensions()
    return {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "base_node": _to_node(base_pos),
        "base_pos": base_pos,
        "enemy_speed": enemy_speed,
        "soldier_speed": soldier_speed,
    }


def _can_soldier_intercept(soldier_pos, enemy_node, enemy_time, ctx):
    """Determine whether a soldier can intercept a specific enemy."""
    if not _position_in_bounds(soldier_pos):
        return False

    soldier_node = _to_node(soldier_pos)
    if not _node_in_bounds(soldier_node, ctx["grid_w"], ctx["grid_h"]):
        return False

    travel_time = _travel_time(
        soldier_node,
        enemy_node,
        ctx["grid_w"],
        ctx["grid_h"],
        ctx["soldier_speed"],
    )
    if travel_time is None:
        return False

    return travel_time < enemy_time * INTERCEPT_TIME_MARGIN


def _needs_reinforcement(enemy_pos, soldier_positions, ctx):
    """Return True if no existing soldier can intercept the given enemy."""
    if not _position_in_bounds(enemy_pos):
        return False

    if math.hypot(
        enemy_pos[0] - ctx["base_pos"][0],
        enemy_pos[1] - ctx["base_pos"][1],
    ) > MAX_ENEMY_DISTANCE:
        return False

    enemy_node = _to_node(enemy_pos)
    if not _node_in_bounds(enemy_node, ctx["grid_w"], ctx["grid_h"]):
        return False

    enemy_time = _travel_time(
        enemy_node,
        ctx["base_node"],
        ctx["grid_w"],
        ctx["grid_h"],
        ctx["enemy_speed"],
    )
    if enemy_time is None:
        return False

    if enemy_time < 2.0:
        return True

    for soldier_pos in soldier_positions:
        if _can_soldier_intercept(soldier_pos, enemy_node, enemy_time, ctx):
            return False

    return True


# =========================================================
# Public API: Should Winterfell deploy soldiers now?
# =========================================================

def should_deploy_soldiers(enemy_positions, soldier_positions, base_pos, enemy_speed, soldier_speed):
    """
    Determine whether new soldiers need to be deployed.

    Returns True only when existing soldiers cannot intercept visible enemies.
    """
    if not enemy_positions:
        return False

    if not soldier_positions:
        return True

    context = _build_context(base_pos, enemy_speed, soldier_speed)
    for enemy_pos in enemy_positions:
        if _needs_reinforcement(enemy_pos, soldier_positions, context):
            return True

    return False
