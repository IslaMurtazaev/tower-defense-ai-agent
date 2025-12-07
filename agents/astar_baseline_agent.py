"""
A* Baseline Agent for Tower Defense

Uses A* pathfinding and heuristic analysis to make placement decisions.
This serves as a baseline to compare against RL algorithms.
"""
import numpy as np
import math
from typing import Dict

from environment.astar_controller import should_deploy_soldiers
from environment import td_pyastar as game


class AStarBaselineAgent:
    """Agent that uses A* pathfinding heuristics for soldier placement."""

    def __init__(self, observation_space, action_space):
        self.obs_space = observation_space
        self.action_space = action_space
        self.grid_size = int(action_space.nvec[0])

    def select_action(self, obs: Dict) -> np.ndarray:
        """Select placement position using A* heuristics."""
        grid = obs["grid"]

        # Get current game state
        soldiers_remaining = int(obs["soldiers_remaining"][0])

        # If no soldiers to place, return center (will be ignored)
        if soldiers_remaining <= 0:
            return np.array([self.grid_size // 2, self.grid_size // 2])

        # Extract enemy and soldier positions from grid
        enemy_positions = []
        soldier_positions = []

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_val = grid[y, x]
                world_x = (x / self.grid_size) * game.SCREEN_W
                world_y = (y / self.grid_size) * game.SCREEN_H

                if cell_val == 3:  # Enemy/wight
                    enemy_positions.append((world_x, world_y))
                elif cell_val == 2:  # Soldier
                    soldier_positions.append((world_x, world_y))

        # Use A* controller to decide if we should deploy
        should_deploy = should_deploy_soldiers(
            enemy_positions,
            soldier_positions,
            game.BASE_POS,
            game.ENEMY_SPEED,
            game.SOLDIER_SPEED
        )

        # Always find optimal placement (A* controller just tells us if needed, but we still place)
        best_pos = self._find_optimal_placement(grid, enemy_positions)
        return best_pos

    def _find_optimal_placement(self, grid: np.ndarray, enemy_positions: list) -> np.ndarray:
        """Find optimal placement using A* pathfinding analysis."""
        if not enemy_positions:
            # No enemies: place in defensive ring around base
            return self._place_defensive_ring(grid)

        # Find choke points: positions where multiple enemy paths converge
        choke_points = self._find_choke_points(enemy_positions)

        if choke_points:
            # Place at best choke point
            best_choke = max(choke_points, key=lambda p: p[1])  # Highest score
            grid_x = int((best_choke[0][0] / SCREEN_W) * self.grid_size)
            grid_y = int((best_choke[0][1] / SCREEN_H) * self.grid_size)
            grid_x = np.clip(grid_x, 0, self.grid_size - 1)
            grid_y = np.clip(grid_y, 0, self.grid_size - 1)
            return np.array([grid_x, grid_y])

        # Fallback: place at intercept point for nearest enemy
        return self._place_intercept_position(grid, enemy_positions[0])

    def _find_choke_points(self, enemy_positions: list) -> list:
        """Identify choke points where enemy paths converge."""
        choke_points = []

        # Create grid of path convergence
        convergence_grid = np.zeros((self.grid_size, self.grid_size))

        for enemy_pos in enemy_positions:
            # Find path from enemy to base using A* heuristic
            path = self._estimate_path_to_base(enemy_pos)

            # Mark cells along path
            for point in path:
                grid_x = int((point[0] / SCREEN_W) * self.grid_size)
                grid_y = int((point[1] / SCREEN_H) * self.grid_size)
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    convergence_grid[grid_y, grid_x] += 1

        # Find cells with high convergence (multiple paths pass through)
        threshold = max(2, len(enemy_positions) // 2)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if convergence_grid[y, x] >= threshold:
                    world_x = (x / self.grid_size) * SCREEN_W
                    world_y = (y / self.grid_size) * SCREEN_H
                    score = convergence_grid[y, x]
                    choke_points.append(((world_x, world_y), score))

        return choke_points

    def _estimate_path_to_base(self, start_pos: tuple) -> list:
        """Estimate path from position to base (simplified A* heuristic)."""
        path = [start_pos]
        current = start_pos

        # Simple path: move towards base in steps
        steps = 10
        for _ in range(steps):
            dx = game.BASE_POS[0] - current[0]
            dy = game.BASE_POS[1] - current[1]
            dist = math.hypot(dx, dy)

            if dist < 50:  # Close enough to base
                break

            # Move towards base
            step_size = min(50, dist / steps)
            current = (
                current[0] + (dx / dist) * step_size,
                current[1] + (dy / dist) * step_size
            )
            path.append(current)

        return path

    def _place_intercept_position(self, grid: np.ndarray, enemy_pos: tuple) -> np.ndarray:
        """Place soldier at position to intercept enemy."""
        # Place between enemy and base
        mid_x = (enemy_pos[0] + game.BASE_POS[0]) / 2
        mid_y = (enemy_pos[1] + game.BASE_POS[1]) / 2

        # Ensure minimum distance from base
        dist_to_base = math.hypot(mid_x - game.BASE_POS[0], mid_y - game.BASE_POS[1])
        if dist_to_base < game.BASE_RADIUS + 100:
            angle = math.atan2(mid_y - game.BASE_POS[1], mid_x - game.BASE_POS[0])
            mid_x = game.BASE_POS[0] + math.cos(angle) * (game.BASE_RADIUS + 100)
            mid_y = game.BASE_POS[1] + math.sin(angle) * (game.BASE_RADIUS + 100)

        grid_x = int((mid_x / game.SCREEN_W) * self.grid_size)
        grid_y = int((mid_y / game.SCREEN_H) * self.grid_size)
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)

        return np.array([grid_x, grid_y])

    def _place_defensive_ring(self, grid: np.ndarray) -> np.ndarray:
        """Place soldier in defensive ring around base."""
        # Count existing soldiers
        soldier_count = np.sum(grid == 2)

        # Place at angle based on soldier count (even distribution)
        angle = (soldier_count * 2 * math.pi / 4) % (2 * math.pi)
        radius = game.BASE_RADIUS + 120

        world_x = game.BASE_POS[0] + math.cos(angle) * radius
        world_y = game.BASE_POS[1] + math.sin(angle) * radius

        grid_x = int((world_x / game.SCREEN_W) * self.grid_size)
        grid_y = int((world_y / game.SCREEN_H) * self.grid_size)
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)

        return np.array([grid_x, grid_y])

    def _find_coverage_position(self, grid: np.ndarray, soldier_positions: list) -> np.ndarray:
        """Find position that maximizes coverage with existing soldiers."""
        if not soldier_positions:
            return self._place_defensive_ring(grid)

        # Find area with least soldier coverage
        coverage_grid = np.zeros((self.grid_size, self.grid_size))

        for soldier_pos in soldier_positions:
            grid_x = int((soldier_pos[0] / SCREEN_W) * self.grid_size)
            grid_y = int((soldier_pos[1] / SCREEN_H) * self.grid_size)

            # Mark coverage area (soldier attack range)
            coverage_radius = 3  # grid cells
            for dy in range(-coverage_radius, coverage_radius + 1):
                for dx in range(-coverage_radius, coverage_radius + 1):
                    if dx*dx + dy*dy <= coverage_radius * coverage_radius:
                        cy = grid_y + dy
                        cx = grid_x + dx
                        if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                            coverage_grid[cy, cx] += 1

        # Find position with minimum coverage (but not too close to base)
        min_coverage = float('inf')
        best_pos = None

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                world_x = (x / self.grid_size) * game.SCREEN_W
                world_y = (y / self.grid_size) * game.SCREEN_H
                dist_to_base = math.hypot(world_x - game.BASE_POS[0], world_y - game.BASE_POS[1])

                if dist_to_base < game.BASE_RADIUS + 80:
                    continue

                if coverage_grid[y, x] < min_coverage:
                    min_coverage = coverage_grid[y, x]
                    best_pos = np.array([x, y])

        return best_pos if best_pos is not None else np.array([self.grid_size // 2, self.grid_size // 2])
