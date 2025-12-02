"""
Gymnasium environment wrapper for the Tower Defense game.

Converts the game into a reinforcement learning environment compatible with
Gymnasium. Handles observations, actions, rewards, and game state management.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import pygame
import math
import random
from collections import deque

from . import td_pyastar as game

# Game constants (from td_pyastar.py)
SCREEN_W, SCREEN_H = game.SCREEN_W, game.SCREEN_H
BASE_POS = game.BASE_POS
BASE_RADIUS = game.BASE_RADIUS
BASE_HP_MAX = game.BASE_HP_MAX
SOLDIER_MAX = game.SOLDIER_MAX
GRID_SIZE = 32  # For observation grid


class HeadlessWarriorGame:
    """Game state manager for headless RL training without Pygame rendering."""

    def __init__(self):
        self.max_runtime = 180.0
        try:
            self.sound_engine = game.SoundEngine()
        except Exception:
            self.sound_engine = type("SE", (), {
                "sfx_enabled": False,
                "sounds": {},
                "ambience": None,
                "ambience_playing": False,
                "toggle_ambience": lambda self=None: None,
                "set_sfx": lambda self, on=None: None,
                "play_sound": lambda self, key, volume=0.7: None
            })()
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        game.dynamic_obstacles.clear()

        self.base_hp = BASE_HP_MAX
        self.soldiers: List[game.Soldier] = []
        self.enemies: deque = deque()
        self.nks: List[game.NightKing] = []
        self.heroes: List = []
        self.burns: List[game.BurningEffect] = []
        self.stats = {
            'wights_killed': 0,
            'soldiers_killed': 0,
            'nk_kills': 0,
            'soldiers_deployed': 0,
            'base_hits': 0,
        }
        self.runtime = 0.0
        self.spawn_timer = 0.0
        self.current_wave = 0
        self.placement_phase = True
        self.combat_phase = False
        self.game_over = False
        self.victory = False

        # Night King scheduling
        self.nk_schedule = []
        for idx, t in enumerate(game.NK_SCHEDULE_TIMES):
            path_key = random.randrange(game.NUM_PATHS)
            spawn_frac = 0.02 + random.uniform(0.0, 0.06)
            self.nk_schedule.append({
                'index': idx,
                'spawn_time': t,
                'path_key': path_key,
                'spawn_frac': spawn_frac,
                'spawned': False
            })
        self.nk_state = "idle"  # 'idle', 'active', 'cooldown'
        self.current_nk = None
        self.nk_cooldown_timer = 0.0
        self.next_nk_delay = game.NK_RESPAWN_COOLDOWN
        self.current_wave_pressure = 1.0
        self.max_wave_pressure = 1.0

        # light initial pressure
        for _ in range(6):
            self.spawn_wight(initial=True)

    # ------------------------------------------------------------------
    # Soldier placement and combat control
    # ------------------------------------------------------------------
    def place_soldier(self, x: float, y: float) -> bool:
        """Place a soldier at position (x, y). Returns True if successful."""
        if not self.placement_phase:
            return False
        if len(self.soldiers) >= SOLDIER_MAX:
            return False

        dist_to_base = math.hypot(x - BASE_POS[0], y - BASE_POS[1])
        if dist_to_base < BASE_RADIUS + 80:
            return False

        if not (50 < x < SCREEN_W - 50 and 50 < y < SCREEN_H - 150):
            return False

        ang = math.atan2(y - BASE_POS[1], x - BASE_POS[0])
        soldier = game.Soldier((x, y), spawn_angle=ang)
        self.soldiers.append(soldier)
        self.stats['soldiers_deployed'] += 1
        return True

    def start_combat(self):
        if self.placement_phase:
            self.placement_phase = False
            self.combat_phase = True

    def spawn_wight(self, initial: bool = False):
        key = random.randrange(game.NUM_PATHS)
        delay = random.uniform(0.0, 0.25) if initial else 0.0
        self.enemies.append(game.Enemy(key, delay=delay))

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def update(self, dt: float):
        if not self.combat_phase or self.game_over:
            return

        self.runtime += dt
        self.spawn_timer += dt
        self.current_wave = int(min(5, self.runtime // 45))

        # Night King spawning logic
        def next_ns():
            for ns in self.nk_schedule:
                if not ns['spawned']:
                    return ns
            return None

        ns = next_ns()
        if self.nk_state == "idle":
            if ns and self.runtime >= ns['spawn_time']:
                # Spawn Night King
                nk = game.NightKing(ns['path_key'], ns['spawn_frac'], ns['index'])
                self.nks.append(nk)
                ns['spawned'] = True
                self.nk_state = "active"
                self.current_nk = nk
                wave_idx = min(ns['index'], len(game.WAVE_PRESSURE_FACTORS) - 1)
                self.current_wave_pressure = game.WAVE_PRESSURE_FACTORS[wave_idx]
                self.max_wave_pressure = max(self.max_wave_pressure, self.current_wave_pressure)
                # Spawn burst of wights
                burst = max(game.WAVE_BURST_BASE, int(math.ceil(game.WAVE_BURST_BASE * self.current_wave_pressure)))
                for _ in range(burst):
                    self.spawn_wight()
                # Deploy hero (Jon or Daenerys)
                if ns['index'] % 2 == 0:
                    hero = game.Jon(math.atan2(nk.pos()[1] - BASE_POS[1], nk.pos()[0] - BASE_POS[0]))
                    hero.deploy_for(nk)
                    self.heroes.append(hero)
                else:
                    hero = game.Daenerys(math.atan2(nk.pos()[1] - BASE_POS[1], nk.pos()[0] - BASE_POS[0]))
                    hero.deploy_for(nk)
                    self.heroes.append(hero)
        elif self.nk_state == "active":
            if self.current_nk and (not self.current_nk.alive or self.current_nk.reached):
                if not self.current_nk.alive and not getattr(self.current_nk, "death_processed", False):
                    self.current_nk.death_processed = True
                    self.next_nk_delay += 12.0
                    # Set heroes to return
                    for h in self.heroes:
                        if isinstance(h, game.Jon) or isinstance(h, game.Daenerys):
                            h.returning = True
                self.current_nk = None
                self.nk_state = "cooldown"
                self.nk_cooldown_timer = 0.0
                self.current_wave_pressure = self.max_wave_pressure
        elif self.nk_state == "cooldown":
            self.nk_cooldown_timer += dt
            if self.nk_cooldown_timer >= self.next_nk_delay:
                self.next_nk_delay = game.NK_RESPAWN_COOLDOWN
                self.nk_state = "idle"

        # Spawn regular wights (fewer while NK active)
        spawn_interval = max(game.SPAWN_MIN, game.SPAWN_BASE_RATE - self.runtime * game.SPAWN_ACCEL)
        if self.nk_state == "active":
            active_factor = game.BASE_ACTIVE_SPAWN_FACTOR / max(self.current_wave_pressure, 1.0)
            active_factor = max(game.ACTIVE_SPAWN_FACTOR_FLOOR, active_factor)
            while self.spawn_timer >= spawn_interval * active_factor:
                self.spawn_wight()
                self.spawn_timer -= spawn_interval * active_factor
        else:
            passive_factor = 1.0 / max(self.max_wave_pressure, 1.0)
            passive_factor = max(game.PASSIVE_SPAWN_FACTOR_FLOOR, passive_factor)
            while self.spawn_timer >= spawn_interval * passive_factor:
                self.spawn_wight()
                self.spawn_timer -= spawn_interval * passive_factor

        # Update Night Kings
        for nk in list(self.nks):
            if not nk.alive:
                try:
                    self.nks.remove(nk)
                except ValueError:
                    pass
                continue
            res = nk.step(dt)
            if res == "sweep":
                # Perform sweep: kill all soldiers within sweep radius
                kx, ky = nk.pos()
                for soldier in list(self.soldiers):
                    if math.hypot(soldier.x - kx, soldier.y - ky) <= game.NK_SWEEP_RADIUS:
                        try:
                            if hasattr(soldier, "_release_target"):
                                soldier._release_target()
                            self.soldiers.remove(soldier)
                        except ValueError:
                            pass
                        self.stats['soldiers_killed'] += 1
                        if self.sound_engine and getattr(self.sound_engine, "sfx_enabled", False):
                            if hasattr(self.sound_engine, "play_sound"):
                                self.sound_engine.play_sound("shock", volume=0.7)
                            elif self.sound_engine.sounds.get("shock"):
                                try:
                                    self.sound_engine.sounds["shock"].set_volume(0.7)
                                    self.sound_engine.sounds["shock"].play()
                                except Exception:
                                    pass
            # Deploy hero if NK is locked for battle and hero not present
            if nk.locked_for_battle and nk.alive:
                hero_present = any(
                    (isinstance(h, game.Jon) and h.active and h.target is nk) or
                    (isinstance(h, game.Daenerys) and h.active and h.target is nk)
                    for h in self.heroes
                )
                if not hero_present:
                    if nk.index % 2 == 0:
                        new_jon = game.Jon(math.atan2(nk.pos()[1] - BASE_POS[1], nk.pos()[0] - BASE_POS[0]))
                        new_jon.deploy_for(nk)
                        self.heroes.append(new_jon)
                    else:
                        new_daen = game.Daenerys(math.atan2(nk.pos()[1] - BASE_POS[1], nk.pos()[0] - BASE_POS[0]))
                        new_daen.deploy_for(nk)
                        self.heroes.append(new_daen)

        # Update enemies
        for enemy in list(self.enemies):
            enemy.step(dt)
            if not enemy.alive:
                try:
                    self.enemies.remove(enemy)
                except ValueError:
                    pass
                if enemy.reached:
                    self.base_hp -= 1
                    self.stats['base_hits'] += 1
                    # Play base hit sound
                    if self.sound_engine and getattr(self.sound_engine, "sfx_enabled", False):
                        if hasattr(self.sound_engine, "play_sound"):
                            self.sound_engine.play_sound("base", volume=0.6)
                        elif self.sound_engine.sounds.get("base"):
                            try:
                                self.sound_engine.sounds["base"].set_volume(0.6)
                                self.sound_engine.sounds["base"].play()
                            except Exception:
                                pass

        # Update soldiers (include NKs in target list)
        active_enemies = list(self.enemies) + list(self.nks)
        for soldier in list(self.soldiers):
            soldier.step(dt, active_enemies, self.burns, self.sound_engine, self.stats)

        # Update heroes
        for hero in list(self.heroes):
            if isinstance(hero, game.Jon):
                hero.step(dt, self.stats, self.sound_engine)
            elif isinstance(hero, game.Daenerys):
                hero.step(dt, self.sound_engine, self.stats)
            # Remove inactive heroes
            if hasattr(hero, 'active') and not hero.active:
                try:
                    self.heroes.remove(hero)
                except ValueError:
                    pass

        # Update burn effects to avoid unbounded growth
        for burn in list(self.burns):
            if burn.step(dt):
                try:
                    self.burns.remove(burn)
                except ValueError:
                    pass

        if self.base_hp <= 0:
            self.game_over = True
            self.combat_phase = False
        elif self.runtime >= self.max_runtime:
            self.victory = True
            self.combat_phase = False

    def is_game_over(self) -> bool:
        return self.game_over or self.victory


class TowerDefenseWarriorEnv(gym.Env):
    """
    Gymnasium environment for Warrior-Based Tower Defense game.

    Observation Space:
        - Grid representation (32x32) showing:
          - Empty cells (0)
          - Base (1)
          - Soldiers (2)
          - Enemies/Wights (3)
          - Night Kings (4)
        - Additional features:
          - Soldiers remaining to place
          - Base HP ratio
          - Current wave number
          - Night King active (bool)

    Action Space:
        - MultiDiscrete([32, 32]):
          - Grid X position (0-31)
          - Grid Y position (0-31)
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    # Grid size for observation
    GRID_SIZE = 32

    def __init__(self, render_mode: Optional[str] = None, fast_mode: bool = True):
        super().__init__()

        self.render_mode = render_mode
        self.fast_mode = fast_mode
        self.sound_engine_initialized = False

        # Create headless game simulation
        self.game_state = HeadlessWarriorGame()

        # Define observation space
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0, high=4, shape=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32
            ),
            'soldiers_remaining': spaces.Box(
                low=0, high=SOLDIER_MAX, shape=(1,), dtype=np.int32
            ),
            'base_hp_ratio': spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            'current_wave': spaces.Box(
                low=0, high=5, shape=(1,), dtype=np.int32
            ),
            'night_king_active': spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.int32
            )
        })

        # Define action space
        # [grid_x, grid_y] - placement position
        # Can expand to [warrior_type, grid_x, grid_y] later
        self.action_space = spaces.MultiDiscrete([self.GRID_SIZE, self.GRID_SIZE])

        # Episode tracking
        self.placement_actions_taken = 0
        self.episode_reward = 0.0

        # Reward tracking
        self.last_wights_killed = 0
        self.last_soldiers_killed = 0
        self.last_base_hp = BASE_HP_MAX
        self.last_nk_kills = 0

        # For rendering
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font_small = None

        # Combat simulation state
        self.combat_running = False
        self.combat_time = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset game state
        self.game_state.reset(seed=seed)

        # Reset tracking
        self.placement_actions_taken = 0
        self.episode_reward = 0.0
        self.last_wights_killed = 0
        self.last_soldiers_killed = 0
        self.last_base_hp = BASE_HP_MAX
        self.last_nk_kills = 0
        self.combat_running = False
        self.combat_time = 0.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        During placement phase: Place soldier at specified location
        During combat phase: Auto-run until game over
        """
        reward = 0.0
        terminated = False
        truncated = False

        # Placement phase
        if self.game_state.placement_phase:
            grid_x, grid_y = action

            # Convert grid coordinates to world coordinates
            world_x = (grid_x / self.GRID_SIZE) * SCREEN_W
            world_y = (grid_y / self.GRID_SIZE) * SCREEN_H

            # Try to place soldier
            success = self.game_state.place_soldier(world_x, world_y)

            if success:
                reward += 5.0
            else:
                reward -= 1.0

            self.placement_actions_taken += 1

            # If we've placed all soldiers (or reached max), start combat
            if self.placement_actions_taken >= SOLDIER_MAX or len(self.game_state.soldiers) >= SOLDIER_MAX:
                if len(self.game_state.soldiers) >= SOLDIER_MAX:
                reward += 20.0
            else:
                reward -= 10.0
                self.game_state.start_combat()
                self.combat_running = True

        # Combat phase - auto-run the game
        if self.game_state.combat_phase:
            if self.fast_mode:
                updates_per_step = 5
                dt = 1.0 / 60.0
                for _ in range(updates_per_step):
                    self._simulate_combat_step(dt)
                    if self.game_state.is_game_over():
                        break
            else:
                dt = 1.0 / 60.0
                self._simulate_combat_step(dt)

            # Calculate rewards based on state changes
            reward += self._calculate_step_reward()

        # Check if game is over
        if self.game_state.is_game_over():
            terminated = True
            reward += self._calculate_final_reward()

        self.episode_reward += reward

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _simulate_combat_step(self, dt: float):
        """Advance the headless game simulation."""
        self.combat_time += dt
        self.game_state.update(dt)

    def _calculate_step_reward(self) -> float:
        """Calculate reward for a single combat step"""
        reward = 0.0

        if self.game_state.combat_phase:
            reward += 1.0

        wights_killed_this_step = self.game_state.stats['wights_killed'] - self.last_wights_killed
        reward += wights_killed_this_step * 10.0
        self.last_wights_killed = self.game_state.stats['wights_killed']

        nk_kills_this_step = self.game_state.stats['nk_kills'] - self.last_nk_kills
        reward += nk_kills_this_step * 20.0
        self.last_nk_kills = self.game_state.stats['nk_kills']

        soldiers_killed_this_step = self.game_state.stats['soldiers_killed'] - self.last_soldiers_killed
        reward -= soldiers_killed_this_step * 15.0
        self.last_soldiers_killed = self.game_state.stats['soldiers_killed']

        base_damage_this_step = self.last_base_hp - self.game_state.base_hp
        reward -= base_damage_this_step * 5.0
        self.last_base_hp = self.game_state.base_hp

        return reward

    def _calculate_final_reward(self) -> float:
        """Calculate final reward at end of episode"""
        reward = 0.0

        if self.game_state.victory:
            reward += 2000.0
            hp_ratio = self.game_state.base_hp / BASE_HP_MAX
            reward += hp_ratio * 500.0
            soldiers_alive = len(self.game_state.soldiers)
            reward += soldiers_alive * 100.0
        else:
            reward -= 500.0
            reward += self.game_state.current_wave * 50.0

        return reward

    def _get_observation(self) -> Dict:
        """Get current observation"""
        grid = self._create_grid_observation()

        soldiers_remaining = SOLDIER_MAX - len(self.game_state.soldiers)
        base_hp_ratio = self.game_state.base_hp / BASE_HP_MAX
        current_wave = self.game_state.current_wave
        night_king_active = 1 if len(self.game_state.nks) > 0 else 0

        return {
            'grid': grid,
            'soldiers_remaining': np.array([soldiers_remaining], dtype=np.int32),
            'base_hp_ratio': np.array([base_hp_ratio], dtype=np.float32),
            'current_wave': np.array([current_wave], dtype=np.int32),
            'night_king_active': np.array([night_king_active], dtype=np.int32)
        }

    def _create_grid_observation(self) -> np.ndarray:
        """
        Create grid representation of game state.

        Values:
            0 = Empty
            1 = Base
            2 = Soldier
            3 = Enemy/Wight
            4 = Night King
        """
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)

        # Add base
        base_grid_x = int((BASE_POS[0] / SCREEN_W) * self.GRID_SIZE)
        base_grid_y = int((BASE_POS[1] / SCREEN_H) * self.GRID_SIZE)
        base_grid_x = np.clip(base_grid_x, 0, self.GRID_SIZE - 1)
        base_grid_y = np.clip(base_grid_y, 0, self.GRID_SIZE - 1)

        # Base occupies 2x2 area
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                gx = base_grid_x + dx
                gy = base_grid_y + dy
                if 0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE:
                    grid[gy, gx] = 1

        # Add soldiers
        for soldier in self.game_state.soldiers:
            grid_x = int((soldier.x / SCREEN_W) * self.GRID_SIZE)
            grid_y = int((soldier.y / SCREEN_H) * self.GRID_SIZE)
            grid_x = np.clip(grid_x, 0, self.GRID_SIZE - 1)
            grid_y = np.clip(grid_y, 0, self.GRID_SIZE - 1)
            grid[grid_y, grid_x] = 2

        # Add enemies
        for enemy in self.game_state.enemies:
            if enemy.alive and enemy.spawn_delay <= 0:
                ex, ey = enemy.pos()
                grid_x = int((ex / SCREEN_W) * self.GRID_SIZE)
                grid_y = int((ey / SCREEN_H) * self.GRID_SIZE)
                if 0 <= grid_x < self.GRID_SIZE and 0 <= grid_y < self.GRID_SIZE:
                    if grid[grid_y, grid_x] == 0:
                        grid[grid_y, grid_x] = 3

        # Add Night Kings
        for nk in self.game_state.nks:
            if nk.alive:
                nk_x, nk_y = nk.pos()
                grid_x = int((nk_x / SCREEN_W) * self.GRID_SIZE)
                grid_y = int((nk_y / SCREEN_H) * self.GRID_SIZE)
                if 0 <= grid_x < self.GRID_SIZE and 0 <= grid_y < self.GRID_SIZE:
                    # NKs override other entities in grid
                    grid[grid_y, grid_x] = 4

        return grid

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        return {
            'phase': 'placement' if self.game_state.placement_phase else 'combat',
            'game_time': self.game_state.runtime,
            'current_wave': self.game_state.current_wave,
            'stats': self.game_state.stats.copy(),
            'episode_reward': self.episode_reward,
            'placement_actions_taken': self.placement_actions_taken,
            'base_hp': self.game_state.base_hp,
            'soldiers_alive': len(self.game_state.soldiers),
            'victory': self.game_state.victory,
            'game_over': self.game_state.game_over
        }

    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return None

        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            return self._render_human()

    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array using the same style as td_pyastar.py"""
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self.font = pygame.font.SysFont("Verdana", 16)
            self.small_font = pygame.font.SysFont("Arial", 12)
            self.font_small = pygame.font.Font(None, 24)
            self.font_tiny = pygame.font.Font(None, 18)
            # Initialize snow particles
            self.snow = [[random.uniform(0, SCREEN_W), random.uniform(-SCREEN_H, 0),
                         random.uniform(20, 120), random.uniform(1.2, 3.8),
                         random.uniform(-18, 18), random.uniform(100, 240)]
                        for _ in range(game.SNOW_BASE)]
            self.snow_time = 0.0
            self.pygame_initialized = True

        # Update snow
        if hasattr(self, 'snow_time'):
            self.snow_time += 1.0 / 60.0  # Approximate dt
        if hasattr(self, 'snow'):
            for p in self.snow:
                p[1] += p[2] * (1.0 / 60.0)
                p[0] += p[4] * (1.0 / 60.0)
                if p[1] > SCREEN_H + 12 or p[0] < -24 or p[0] > SCREEN_W + 24:
                    p[0] = random.uniform(0, SCREEN_W)
                    p[1] = random.uniform(-SCREEN_H, 0)

        # Clear screen
        self.screen.fill((18, 20, 28))

        # Background gradient
        bg = pygame.Surface((SCREEN_W, SCREEN_H))
        for y in range(SCREEN_H):
            v = int(18 + (y / SCREEN_H) * 36)
            bg.fill((v + 8, v + 12, v + 18), (0, y, SCREEN_W, 1))
        self.screen.blit(bg, (0, 0))

        # Atmospheric speckles
        speck = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        for i in range(60):
            sx = (i * 173) % SCREEN_W
            sy = (i * 97) % (SCREEN_H // 3)
            pygame.draw.circle(speck, (255, 255, 255, 14), (sx, sy + 30), 1)
        self.screen.blit(speck, (0, 0))

        # Fog layer
        fog_layer = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        fog_layer.fill((160, 180, 200, game.FOG_ALPHA))
        self.screen.blit(fog_layer, (0, 0))

        # Draw static obstacles
        for obs in game.combined_obstacles():
            if obs.get("type") == "rect":
                half_w = obs.get("half_w", 30)
                half_h = obs.get("half_h", 30)
                rect = pygame.Rect(
                    int(obs["pos"][0] - half_w),
                    int(obs["pos"][1] - half_h),
                    int(half_w * 2),
                    int(half_h * 2)
                )
                pygame.draw.rect(self.screen, (48, 54, 66), rect, border_radius=6)
                pygame.draw.rect(self.screen, (30, 34, 44), rect, 3, border_radius=6)
            else:
                ox, oy = int(obs["pos"][0]), int(obs["pos"][1])
                radius = obs.get("radius", 30)
                pygame.draw.circle(self.screen, (48, 54, 66), (ox, oy), radius)
                pygame.draw.circle(self.screen, (30, 34, 44), (ox, oy), radius, 3)

        # Draw burn effects (wight death flashes)
        for burn in self.game_state.burns:
            burn.draw(self.screen)

        # Draw soldiers
        for soldier in self.game_state.soldiers:
            if hasattr(soldier, 'draw'):
                soldier.draw(self.screen, self.font)
            else:
                pygame.draw.circle(self.screen, (22, 22, 26), (int(soldier.x), int(soldier.y)), game.SOLDIER_RADIUS + 4)
                pygame.draw.circle(self.screen, soldier.color if hasattr(soldier, 'color') else game.SOLDIER_COLOR,
                                  (int(soldier.x), int(soldier.y)), game.SOLDIER_RADIUS)
                pygame.draw.rect(self.screen, (200, 200, 200),
                               (int(soldier.x + 8), int(soldier.y - 6), 6, 10))

        # Draw enemies (wights) with aura
        for enemy in sorted(list(self.game_state.enemies), key=lambda z: -getattr(z, 's', 0)):
            if enemy.spawn_delay > 0:
                continue
            if enemy.alive:
                x, y = enemy.pos()
                # Aura effect
                aura = pygame.Surface((game.ENEMY_RADIUS * 5, game.ENEMY_RADIUS * 5), pygame.SRCALPHA)
                pygame.draw.circle(aura, (255, 120, 120, 64), (game.ENEMY_RADIUS * 2, game.ENEMY_RADIUS * 2),
                                 game.ENEMY_RADIUS * 2)
                self.screen.blit(aura, (int(x - game.ENEMY_RADIUS * 2), int(y - game.ENEMY_RADIUS * 2)),
                               special_flags=pygame.BLEND_RGBA_ADD)
                pygame.draw.circle(self.screen, game.ENEMY_COLOR, (int(x), int(y)), game.ENEMY_RADIUS)
                pygame.draw.circle(self.screen, (24, 24, 24), (int(x), int(y)), game.ENEMY_RADIUS, 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (int(x + 3), int(y - 4)), 3)

        # Draw Night Kings
        import time as time_module
        for nk in list(self.game_state.nks):
            if not nk.alive:
                continue
            x, y = nk.pos()
            # Aura
            aura = pygame.Surface((game.NK_RADIUS * 6, game.NK_RADIUS * 6), pygame.SRCALPHA)
            pygame.draw.circle(aura, game.NK_AURA, (game.NK_RADIUS * 3, game.NK_RADIUS * 3), game.NK_RADIUS * 3)
            self.screen.blit(aura, (int(x - game.NK_RADIUS * 3), int(y - game.NK_RADIUS * 3)),
                           special_flags=pygame.BLEND_RGBA_ADD)
            pulse = 1.0 + 0.05 * math.sin(time_module.time() * 2.5 + nk.index)
            rcore = int(game.NK_RADIUS * pulse)
            pygame.draw.circle(self.screen, game.NK_COLOR, (int(x), int(y)), rcore)
            pygame.draw.circle(self.screen, (18, 22, 28), (int(x), int(y)), rcore, 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x + 6), int(y - 6)), 5)
            # Label
            lfont = pygame.font.SysFont("Georgia", 18, bold=True)
            lbl = lfont.render("NIGHT KING", True, (200, 240, 255))
            self.screen.blit(lbl, (int(x - lbl.get_width() / 2), int(y - game.NK_RADIUS - 26)))
            # HP bar
            bw, bh = 120, 12
            bx = x - bw / 2
            by = y - game.NK_RADIUS - 16
            pygame.draw.rect(self.screen, (20, 20, 26), (bx, by, bw, bh), border_radius=4)
            max_hp = game.NK_HP_BASE * (1.0 + 0.08 * nk.index)
            frac = max(0.0, min(1.0, nk.hp / max_hp))
            pygame.draw.rect(self.screen, (160, 220, 255), (bx + 4, by + 3, int((bw - 8) * frac), bh - 6),
                           border_radius=3)
            if nk.locked_for_battle:
                small = pygame.font.SysFont("Verdana", 14, bold=True)
                st = small.render("ENGAGED", True, (255, 220, 180))
                self.screen.blit(st, (int(x - st.get_width() / 2), int(y + game.NK_RADIUS + 8)))

        # Draw heroes
        for hero in self.game_state.heroes:
            if isinstance(hero, game.Daenerys):
                if getattr(hero, "breathing", False) and hero.target and hero.target.alive:
                    tx, ty = hero.target.pos()
                    beam = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                    pygame.draw.line(beam, (255, 140, 40, 160), (int(hero.x), int(hero.y)), (int(tx), int(ty)), 22)
                    self.screen.blit(beam, (0, 0))
                hero.draw(self.screen)
                if hero.active:
                    ttxt = self.small_font.render("Daenerys & Drogon", True, (240, 220, 200))
                    self.screen.blit(ttxt, (int(hero.x - ttxt.get_width() / 2), int(hero.y - 48)))
            elif isinstance(hero, game.Jon):
                hero.draw(self.screen)
                if hero.active:
                    ttxt = self.small_font.render("Jon Snow — Longclaw", True, (220, 220, 230))
                    self.screen.blit(ttxt, (int(hero.x - ttxt.get_width() / 2), int(hero.y - 58)))


        # Draw base
        pygame.draw.circle(self.screen, (50, 50, 60), BASE_POS, BASE_RADIUS)
        batt_w = 10
        for i in range(-3, 4):
            bx = int(BASE_POS[0] + i * (batt_w + 2))
            by = int(BASE_POS[1] - BASE_RADIUS - 6)
            pygame.draw.rect(self.screen, (38, 38, 48), (bx, by, batt_w, 12))
            pygame.draw.rect(self.screen, (18, 18, 26), (bx, by, batt_w, 12), 1)
        flag_x = int(BASE_POS[0] + BASE_RADIUS + 12)
        flag_y = int(BASE_POS[1] - BASE_RADIUS + 8)
        pygame.draw.rect(self.screen, (28, 28, 36), (flag_x, flag_y, 6, 32))
        pygame.draw.polygon(self.screen, (210, 210, 210),
                          [(flag_x + 6, flag_y + 4), (flag_x + 36, flag_y + 12), (flag_x + 6, flag_y + 28)])
        lbl = self.font.render("Winterfell Keep", True, (235, 235, 235))
        self.screen.blit(lbl, (int(BASE_POS[0] - lbl.get_width() / 2), int(BASE_POS[1] - BASE_RADIUS - 64)))
        # HP bar
        bar_w, bar_h = 280, 16
        bx = BASE_POS[0] - bar_w / 2
        by = BASE_POS[1] - BASE_RADIUS - 44
        pygame.draw.rect(self.screen, (28, 28, 36), (bx - 2, by - 2, bar_w + 4, bar_h + 4), border_radius=6)
        pygame.draw.rect(self.screen, (150, 150, 150), (bx, by, bar_w, bar_h), border_radius=6)
        base_hp_ratio = self.game_state.base_hp / BASE_HP_MAX
        frac = max(0.0, min(1.0, base_hp_ratio))
        pygame.draw.rect(self.screen, (90, 200, 120), (bx + 4, by + 4, int((bar_w - 8) * frac), bar_h - 8),
                        border_radius=6)
        if frac < 0.25:
            pygame.draw.rect(self.screen, (220, 40, 40),
                           (bx + 4 + int((bar_w - 8) * frac), by + 4, int((bar_w - 8) * (0.25 - frac)), bar_h - 8),
                           border_radius=4)

        # Snow
        if hasattr(self, 'snow'):
            for p in self.snow:
                pygame.draw.circle(self.screen, (255, 255, 255, int(p[5])), (int(p[0]), int(p[1])), int(p[3]))

        # Vignette
        v = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        for i in range(140):
            a = int(80 * (i / 140) ** 1.2)
            pygame.draw.rect(v, (6, 10, 14, a), (i, i, SCREEN_W - 2 * i, SCREEN_H - 2 * i), 1)
        self.screen.blit(v, (0, 0))

        # HUD
        active_enemies = len([e for e in self.game_state.enemies if e.spawn_delay <= 0 and e.alive])
        hud = self.font.render(f"Enemies: {active_enemies}   Wall Integrity: {self.game_state.base_hp}", True,
                             (220, 220, 220))
        self.screen.blit(hud, (14, 12))
        soldier_hud = self.font.render(
            f"Soldiers Alive: {len(self.game_state.soldiers)}   Killed: {self.game_state.stats['soldiers_killed']}   Deployed: {self.game_state.stats['soldiers_deployed']}",
            True, (220, 220, 200))
        self.screen.blit(soldier_hud, (14, 40))
        killed_hud = self.font.render(
            f"Wights Killed: {self.game_state.stats['wights_killed']}   NKs Defeated: {self.game_state.stats['nk_kills']}/5   Time: {int(self.game_state.runtime)}s",
            True, (220, 220, 200))
        self.screen.blit(killed_hud, (14, 68))

        # Phase indicator
        phase_text = "PLACEMENT PHASE" if self.game_state.placement_phase else "COMBAT PHASE"
        phase_color = (100, 200, 100) if self.game_state.placement_phase else (200, 100, 100)
        phase_surf = self.font_small.render(phase_text, True, phase_color)
        self.screen.blit(phase_surf, (SCREEN_W - phase_surf.get_width() - 14, 12))

        # Convert to numpy array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def _render_human(self):
        """Render for human viewing"""
        if not self.pygame_initialized:
            pygame.init()
            # Initialize mixer for sound
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            except Exception:
                pass
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Tower Defense - RL Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Verdana", 16)
            self.small_font = pygame.font.SysFont("Arial", 12)
            self.font_small = pygame.font.Font(None, 24)
            self.font_tiny = pygame.font.Font(None, 18)
            # Initialize snow particles
            self.snow = [[random.uniform(0, SCREEN_W), random.uniform(-SCREEN_H, 0),
                         random.uniform(20, 120), random.uniform(1.2, 3.8),
                         random.uniform(-18, 18), random.uniform(100, 240)]
                        for _ in range(game.SNOW_BASE)]
            self.snow_time = 0.0
            self.pygame_initialized = True

            if not self.sound_engine_initialized:
                if self.game_state.sound_engine and self.game_state.sound_engine.ambience:
                    try:
                        if self.game_state.sound_engine.ambience_enabled:
                            self.game_state.sound_engine.ambience.set_volume(0.5)
                            self.game_state.sound_engine.ambience.play(loops=-1)
                            self.game_state.sound_engine.ambience_playing = True
                    except Exception:
                        pass
                self.sound_engine_initialized = True

        rgb_array = self._render_rgb_array()
        surf = pygame.surfarray.make_surface(np.transpose(rgb_array, axes=(1, 0, 2)))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

        if self.clock:
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Clean up"""
        # Stop ambience if playing
        if hasattr(self, 'game_state') and self.game_state.sound_engine:
            if getattr(self.game_state.sound_engine, "ambience_playing", False):
                try:
                    if self.game_state.sound_engine.ambience:
                        self.game_state.sound_engine.ambience.stop()
                except Exception:
                    pass
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False


def test_environment():
    """Test the environment with random actions"""
    print("Testing Tower Defense Warrior Gymnasium Environment...")

    env = TowerDefenseWarriorEnv()

    # Test reset
    observation, info = env.reset(seed=42)
    print(f"Initial observation shape: grid={observation['grid'].shape}")
    print(f"Initial info: {info}")

    # Test random episode
    episode_reward = 0
    done = False
    step_count = 0

    while not done and step_count < 200:
        # Random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        done = terminated or truncated
        step_count += 1

        if step_count % 20 == 0:
            print(f"Step {step_count}: Phase={info['phase']}, Reward={reward:.2f}, Total={episode_reward:.2f}")

    print(f"\nEpisode finished after {step_count} steps")
    print(f"Final phase: {info['phase']}")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Stats: {info['stats']}")

    env.close()
    print("\nEnvironment test completed successfully!")


if __name__ == "__main__":
    test_environment()
