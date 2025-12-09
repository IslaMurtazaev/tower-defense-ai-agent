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
        self.base_hit_effects: List[game.BaseHitEffect] = []
        self.stats = {
            'wights_killed': 0,
            'soldier_nk_attacks': 0,  # Track when soldiers attack Night Kings
            'soldiers_killed_by_nk_sweep': 0,  # Track soldiers killed by NK sweeps
            'soldiers_killed': 0,
            'nk_kills': 0,
            'soldiers_deployed': 0,
            'base_hits': 0,
            'soldiers_far_from_base': 0,  # Track soldiers too far from base
            'enemies_near_base': 0,  # Track enemies close to base
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
    # Unit placement and combat control
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

    def place_hero(self, x: float, y: float, hero_type: int = 0) -> bool:
        """Place a hero at position (x, y). hero_type: 0=Jon, 1=Daenerys. Returns True if successful."""
        if not self.placement_phase:
            return False
        if len(self.heroes) >= 2:  # Maximum 2 heroes allowed
            return False

        dist_to_base = math.hypot(x - BASE_POS[0], y - BASE_POS[1])
        if dist_to_base < BASE_RADIUS + 80:
            return False

        if not (50 < x < SCREEN_W - 50 and 50 < y < SCREEN_H - 150):
            return False

        ang = math.atan2(y - BASE_POS[1], x - BASE_POS[0])
        if hero_type == 0:
            hero = game.Jon(ang)
        else:
            # Daenerys always starts hovering on the tower, regardless of placement position
            hero = game.Daenerys(ang)
            # Override position to tower center (her __init__ already does this, but ensure it)
            hero.x = BASE_POS[0]
            hero.y = BASE_POS[1]
        # Hero starts active and will target nearest Night King when one appears
        hero.active = True
        self.heroes.append(hero)
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

        # Night King state machine: idle -> active -> cooldown -> idle
        def next_ns():
            for ns in self.nk_schedule:
                if not ns['spawned']:
                    return ns
            return None

        ns = next_ns()
        if self.nk_state == "idle":
            if ns and self.runtime >= ns['spawn_time']:
                nk = game.NightKing(ns['path_key'], ns['spawn_frac'], ns['index'])
                self.nks.append(nk)
                ns['spawned'] = True
                self.nk_state = "active"
                self.current_nk = nk

                # Increase wave pressure and spawn wight burst
                wave_idx = min(ns['index'], len(game.WAVE_PRESSURE_FACTORS) - 1)
                self.current_wave_pressure = game.WAVE_PRESSURE_FACTORS[wave_idx]
                self.max_wave_pressure = max(self.max_wave_pressure, self.current_wave_pressure)
                burst = max(game.WAVE_BURST_BASE, int(math.ceil(game.WAVE_BURST_BASE * self.current_wave_pressure)))
                for _ in range(burst):
                    self.spawn_wight()

                # Heroes are now deployed by RL agent during placement phase
                # They will automatically target Night Kings when they engage
                # No auto-deployment here - heroes are already on the field
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

        # Update Night Kings: movement, sweeps, hero deployment
        for nk in list(self.nks):
            if not nk.alive:
                try:
                    self.nks.remove(nk)
                except ValueError:
                    pass
                continue
            res = nk.step(dt)
            if res == "sweep":
                # NK sweep attack: instant kill all soldiers in radius
                # This is a learning signal for RL: soldiers near NKs get killed
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
                        self.stats['soldiers_killed_by_nk_sweep'] += 1
                        if self.sound_engine and getattr(self.sound_engine, "sfx_enabled", False):
                            if hasattr(self.sound_engine, "play_sound"):
                                self.sound_engine.play_sound("shock", volume=0.7)
                            elif self.sound_engine.sounds.get("shock"):
                                try:
                                    self.sound_engine.sounds["shock"].set_volume(0.7)
                                    self.sound_engine.sounds["shock"].play()
                                except Exception:
                                    pass
            # Heroes are deployed by RL agent during placement phase
            # When Night King engages, assign available heroes to target it
            if nk.locked_for_battle and nk.alive:
                # Find heroes without a target or with dead target
                available_heroes = [h for h in self.heroes if h.active and (h.target is None or not h.target.alive)]
                for hero in available_heroes:
                    # Assign hero to this Night King if not already targeting it
                    if hero.target is not nk:
                        hero.deploy_for(nk)
                        # With 2 heroes, we can assign both to the same NK (faster kill) or different NKs
                        # This will naturally distribute if multiple NKs are active

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
                    # Create visual impact effect
                    self.base_hit_effects.append(game.BaseHitEffect())
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

        # Update soldiers: they target both wights and Night Kings
        # RL agents must LEARN to place soldiers away from NKs to avoid sweep deaths
        # The reward penalties (soldiers_killed_by_nk_sweep) provide the learning signal
        active_enemies = list(self.enemies) + list(self.nks)

        # Track how many soldiers are defending the base vs chasing enemies far away
        # This helps the agent learn to keep soldiers near base when enemies are close
        BASE_DEFENSE_RADIUS = 200.0  # Soldiers within 200 pixels count as "defending"
        ENEMY_THREAT_RADIUS = 250.0  # Enemies within 250 pixels are "threatening" the base
        soldiers_near_base = 0
        enemies_near_base = 0

        for soldier in list(self.soldiers):
            soldier.step(dt, active_enemies, self.burns, self.sound_engine, self.stats)
            # Check if this soldier is close enough to the base to defend it
            dist_to_base = math.hypot(soldier.x - BASE_POS[0], soldier.y - BASE_POS[1])
            if dist_to_base <= BASE_DEFENSE_RADIUS:
                soldiers_near_base += 1

        # Count how many enemies are close to the base (they're a threat)
        for enemy in active_enemies:
            if enemy.alive and enemy.spawn_delay <= 0:
                if hasattr(enemy, 'pos'):
                    enemy_pos = enemy.pos()
                else:
                    enemy_pos = (enemy.x, enemy.y) if hasattr(enemy, 'x') else (0, 0)
                dist_to_base = math.hypot(enemy_pos[0] - BASE_POS[0], enemy_pos[1] - BASE_POS[1])
                if dist_to_base <= ENEMY_THREAT_RADIUS:
                    enemies_near_base += 1

        # Save these stats so we can use them for rewards later
        # If soldiers are far from base when enemies are near, that's bad
        self.stats['soldiers_far_from_base'] = len(self.soldiers) - soldiers_near_base
        self.stats['enemies_near_base'] = enemies_near_base

        # Update heroes - prioritize Night Kings, but also attack wights when no NKs present
        for hero in list(self.heroes):
            # If hero is inactive (returned to base) or has no target, find a new target
            if not hero.active or (hero.target is None or not hero.target.alive):
                # Priority 1: Find nearest active Night King (if any)
                best_nk = None
                best_nk_dist = float('inf')
                for nk in self.nks:
                    if nk.alive and nk.locked_for_battle:
                        nk_pos = nk.pos()
                        hero_pos = (hero.x, hero.y)
                        dist = math.hypot(nk_pos[0] - hero_pos[0], nk_pos[1] - hero_pos[1])
                        if dist < best_nk_dist:
                            best_nk_dist = dist
                            best_nk = nk

                # Priority 2: If no Night King, find nearest wight
                best_wight = None
                best_wight_dist = float('inf')
                if not best_nk:
                    for enemy in self.enemies:
                        if enemy.alive and enemy.spawn_delay <= 0:
                            enemy_pos = enemy.pos()
                            hero_pos = (hero.x, hero.y)
                            dist = math.hypot(enemy_pos[0] - hero_pos[0], enemy_pos[1] - hero_pos[1])
                            if dist < best_wight_dist:
                                best_wight_dist = dist
                                best_wight = enemy

                # Assign target (NK takes priority)
                if best_nk:
                    # Reactivate hero and deploy for Night King
                    hero.deploy_for(best_nk)
                elif best_wight:
                    # Assign wight as target and activate hero
                    hero.target = best_wight
                    hero.active = True
                    hero.returning = False

            if isinstance(hero, game.Jon):
                hero.step(dt, self.stats, self.sound_engine)
            elif isinstance(hero, game.Daenerys):
                hero.step(dt, self.sound_engine, self.stats, self.nks)  # Pass nks so she can check for Night Kings in range
            # DO NOT remove inactive heroes - they should remain available for next Night King
            # Heroes will be reactivated when a new Night King appears via deploy_for() above

        # Update burn effects to avoid unbounded growth
        for burn in list(self.burns):
            if burn.step(dt):
                try:
                    self.burns.remove(burn)
                except ValueError:
                    pass

        # Update base hit effects
        for hit_effect in list(self.base_hit_effects):
            if hit_effect.step(dt):
                try:
                    self.base_hit_effects.remove(hit_effect)
                except ValueError:
                    pass

        if self.base_hp <= 0:
            self.game_over = True
            self.combat_phase = False
        else:
            # Victory check: all Night Kings spawned and all killed
            all_spawned = all(ns['spawned'] for ns in self.nk_schedule)
            active_nks = any(nk for nk in self.nks if nk.alive)
            if all_spawned and not active_nks:
                # Check if all scheduled NKs have been killed
                if self.stats.get('nk_kills', 0) >= len(self.nk_schedule):
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

    def __init__(self, render_mode: Optional[str] = None, fast_mode: bool = True, fast_multiplier: int = 5, agent_type: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode
        self.fast_mode = fast_mode
        self.fast_multiplier = fast_multiplier
        self.agent_type = agent_type  # For window title customization
        self.sound_engine_initialized = False

        # Create headless game simulation
        self.game_state = HeadlessWarriorGame()

        # Define observation space
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0, high=5, shape=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32
            ),
            'soldiers_remaining': spaces.Box(
                low=0, high=SOLDIER_MAX, shape=(1,), dtype=np.int32
            ),
            'heroes_remaining': spaces.Box(
                low=0, high=2, shape=(1,), dtype=np.int32
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
        # [unit_type, grid_x, grid_y] - unit_type: 0=soldier, 1=hero
        # Total units: 6 soldiers + 2 heroes = 8 units (increased from 6)
        self.action_space = spaces.MultiDiscrete([2, self.GRID_SIZE, self.GRID_SIZE])

        # Episode tracking
        self.placement_actions_taken = 0
        self.episode_reward = 0.0

        # Reward tracking
        self.last_wights_killed = 0
        self.last_soldiers_killed = 0
        self.last_soldier_nk_attacks = 0
        self.last_soldiers_killed_by_nk_sweep = 0
        self.last_base_hp = BASE_HP_MAX
        self.last_nk_kills = 0
        self.last_wave = 0
        self._last_soldiers_alive = 6  # Track how many soldiers are alive (we have 6 max now)
        self.last_soldiers_far_from_base = 0
        self.last_enemies_near_base = 0

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
        self.last_soldier_nk_attacks = 0
        self.last_soldiers_killed_by_nk_sweep = 0
        self.last_nk_kills = 0
        self.last_wave = 0
        self.last_soldiers_far_from_base = 0
        self.last_enemies_near_base = 0
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

        if self.game_state.placement_phase:
            unit_type, grid_x, grid_y = action

            unit_type = int(np.clip(unit_type, 0, 1))
            grid_x = int(np.clip(grid_x, 0, self.GRID_SIZE - 1))
            grid_y = int(np.clip(grid_y, 0, self.GRID_SIZE - 1))

            # Convert grid coordinates to world coordinates
            world_x = (grid_x / self.GRID_SIZE) * SCREEN_W
            world_y = (grid_y / self.GRID_SIZE) * SCREEN_H

            # Place unit based on type
            if unit_type == 0:
                # Place soldier
                success = self.game_state.place_soldier(world_x, world_y)
            else:
                # Place hero (hero_type: 0=Jon, 1=Daenerys)
                # Alternate between Jon and Daenerys: first hero is Jon, second is Daenerys
                hero_type = len(self.game_state.heroes) % 2  # 0 for first hero (Jon), 1 for second (Daenerys)
                success = self.game_state.place_hero(world_x, world_y, hero_type)

            if success:
                if unit_type == 1:  # Hero placement
                    reward += 5.0  # Normalized from 50.0
                else:  # Soldier placement
                    reward += 1.0  # Normalized from 10.0
            else:
                reward -= 0.1  # Normalized from 1.0

            self.placement_actions_taken += 1

            total_units = len(self.game_state.soldiers) + len(self.game_state.heroes)
            heroes_placed = len(self.game_state.heroes)
            soldiers_placed = len(self.game_state.soldiers)
            if heroes_placed == 1:
                reward += 2.0
            elif heroes_placed == 2:
                reward += 5.0

            # Reward for placing soldiers - we want all 6, not just 4
            if soldiers_placed >= 4:
                reward += 1.0  # Good, but we want more
            if soldiers_placed >= 6:
                reward += 2.0  # Great! All 6 soldiers placed for better defense

            # Start combat once all units are placed (6 soldiers + 2 heroes = 8 total)
            if total_units >= 8:
                # All units successfully placed - STRONG reward
                reward += 15.0  # INCREASED from 10.0 - very strong incentive to place all 8 units
                self.game_state.start_combat()
                self.combat_running = True
            elif self.placement_actions_taken >= 12:
                # Force combat after 12 attempts (even if not all units placed)
                # This prevents infinite placement phase if agent keeps failing
                # STRONG penalty for not placing all units
                missing_units = 8 - total_units
                reward -= 5.0 * missing_units  # INCREASED penalty - 5.0 per missing unit
                self.game_state.start_combat()
                self.combat_running = True

        # Combat phase: run game simulation and calculate step rewards
        if self.game_state.combat_phase:
            if self.fast_mode:
                # Fast mode: simulate multiple frames per step for faster training
                updates_per_step = self.fast_multiplier
                dt = 1.0 / 60.0
                for _ in range(updates_per_step):
                    self._simulate_combat_step(dt)
                    if self.game_state.is_game_over():
                        break
            else:
                dt = 1.0 / 60.0
                self._simulate_combat_step(dt)

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
        """Calculate reward for a single combat step - NORMALIZED for stable Q-learning"""
        reward = 0.0

        # Survival bonus: reward for staying alive each step
        if self.game_state.combat_phase:
            reward += 0.1  # Reduced from 1.0

        # Wave completion bonus (reward for progressing through waves)
        wave_completed_this_step = self.game_state.current_wave - self.last_wave
        if wave_completed_this_step > 0:
            reward += wave_completed_this_step * 10.0  # Reduced from 100.0
        self.last_wave = self.game_state.current_wave

        # Track kills and deaths (delta rewards) - NORMALIZED
        wights_killed_this_step = self.game_state.stats['wights_killed'] - self.last_wights_killed
        reward += wights_killed_this_step * 1.0  # Normalized from 2.0
        self.last_wights_killed = self.game_state.stats['wights_killed']

        nk_kills_this_step = self.game_state.stats['nk_kills'] - self.last_nk_kills
        if nk_kills_this_step > 0:
            reward += nk_kills_this_step * 50.0
        self.last_nk_kills = self.game_state.stats['nk_kills']

        # Reward for soldiers surviving (positive signal for good placement)
        soldiers_alive_now = len(self.game_state.soldiers)
        soldiers_alive_before = getattr(self, '_last_soldiers_alive', 4)
        if soldiers_alive_now < soldiers_alive_before:
            # Soldiers died - already penalized elsewhere
            pass
        elif self.game_state.combat_phase:
            # Small bonus for each soldier that survives each step (encourages safe placement)
            reward += soldiers_alive_now * 0.05  # Reduced from 0.5
        self._last_soldiers_alive = soldiers_alive_now

        soldiers_killed_this_step = self.game_state.stats['soldiers_killed'] - self.last_soldiers_killed
        reward -= soldiers_killed_this_step * 5.0  # Normalized from 15.0
        self.last_soldiers_killed = self.game_state.stats['soldiers_killed']

        # Base damage penalty - NORMALIZED
        base_damage_this_step = self.last_base_hp - self.game_state.base_hp
        reward -= base_damage_this_step * 20.0  # Keep at 20.0 (already reasonable)
        # Bonus for keeping base HP high (strategic positioning)
        if self.game_state.base_hp > 80:
            reward += 0.01  # Reduced from 0.1
        self.last_base_hp = self.game_state.base_hp

        soldier_nk_attacks_this_step = self.game_state.stats.get('soldier_nk_attacks', 0) - self.last_soldier_nk_attacks
        reward -= soldier_nk_attacks_this_step * 10.0
        self.last_soldier_nk_attacks = self.game_state.stats.get('soldier_nk_attacks', 0)

        soldiers_killed_by_nk_this_step = self.game_state.stats.get('soldiers_killed_by_nk_sweep', 0) - self.last_soldiers_killed_by_nk_sweep
        reward -= soldiers_killed_by_nk_this_step * 15.0
        self.last_soldiers_killed_by_nk_sweep = self.game_state.stats.get('soldiers_killed_by_nk_sweep', 0)

        # Small bonus for keeping soldiers alive when Night Kings are active
        if self.game_state.combat_phase and len(self.game_state.nks) > 0:
            soldiers_alive = len(self.game_state.soldiers)
            reward += soldiers_alive * 0.2  # Reward for surviving NK threats

        # Base defense: punish soldiers that leave base undefended
        # If enemies are close to base, soldiers should stay nearby to defend
        soldiers_far_from_base = self.game_state.stats.get('soldiers_far_from_base', 0)
        enemies_near_base = self.game_state.stats.get('enemies_near_base', 0)

        # When enemies are threatening, soldiers should be defending
        if enemies_near_base > 0:
            # Bad: soldiers leaving base when enemies are near
            soldiers_far_this_step = soldiers_far_from_base - self.last_soldiers_far_from_base
            if soldiers_far_this_step > 0:
                reward -= soldiers_far_this_step * 3.0  # Penalty for leaving base undefended

            # Good: soldiers staying near base to defend
            soldiers_near_base = len(self.game_state.soldiers) - soldiers_far_from_base
            if soldiers_near_base > 0:
                reward += soldiers_near_base * 0.5  # Bonus for defending

        # Extra penalty if lots of enemies are near and most soldiers are far away
        if enemies_near_base >= 3:  # Multiple enemies near base
            if soldiers_far_from_base > len(self.game_state.soldiers) / 2:  # More than half soldiers far
                reward -= 5.0  # Big penalty - base is vulnerable!

        self.last_soldiers_far_from_base = soldiers_far_from_base
        self.last_enemies_near_base = enemies_near_base

        # Penalty for redundant targeting (multiple units attacking same enemy)
        if self.game_state.combat_phase:
            # Track which enemies are being targeted
            target_counts = {}  # enemy_id -> count of units targeting it

            # Count soldier targets
            for soldier in self.game_state.soldiers:
                if hasattr(soldier, 'current_target') and soldier.current_target:
                    enemy_id = id(soldier.current_target)
                    target_counts[enemy_id] = target_counts.get(enemy_id, 0) + 1

            # Count hero targets
            for hero in self.game_state.heroes:
                if hero.active and hasattr(hero, 'target') and hero.target:
                    enemy_id = id(hero.target)
                    target_counts[enemy_id] = target_counts.get(enemy_id, 0) + 1

            # Apply penalty for redundant targeting (2+ units on same enemy)
            redundant_targets = sum(max(0, count - 1) for count in target_counts.values())
            if redundant_targets > 0:
                reward -= redundant_targets * 0.1  # Reduced from 0.5

        # Clip reward to prevent explosion
        reward = np.clip(reward, -50.0, 50.0)
        return reward

    def _calculate_final_reward(self) -> float:
        """Calculate final reward at end of episode - NORMALIZED for stable Q-learning"""
        reward = 0.0

        if self.game_state.victory:
            # Victory bonus - NORMALIZED to prevent reward explosion
            reward += 300.0  # Normalized from 10000.0
            hp_ratio = self.game_state.base_hp / BASE_HP_MAX
            reward += hp_ratio * 50.0  # Normalized from 2000.0
            soldiers_alive = len(self.game_state.soldiers)
            reward += soldiers_alive * 15.0  # Bonus for keeping soldiers alive - we have 6 max now
        else:
            # Defeat penalty - NORMALIZED
            reward -= 300.0  # Normalized from 500.0
            reward += self.game_state.current_wave * 10.0  # Normalized from 200.0
            # Bonus for getting close to victory (3/4 NK kills)
            nk_kills = self.game_state.stats.get('nk_kills', 0)
            if nk_kills >= 3:
                reward += 50.0  # Normalized from 500.0

        # Clip final reward to prevent explosion
        reward = np.clip(reward, -300.0, 400.0)
        return reward

    def _get_observation(self) -> Dict:
        """Get current observation"""
        grid = self._create_grid_observation()

        soldiers_remaining = SOLDIER_MAX - len(self.game_state.soldiers)  # Can place up to 6 soldiers now
        heroes_remaining = 2 - len(self.game_state.heroes)
        base_hp_ratio = self.game_state.base_hp / BASE_HP_MAX
        current_wave = self.game_state.current_wave
        night_king_active = 1 if len(self.game_state.nks) > 0 else 0

        return {
            'grid': grid,
            'soldiers_remaining': np.array([soldiers_remaining], dtype=np.int32),
            'heroes_remaining': np.array([heroes_remaining], dtype=np.int32),
            'base_hp_ratio': np.array([base_hp_ratio], dtype=np.float32),
            'current_wave': np.array([current_wave], dtype=np.int32),
            'night_king_active': np.array([night_king_active], dtype=np.int32)
        }

    def _create_grid_observation(self) -> np.ndarray:
        """Create 32x32 grid representation of game state.

        Grid values: 0=empty, 1=base, 2=soldier, 3=wight, 4=Night King, 5=hero
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

        # Add heroes
        for hero in self.game_state.heroes:
            if hero.active:
                grid_x = int((hero.x / SCREEN_W) * self.GRID_SIZE)
                grid_y = int((hero.y / SCREEN_H) * self.GRID_SIZE)
                grid_x = np.clip(grid_x, 0, self.GRID_SIZE - 1)
                grid_y = np.clip(grid_y, 0, self.GRID_SIZE - 1)
                grid[grid_y, grid_x] = 5

        # Add enemies
        for enemy in self.game_state.enemies:
            if enemy.alive and enemy.spawn_delay <= 0:
                ex, ey = enemy.pos()
                grid_x = int((ex / SCREEN_W) * self.GRID_SIZE)
                grid_y = int((ey / SCREEN_H) * self.GRID_SIZE)
                if 0 <= grid_x < self.GRID_SIZE and 0 <= grid_y < self.GRID_SIZE:
                    if grid[grid_y, grid_x] == 0:
                        grid[grid_y, grid_x] = 3

        # Add Night Kings (override other entities)
        for nk in self.game_state.nks:
            if nk.alive:
                nk_x, nk_y = nk.pos()
                grid_x = int((nk_x / SCREEN_W) * self.GRID_SIZE)
                grid_y = int((nk_y / SCREEN_H) * self.GRID_SIZE)
                if 0 <= grid_x < self.GRID_SIZE and 0 <= grid_y < self.GRID_SIZE:
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
            'victory': self.game_state.victory,
            'soldiers_alive': len(self.game_state.soldiers),
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
            self.big_font = pygame.font.SysFont("Georgia", 36, bold=True)
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

        # Draw burn effects (wight death flashes)
        for burn in self.game_state.burns:
            burn.draw(self.screen)

        # Draw base hit effects (before base so base appears on top)
        for hit_effect in self.game_state.base_hit_effects:
            hit_effect.draw(self.screen, BASE_POS, BASE_RADIUS)

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
            f"Wights Killed: {self.game_state.stats['wights_killed']}   "
            f"NKs Defeated: {self.game_state.stats['nk_kills']}/{len(game.NK_SCHEDULE_TIMES)}   "
            f"Time: {int(self.game_state.runtime)}s",
            True,
            (220, 220, 200),
        )
        self.screen.blit(killed_hud, (14, 68))

        # Phase indicator
        phase_text = "PLACEMENT PHASE" if self.game_state.placement_phase else "COMBAT PHASE"
        phase_color = (100, 200, 100) if self.game_state.placement_phase else (200, 100, 100)
        phase_surf = self.font_small.render(phase_text, True, phase_color)
        self.screen.blit(phase_surf, (SCREEN_W - phase_surf.get_width() - 14, 12))

        # Game over / victory overlay
        if self.game_state.is_game_over():
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 190))
            self.screen.blit(overlay, (0, 0))
            victory = self.game_state.victory
            title = self.big_font.render(
                "WINTERFELL PREVAILS" if victory else "WINTERFELL HAS FALLEN",
                True,
                (200, 230, 210) if victory else (255, 190, 180)
            )
            self.screen.blit(title, (SCREEN_W / 2 - title.get_width() / 2, SCREEN_H / 2 - 220))
            mid_font = pygame.font.SysFont("Arial", 24)
            lines = [
                f"Time Survived: {int(self.game_state.runtime)} s",
                f"Soldiers Deployed: {self.game_state.stats['soldiers_deployed']}",
                f"Soldiers Killed: {self.game_state.stats['soldiers_killed']}",
                f"Wights Killed: {self.game_state.stats['wights_killed']}",
                f"Night Kings Defeated: {self.game_state.stats['nk_kills']}/{len(game.NK_SCHEDULE_TIMES)}",
                "",
                "Press R to Restart • Press Q to Quit"
            ]
            y = SCREEN_H / 2 - 60
            for line in lines:
                txt = mid_font.render(line, True, (240, 240, 240))
                self.screen.blit(txt, (SCREEN_W / 2 - txt.get_width() / 2, y))
                y += 36

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
            # Set window title based on agent type
            if self.agent_type:
                caption = f"Tower Defense - {self.agent_type} Agent"
            else:
                caption = "Tower Defense - RL Agent"
            pygame.display.set_caption(caption)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Verdana", 16)
            self.small_font = pygame.font.SysFont("Arial", 12)
            self.font_small = pygame.font.Font(None, 24)
            self.font_tiny = pygame.font.Font(None, 18)
            self.big_font = pygame.font.SysFont("Georgia", 36, bold=True)
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
