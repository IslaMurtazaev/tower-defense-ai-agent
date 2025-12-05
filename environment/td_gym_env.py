"""
Gymnasium Wrapper for Tower Defense Game
For training RL agents
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from td_game_core import (
    TowerDefenseGame, SoldierType, GamePhase, Position
)


class TowerDefenseEnv(gym.Env):
    """
    Gymnasium environment for Tower Defense game.
    
    Observation Space:
        - Grid representation (32x32) showing:
          - Empty cells (0)
          - Castle (1)
          - Footmen (2)
          - Archers (3)
          - Wights (4)
        - Additional features:
          - Soldiers remaining to place
          - Current wave number
          - Castle HP ratio
    
    Action Space:
        - MultiDiscrete([2, 32, 32]):
          - Soldier type (0=Footman, 1=Archer)
          - Grid X position (0-31)
          - Grid Y position (0-31)
    
    Episode Structure:
        1. Placement phase: Agent takes MAX_SOLDIERS actions to place soldiers
        2. Combat phase: Game auto-runs until completion
        3. Episode ends with final reward based on performance
    """
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}
    
    # Grid size for observation
    GRID_SIZE = 32
    
    def __init__(self, render_mode: Optional[str] = None, fast_mode: bool = True):
        super().__init__()
        
        self.render_mode = render_mode
        self.fast_mode = fast_mode  # Run simulation faster
        
        # Create game instance
        self.game = TowerDefenseGame()
        
        # Define observation space
        # Grid (32x32) + additional features (3 values)
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0, high=4, shape=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32
            ),
            'soldiers_remaining': spaces.Box(
                low=0, high=TowerDefenseGame.MAX_SOLDIERS, shape=(1,), dtype=np.int32
            ),
            'current_wave': spaces.Box(
                low=0, high=len(TowerDefenseGame.WAVE_DEFINITIONS), shape=(1,), dtype=np.int32
            ),
            'castle_hp_ratio': spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        })
        
        # Define action space
        # [soldier_type, grid_x, grid_y]
        self.action_space = spaces.MultiDiscrete([2, self.GRID_SIZE, self.GRID_SIZE])
        
        # Episode tracking
        self.placement_actions_taken = 0
        self.episode_reward = 0.0
        
        # Reward tracking
        self.last_wights_killed = 0
        self.last_soldiers_killed = 0
        self.last_castle_hp = self.game.castle.hp
        
        # For rendering
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font_small = None
        self.font_medium = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game
        self.game.reset()
        
        # Reset tracking
        self.placement_actions_taken = 0
        self.episode_reward = 0.0
        self.last_wights_killed = 0
        self.last_soldiers_killed = 0
        self.last_castle_hp = self.game.castle.hp
        
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
        if self.game.phase == GamePhase.PLACEMENT:
            soldier_type_idx, grid_x, grid_y = action
            
            # Convert grid coordinates to world coordinates
            world_x = (grid_x / self.GRID_SIZE) * TowerDefenseGame.WIDTH
            world_y = (grid_y / self.GRID_SIZE) * TowerDefenseGame.HEIGHT
            
            # Select soldier type
            soldier_type = SoldierType.FOOTMAN if soldier_type_idx == 0 else SoldierType.ARCHER
            
            # Try to place soldier
            success = self.game.place_soldier(soldier_type, Position(world_x, world_y))
            
            # Small penalty for invalid placements
            if not success:
                reward -= 1.0
            
            self.placement_actions_taken += 1
            
            # If we've placed all soldiers (or reached max), start combat
            if self.placement_actions_taken >= TowerDefenseGame.MAX_SOLDIERS or not self.game.can_place_soldier():
                self.game.start_combat_phase()
        
        # Combat phase - auto-run the game
        if self.game.phase == GamePhase.COMBAT:
            # Run combat faster in fast mode
            if self.fast_mode:
                # Run multiple updates per step for faster training
                updates_per_step = 5  # Simulate 5 frames at once
                dt = 1.0 / 60.0
                for _ in range(updates_per_step):
                    self.game.update(dt)
                    if self.game.is_game_over():
                        break
            else:
                # Normal speed for visualization
                dt = 1.0 / 60.0
                self.game.update(dt)
            
            # Calculate rewards based on state changes
            reward += self._calculate_step_reward()
        
        # Check if game is over
        if self.game.is_game_over():
            terminated = True
            reward += self._calculate_final_reward()
        
        self.episode_reward += reward
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_step_reward(self) -> float:
        """Calculate reward for a single combat step"""
        reward = 0.0
        stats = self.game.stats
        
        # Reward for killing wights
        wights_killed_this_step = stats['wights_killed'] - self.last_wights_killed
        reward += wights_killed_this_step * 10.0  # +10 per wight killed
        self.last_wights_killed = stats['wights_killed']
        
        # SIGNIFICANT penalty for losing soldiers (new mechanic!)
        soldiers_killed_this_step = stats['soldiers_killed'] - self.last_soldiers_killed
        reward -= soldiers_killed_this_step * 30.0  # -30 per soldier lost
        self.last_soldiers_killed = stats['soldiers_killed']
        
        # Penalty for castle taking damage (INCREASED from -5 to -10)
        castle_damage_this_step = self.last_castle_hp - self.game.castle.hp
        reward -= castle_damage_this_step * 10.0
        self.last_castle_hp = self.game.castle.hp
        
        return reward
    
    def _calculate_final_reward(self) -> float:
        """Calculate final reward at end of episode"""
        reward = 0.0
        
        if self.game.phase == GamePhase.VICTORY:
            # Victory bonus
            reward += 500.0
            
            # Bonus for castle HP remaining
            hp_ratio = self.game.castle.hp / self.game.castle.max_hp
            reward += hp_ratio * 100.0
            
            # MAJOR bonus for soldiers surviving (preservation reward)
            soldiers_alive = sum(1 for s in self.game.soldiers if s.alive)
            reward += soldiers_alive * 100.0  # +100 per soldier alive at end
            
            # Additional bonus for soldier HP remaining (health preservation)
            total_soldier_hp = sum(s.hp for s in self.game.soldiers if s.alive)
            reward += total_soldier_hp * 2.0  # +2 per HP point
        else:
            # Defeat penalty
            reward -= 200.0
            
            # Partial credit for waves completed
            reward += self.game.stats['waves_completed'] * 50.0
        
        return reward
    
    def _get_observation(self) -> Dict:
        """Get current observation"""
        grid = self._create_grid_observation()
        
        soldiers_remaining = TowerDefenseGame.MAX_SOLDIERS - len(self.game.soldiers)
        current_wave = self.game.current_wave if self.game.phase == GamePhase.COMBAT else 0
        castle_hp_ratio = self.game.castle.hp / self.game.castle.max_hp
        
        return {
            'grid': grid,
            'soldiers_remaining': np.array([soldiers_remaining], dtype=np.int32),
            'current_wave': np.array([current_wave], dtype=np.int32),
            'castle_hp_ratio': np.array([castle_hp_ratio], dtype=np.float32)
        }
    
    def _create_grid_observation(self) -> np.ndarray:
        """
        Create grid representation of game state.
        
        Values:
            0 = Empty
            1 = Castle
            2 = Footman
            3 = Archer
            4 = Wight
        """
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        
        # Add castle
        castle_grid_x = int((self.game.castle.position.x / TowerDefenseGame.WIDTH) * self.GRID_SIZE)
        castle_grid_y = int((self.game.castle.position.y / TowerDefenseGame.HEIGHT) * self.GRID_SIZE)
        castle_grid_x = np.clip(castle_grid_x, 0, self.GRID_SIZE - 1)
        castle_grid_y = np.clip(castle_grid_y, 0, self.GRID_SIZE - 1)
        
        # Castle occupies 2x2 area
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                gx = castle_grid_x + dx
                gy = castle_grid_y + dy
                if 0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE:
                    grid[gy, gx] = 1
        
        # Add soldiers
        for soldier in self.game.soldiers:
            if soldier.alive:
                grid_x = int((soldier.position.x / TowerDefenseGame.WIDTH) * self.GRID_SIZE)
                grid_y = int((soldier.position.y / TowerDefenseGame.HEIGHT) * self.GRID_SIZE)
                grid_x = np.clip(grid_x, 0, self.GRID_SIZE - 1)
                grid_y = np.clip(grid_y, 0, self.GRID_SIZE - 1)
                
                value = 2 if soldier.type == SoldierType.FOOTMAN else 3
                grid[grid_y, grid_x] = value
        
        # Add wights
        for wight in self.game.wights:
            if wight.alive:
                grid_x = int((wight.position.x / TowerDefenseGame.WIDTH) * self.GRID_SIZE)
                grid_y = int((wight.position.y / TowerDefenseGame.HEIGHT) * self.GRID_SIZE)
                grid_x = np.clip(grid_x, 0, self.GRID_SIZE - 1)
                grid_y = np.clip(grid_y, 0, self.GRID_SIZE - 1)
                
                grid[grid_y, grid_x] = 4
        
        return grid
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        state = self.game.get_state()
        return {
            'phase': state['phase'].value,
            'game_time': state['game_time'],
            'current_wave': state['current_wave'],
            'stats': state['stats'],
            'episode_reward': self.episode_reward,
            'placement_actions_taken': self.placement_actions_taken
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
        """Render as RGB array with full game details"""
        import pygame
        
        # Initialize pygame if needed (for rgb_array mode)
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.Surface((TowerDefenseGame.WIDTH, TowerDefenseGame.HEIGHT))
            self.font_small = pygame.font.Font(None, 24)
            self.font_medium = pygame.font.Font(None, 32)
            self.pygame_initialized = True
        
        # Colors
        COLOR_BG = (20, 20, 30)
        COLOR_CASTLE = (100, 100, 120)
        COLOR_FOOTMAN = (50, 100, 200)
        COLOR_ARCHER = (50, 200, 100)
        COLOR_WIGHT = (200, 50, 50)
        COLOR_TEXT = (255, 255, 255)
        COLOR_HP_BAR_BG = (60, 60, 60)
        COLOR_HP_BAR = (0, 200, 0)
        COLOR_HP_BAR_LOW = (255, 165, 0)
        COLOR_HP_BAR_CRITICAL = (255, 50, 50)
        
        # Clear screen
        self.screen.fill(COLOR_BG)
        
        # Draw castle with HP
        castle_pos = self.game.castle.position
        castle_width, castle_height = 120, 80
        
        # Castle structure
        castle_rect = pygame.Rect(
            castle_pos.x - castle_width // 2,
            castle_pos.y - castle_height // 2,
            castle_width, castle_height
        )
        pygame.draw.rect(self.screen, COLOR_CASTLE, castle_rect)
        pygame.draw.rect(self.screen, COLOR_TEXT, castle_rect, 2)
        
        # Castle towers
        tower_size = 20
        pygame.draw.rect(self.screen, COLOR_CASTLE,
                        (castle_pos.x - castle_width // 2 - 10, 
                         castle_pos.y - castle_height // 2 - 10,
                         tower_size, tower_size + 10))
        pygame.draw.rect(self.screen, COLOR_CASTLE,
                        (castle_pos.x + castle_width // 2 - 10,
                         castle_pos.y - castle_height // 2 - 10,
                         tower_size, tower_size + 10))
        
        # Castle HP bar
        hp_bar_width = castle_width
        hp_bar_height = 10
        hp_bar_x = castle_pos.x - hp_bar_width // 2
        hp_bar_y = castle_pos.y + castle_height // 2 + 10
        
        pygame.draw.rect(self.screen, COLOR_HP_BAR_BG,
                        (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
        
        hp_ratio = self.game.castle.hp / self.game.castle.max_hp
        # Color based on HP
        if hp_ratio > 0.6:
            hp_color = COLOR_HP_BAR
        elif hp_ratio > 0.3:
            hp_color = COLOR_HP_BAR_LOW
        else:
            hp_color = COLOR_HP_BAR_CRITICAL
            
        pygame.draw.rect(self.screen, hp_color,
                        (hp_bar_x, hp_bar_y, int(hp_bar_width * hp_ratio), hp_bar_height))
        
        # Castle HP text
        hp_text = self.font_small.render(
            f"Castle: {self.game.castle.hp}/{self.game.castle.max_hp} HP",
            True, COLOR_TEXT
        )
        self.screen.blit(hp_text, (hp_bar_x, hp_bar_y + hp_bar_height + 3))
        
        # Draw soldiers with type distinction
        for soldier in self.game.soldiers:
            if soldier.alive:
                pos = soldier.position
                
                if soldier.type == SoldierType.FOOTMAN:
                    # Footman: Blue circle
                    pygame.draw.circle(self.screen, COLOR_FOOTMAN,
                                     (int(pos.x), int(pos.y)), 12)
                    pygame.draw.circle(self.screen, COLOR_TEXT,
                                     (int(pos.x), int(pos.y)), 12, 2)
                else:
                    # Archer: Green triangle
                    points = [
                        (pos.x, pos.y - 12),
                        (pos.x - 10, pos.y + 8),
                        (pos.x + 10, pos.y + 8)
                    ]
                    pygame.draw.polygon(self.screen, COLOR_ARCHER, points)
                    pygame.draw.polygon(self.screen, COLOR_TEXT, points, 2)
        
        # Draw wights with HP bars
        for wight in self.game.wights:
            if wight.alive:
                pos = wight.position
                
                # Wight body (red square)
                size = 14
                rect = pygame.Rect(int(pos.x - size // 2), int(pos.y - size // 2), size, size)
                pygame.draw.rect(self.screen, COLOR_WIGHT, rect)
                pygame.draw.rect(self.screen, COLOR_TEXT, rect, 2)
                
                # Wight HP bar
                hp_bar_width = 20
                hp_bar_height = 3
                hp_bar_x = pos.x - hp_bar_width // 2
                hp_bar_y = pos.y - 20
                
                pygame.draw.rect(self.screen, COLOR_HP_BAR_BG,
                               (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
                
                wight_hp_ratio = wight.hp / wight.max_hp
                pygame.draw.rect(self.screen, COLOR_HP_BAR,
                               (hp_bar_x, hp_bar_y, int(hp_bar_width * wight_hp_ratio), hp_bar_height))
        
        # Draw UI overlay
        stats = self.game.stats
        ui_lines = [
            f"AI Agent Playing",
            f"Wave: {self.game.current_wave + 1}/5",
            f"Soldiers: {sum(1 for s in self.game.soldiers if s.alive)}/{len(self.game.soldiers)}",
            f"Wights Killed: {stats['wights_killed']}/300",
            f"Phase: {self.game.phase.value.upper()}"
        ]
        
        # Draw semi-transparent background for UI
        ui_bg = pygame.Surface((250, 150))
        ui_bg.set_alpha(180)
        ui_bg.fill((0, 0, 0))
        self.screen.blit(ui_bg, (10, 10))
        
        # Draw UI text
        y_offset = 20
        for line in ui_lines:
            text = self.font_small.render(line, True, COLOR_TEXT)
            self.screen.blit(text, (20, y_offset))
            y_offset += 28
        
        # Draw legend
        legend_y = TowerDefenseGame.HEIGHT - 100
        legend_bg = pygame.Surface((200, 90))
        legend_bg.set_alpha(180)
        legend_bg.fill((0, 0, 0))
        self.screen.blit(legend_bg, (10, legend_y))
        
        # Footman icon
        pygame.draw.circle(self.screen, COLOR_FOOTMAN, (30, legend_y + 20), 10)
        legend_text = self.font_small.render("Footman (Melee)", True, COLOR_TEXT)
        self.screen.blit(legend_text, (50, legend_y + 12))
        
        # Archer icon
        points = [(30, legend_y + 43), (20, legend_y + 58), (40, legend_y + 58)]
        pygame.draw.polygon(self.screen, COLOR_ARCHER, points)
        legend_text = self.font_small.render("Archer (Ranged)", True, COLOR_TEXT)
        self.screen.blit(legend_text, (50, legend_y + 45))
        
        # Convert to numpy array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
    
    def _render_human(self):
        """Render for human viewing"""
        if not self.pygame_initialized:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((TowerDefenseGame.WIDTH, TowerDefenseGame.HEIGHT))
            pygame.display.set_caption("Tower Defense - RL Agent")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 24)
            self.font_medium = pygame.font.Font(None, 32)
            self.pygame_initialized = True
        
        # Get RGB array and display it
        rgb_array = self._render_rgb_array()
        import pygame
        surf = pygame.surfarray.make_surface(np.transpose(rgb_array, axes=(1, 0, 2)))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        """Clean up"""
        if self.pygame_initialized:
            import pygame
            pygame.quit()
            self.pygame_initialized = False


def test_environment():
    """Test the environment with random actions"""
    print("Testing Tower Defense Gymnasium Environment...")
    
    env = TowerDefenseEnv()
    
    # Test reset
    observation, info = env.reset(seed=42)
    print(f"Initial observation shape: grid={observation['grid'].shape}")
    print(f"Initial info: {info}")
    
    # Test random episode
    episode_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < 2000:
        # Random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        done = terminated or truncated
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Step {step_count}: Phase={info['phase']}, Reward={reward:.2f}, Total={episode_reward:.2f}")
    
    print(f"\nEpisode finished after {step_count} steps")
    print(f"Final phase: {info['phase']}")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Stats: {info['stats']}")
    
    env.close()
    print("\nEnvironment test completed successfully!")


if __name__ == "__main__":
    test_environment()

