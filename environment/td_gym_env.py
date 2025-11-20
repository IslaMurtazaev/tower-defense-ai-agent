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
    
    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_combat_steps = max_steps
        
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
        self.combat_steps = 0
        self.episode_reward = 0.0
        
        # Reward tracking
        self.last_wights_killed = 0
        self.last_soldiers_killed = 0
        self.last_castle_hp = self.game.castle.hp
        
        # For rendering
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game
        self.game.reset()
        
        # Reset tracking
        self.placement_actions_taken = 0
        self.combat_steps = 0
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
            # Run combat for a time step
            dt = 1.0 / 60.0  # 60 FPS simulation
            self.game.update(dt)
            self.combat_steps += 1
            
            # Calculate rewards based on state changes
            reward += self._calculate_step_reward()
            
            # Check if combat is too long
            if self.combat_steps >= self.max_combat_steps:
                truncated = True
                reward -= 100  # Penalty for taking too long
        
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
        reward += wights_killed_this_step * 10.0
        self.last_wights_killed = stats['wights_killed']
        
        # Penalty for losing soldiers (currently no soldier death mechanic, but keeping for future)
        soldiers_killed_this_step = stats['soldiers_killed'] - self.last_soldiers_killed
        reward -= soldiers_killed_this_step * 50.0
        self.last_soldiers_killed = stats['soldiers_killed']
        
        # Penalty for castle taking damage
        castle_damage_this_step = self.last_castle_hp - self.game.castle.hp
        reward -= castle_damage_this_step * 5.0
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
            reward += hp_ratio * 200.0
            
            # Bonus for soldiers surviving (if applicable)
            soldiers_alive = sum(1 for s in self.game.soldiers if s.alive)
            reward += soldiers_alive * 10.0
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
            'placement_actions_taken': self.placement_actions_taken,
            'combat_steps': self.combat_steps
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
        """Render as RGB array"""
        # Initialize pygame if needed
        if not self.pygame_initialized:
            import pygame
            pygame.init()
            self.screen = pygame.Surface((TowerDefenseGame.WIDTH, TowerDefenseGame.HEIGHT))
            self.pygame_initialized = True
        
        # Simple rendering
        import pygame
        
        # Clear
        self.screen.fill((20, 20, 30))
        
        # Draw castle
        castle_pos = self.game.castle.position
        pygame.draw.rect(self.screen, (100, 100, 120),
                        (castle_pos.x - 60, castle_pos.y - 40, 120, 80))
        
        # Draw soldiers
        for soldier in self.game.soldiers:
            if soldier.alive:
                color = (50, 100, 200) if soldier.type == SoldierType.FOOTMAN else (50, 200, 100)
                pygame.draw.circle(self.screen, color,
                                 (int(soldier.position.x), int(soldier.position.y)), 10)
        
        # Draw wights
        for wight in self.game.wights:
            if wight.alive:
                pygame.draw.rect(self.screen, (200, 50, 50),
                               (int(wight.position.x) - 7, int(wight.position.y) - 7, 14, 14))
        
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

