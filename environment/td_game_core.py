"""
Tower Defense Game Core Logic
No rendering - pure game state and mechanics
"""
import math
import random
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
import heapq


class GamePhase(Enum):
    PLACEMENT = "placement"
    COMBAT = "combat"
    VICTORY = "victory"
    DEFEAT = "defeat"


class SoldierType(Enum):
    FOOTMAN = "footman"
    ARCHER = "archer"


@dataclass
class Position:
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def move_towards(self, target: 'Position', speed: float) -> 'Position':
        """Move towards target at given speed, return new position"""
        dist = self.distance_to(target)
        if dist <= speed:
            return Position(target.x, target.y)
        
        ratio = speed / dist
        new_x = self.x + (target.x - self.x) * ratio
        new_y = self.y + (target.y - self.y) * ratio
        return Position(new_x, new_y)
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return abs(self.x - other.x) < 0.1 and abs(self.y - other.y) < 0.1
    
    def __hash__(self):
        return hash((round(self.x, 1), round(self.y, 1)))


class AStarPathfinder:
    """A* pathfinding for soldiers"""
    
    @staticmethod
    def find_path(start: Position, goal: Position, obstacles: List[Position] = None, grid_size: int = 20) -> List[Position]:
        """
        Find path from start to goal using A* algorithm.
        Uses a simplified grid-based approach.
        """
        if obstacles is None:
            obstacles = []
        
        # Simple direct path if close enough
        if start.distance_to(goal) < grid_size * 2:
            return [goal]
        
        # Convert to grid coordinates
        def to_grid(pos: Position) -> Tuple[int, int]:
            return (int(pos.x / grid_size), int(pos.y / grid_size))
        
        def from_grid(gx: int, gy: int) -> Position:
            return Position(gx * grid_size + grid_size / 2, gy * grid_size + grid_size / 2)
        
        start_grid = to_grid(start)
        goal_grid = to_grid(goal)
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: AStarPathfinder._heuristic(start_grid, goal_grid)}
        
        closed_set = set()
        max_iterations = 200
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(from_grid(current[0], current[1]))
                    current = came_from[current]
                path.reverse()
                path.append(goal)
                return path if path else [goal]
            
            closed_set.add(current)
            
            # Check neighbors (8 directions)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Bounds check (simplified)
                if neighbor[0] < 0 or neighbor[1] < 0 or neighbor[0] > 64 or neighbor[1] > 40:
                    continue
                
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + (1.414 if dx != 0 and dy != 0 else 1)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + AStarPathfinder._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found, return direct
        return [goal]
    
    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class Soldier:
    """Soldier class with movement and A* pathfinding"""
    
    def __init__(self, soldier_type: SoldierType, position: Position):
        self.type = soldier_type
        self.position = position
        self.home_position = Position(position.x, position.y)  # Remember starting position
        self.alive = True
        self.last_attack_time = 0.0
        
        # Stats based on type
        if soldier_type == SoldierType.FOOTMAN:
            self.hp = 200
            self.max_hp = 200
            self.attack_range = 30
            self.damage = 30
            self.attack_speed = 0.5  # seconds between attacks (FASTER - was 1.0)
            self.detection_radius = 100  # VERY SHORT range - only engage very close enemies
            self.move_speed = 50  # pixels per second
            self.is_static = False  # Mobile unit
        else:  # ARCHER
            self.hp = 50
            self.max_hp = 50
            self.attack_range = 400
            self.damage = 10
            self.attack_speed = 0.9
            self.detection_radius = 400  # Long range - matches attack range
            self.move_speed = 0  # STATIC - doesn't move
            self.is_static = True  # Static unit
        
        # Movement state
        self.current_path: List[Position] = []
        self.current_target: Optional['Wight'] = None
        self.is_returning_home = False
        self.path_update_timer = 0.0
        self.path_update_interval = 0.3  # Recalculate path every 0.3 seconds
        self.target_last_position: Optional[Position] = None
    
    def take_damage(self, damage: int):
        """Take damage and check if dead"""
        self.hp -= damage
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
    
    def can_attack(self, current_time: float) -> bool:
        """Check if enough time has passed to attack again"""
        return current_time - self.last_attack_time >= self.attack_speed
    
    def attack(self, target: 'Wight', current_time: float) -> bool:
        """Attack target if in range and ready. Returns True if attack happened."""
        if not self.alive or not target.alive:
            return False
        
        if self.position.distance_to(target.position) <= self.attack_range:
            if self.can_attack(current_time):
                target.take_damage(self.damage)
                self.last_attack_time = current_time
                return True
        return False
    
    def find_nearest_enemy(self, wights: List['Wight']) -> Optional['Wight']:
        """Find nearest alive wight within detection radius"""
        nearest = None
        min_dist = float('inf')
        
        for wight in wights:
            if wight.alive:
                dist = self.position.distance_to(wight.position)
                if dist <= self.detection_radius and dist < min_dist:
                    min_dist = dist
                    nearest = wight
        
        return nearest
    
    def update(self, dt: float, current_time: float, wights: List['Wight']):
        """Update soldier state - movement and combat"""
        if not self.alive:
            return
        
        # Find nearest enemy
        nearest_enemy = self.find_nearest_enemy(wights)
        
        if nearest_enemy:
            # Track current target
            self.current_target = nearest_enemy
            
            # Check if in attack range
            distance_to_enemy = self.position.distance_to(nearest_enemy.position)
            
            if distance_to_enemy <= self.attack_range:
                # In range - attack!
                self.attack(nearest_enemy, current_time)
            else:
                # Not in range
                if self.is_static:
                    # ARCHERS: Static units don't move, just wait for enemies in range
                    pass
                else:
                    # FOOTMEN: Mobile units move toward enemy
                    self.is_returning_home = False
                    self.path_update_timer += dt
                    
                    # Check if target moved significantly
                    target_moved = False
                    if self.target_last_position:
                        distance_moved = self.target_last_position.distance_to(nearest_enemy.position)
                        target_moved = distance_moved > 50  # Enemy moved more than 50 pixels
                    
                    # Check if we need a new path (fixed latency issue)
                    needs_new_path = (
                        not self.current_path or
                        self.path_update_timer >= self.path_update_interval or  # Time-based update
                        target_moved  # Distance-based update
                    )
                    
                    if needs_new_path:
                        self.current_path = AStarPathfinder.find_path(self.position, nearest_enemy.position)
                        self.target_last_position = Position(nearest_enemy.position.x, nearest_enemy.position.y)
                        self.path_update_timer = 0.0
                    
                    # Move towards enemy
                    if self.current_path:
                        next_waypoint = self.current_path[0]
                        move_distance = self.move_speed * dt
                        self.position = self.position.move_towards(next_waypoint, move_distance)
                        
                        # Check if reached waypoint
                        if self.position.distance_to(next_waypoint) < 5:
                            self.current_path.pop(0)
                    else:
                        # Move directly toward enemy
                        move_distance = self.move_speed * dt
                        self.position = self.position.move_towards(nearest_enemy.position, move_distance)
        else:
            # No enemies nearby
            self.current_target = None
            
            if self.is_static:
                # ARCHERS: Static units stay at home position (don't need to return)
                self.is_returning_home = False
            else:
                # FOOTMEN: Return to home position
                self.current_path = []
                self.target_last_position = None
                self.path_update_timer = 0.0
                
                distance_to_home = self.position.distance_to(self.home_position)
                if distance_to_home > 5:
                    self.is_returning_home = True
                    move_distance = self.move_speed * dt
                    self.position = self.position.move_towards(self.home_position, move_distance)
                else:
                    self.is_returning_home = False


class Wight:
    """Enemy wight class"""
    
    def __init__(self, spawn_position: Position, target_position: Position):
        self.position = spawn_position
        self.target = target_position
        self.hp = 30
        self.max_hp = 30
        self.speed = 50  # pixels per second - FASTER (was 30)
        self.castle_damage = 10  # Damage to castle
        self.soldier_damage = 10  # Damage to soldiers
        self.alive = True
        self.last_attack_time = 0.0
        self.attack_speed = 1.5  # seconds between attacks
        self.detection_radius = 100  # How far they can detect soldiers
        self.current_soldier_target: Optional[Soldier] = None
    
    def take_damage(self, damage: int):
        """Take damage and check if dead"""
        self.hp -= damage
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
    
    def find_nearest_soldier(self, soldiers: List[Soldier]) -> Optional[Soldier]:
        """Find nearest alive soldier within detection radius"""
        nearest = None
        min_dist = float('inf')
        
        for soldier in soldiers:
            if soldier.alive:
                dist = self.position.distance_to(soldier.position)
                if dist <= self.detection_radius and dist < min_dist:
                    min_dist = dist
                    nearest = soldier
        
        return nearest
    
    def can_attack(self, current_time: float) -> bool:
        """Check if enough time has passed to attack again"""
        return current_time - self.last_attack_time >= self.attack_speed
    
    def attack_soldier(self, soldier: Soldier, current_time: float) -> bool:
        """Attack soldier if ready. Returns True if attack happened."""
        if not self.alive or not soldier.alive:
            return False
        
        if self.can_attack(current_time):
            soldier.take_damage(self.soldier_damage)
            self.last_attack_time = current_time
            return True
        return False
    
    def update(self, dt: float, current_time: float, soldiers: List[Soldier]) -> bool:
        """Move towards target or engage soldiers. Returns True if reached castle."""
        if not self.alive:
            return False
        
        # Find nearest soldier
        nearest_soldier = self.find_nearest_soldier(soldiers)
        
        if nearest_soldier:
            # Engage soldier
            self.current_soldier_target = nearest_soldier
            distance_to_soldier = self.position.distance_to(nearest_soldier.position)
            
            if distance_to_soldier <= 30:  # Melee range
                # Stop and attack
                self.attack_soldier(nearest_soldier, current_time)
            else:
                # Move towards soldier
                move_distance = self.speed * dt
                self.position = self.position.move_towards(nearest_soldier.position, move_distance)
        else:
            # No soldiers nearby - move towards castle
            self.current_soldier_target = None
            move_distance = self.speed * dt
            self.position = self.position.move_towards(self.target, move_distance)
            
            # Check if reached castle (within 20 pixels)
            if self.position.distance_to(self.target) < 20:
                return True
        
        return False


class Castle:
    """The castle to defend"""
    
    def __init__(self, position: Position, hp: int = 100):
        self.position = position
        self.hp = hp
        self.max_hp = hp
    
    def take_damage(self, damage: int):
        """Take damage from wight"""
        self.hp -= damage
        if self.hp < 0:
            self.hp = 0
    
    def is_destroyed(self) -> bool:
        return self.hp <= 0


class TowerDefenseGame:
    """Main game logic class"""
    
    # Map dimensions
    WIDTH = 1280
    HEIGHT = 800
    
    # Game constants
    MAX_SOLDIERS = 10
    WAVE_DEFINITIONS = [25, 40, 60, 75, 100]  # Number of wights per wave (5x multiplier)
    TIME_BETWEEN_WAVES = 0.0  # seconds
    
    def __init__(self):
        # Game state
        self.phase = GamePhase.PLACEMENT
        self.game_time = 0.0
        
        # Castle at bottom center
        self.castle = Castle(Position(self.WIDTH / 2, self.HEIGHT - 100))
        
        # Game entities
        self.soldiers: List[Soldier] = []
        self.wights: List[Wight] = []
        
        # Wave management
        self.current_wave = 0
        self.wave_spawn_timer = 0.0
        self.wights_to_spawn_this_wave = 0
        self.spawn_interval = 0.05  # seconds between spawns - VERY FAST (burst spawning!)
        self.time_since_last_spawn = 0.0
        self.waiting_for_next_wave = False
        self.next_wave_timer = 0.0
        
        # Statistics
        self.stats = {
            'soldiers_placed': 0,
            'soldiers_killed': 0,
            'wights_killed': 0,
            'waves_completed': 0,
            'castle_damage_taken': 0
        }
    
    def can_place_soldier(self) -> bool:
        """Check if player can place another soldier"""
        return (self.phase == GamePhase.PLACEMENT and 
                len(self.soldiers) < self.MAX_SOLDIERS)
    
    def place_soldier(self, soldier_type: SoldierType, position: Position) -> bool:
        """Place a soldier at position. Returns True if successful."""
        if not self.can_place_soldier():
            return False
        
        # Check if position is valid (not too close to castle, within bounds)
        if position.distance_to(self.castle.position) < 80:
            return False
        
        if not (50 < position.x < self.WIDTH - 50 and 50 < position.y < self.HEIGHT - 150):
            return False
        
        soldier = Soldier(soldier_type, position)
        self.soldiers.append(soldier)
        self.stats['soldiers_placed'] += 1
        return True
    
    def start_combat_phase(self):
        """Transition from placement to combat"""
        if self.phase == GamePhase.PLACEMENT:
            self.phase = GamePhase.COMBAT
            self.current_wave = 0
            self._start_wave(0)
    
    def _start_wave(self, wave_index: int):
        """Start spawning a specific wave"""
        if wave_index >= len(self.WAVE_DEFINITIONS):
            return
        
        self.wights_to_spawn_this_wave = self.WAVE_DEFINITIONS[wave_index]
        self.time_since_last_spawn = 0.0
        self.waiting_for_next_wave = False
    
    def _spawn_wight(self):
        """Spawn a single wight at a random edge position"""
        # Random spawn from edges (top or sides)
        edge = random.choice(['top', 'left', 'right'])
        
        if edge == 'top':
            spawn_pos = Position(random.uniform(100, self.WIDTH - 100), 50)
        elif edge == 'left':
            spawn_pos = Position(50, random.uniform(50, self.HEIGHT / 2))
        else:  # right
            spawn_pos = Position(self.WIDTH - 50, random.uniform(50, self.HEIGHT / 2))
        
        wight = Wight(spawn_pos, self.castle.position)
        self.wights.append(wight)
    
    def update(self, dt: float):
        """Update game state. dt is delta time in seconds."""
        if self.phase not in [GamePhase.COMBAT]:
            return
        
        self.game_time += dt
        
        # Wave spawning logic
        if not self.waiting_for_next_wave:
            if self.wights_to_spawn_this_wave > 0:
                self.time_since_last_spawn += dt
                if self.time_since_last_spawn >= self.spawn_interval:
                    self._spawn_wight()
                    self.wights_to_spawn_this_wave -= 1
                    self.time_since_last_spawn = 0.0
                    
                    # If that was the last wight of this wave, start waiting
                    if self.wights_to_spawn_this_wave == 0:
                        self.waiting_for_next_wave = True
                        self.next_wave_timer = 0.0
        else:
            # Check if all wights are dead
            if not any(w.alive for w in self.wights):
                self.next_wave_timer += dt
                if self.next_wave_timer >= self.TIME_BETWEEN_WAVES:
                    # Start next wave
                    self.stats['waves_completed'] += 1
                    self.current_wave += 1
                    
                    if self.current_wave >= len(self.WAVE_DEFINITIONS):
                        # Victory!
                        self.phase = GamePhase.VICTORY
                        return
                    else:
                        self._start_wave(self.current_wave)
        
        # Update soldiers first (they now move and attack)
        alive_wights = [w for w in self.wights if w.alive]
        for soldier in self.soldiers:
            if soldier.alive:
                soldier.update(dt, self.game_time, alive_wights)
        
        # Update wights (they can now attack soldiers)
        alive_soldiers = [s for s in self.soldiers if s.alive]
        for wight in self.wights:
            if wight.alive:
                reached_castle = wight.update(dt, self.game_time, alive_soldiers)
                if reached_castle:
                    self.castle.take_damage(wight.castle_damage)
                    self.stats['castle_damage_taken'] += wight.castle_damage
                    wight.alive = False
                    
                    if self.castle.is_destroyed():
                        self.phase = GamePhase.DEFEAT
                        return
        
        # Check for killed wights and soldiers
        current_alive_wights = sum(1 for w in self.wights if w.alive)
        total_wights = len(self.wights)
        expected_alive_wights = total_wights - self.stats['wights_killed']
        if current_alive_wights < expected_alive_wights:
            self.stats['wights_killed'] = total_wights - current_alive_wights
        
        current_alive_soldiers = sum(1 for s in self.soldiers if s.alive)
        total_soldiers = len(self.soldiers)
        expected_alive_soldiers = total_soldiers - self.stats['soldiers_killed']
        if current_alive_soldiers < expected_alive_soldiers:
            self.stats['soldiers_killed'] = total_soldiers - current_alive_soldiers
    
    def reset(self):
        """Reset game to initial state"""
        self.__init__()
    
    def get_state(self) -> Dict:
        """Get current game state for observation"""
        return {
            'phase': self.phase,
            'game_time': self.game_time,
            'castle_hp': self.castle.hp,
            'castle_max_hp': self.castle.max_hp,
            'soldiers': [(s.type, s.position.x, s.position.y, s.alive, s.hp) for s in self.soldiers],
            'wights': [(w.position.x, w.position.y, w.alive, w.hp) for w in self.wights],
            'current_wave': self.current_wave,
            'total_waves': len(self.WAVE_DEFINITIONS),
            'stats': self.stats.copy(),
            'waiting_for_next_wave': self.waiting_for_next_wave,
            'next_wave_timer': self.next_wave_timer if self.waiting_for_next_wave else 0
        }
    
    def is_game_over(self) -> bool:
        """Check if game is over (victory or defeat)"""
        return self.phase in [GamePhase.VICTORY, GamePhase.DEFEAT]
    
    def get_result(self) -> str:
        """Get game result"""
        if self.phase == GamePhase.VICTORY:
            return "victory"
        elif self.phase == GamePhase.DEFEAT:
            return "defeat"
        return "ongoing"
