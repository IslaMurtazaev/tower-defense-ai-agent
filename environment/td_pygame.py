"""
Tower Defense Pygame Visualization
Human-playable interface for the Winterfell Tower Defense game
"""
import pygame
import sys
import math
from td_game_core import (
    TowerDefenseGame, SoldierType, GamePhase, Position
)


class TowerDefenseVisualization:
    """Pygame visualization for Tower Defense game"""
    
    # Colors
    COLOR_BACKGROUND = (20, 20, 30)
    COLOR_CASTLE = (100, 100, 120)
    COLOR_FOOTMAN = (50, 100, 200)
    COLOR_ARCHER = (50, 200, 100)
    COLOR_WIGHT = (200, 50, 50)
    COLOR_HP_BAR_BG = (60, 60, 60)
    COLOR_HP_BAR = (0, 200, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GRID = (40, 40, 50)
    COLOR_PLACEMENT_INDICATOR = (255, 255, 100)
    COLOR_BUTTON = (70, 70, 90)
    COLOR_BUTTON_HOVER = (90, 90, 110)
    COLOR_BUTTON_DISABLED = (50, 50, 60)
    
    def __init__(self):
        pygame.init()
        
        # Display
        self.screen = pygame.display.set_mode((TowerDefenseGame.WIDTH, TowerDefenseGame.HEIGHT))
        pygame.display.set_caption("Winterfell Tower Defense - Battle Against the Night King")
        
        # Clock for frame rate
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Game instance
        self.game = TowerDefenseGame()
        
        # UI state
        self.selected_soldier_type = SoldierType.FOOTMAN
        self.show_grid = False
        self.mouse_pos = (0, 0)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Button for starting combat
        self.start_button_rect = pygame.Rect(
            TowerDefenseGame.WIDTH // 2 - 100, 20, 200, 50
        )
        
        # Running flag
        self.running = True
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_left_click(event.pos)
            
            elif event.type == pygame.KEYDOWN:
                self._handle_keypress(event.key)
    
    def _handle_left_click(self, pos: tuple):
        """Handle left mouse click"""
        if self.game.phase == GamePhase.PLACEMENT:
            # Check if clicking start button
            if self.start_button_rect.collidepoint(pos):
                if len(self.game.soldiers) > 0:
                    self.game.start_combat_phase()
            # Check if clicking soldier type selector
            elif self._in_soldier_selector(pos):
                self._cycle_soldier_type()
            # Otherwise, place soldier
            else:
                position = Position(pos[0], pos[1])
                self.game.place_soldier(self.selected_soldier_type, position)
    
    def _in_soldier_selector(self, pos: tuple) -> bool:
        """Check if click is in soldier type selector area"""
        selector_rect = pygame.Rect(20, 20, 150, 120)
        return selector_rect.collidepoint(pos)
    
    def _cycle_soldier_type(self):
        """Cycle between soldier types"""
        if self.selected_soldier_type == SoldierType.FOOTMAN:
            self.selected_soldier_type = SoldierType.ARCHER
        else:
            self.selected_soldier_type = SoldierType.FOOTMAN
    
    def _handle_keypress(self, key):
        """Handle keyboard input"""
        if key == pygame.K_r:
            # Reset game
            self.game.reset()
            self.selected_soldier_type = SoldierType.FOOTMAN
        elif key == pygame.K_g:
            # Toggle grid
            self.show_grid = not self.show_grid
        elif key == pygame.K_q:
            # Quit
            self.running = False
        elif key == pygame.K_SPACE:
            # Start combat if in placement phase
            if self.game.phase == GamePhase.PLACEMENT and len(self.game.soldiers) > 0:
                self.game.start_combat_phase()
    
    def update(self):
        """Update game state"""
        if self.game.phase == GamePhase.COMBAT:
            dt = self.clock.get_time() / 1000.0  # Convert to seconds
            self.game.update(dt)
    
    def render(self):
        """Render everything"""
        # Clear screen
        self.screen.fill(self.COLOR_BACKGROUND)
        
        # Draw grid if enabled
        if self.show_grid:
            self._draw_grid()
        
        # Draw castle
        self._draw_castle()
        
        # Draw entities
        self._draw_wights()
        self._draw_soldiers()
        
        # Draw UI
        if self.game.phase == GamePhase.PLACEMENT:
            self._draw_placement_ui()
        else:
            self._draw_combat_ui()
        
        # Draw game over screen
        if self.game.is_game_over():
            self._draw_game_over()
        
        # Update display
        pygame.display.flip()
    
    def _draw_grid(self):
        """Draw background grid"""
        grid_size = 40
        for x in range(0, TowerDefenseGame.WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, TowerDefenseGame.HEIGHT))
        for y in range(0, TowerDefenseGame.HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (TowerDefenseGame.WIDTH, y))
    
    def _draw_castle(self):
        """Draw the castle"""
        castle_pos = self.game.castle.position
        
        # Draw castle structure (simplified)
        castle_width = 120
        castle_height = 80
        castle_rect = pygame.Rect(
            castle_pos.x - castle_width // 2,
            castle_pos.y - castle_height // 2,
            castle_width,
            castle_height
        )
        pygame.draw.rect(self.screen, self.COLOR_CASTLE, castle_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, castle_rect, 2)
        
        # Draw towers
        tower_size = 20
        # Left tower
        pygame.draw.rect(self.screen, self.COLOR_CASTLE,
                        (castle_pos.x - castle_width // 2 - 10, castle_pos.y - castle_height // 2 - 10,
                         tower_size, tower_size + 10))
        # Right tower
        pygame.draw.rect(self.screen, self.COLOR_CASTLE,
                        (castle_pos.x + castle_width // 2 - 10, castle_pos.y - castle_height // 2 - 10,
                         tower_size, tower_size + 10))
        
        # Draw HP bar
        hp_bar_width = castle_width
        hp_bar_height = 8
        hp_bar_x = castle_pos.x - hp_bar_width // 2
        hp_bar_y = castle_pos.y + castle_height // 2 + 10
        
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR_BG,
                        (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
        
        hp_ratio = self.game.castle.hp / self.game.castle.max_hp
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR,
                        (hp_bar_x, hp_bar_y, int(hp_bar_width * hp_ratio), hp_bar_height))
        
        # Draw HP text
        hp_text = self.font_small.render(f"Castle HP: {self.game.castle.hp}/{self.game.castle.max_hp}",
                                         True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (hp_bar_x, hp_bar_y + hp_bar_height + 5))
    
    def _draw_soldiers(self):
        """Draw all soldiers"""
        for soldier in self.game.soldiers:
            if soldier.alive:
                pos = soldier.position
                
                # Draw home position marker (small dot)
                home_pos = soldier.home_position
                pygame.draw.circle(self.screen, (100, 100, 100),
                                 (int(home_pos.x), int(home_pos.y)), 3)
                
                # Draw line to home if returning
                if soldier.is_returning_home:
                    pygame.draw.line(self.screen, (150, 150, 150),
                                   (int(pos.x), int(pos.y)),
                                   (int(home_pos.x), int(home_pos.y)), 1)
                
                # Draw detection radius (very faint)
                if self.game.phase == GamePhase.PLACEMENT or soldier.current_target:
                    pygame.draw.circle(self.screen, (80, 80, 100),
                                     (int(pos.x), int(pos.y)), int(soldier.detection_radius), 1)
                
                # Draw line to target if attacking
                if soldier.current_target and soldier.current_target.alive:
                    target_pos = soldier.current_target.position
                    color = (255, 100, 100) if pos.distance_to(target_pos) <= soldier.attack_range else (150, 150, 100)
                    pygame.draw.line(self.screen, color,
                                   (int(pos.x), int(pos.y)),
                                   (int(target_pos.x), int(target_pos.y)), 1)
                
                # Draw based on type
                if soldier.type == SoldierType.FOOTMAN:
                    # Blue circle for footman
                    pygame.draw.circle(self.screen, self.COLOR_FOOTMAN,
                                     (int(pos.x), int(pos.y)), 12)
                    pygame.draw.circle(self.screen, self.COLOR_TEXT,
                                     (int(pos.x), int(pos.y)), 12, 2)
                else:  # ARCHER
                    # Green triangle for archer
                    points = [
                        (pos.x, pos.y - 12),
                        (pos.x - 10, pos.y + 8),
                        (pos.x + 10, pos.y + 8)
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_ARCHER, points)
                    pygame.draw.polygon(self.screen, self.COLOR_TEXT, points, 2)
                
                # Draw attack range circle (faint) when in combat
                if self.game.phase == GamePhase.COMBAT and soldier.current_target:
                    pygame.draw.circle(self.screen, (200, 200, 100),
                                     (int(pos.x), int(pos.y)), int(soldier.attack_range), 1)
    
    def _draw_wights(self):
        """Draw all wights"""
        for wight in self.game.wights:
            if wight.alive:
                pos = wight.position
                
                # Draw red square for wight
                size = 14
                rect = pygame.Rect(int(pos.x - size // 2), int(pos.y - size // 2), size, size)
                pygame.draw.rect(self.screen, self.COLOR_WIGHT, rect)
                pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2)
                
                # Draw HP bar above wight
                hp_bar_width = 20
                hp_bar_height = 3
                hp_bar_x = pos.x - hp_bar_width // 2
                hp_bar_y = pos.y - 20
                
                pygame.draw.rect(self.screen, self.COLOR_HP_BAR_BG,
                               (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
                
                hp_ratio = wight.hp / wight.max_hp
                pygame.draw.rect(self.screen, self.COLOR_HP_BAR,
                               (hp_bar_x, hp_bar_y, int(hp_bar_width * hp_ratio), hp_bar_height))
    
    def _draw_placement_ui(self):
        """Draw placement phase UI"""
        # Soldier type selector
        selector_rect = pygame.Rect(20, 20, 150, 120)
        pygame.draw.rect(self.screen, self.COLOR_BUTTON, selector_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, selector_rect, 2)
        
        # Title
        title_text = self.font_small.render("Click to Switch:", True, self.COLOR_TEXT)
        self.screen.blit(title_text, (30, 30))
        
        # Show current selection
        if self.selected_soldier_type == SoldierType.FOOTMAN:
            type_text = "FOOTMAN"
            color = self.COLOR_FOOTMAN
            info = "Melee - Short Range"
            info2 = "High Damage"
        else:
            type_text = "ARCHER"
            color = self.COLOR_ARCHER
            info = "Ranged - Long Range"
            info2 = "Lower Damage"
        
        type_surface = self.font_medium.render(type_text, True, color)
        info_surface = self.font_small.render(info, True, self.COLOR_TEXT)
        info2_surface = self.font_small.render(info2, True, self.COLOR_TEXT)
        
        self.screen.blit(type_surface, (30, 55))
        self.screen.blit(info_surface, (30, 90))
        self.screen.blit(info2_surface, (30, 110))
        
        # Start button
        can_start = len(self.game.soldiers) > 0
        button_color = self.COLOR_BUTTON_HOVER if (can_start and self.start_button_rect.collidepoint(self.mouse_pos)) else (self.COLOR_BUTTON if can_start else self.COLOR_BUTTON_DISABLED)
        
        pygame.draw.rect(self.screen, button_color, self.start_button_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, self.start_button_rect, 2)
        
        start_text = self.font_medium.render("START BATTLE", True, self.COLOR_TEXT if can_start else (100, 100, 100))
        text_rect = start_text.get_rect(center=self.start_button_rect.center)
        self.screen.blit(start_text, text_rect)
        
        # Soldier count
        count_text = self.font_small.render(
            f"Soldiers: {len(self.game.soldiers)}/{TowerDefenseGame.MAX_SOLDIERS}",
            True, self.COLOR_TEXT
        )
        self.screen.blit(count_text, (TowerDefenseGame.WIDTH // 2 - 80, 80))
        
        # Instructions
        instructions = [
            "Left Click: Place Soldier",
            "SPACE: Start Battle",
            "R: Reset  |  G: Toggle Grid  |  Q: Quit"
        ]
        y_offset = TowerDefenseGame.HEIGHT - 80
        for instruction in instructions:
            text = self.font_small.render(instruction, True, self.COLOR_TEXT)
            self.screen.blit(text, (20, y_offset))
            y_offset += 25
    
    def _draw_combat_ui(self):
        """Draw combat phase UI"""
        # Stats panel
        stats = self.game.stats
        info_lines = [
            f"Wave: {self.game.current_wave + 1}/{len(TowerDefenseGame.WAVE_DEFINITIONS)}",
            f"Soldiers Alive: {sum(1 for s in self.game.soldiers if s.alive)}",
            f"Soldiers Placed: {stats['soldiers_placed']}",
            f"Wights Killed: {stats['wights_killed']}",
            f"Castle HP: {self.game.castle.hp}/{self.game.castle.max_hp}"
        ]
        
        y_offset = 20
        for line in info_lines:
            text = self.font_small.render(line, True, self.COLOR_TEXT)
            self.screen.blit(text, (20, y_offset))
            y_offset += 25
        
        # Wave timer
        if self.game.waiting_for_next_wave and any(w.alive for w in self.game.wights):
            timer_text = "Clearing Wave..."
        elif self.game.waiting_for_next_wave:
            remaining = TowerDefenseGame.TIME_BETWEEN_WAVES - self.game.next_wave_timer
            timer_text = f"Next Wave in: {remaining:.1f}s"
        else:
            timer_text = f"Spawning Wights: {self.game.wights_to_spawn_this_wave} remaining"
        
        timer_surface = self.font_small.render(timer_text, True, self.COLOR_PLACEMENT_INDICATOR)
        self.screen.blit(timer_surface, (20, 150))
        
        # Controls reminder
        controls = self.font_small.render("R: Reset  |  G: Grid  |  Q: Quit", True, self.COLOR_TEXT)
        self.screen.blit(controls, (20, TowerDefenseGame.HEIGHT - 30))
    
    def _draw_game_over(self):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((TowerDefenseGame.WIDTH, TowerDefenseGame.HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Result text
        if self.game.phase == GamePhase.VICTORY:
            result_text = "VICTORY!"
            result_color = (0, 255, 0)
            subtitle = "Winterfell Stands!"
        else:
            result_text = "DEFEAT"
            result_color = (255, 0, 0)
            subtitle = "The Night King Has Won..."
        
        result_surface = self.font_large.render(result_text, True, result_color)
        result_rect = result_surface.get_rect(center=(TowerDefenseGame.WIDTH // 2, TowerDefenseGame.HEIGHT // 2 - 50))
        self.screen.blit(result_surface, result_rect)
        
        subtitle_surface = self.font_medium.render(subtitle, True, self.COLOR_TEXT)
        subtitle_rect = subtitle_surface.get_rect(center=(TowerDefenseGame.WIDTH // 2, TowerDefenseGame.HEIGHT // 2))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Stats
        stats = self.game.stats
        stats_lines = [
            f"Waves Completed: {stats['waves_completed']}/{len(TowerDefenseGame.WAVE_DEFINITIONS)}",
            f"Wights Killed: {stats['wights_killed']}",
            f"Soldiers Placed: {stats['soldiers_placed']}",
            f"Final Castle HP: {self.game.castle.hp}/{self.game.castle.max_hp}"
        ]
        
        y_offset = TowerDefenseGame.HEIGHT // 2 + 60
        for line in stats_lines:
            text = self.font_small.render(line, True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=(TowerDefenseGame.WIDTH // 2, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 30
        
        # Restart prompt
        restart_text = self.font_medium.render("Press R to Restart", True, self.COLOR_PLACEMENT_INDICATOR)
        restart_rect = restart_text.get_rect(center=(TowerDefenseGame.WIDTH // 2, TowerDefenseGame.HEIGHT - 100))
        self.screen.blit(restart_text, restart_rect)
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(self.fps)
        
        pygame.quit()
        sys.exit()


def main():
    """Entry point"""
    game = TowerDefenseVisualization()
    game.run()


if __name__ == "__main__":
    main()

