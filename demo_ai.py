"""
Quick demo of the soldier AI with A* pathfinding
Shows soldiers moving, attacking, and returning home
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))

from td_game_core import TowerDefenseGame, SoldierType, Position


def demo_soldier_ai():
    """Demonstrate the AI behavior with detailed output"""
    print("=" * 70)
    print("WINTERFELL TOWER DEFENSE - SOLDIER AI DEMO")
    print("=" * 70)
    print("\nSoldier AI Features:")
    print("  ✓ A* pathfinding for intelligent movement")
    print("  ✓ Detection radius for enemy awareness")
    print("  ✓ Automatic pursuit and attack")
    print("  ✓ Return to home position when idle")
    print()
    
    game = TowerDefenseGame()
    
    # Place soldiers in strategic positions
    print("Placing defenders...")
    placements = [
        (SoldierType.FOOTMAN, Position(500, 400)),
        (SoldierType.ARCHER, Position(640, 300)),
        (SoldierType.FOOTMAN, Position(780, 400)),
        (SoldierType.ARCHER, Position(640, 500)),
        (SoldierType.FOOTMAN, Position(400, 500)),
        (SoldierType.ARCHER, Position(880, 500)),
    ]
    
    for soldier_type, position in placements:
        game.place_soldier(soldier_type, position)
        name = "Footman" if soldier_type == SoldierType.FOOTMAN else "Archer"
        print(f"  ✓ {name} at ({position.x:.0f}, {position.y:.0f})")
    
    print(f"\nTotal soldiers placed: {len(game.soldiers)}")
    print(f"  - Footmen: Detection {game.soldiers[0].detection_radius}px, Speed {game.soldiers[0].move_speed}px/s")
    print(f"  - Archers: Detection {game.soldiers[1].detection_radius}px, Speed {game.soldiers[1].move_speed}px/s")
    
    # Start combat
    print("\n" + "=" * 70)
    print("COMBAT PHASE - Watch the AI in action!")
    print("=" * 70)
    game.start_combat_phase()
    
    # Track soldier behavior
    soldier_to_track = game.soldiers[0]
    tracking_log = []
    
    # Run simulation
    for frame in range(1800):  # 30 seconds at 60 FPS
        game.update(1.0 / 60.0)
        
        # Log interesting events
        if frame % 120 == 0:  # Every 2 seconds
            time = frame / 60.0
            alive_wights = sum(1 for w in game.wights if w.alive)
            
            # Track first soldier
            s = soldier_to_track
            home_dist = s.position.distance_to(s.home_position)
            status = "IDLE"
            if s.current_target:
                status = "ENGAGING"
                target_dist = s.position.distance_to(s.current_target.position)
                if target_dist <= s.attack_range:
                    status = "ATTACKING"
            elif s.is_returning_home:
                status = "RETURNING"
            
            log_entry = {
                'time': time,
                'wights_alive': alive_wights,
                'wights_killed': game.stats['wights_killed'],
                'wave': game.current_wave + 1,
                'soldier_status': status,
                'soldier_pos': (s.position.x, s.position.y),
                'home_distance': home_dist
            }
            tracking_log.append(log_entry)
            
            print(f"\n[{time:5.1f}s] Wave {log_entry['wave']}/5")
            print(f"  Wights: {alive_wights} alive | {log_entry['wights_killed']} killed")
            print(f"  Soldier-1: {status} at ({s.position.x:.0f}, {s.position.y:.0f})")
            if s.current_target:
                print(f"    → Target at ({s.current_target.position.x:.0f}, {s.current_target.position.y:.0f})")
            elif home_dist > 10:
                print(f"    → {home_dist:.0f}px from home")
        
        # Check for game over
        if game.is_game_over():
            break
    
    # Final results
    print("\n" + "=" * 70)
    print("BATTLE RESULTS")
    print("=" * 70)
    result = game.get_result()
    
    if result == "victory":
        print("✅ VICTORY! Winterfell stands!")
    else:
        print("❌ DEFEAT! The castle has fallen...")
    
    print(f"\nStatistics:")
    print(f"  Waves completed: {game.stats['waves_completed']}/{len(game.WAVE_DEFINITIONS)}")
    print(f"  Wights killed: {game.stats['wights_killed']}")
    print(f"  Castle HP: {game.castle.hp}/{game.castle.max_hp}")
    print(f"  Battle duration: {game.game_time:.1f} seconds")
    
    soldiers_alive = sum(1 for s in game.soldiers if s.alive)
    print(f"  Soldiers surviving: {soldiers_alive}/{len(game.soldiers)}")
    
    # Show AI effectiveness
    print(f"\nAI Effectiveness:")
    avg_kills_per_soldier = game.stats['wights_killed'] / len(game.soldiers) if game.soldiers else 0
    print(f"  Average kills per soldier: {avg_kills_per_soldier:.1f}")
    print(f"  Damage prevented: {game.stats['wights_killed'] * 20} HP (from reaching castle)")
    
    print("\n" + "=" * 70)
    print("Demo complete! Run 'python environment/td_pygame.py' to see it visually!")
    print("=" * 70)


if __name__ == "__main__":
    demo_soldier_ai()

