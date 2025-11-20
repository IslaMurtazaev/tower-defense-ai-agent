"""
Test script for Tower Defense game
Tests both the Pygame interface and Gymnasium environment
"""
import sys
import os

# Add environment directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))

from td_game_core import TowerDefenseGame, SoldierType, Position, GamePhase


def test_core_game():
    """Test core game mechanics"""
    print("=" * 60)
    print("Testing Core Game Engine")
    print("=" * 60)
    
    game = TowerDefenseGame()
    
    # Test initial state
    assert game.phase == GamePhase.PLACEMENT
    assert game.castle.hp == game.castle.max_hp
    assert len(game.soldiers) == 0
    assert len(game.wights) == 0
    print("✓ Initial state correct")
    
    # Test soldier placement
    pos1 = Position(400, 300)
    success = game.place_soldier(SoldierType.FOOTMAN, pos1)
    assert success
    assert len(game.soldiers) == 1
    print("✓ Soldier placement works")
    
    # Test placing soldiers at different positions
    for i in range(5):
        pos = Position(300 + i * 50, 400)
        game.place_soldier(SoldierType.ARCHER if i % 2 else SoldierType.FOOTMAN, pos)
    
    assert len(game.soldiers) == 6
    print(f"✓ Placed {len(game.soldiers)} soldiers")
    
    # Test invalid placement (too close to castle)
    castle_pos = game.castle.position
    invalid_pos = Position(castle_pos.x + 10, castle_pos.y + 10)
    success = game.place_soldier(SoldierType.FOOTMAN, invalid_pos)
    assert not success
    print("✓ Invalid placement correctly rejected")
    
    # Test max soldiers
    while game.can_place_soldier():
        pos = Position(200 + len(game.soldiers) * 30, 200)
        game.place_soldier(SoldierType.FOOTMAN, pos)
    
    assert len(game.soldiers) == TowerDefenseGame.MAX_SOLDIERS
    assert not game.can_place_soldier()
    print(f"✓ Max soldiers limit ({TowerDefenseGame.MAX_SOLDIERS}) enforced")
    
    # Test starting combat
    game.start_combat_phase()
    assert game.phase == GamePhase.COMBAT
    print("✓ Combat phase started")
    
    # Simulate some game updates
    for i in range(100):
        game.update(1.0 / 60.0)  # 60 FPS
    
    assert len(game.wights) > 0
    print(f"✓ Wights spawning ({len(game.wights)} spawned)")
    
    # Check stats
    stats = game.get_state()
    print(f"✓ Game state accessible: {stats['phase']}")
    
    print("\n✅ Core game engine tests passed!\n")
    return True


def test_gym_environment():
    """Test Gymnasium environment"""
    print("=" * 60)
    print("Testing Gymnasium Environment")
    print("=" * 60)
    
    try:
        from td_gym_env import TowerDefenseEnv
        import numpy as np
        
        # Create environment
        env = TowerDefenseEnv()
        print("✓ Environment created")
        
        # Test reset
        observation, info = env.reset(seed=42)
        assert 'grid' in observation
        assert 'soldiers_remaining' in observation
        assert observation['grid'].shape == (32, 32)
        print("✓ Reset works, observation space correct")
        
        # Test action space
        action = env.action_space.sample()
        assert len(action) == 3
        print(f"✓ Action space correct (sample: {action})")
        
        # Take a few steps
        total_reward = 0
        for i in range(30):  # Place some soldiers
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"✓ Environment step works (reward after 30 steps: {total_reward:.2f})")
        print(f"  Phase: {info['phase']}, Soldiers placed: {info['placement_actions_taken']}")
        
        # Close environment
        env.close()
        print("✓ Environment closes properly")
        
        print("\n✅ Gymnasium environment tests passed!\n")
        return True
        
    except ImportError as e:
        print(f"⚠️  Gymnasium not installed: {e}")
        print("   Install with: pip install gymnasium numpy")
        return False


def quick_game_simulation():
    """Run a quick game simulation to ensure everything works"""
    print("=" * 60)
    print("Quick Game Simulation")
    print("=" * 60)
    
    game = TowerDefenseGame()
    
    # Place soldiers in a defensive formation
    print("Placing soldiers in defensive formation...")
    
    # Front line footmen
    for i in range(5):
        x = 400 + i * 80
        y = 500
        game.place_soldier(SoldierType.FOOTMAN, Position(x, y))
    
    # Back line archers
    for i in range(5):
        x = 400 + i * 80
        y = 580
        game.place_soldier(SoldierType.ARCHER, Position(x, y))
    
    print(f"✓ Placed {len(game.soldiers)} soldiers")
    
    # Start combat
    game.start_combat_phase()
    print("✓ Combat started")
    
    # Run simulation
    print("\nRunning battle simulation...")
    max_time = 120.0  # 2 minutes max
    dt = 1.0 / 60.0
    time_elapsed = 0.0
    last_wave = -1
    
    while not game.is_game_over() and time_elapsed < max_time:
        game.update(dt)
        time_elapsed += dt
        
        # Report wave changes
        if game.current_wave != last_wave:
            last_wave = game.current_wave
            print(f"  Wave {game.current_wave + 1}/{len(TowerDefenseGame.WAVE_DEFINITIONS)} - "
                  f"Castle HP: {game.castle.hp}/{game.castle.max_hp}")
    
    # Report results
    print(f"\n{'='*60}")
    print("Battle Results")
    print('='*60)
    result = game.get_result()
    print(f"Result: {result.upper()}")
    print(f"Time elapsed: {time_elapsed:.1f}s")
    print(f"Waves completed: {game.stats['waves_completed']}/{len(TowerDefenseGame.WAVE_DEFINITIONS)}")
    print(f"Wights killed: {game.stats['wights_killed']}")
    print(f"Final castle HP: {game.castle.hp}/{game.castle.max_hp}")
    print(f"Soldiers alive: {sum(1 for s in game.soldiers if s.alive)}/{len(game.soldiers)}")
    
    if result == "victory":
        print("\n✅ Victory! The Night King has been defeated!")
    else:
        print("\n❌ Defeat. Winterfell has fallen...")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TOWER DEFENSE GAME TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test core game
    try:
        results.append(("Core Game Engine", test_core_game()))
    except Exception as e:
        print(f"❌ Core game test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Core Game Engine", False))
    
    # Test gym environment
    try:
        results.append(("Gymnasium Environment", test_gym_environment()))
    except Exception as e:
        print(f"❌ Gym environment test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Gymnasium Environment", False))
    
    # Quick simulation
    try:
        results.append(("Game Simulation", quick_game_simulation()))
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Game Simulation", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Game is ready to play!")
        print("\nTo play the game:")
        print("  cd environment")
        print("  python td_pygame.py")
        print("\nTo test with RL agents:")
        print("  python environment/td_gym_env.py")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

