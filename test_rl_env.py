"""
Test the RL environment without requiring full RL stack
Quick verification that the Gymnasium environment works
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))

def test_basic_env():
    """Test basic environment functionality"""
    print("=" * 70)
    print("TESTING RL ENVIRONMENT")
    print("=" * 70)
    
    try:
        from td_gym_env import TowerDefenseEnv
        import numpy as np
        print("✓ Imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\nPlease install required packages:")
        print("  pip install gymnasium numpy")
        return False
    
    # Create environment
    print("\n[1/6] Creating environment...")
    env = TowerDefenseEnv()
    print("✓ Environment created")
    
    # Test reset
    print("\n[2/6] Testing reset...")
    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  Observation keys: {obs.keys()}")
    print(f"  Grid shape: {obs['grid'].shape}")
    print(f"  Soldiers remaining: {obs['soldiers_remaining'][0]}")
    
    # Test action space
    print("\n[3/6] Testing action space...")
    action = env.action_space.sample()
    print(f"✓ Action space works")
    print(f"  Sample action: {action}")
    print(f"  Action space: MultiDiscrete([2, 32, 32])")
    
    # Test observation space
    print("\n[4/6] Testing observation space...")
    assert obs['grid'].shape == (32, 32), "Grid shape incorrect"
    assert 0 <= obs['soldiers_remaining'][0] <= 10, "Soldiers remaining out of range"
    assert 0 <= obs['castle_hp_ratio'][0] <= 1.0, "HP ratio out of range"
    print("✓ Observation space validated")
    
    # Test episode
    print("\n[5/6] Running test episode...")
    total_reward = 0
    steps = 0
    max_steps = 3000
    
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
        
        if step % 500 == 0:
            print(f"  Step {step}: Phase={info['phase']}, Reward={total_reward:.1f}")
    
    print(f"✓ Episode completed")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final phase: {info['phase']}")
    print(f"  Waves completed: {info['stats']['waves_completed']}/5")
    print(f"  Wights killed: {info['stats']['wights_killed']}/300")
    
    # Test multiple episodes
    print("\n[6/6] Testing multiple episodes...")
    rewards = []
    for ep in range(5):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
    
    print(f"✓ Completed 5 episodes")
    print(f"  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Reward range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe RL environment is ready for training!")
    print("\nNext steps:")
    print("  1. Install RL dependencies: pip install stable-baselines3 tensorboard torch")
    print("  2. Start training: python train_rl_agent.py")
    print("  3. Monitor progress: tensorboard --logdir trained_models/logs")
    print("=" * 70)
    
    return True


def test_with_wrapper():
    """Test the flattened wrapper used in training"""
    print("\n" + "=" * 70)
    print("TESTING TRAINING WRAPPER")
    print("=" * 70)
    
    try:
        from td_gym_env import TowerDefenseEnv
        from train_rl_agent import TowerDefenseWrapper
        import numpy as np
        print("✓ Imports successful")
    except ImportError as e:
        print(f"❌ Cannot test wrapper without stable-baselines3")
        print(f"   This is OK - wrapper only needed for training")
        return True
    
    env = TowerDefenseEnv()
    wrapped_env = TowerDefenseWrapper(env)
    
    print("\n[1/2] Testing wrapped reset...")
    obs, info = wrapped_env.reset()
    print(f"✓ Reset successful")
    print(f"  Flattened observation shape: {obs.shape}")
    print(f"  Expected: (1027,) = 32×32 grid + 3 features")
    
    print("\n[2/2] Testing wrapped step...")
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print(f"✓ Step successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Reward: {reward:.2f}")
    
    wrapped_env.close()
    
    print("\n✅ WRAPPER TESTS PASSED!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_basic_env()
    
    if success:
        test_with_wrapper()
    
    sys.exit(0 if success else 1)

