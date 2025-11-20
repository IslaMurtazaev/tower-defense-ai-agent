"""
Evaluate a trained RL agent
Can run with or without visualization
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))

import numpy as np
from stable_baselines3 import PPO
from td_gym_env import TowerDefenseEnv
from train_rl_agent import TowerDefenseWrapper


def evaluate_agent(model_path, n_episodes=10, render=False, verbose=True):
    """Evaluate trained agent"""
    
    if verbose:
        print("=" * 70)
        print("EVALUATING TRAINED AGENT")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Episodes: {n_episodes}")
        print(f"Render: {render}")
        print("=" * 70 + "\n")
    
    # Load model
    model = PPO.load(model_path)
    if verbose:
        print("✓ Model loaded\n")
    
    # Create environment
    env = TowerDefenseEnv(render_mode="human" if render else None)
    env = TowerDefenseWrapper(env)
    
    # Run episodes
    results = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print('='*70)
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render()
                time.sleep(0.016)  # ~60 FPS
        
        # Collect results
        result = {
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': steps,
            'phase': info['phase'],
            'waves_completed': info['stats']['waves_completed'],
            'wights_killed': info['stats']['wights_killed'],
            'soldiers_placed': info['stats']['soldiers_placed'],
            'castle_hp': info['stats'].get('castle_hp', 0)
        }
        results.append(result)
        
        if verbose:
            print(f"\nResults:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Phase: {info['phase']}")
            print(f"  Waves Completed: {info['stats']['waves_completed']}/5")
            print(f"  Wights Killed: {info['stats']['wights_killed']}/300")
            print(f"  Soldiers Placed: {info['stats']['soldiers_placed']}/10")
            
            if info['phase'] == 'victory':
                print(f"  🎉 VICTORY!")
            else:
                print(f"  ❌ Defeat")
    
    env.close()
    
    # Compute statistics
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
    
    rewards = [r['reward'] for r in results]
    victories = sum(1 for r in results if r['phase'] == 'victory')
    avg_waves = np.mean([r['waves_completed'] for r in results])
    avg_kills = np.mean([r['wights_killed'] for r in results])
    
    summary = {
        'n_episodes': n_episodes,
        'victories': victories,
        'win_rate': victories / n_episodes * 100,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'avg_waves_completed': avg_waves,
        'avg_wights_killed': avg_kills
    }
    
    if verbose:
        print(f"Win Rate: {summary['win_rate']:.1f}% ({victories}/{n_episodes})")
        print(f"Average Reward: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"Reward Range: [{summary['min_reward']:.2f}, {summary['max_reward']:.2f}]")
        print(f"Avg Waves Completed: {summary['avg_waves_completed']:.2f}/5")
        print(f"Avg Wights Killed: {summary['avg_wights_killed']:.1f}/300")
        print("=" * 70)
    
    return results, summary


def compare_with_random(model_path, n_episodes=10):
    """Compare trained agent with random policy"""
    
    print("\n" + "=" * 70)
    print("COMPARING: TRAINED AGENT vs RANDOM POLICY")
    print("=" * 70)
    
    # Evaluate trained agent
    print("\n[1/2] Evaluating TRAINED agent...")
    trained_results, trained_summary = evaluate_agent(model_path, n_episodes, verbose=False)
    
    # Evaluate random policy
    print("\n[2/2] Evaluating RANDOM policy...")
    env = TowerDefenseEnv()
    env = TowerDefenseWrapper(env)
    
    random_results = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        random_results.append({
            'reward': episode_reward,
            'phase': info['phase'],
            'waves_completed': info['stats']['waves_completed'],
            'wights_killed': info['stats']['wights_killed']
        })
    
    env.close()
    
    random_victories = sum(1 for r in random_results if r['phase'] == 'victory')
    random_summary = {
        'victories': random_victories,
        'win_rate': random_victories / n_episodes * 100,
        'avg_reward': np.mean([r['reward'] for r in random_results]),
        'avg_waves': np.mean([r['waves_completed'] for r in random_results]),
        'avg_kills': np.mean([r['wights_killed'] for r in random_results])
    }
    
    # Display comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Trained':<20} {'Random':<20} {'Improvement'}")
    print("-" * 70)
    print(f"{'Win Rate':<30} {trained_summary['win_rate']:>6.1f}% {random_summary['win_rate']:>16.1f}% {trained_summary['win_rate'] - random_summary['win_rate']:>15.1f}%")
    print(f"{'Average Reward':<30} {trained_summary['avg_reward']:>19.2f} {random_summary['avg_reward']:>19.2f} {trained_summary['avg_reward'] - random_summary['avg_reward']:>+15.2f}")
    print(f"{'Avg Waves Completed':<30} {trained_summary['avg_waves_completed']:>19.2f} {random_summary['avg_waves']:>19.2f} {trained_summary['avg_waves_completed'] - random_summary['avg_waves']:>+15.2f}")
    print(f"{'Avg Wights Killed':<30} {trained_summary['avg_wights_killed']:>19.1f} {random_summary['avg_kills']:>19.1f} {trained_summary['avg_wights_killed'] - random_summary['avg_kills']:>+15.1f}")
    print("=" * 70)
    
    # Performance multiplier
    if random_summary['avg_reward'] > 0:
        multiplier = trained_summary['avg_reward'] / random_summary['avg_reward']
        print(f"\nTrained agent performs {multiplier:.2f}x better than random policy! 🚀")
    else:
        print(f"\nTrained agent significantly outperforms random policy! 🚀")
    print("=" * 70)


def visualize_agent(model_path, n_episodes=1):
    """Visualize agent playing with Pygame"""
    print("\n" + "=" * 70)
    print("VISUAL DEMONSTRATION")
    print("=" * 70)
    print("Watch the AI agent play in real-time!")
    print("Close the window or press Q to stop")
    print("=" * 70 + "\n")
    
    evaluate_agent(model_path, n_episodes, render=True, verbose=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Tower Defense agent")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.zip)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to evaluate (default: 10)")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes with Pygame visualization")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with random policy")
    parser.add_argument("--visualize", action="store_true",
                       help="Visual demonstration (implies --render)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found at {args.model}")
        return 1
    
    try:
        if args.visualize:
            visualize_agent(args.model, args.episodes)
        elif args.compare:
            compare_with_random(args.model, args.episodes)
        else:
            evaluate_agent(args.model, args.episodes, args.render)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

