"""
Compare different algorithms: A* baseline, Q-Learning, and PPO.

This script evaluates all algorithms and generates a comparison report.
"""
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json

from environment.td_warrior_gym_env import TowerDefenseWarriorEnv
from agents.astar_baseline_agent import AStarBaselineAgent
from agents.approx_q_agent import ApproximateQAgent
from agents.ppo_agent import PPOAgent


def evaluate_agent(agent, env, episodes: int, max_steps: int, seed: int = 0) -> Dict[str, Any]:
    """Evaluate an agent and return performance metrics."""
    episode_rewards = []
    episode_lengths = []
    victories = 0
    base_hp_remaining = []
    soldiers_deployed = []
    wights_killed = []
    nk_kills = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)

        # Reset PPO agent buffers before each episode (critical for proper evaluation)
        if hasattr(agent, 'reset_buffers'):
            agent.reset_buffers()

        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done and ep_length < max_steps:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1
            obs = next_obs

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        if info.get('victory', False):
            victories += 1

        base_hp_remaining.append(info.get('base_hp', 0))
        soldiers_deployed.append(info.get('stats', {}).get('soldiers_deployed', 0))
        wights_killed.append(info.get('stats', {}).get('wights_killed', 0))
        nk_kills.append(info.get('stats', {}).get('nk_kills', 0))

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'victory_rate': victories / episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_base_hp': float(np.mean(base_hp_remaining)),
        'mean_soldiers_deployed': float(np.mean(soldiers_deployed)),
        'mean_wights_killed': float(np.mean(wights_killed)),
        'mean_nk_kills': float(np.mean(nk_kills)),
    }


def compare_algorithms(args):
    """Compare all implemented algorithms."""
    env = TowerDefenseWarriorEnv(fast_mode=True)

    algorithms = {}

    # A* Baseline
    print("Loading A* baseline agent...")
    algorithms['A* Baseline'] = AStarBaselineAgent(
        env.observation_space,
        env.action_space
    )

    # Q-Learning (if model exists)
    if args.q_model_path and Path(args.q_model_path).exists():
        print(f"Loading Q-Learning agent from {args.q_model_path}...")
        algorithms['Q-Learning'] = ApproximateQAgent.load(
            args.q_model_path,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        algorithms['Q-Learning'].epsilon_start = 0.0
        algorithms['Q-Learning'].epsilon_end = 0.0
    else:
        print("⚠️  Q-Learning model not found, skipping...")

    # PPO (if model exists)
    if args.ppo_model_path and Path(args.ppo_model_path).exists():
        print(f"Loading PPO agent from {args.ppo_model_path}...")
        algorithms['PPO'] = PPOAgent.load(
            args.ppo_model_path,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
    elif args.ppo_model_path:
        print("⚠️  PPO model not found, skipping...")

    # Evaluate each algorithm
    results = {}
    print("\n" + "=" * 60)
    print("EVALUATING ALGORITHMS")
    print("=" * 60)

    for name, agent in algorithms.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_agent(
            agent,
            env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed
        )
        print(f"  Victory Rate: {results[name]['victory_rate']*100:.1f}%")
        print(f"  Mean Reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}")

    env.close()

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Victory %':<12} {'Mean Reward':<15} {'Base HP':<12} {'Wights Killed':<15} {'NK Kills':<12}")
    print("-" * 80)

    for name, metrics in results.items():
        print(f"{name:<20} {metrics['victory_rate']*100:>6.1f}%     "
              f"{metrics['mean_reward']:>8.2f} ± {metrics['std_reward']:>4.2f}  "
              f"{metrics['mean_base_hp']:>6.1f}      "
              f"{metrics['mean_wights_killed']:>8.1f}        "
              f"{metrics['mean_nk_kills']:>6.1f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON
        json_results = {}
        for name, metrics in results.items():
            json_results[name] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in metrics.items()
            }

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different algorithms")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes per algorithm")
    parser.add_argument("--max-steps", type=int, default=2500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--q-model-path", type=str, default="trained_models/q_learning/q_agent_final.pkl",
                       help="Path to Q-Learning model")
    parser.add_argument("--ppo-model-path", type=str, default="trained_models/ppo/ppo_agent_final.pth",
                       help="Path to PPO model")
    parser.add_argument("--output", type=str, default="comparison_results.json",
                       help="Output file for results")

    args = parser.parse_args()
    compare_algorithms(args)
