"""Evaluate a trained Q-Learning agent on TowerDefenseWarriorEnv."""
import argparse
import numpy as np
from pathlib import Path

from environment.td_warrior_gym_env import TowerDefenseWarriorEnv
from agents.approx_q_agent import ApproximateQAgent


def evaluate(args):
    env = TowerDefenseWarriorEnv()

    # Load agent
    print(f"Loading agent from {args.model_path}")
    agent = ApproximateQAgent.load(
        args.model_path,
        observation_space=env.observation_space,
        action_space=env.action_space
    )

    # Set epsilon to 0 for evaluation (no exploration)
    agent.epsilon_start = 0.0
    agent.epsilon_end = 0.0

    episode_rewards = []
    episode_lengths = []
    victories = 0

    print(f"\nEvaluating agent for {args.episodes} episodes...")
    print("=" * 60)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done and ep_length < args.max_steps:
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

        if (ep + 1) % max(1, args.log_interval) == 0:
            recent_rewards = episode_rewards[-args.log_interval:]
            mean_r = float(np.mean(recent_rewards))
            print(f"Episode {ep+1}/{args.episodes} | Reward: {ep_reward:.1f} | "
                  f"Recent mean: {mean_r:.1f} | Victory: {info.get('victory', False)}")

    env.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Victory rate: {victories}/{args.episodes} ({100*victories/args.episodes:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Q-Learning agent")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=2500, help="Max steps per episode (default allows ~180s game time in fast mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=5, help="Logging interval")

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return

    evaluate(args)


if __name__ == "__main__":
    main()
