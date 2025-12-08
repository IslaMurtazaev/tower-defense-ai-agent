"""Evaluate a trained PPO agent."""
import argparse
from pathlib import Path

import numpy as np

from agents.ppo_agent import PPOAgent
from environment.td_warrior_gym_env import TowerDefenseWarriorEnv


def evaluate(args):
    """Evaluate the PPO agent."""
    env = TowerDefenseWarriorEnv(fast_mode=True)

    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return

    print(f"Loading PPO agent from {args.model_path}...")
    agent = PPOAgent.load(
        args.model_path,
        observation_space=env.observation_space,
        action_space=env.action_space
    )

    print(f"\nEvaluating agent for {args.episodes} episodes...")
    print("-" * 60)

    episode_rewards = []
    episode_lengths = []
    victories = 0
    base_hp_remaining = []
    soldiers_deployed = []
    wights_killed = []
    nk_kills = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode if args.seed is not None else None)
        agent.reset_buffers()

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done and episode_length < args.max_steps:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            obs = next_obs

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if info.get('victory', False):
            victories += 1

        base_hp_remaining.append(info.get('base_hp', 0))
        soldiers_deployed.append(info.get('stats', {}).get('soldiers_deployed', 0))
        wights_killed.append(info.get('stats', {}).get('wights_killed', 0))
        nk_kills.append(info.get('stats', {}).get('nk_kills', 0))

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{args.episodes}: "
                  f"Reward={episode_reward:.1f}, "
                  f"Victory={info.get('victory', False)}, "
                  f"BaseHP={info.get('base_hp', 0)}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Victory Rate: {victories/args.episodes*100:.1f}% ({victories}/{args.episodes})")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Mean Base HP Remaining: {np.mean(base_hp_remaining):.1f}")
    print(f"Mean Soldiers Deployed: {np.mean(soldiers_deployed):.1f}")
    print(f"Mean Wights Killed: {np.mean(wights_killed):.1f}")
    print(f"Mean NK Kills: {np.mean(nk_kills):.1f}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument("--model-path", type=str, default="trained_models/ppo/ppo_agent_final.pth",
                       help="Path to PPO model (.pth file)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=2500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()
    evaluate(args)
