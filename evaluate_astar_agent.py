"""Evaluate the A* baseline agent."""
import argparse
import numpy as np
from environment.td_warrior_gym_env import TowerDefenseWarriorEnv
from agents.astar_baseline_agent import AStarBaselineAgent


def evaluate(args):
    env = TowerDefenseWarriorEnv()
    agent = AStarBaselineAgent(env.observation_space, env.action_space)

    episode_rewards = []
    episode_lengths = []
    victories = 0

    print(f"\nEvaluating A* baseline agent for {args.episodes} episodes...")
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
    print(f"Victories: {victories} ({100*victories/args.episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate A* baseline agent")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=2500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=5, help="Logging interval")

    args = parser.parse_args()
    evaluate(args)
