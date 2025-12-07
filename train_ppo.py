"""Train a PPO agent on TowerDefenseWarriorEnv."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        raise ImportError("Please install torch or tensorboardX for TensorBoard logging")

from agents.ppo_agent import PPOAgent
from environment.td_warrior_gym_env import TowerDefenseWarriorEnv


def _apply_terminal_reward_shaping(reward: float, info: dict) -> float:
    """Apply terminal reward shaping for episode completion."""
    victory = info.get('victory', False)
    base_hp = info.get('base_hp', 0)
    stats = info.get('stats', {})

    if victory:
        terminal_bonus = 50.0
        terminal_bonus += (base_hp / 100.0) * 30.0

        soldiers_alive = stats.get('soldiers_deployed', 0) - stats.get('soldiers_killed', 0)
        terminal_bonus += soldiers_alive * 15.0

        soldiers_killed_by_nk = stats.get('soldiers_killed_by_nk_sweep', 0)
        if soldiers_killed_by_nk >= 6:
            terminal_bonus -= 50.0

        nk_kills = stats.get('nk_kills', 0)
        terminal_bonus += nk_kills * 20.0

        reward += terminal_bonus
    else:
        reward -= 30.0
        nk_kills = stats.get('nk_kills', 0)
        if nk_kills >= 3:
            reward += 20.0
        elif nk_kills >= 2:
            reward += 10.0
        elif nk_kills >= 1:
            reward += 5.0

    return reward


def train(args):
    """Train the PPO agent."""
    env = TowerDefenseWarriorEnv(fast_mode=args.fast_mode, fast_multiplier=getattr(args, 'fast_multiplier', 20))

    # Create models directory
    model_dir = Path("trained_models/ppo")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard log directory
    log_dir = Path("runs/ppo")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{timestamp}"
    tb_log_dir = log_dir / run_name
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard logs: tensorboard --logdir={tb_log_dir}")

    # Create agent
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        agent = PPOAgent.load(
            args.load_model,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
    else:
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=24,  # Updated: added base defense features (was 21)
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            clip_epsilon=args.clip_epsilon,
            reward_scale=args.reward_scale,
            device=args.device,
            gae_lambda=getattr(args, 'gae_lambda', 0.95),
            entropy_coef=getattr(args, 'entropy_coef', 0.02),
        )

    print(f"\nTraining PPO agent for {args.episodes} episodes")
    print(f"Learning rate: {args.learning_rate}, Gamma: {args.gamma}, Clip epsilon: {args.clip_epsilon}")
    print("-" * 60)

    episode_rewards = []
    episode_lengths = []
    victory_seeds = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode if args.seed is not None else None)
        agent.reset_buffers()
        # Reset NK kills tracker for new episode
        agent._episode_nk_kills = 0

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done and episode_length < args.max_steps:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            shaped_reward = reward
            shaped_reward += 0.5 * next_obs["base_hp_ratio"][0]

            nk_kills = info.get("stats", {}).get("nk_kills", 0)
            if not hasattr(agent, '_episode_nk_kills'):
                agent._episode_nk_kills = 0
            nk_kills_new = max(0, nk_kills - agent._episode_nk_kills)
            if nk_kills_new > 0:
                shaped_reward += nk_kills_new * 5.0
            agent._episode_nk_kills = nk_kills

            shaped_reward += 0.3

            stats = info.get("stats", {})
            if not hasattr(agent, '_episode_soldiers_killed_by_nk'):
                agent._episode_soldiers_killed_by_nk = 0
            soldiers_killed_by_nk = stats.get('soldiers_killed_by_nk_sweep', 0)
            new_soldiers_killed = soldiers_killed_by_nk - agent._episode_soldiers_killed_by_nk
            if new_soldiers_killed > 0:
                shaped_reward -= new_soldiers_killed * 25.0
            agent._episode_soldiers_killed_by_nk = soldiers_killed_by_nk

            base_hp_ratio = next_obs["base_hp_ratio"][0]
            base_hp_before = getattr(agent, '_last_base_hp_ratio', 1.0)
            base_damage = base_hp_before - base_hp_ratio
            if base_damage > 0:
                shaped_reward -= base_damage * 30.0
            agent._last_base_hp_ratio = base_hp_ratio

            enemies_near_base = stats.get('enemies_near_base', 0)
            soldiers_far_from_base = stats.get('soldiers_far_from_base', 0)
            if enemies_near_base > 0 and soldiers_far_from_base > 3:
                shaped_reward -= 5.0

            if enemies_near_base > 0:
                soldiers_near_base = len(env.game_state.soldiers) - soldiers_far_from_base
                if soldiers_near_base > 0:
                    shaped_reward += soldiers_near_base * 2.0

            if done:
                shaped_reward = _apply_terminal_reward_shaping(shaped_reward, info)

            shaped_reward = np.clip(shaped_reward, -500.0, 600.0)
            agent.store_transition(shaped_reward, done)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if done or episode_length % args.update_frequency == 0:
                agent.update()

        if len(agent.states) > 0:
            agent.update()

        final_info = info

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if final_info is not None and final_info.get('victory', False):
            episode_seed = args.seed + episode if args.seed is not None else episode
            victory_seeds.append(episode_seed)

        if (episode + 1) % args.log_interval == 0:
            mean_reward = np.mean(episode_rewards[-args.log_interval:])
            mean_length = np.mean(episode_lengths[-args.log_interval:])
            std_reward = np.std(episode_rewards[-args.log_interval:])

            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"  Reward: {episode_reward:.1f} (mean: {mean_reward:.1f} ± {std_reward:.1f})")
            print(f"  Length: {episode_length} (mean: {mean_length:.1f})")
            print(f"  Victory: {final_info.get('victory', False)}")
            print(f"  Base HP: {final_info.get('base_hp', 0):.0f}")
            print(f"  NK Kills: {final_info.get('stats', {}).get('nk_kills', 0)}")
            print()

            writer.add_scalar("Episode/Reward", episode_reward, episode)
            writer.add_scalar("Episode/Length", episode_length, episode)
            writer.add_scalar("Episode/MeanReward", mean_reward, episode)
            writer.add_scalar("Episode/MeanLength", mean_length, episode)
            writer.add_scalar("Episode/Victory", 1.0 if final_info.get('victory', False) else 0.0, episode)
            writer.add_scalar("Episode/MaxReward", np.max(episode_rewards[-args.log_interval:]), episode)
            writer.add_scalar("Episode/MinReward", np.min(episode_rewards[-args.log_interval:]), episode)
            writer.add_scalar("Training/PolicyEntropy", agent.last_entropy, episode)

        if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
            checkpoint_path = model_dir / f"ppo_agent_ep{episode + 1}.pth"
            agent.save(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")

    final_path = model_dir / "ppo_agent_final.pth"
    agent.save(str(final_path))
    print(f"\nTraining complete! Final model saved: {final_path}")

    if victory_seeds:
        print(f"Found {len(victory_seeds)} victory episodes. Seeds: {victory_seeds[:10]}{'...' if len(victory_seeds) > 10 else ''}")
        print(f"To replay victories, use: python visualize_agent.py --model-path {final_path} --seed <seed>")

    writer.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--update-frequency", type=int, default=32, help="Update frequency (steps)")
    parser.add_argument("--log-interval", type=int, default=5, help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=0, help="Checkpoint save interval (0=only final)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load model from")
    parser.add_argument("--fast-mode", action="store_true", default=True, help="Use fast mode for training")
    parser.add_argument("--fast-multiplier", type=int, default=20, help="Fast mode simulation multiplier")
    parser.add_argument("--reward-scale", type=float, default=20.0, help="Scale factor to divide rewards by")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda for advantage estimation")
    parser.add_argument("--entropy-coef", type=float, default=0.02, help="Entropy coefficient for exploration")
    parser.add_argument("--device", type=str, default=None, help="Device for training (cpu or cuda)")

    args = parser.parse_args()
    train(args)
