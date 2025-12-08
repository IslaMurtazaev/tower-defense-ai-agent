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
            feature_dim=24,  # UPDATED: Enhanced with NK avoidance + base defense features
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

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode if args.seed is not None else None)
        agent.reset_buffers()
        # Reset NK kills tracker for new episode
        agent._episode_nk_kills = 0

        episode_reward = 0.0
        episode_length = 0
        done = False

        # Episodes end naturally when victory or defeat occurs (terminated=True)
        # max_steps is just a safety timeout to prevent infinite loops
        while not done and episode_length < args.max_steps:
            # Select action
            action = agent.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # IMPROVED: Combine environment reward with additional shaping
            # Environment already provides rich rewards (NK kills: +50, soldier deaths: -15, etc.)
            # We add light additional shaping for emphasis, not replace the signal
            shaped_reward = reward  # Start with environment reward (preserves rich signal)

            # Additional shaping for emphasis (environment already has strong signals)
            # Base HP survival bonus (continuous signal)
            shaped_reward += 0.5 * next_obs["base_hp_ratio"][0]  # Reduced from 1.0 (environment already rewards this)

            # Track NK kills increment for additional emphasis
            nk_kills = info.get("stats", {}).get("nk_kills", 0)
            if not hasattr(agent, '_episode_nk_kills'):
                agent._episode_nk_kills = 0
            nk_kills_new = max(0, nk_kills - agent._episode_nk_kills)
            if nk_kills_new > 0:
                # Environment already gives +50 per NK kill, we add +5 for emphasis
                shaped_reward += nk_kills_new * 5.0  # Additional emphasis (was 2.0, now 5.0)
            agent._episode_nk_kills = nk_kills

            # Wave survival bonus (small continuous reward)
            shaped_reward += 0.3  # Reduced from 0.5 (environment already rewards survival)

            # Light penalty for losing soldiers (environment already penalizes -15 per soldier killed by NK)
            soldiers_remaining = next_obs["soldiers_remaining"][0]
            soldiers_lost = 6 - soldiers_remaining  # 6 is max soldiers
            if soldiers_lost > 0:
                shaped_reward -= soldiers_lost * 0.3  # Light additional penalty (was 0.5, now 0.3)

            # Use combined reward (environment + shaping)
            agent.store_transition(shaped_reward, done)
            # Track original reward for logging
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            # Update agent periodically (every N steps or at episode end)
            if done or episode_length % args.update_frequency == 0:
                agent.update()

        # Final update if episode ended
        if len(agent.states) > 0:
            agent.update()

        # Get final info (from last step, which should have the final state)
        final_info = info

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Logging
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

            # TensorBoard logging
            writer.add_scalar("Episode/Reward", episode_reward, episode)
            writer.add_scalar("Episode/Length", episode_length, episode)
            writer.add_scalar("Episode/MeanReward", mean_reward, episode)
            writer.add_scalar("Episode/MeanLength", mean_length, episode)
            writer.add_scalar("Episode/Victory", 1.0 if final_info.get('victory', False) else 0.0, episode)
            writer.add_scalar("Episode/MaxReward", np.max(episode_rewards[-args.log_interval:]), episode)
            writer.add_scalar("Episode/MinReward", np.min(episode_rewards[-args.log_interval:]), episode)
            writer.add_scalar("Training/PolicyEntropy", agent.last_entropy, episode)

        # Save checkpoint (only if save_interval > 0)
        if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
            checkpoint_path = model_dir / f"ppo_agent_ep{episode + 1}.pth"
            agent.save(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = model_dir / "ppo_agent_final.pth"
    agent.save(str(final_path))
    print(f"\nTraining complete! Final model saved: {final_path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes (reduced for speed)")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode (reduced for speed)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (increased from 3e-4 for faster learning)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--update-frequency", type=int, default=32, help="Update frequency (steps, more frequent updates for faster learning)")
    parser.add_argument("--log-interval", type=int, default=5, help="Logging interval (reduced for less I/O)")
    parser.add_argument("--save-interval", type=int, default=0, help="Checkpoint save interval (0=only final, reduces I/O)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load model from")
    parser.add_argument("--fast-mode", action="store_true", default=True, help="Use fast mode for training (default: True)")
    parser.add_argument("--fast-multiplier", type=int, default=20, help="Fast mode simulation multiplier (default: 20, very fast)")
    parser.add_argument("--reward-scale", type=float, default=20.0, help="Scale factor to divide rewards by inside PPO (stabilizes training)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda for advantage estimation (0.95-0.98 recommended)")
    parser.add_argument("--entropy-coef", type=float, default=0.02, help="Entropy coefficient for exploration (0.01-0.02 recommended)")
    parser.add_argument("--device", type=str, default=None, help="Device for training (cpu or cuda). Default: auto-detect")

    args = parser.parse_args()
    train(args)
