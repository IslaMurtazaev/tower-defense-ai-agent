"""Train an Approximate Q-Learning agent on TowerDefenseWarriorEnv."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        raise ImportError("Please install torch or tensorboardX for TensorBoard logging")

from agents.approx_q_agent import ApproximateQAgent
from environment.td_warrior_gym_env import TowerDefenseWarriorEnv


def train(args):
    """Train the Q-learning agent."""
    env = TowerDefenseWarriorEnv()
    obs, _ = env.reset(seed=args.seed)

    # Create models directory if it doesn't exist
    model_dir = Path("trained_models/q_learning")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard log directory
    log_dir = Path("runs/q_learning")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"q_learning_{timestamp}"
    tb_log_dir = log_dir / run_name
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard logs: tensorboard --logdir={tb_log_dir}")

    # Load existing model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        agent = ApproximateQAgent.load(
            args.load_model,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
    else:
        agent = ApproximateQAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=16,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.eps_start,
            epsilon_end=args.eps_end,
            epsilon_decay_steps=args.eps_decay,
        )

    episode_rewards = []
    episode_lengths = []

    # Main training loop
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_length = 0

        # Run episode until done or max steps
        for _ in range(args.max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update
            agent.update(obs, action, reward, next_obs, done)
            ep_reward += reward
            ep_length += 1

            obs = next_obs
            if done:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        # Log metrics to TensorBoard
        current_eps = agent.get_epsilon()
        writer.add_scalar("Episode/Reward", ep_reward, ep + 1)
        writer.add_scalar("Episode/Length", ep_length, ep + 1)
        writer.add_scalar("Training/Epsilon", current_eps, ep + 1)
        writer.add_scalar("Agent/Step_Count", agent.step_count, ep + 1)

        # Moving averages for smoother visualization
        if len(episode_rewards) >= 10:
            recent_rewards = episode_rewards[-10:]
            writer.add_scalar("Episode/Reward_MA10", np.mean(recent_rewards), ep + 1)
        if len(episode_rewards) >= 50:
            recent_rewards = episode_rewards[-50:]
            writer.add_scalar("Episode/Reward_MA50", np.mean(recent_rewards), ep + 1)

        if (ep + 1) % max(1, args.log_interval) == 0:
            recent = episode_rewards[-args.log_interval:]
            mean_r = float(np.mean(recent))
            std_r = float(np.std(recent))
            # Epsilon already calculated above for TensorBoard
            msg = (f"Episode {ep+1}/{args.episodes} | Reward: {ep_reward:.1f} | "
                   f"Recent mean: {mean_r:.1f} ± {std_r:.1f} | Epsilon: {current_eps:.3f}")
            print(msg)

        # Periodic checkpoints
        if args.save_interval > 0 and (ep + 1) % args.save_interval == 0:
            checkpoint_path = model_dir / f"q_agent_ep{ep+1}.pkl"
            agent.save(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    if args.save_model:
        final_path = model_dir / args.save_model
        agent.save(str(final_path))
        print(f"Saved final model to {final_path}")

    # Save training statistics
    stats_path = model_dir / "training_stats.npz"
    np.savez(
        str(stats_path),
        episode_rewards=np.array(episode_rewards),
        episode_lengths=np.array(episode_lengths),
    )
    print(f"Saved training statistics to {stats_path}")

    # Log final statistics
    if len(episode_rewards) > 0:
        writer.add_scalar("Training/Final_Mean_Reward",
                         np.mean(episode_rewards[-args.log_interval:]), 0)
        writer.add_scalar("Training/Total_Episodes", len(episode_rewards), 0)
        writer.add_scalar("Training/Total_Steps", agent.step_count, 0)

    writer.close()
    env.close()
    print("Training complete.")
    print(f"Final mean reward: {np.mean(episode_rewards[-args.log_interval:]):.1f}")
    print(f"View TensorBoard with: tensorboard --logdir={tb_log_dir}")


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train Approximate Q-Learning agent")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--eps-decay", type=int, default=50_000, help="Epsilon decay steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save-model", type=str, default="q_agent_final.pkl",
                        help="Filename for final model")
    parser.add_argument("--load-model", type=str, default=None, help="Path to model to load")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save checkpoint every N episodes (0 to disable)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
