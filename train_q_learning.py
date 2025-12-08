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

    return np.clip(reward, -500.0, 600.0)


def train(args):
    """Train the Q-learning agent."""
    env = TowerDefenseWarriorEnv(
        fast_mode=args.fast_mode,
        fast_multiplier=args.fast_multiplier
    )

    obs, _ = env.reset(seed=args.seed)

    model_dir = Path("trained_models/q_learning")
    model_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("runs/q_learning")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"q_learning_{timestamp}"
    tb_log_dir = log_dir / run_name
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard logs: tensorboard --logdir={tb_log_dir}")

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
            feature_dim=28,  # Updated: added base defense features (was 25)
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.eps_start,
            epsilon_end=args.eps_end,
            epsilon_decay_steps=args.eps_decay,
            epsilon_decay_type=args.eps_decay_type,
            reduce_action_search=args.reduce_action_search,
        )

    episode_rewards = []
    episode_lengths = []
    victory_seeds = []

    last_soldiers_killed_by_nk = 0
    last_nk_kills = 0
    last_base_hp = 100.0
    last_soldiers_far_from_base = 0
    last_enemies_near_base = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_length = 0
        final_info = None

        # Reset tracking variables for new episode
        last_soldiers_killed_by_nk = 0
        last_nk_kills = 0
        last_base_hp = 100.0
        last_soldiers_far_from_base = 0
        last_enemies_near_base = 0

        for _ in range(args.max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            ep_length += 1

            if not done:
                stats = info.get('stats', {})

                # Track deltas for reward shaping
                soldiers_killed_by_nk = stats.get('soldiers_killed_by_nk_sweep', 0)
                new_soldiers_killed = soldiers_killed_by_nk - last_soldiers_killed_by_nk
                if new_soldiers_killed > 0:
                    reward -= new_soldiers_killed * 30.0
                last_soldiers_killed_by_nk = soldiers_killed_by_nk

                nk_kills = stats.get('nk_kills', 0)
                new_nk_kills = nk_kills - last_nk_kills
                if new_nk_kills > 0:
                    reward += new_nk_kills * 25.0
                last_nk_kills = nk_kills

                base_hp = info.get('base_hp', 100.0)
                base_damage = last_base_hp - base_hp
                if base_damage > 0:
                    reward -= base_damage * 30.0
                last_base_hp = base_hp

                enemies_near_base = stats.get('enemies_near_base', 0)
                soldiers_far_from_base = stats.get('soldiers_far_from_base', 0)

                if enemies_near_base > 0 and soldiers_far_from_base > len(env.game_state.soldiers) / 2:
                    reward -= 10.0

                if enemies_near_base > 0:
                    soldiers_near_base = len(env.game_state.soldiers) - soldiers_far_from_base
                    if soldiers_near_base > 0:
                        reward += soldiers_near_base * 2.0

                last_soldiers_far_from_base = soldiers_far_from_base
                last_enemies_near_base = enemies_near_base

            if done:
                reward = _apply_terminal_reward_shaping(reward, info)
                final_info = info

            reward = np.clip(reward, -500.0, 600.0)

            agent.update(obs, action, reward, next_obs, done)

            ep_reward += reward
            obs = next_obs

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        if final_info is not None and final_info.get('victory', False):
            victory_seeds.append(args.seed + ep)

        writer.add_scalar("Episode/Reward", ep_reward, ep)
        writer.add_scalar("Episode/Length", ep_length, ep)
        writer.add_scalar("Training/Epsilon", agent.get_epsilon(), ep)
        writer.add_scalar("Agent/Step_Count", agent.step_count, ep)

        if len(episode_rewards) >= 10:
            recent_rewards = episode_rewards[-10:]
            writer.add_scalar("Episode/Reward_MA10", np.mean(recent_rewards), ep)
        if len(episode_rewards) >= 50:
            recent_rewards = episode_rewards[-50:]
            writer.add_scalar("Episode/Reward_MA50", np.mean(recent_rewards), ep)

        if final_info is not None:
            writer.add_scalar("Episode/Base_HP_End", final_info.get("base_hp", 0), ep)
            writer.add_scalar("Episode/Victory", 1 if final_info.get("victory", False) else 0, ep)
            stats = final_info.get('stats', {})
            writer.add_scalar("Stats/Soldiers_Deployed", stats.get('soldiers_deployed', 0), ep)
            writer.add_scalar("Stats/Wights_Killed", stats.get('wights_killed', 0), ep)
            writer.add_scalar("Stats/NK_Kills", stats.get('nk_kills', 0), ep)
            writer.add_scalar("Stats/Soldiers_Killed_by_NK", stats.get('soldiers_killed_by_nk_sweep', 0), ep)

        if (ep + 1) % args.log_interval == 0 or ep == 0:
            recent = episode_rewards[-args.log_interval:] if len(episode_rewards) >= args.log_interval else episode_rewards
            mean_r = float(np.mean(recent))
            std_r = float(np.std(recent)) if len(recent) > 1 else 0.0
            mean_length = float(np.mean(episode_lengths[-args.log_interval:])) if len(episode_lengths) >= args.log_interval else float(np.mean(episode_lengths))

            print(f"Episode {ep+1}/{args.episodes}")
            print(f"  Reward: {ep_reward:.1f} (mean: {mean_r:.1f} ± {std_r:.1f})")
            print(f"  Length: {ep_length} (mean: {mean_length:.1f})")
            if final_info is not None:
                print(f"  Victory: {final_info.get('victory', False)}")
                print(f"  Base HP: {final_info.get('base_hp', 0):.0f}")
                stats = final_info.get('stats', {})
                print(f"  Soldiers Deployed: {stats.get('soldiers_deployed', 0)}")
                print(f"  Wights Killed: {stats.get('wights_killed', 0)}")
                print(f"  NK Kills: {stats.get('nk_kills', 0)}")
                print(f"  Soldiers Killed by NK: {stats.get('soldiers_killed_by_nk_sweep', 0)}")
            print(f"  Epsilon: {agent.get_epsilon():.3f}")
            print()

        if args.save_interval > 0 and (ep + 1) % args.save_interval == 0:
            save_path = model_dir / f"q_agent_ep{ep+1}.pkl"
            agent.save(str(save_path))
            print(f"[Episode {ep+1}] Saved checkpoint → {save_path}")

    if args.save_model:
        final_path = model_dir / args.save_model
        agent.save(str(final_path))
        print(f"Saved final model to {final_path}")

    stats_path = model_dir / "training_stats.npz"
    np.savez(
        str(stats_path),
        episode_rewards=np.array(episode_rewards),
        episode_lengths=np.array(episode_lengths),
        victory_seeds=np.array(victory_seeds) if victory_seeds else np.array([]),
    )
    print(f"Saved training statistics to {stats_path}")
    if victory_seeds:
        print(f"Found {len(victory_seeds)} victory episodes. Seeds: {victory_seeds[:10]}{'...' if len(victory_seeds) > 10 else ''}")

    if len(episode_rewards) > 0:
        writer.add_scalar("Training/Final_Mean_Reward",
                         np.mean(episode_rewards[-args.log_interval:]), 0)
        writer.add_scalar("Training/Total_Episodes", len(episode_rewards), 0)
        writer.add_scalar("Training/Total_Steps", agent.step_count, 0)

    writer.close()
    env.close()
    print("Training complete.")
    if len(episode_rewards) > 0:
        print(f"Final mean reward: {np.mean(episode_rewards[-args.log_interval:]):.1f}")
    print(f"View TensorBoard with: tensorboard --logdir={tb_log_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Approximate Q-Learning agent")

    p.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    p.add_argument("--max-steps", type=int, default=1500, help="Max steps per episode")

    p.add_argument("--alpha", type=float, default=0.001, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.92, help="Discount factor")

    p.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon")
    p.add_argument("--eps-end", type=float, default=0.1, help="Final epsilon")
    p.add_argument("--eps-decay", type=int, default=100_000, help="Epsilon decay steps")
    p.add_argument("--eps-decay-type", type=str, default="linear", choices=["linear", "exponential"],
                   help="Epsilon decay type: 'linear' (default) or 'exponential'")

    p.add_argument("--fast-mode", action="store_true", default=True, help="Use fast mode for training (default: True)")
    p.add_argument("--fast-multiplier", type=int, default=50, help="Fast mode simulation multiplier")

    p.add_argument("--reduce-action-search", action="store_true", default=True, help="Use reduced action space search (faster, enabled by default)")

    p.add_argument("--save-interval", type=int, default=0, help="Save checkpoint every N episodes (0=only final)")
    p.add_argument("--save-model", type=str, default="q_agent_final.pkl", help="Filename for final model")
    p.add_argument("--load-model", type=str, default=None, help="Path to model to load")

    p.add_argument("--log-interval", type=int, default=10, help="Print stats every N episodes")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
