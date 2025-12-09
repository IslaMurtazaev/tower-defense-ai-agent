"""Visualize a trained RL agent (Q-Learning or PPO) playing the Tower Defense game."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pygame
import torch

from environment import td_pyastar as game
from environment.td_warrior_gym_env import TowerDefenseWarriorEnv
from agents.approx_q_agent import ApproximateQAgent
# Only import PPOAgent if needed (for .pth files)
try:
    from agents.ppo_agent import PPOAgent
except ImportError:
    PPOAgent = None


def visualize(args):
    """Run the agent and visualize its gameplay."""
    # Load agent based on file extension to determine agent type
    model_path = Path(args.model_path)

    if model_path.suffix == '.pkl':
        agent_type = "Q-Learning"
    elif model_path.suffix == '.pth':
        agent_type = "PPO"
    else:
        agent_type = "RL"  # Default fallback

    # Create environment with human rendering and agent type for window title
    env = TowerDefenseWarriorEnv(render_mode="human", fast_mode=False, agent_type=agent_type)

    # Load agent based on file extension
    print(f"Loading agent from {args.model_path}")

    if model_path.suffix == '.pkl':
        # Q-Learning agent
        agent = ApproximateQAgent.load(
            str(model_path),
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        # Set epsilon to 0 for evaluation (no exploration)
        agent.epsilon_start = 0.0
        agent.epsilon_end = 0.0
        # agent_type already set above
    elif model_path.suffix == '.pth':
        # PPO agent (separate policy and value networks)
        if PPOAgent is None:
            raise ImportError("PyTorch is required for PPO agents. Install with: pip install torch")
        agent = PPOAgent.load(
            str(model_path),
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        # agent_type already set above
    else:
        raise ValueError(
            f"Unknown model file extension: {model_path.suffix}. "
            f"Use .pkl for Q-Learning or .pth for PPO"
        )

    print(f"\n{'='*60}")
    print("AGENT VISUALIZATION")
    print(f"{'='*60}")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset episode")
    print("  Q     - Quit")
    print(f"{'='*60}\n")

    # Initialize pygame for event handling
    pygame.init()
    clock = pygame.time.Clock()

    episode = 0
    paused = False

    while episode < args.episodes:
        obs, info = env.reset(seed=args.seed + episode)

        # Reset buffers for PPO if needed
        if agent_type == "PPO" and hasattr(agent, 'reset_buffers'):
            agent.reset_buffers()

        done = False
        ep_reward = 0.0
        ep_length = 0
        placement_step = 0

        print(f"\nEpisode {episode + 1}/{args.episodes}")
        print("-" * 60)

        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif event.key == pygame.K_r:
                        print("Resetting episode...")
                        done = True
                    elif event.key == pygame.K_q:
                        print("Quitting...")
                        env.close()
                        return

            if paused:
                clock.tick(60)
                continue

            # Agent selects action (only during placement phase to avoid expensive Q-value computation during combat)
            # During combat, the game auto-runs and actions are ignored, so we skip the expensive select_action() call
            if info.get('phase', 'placement') == 'placement':
                action = agent.select_action(obs)
            else:
                # Combat phase: use a dummy action (won't be used, but needed for step())
                # This avoids calling select_action() which computes Q-values and features every frame
                action = np.array([0, 0, 0], dtype=np.int64)

            # Track units before step
            soldiers_before = len(env.game_state.soldiers)
            heroes_before = len(env.game_state.heroes)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            # Print placement info with Q-Learning decision details
            if info['phase'] == 'placement':
                placement_step += 1
                # Action format: [unit_type, grid_x, grid_y]
                unit_type, grid_x, grid_y = action
                world_x = int((grid_x / env.GRID_SIZE) * 1280)
                world_y = int((grid_y / env.GRID_SIZE) * 800)
                unit_name = "Soldier" if unit_type == 0 else "Hero"

                # Check if placement succeeded
                soldiers_after = len(env.game_state.soldiers)
                heroes_after = len(env.game_state.heroes)
                if unit_type == 0:
                    placement_success = soldiers_after > soldiers_before
                else:
                    placement_success = heroes_after > heroes_before
                success_str = "SUCCESS" if placement_success else "FAILED"

                # Calculate distance to base for debugging
                base_pos = game.BASE_POS
                base_radius = game.BASE_RADIUS
                dist_to_base = math.hypot(world_x - base_pos[0], world_y - base_pos[1])
                min_dist_required = base_radius + 80

                print(f"  Placement {placement_step}: {unit_name} at Grid({grid_x}, {grid_y}) -> World({world_x}, {world_y}) [{success_str}]")
                print(f"    Reward: {reward:.1f}")
                print(f"    Distance to base: {dist_to_base:.1f} (min required: {min_dist_required:.1f})")

                # Show agent-specific info
                if agent_type == "Q-Learning":
                    q_val = agent._q_value(obs, action)
                    features = agent._features(obs, action)
                    print(f"    Q-Value: {q_val:.2f}")
                    print(f"    Features: BaseHP={features[6]:.2f}, Wave={features[7]:.2f}, NK={features[8]:.1f}, DistToBase={features[2]:.2f}")

                total_units = soldiers_after + heroes_after
                print(f"    Units deployed: {soldiers_after} soldiers, {heroes_after} heroes (total: {total_units}/8)")
            elif info['phase'] == 'combat' and placement_step < 10:
                print(f"  Combat phase started! Runtime: {info.get('game_time', 0):.1f}s")
                placement_step = 10  # Mark that we've entered combat

            # Render
            env.render()

            # Update observation
            obs = next_obs

            # Control frame rate
            clock.tick(60)

            # If game is done, show final state until user presses R or Q
            if done:
                print(f"\n  Game Over!")
                print(f"    Victory: {info.get('victory', False)}")
                print(f"    Base HP: {info.get('base_hp', 0)}")
                print(f"    Game Time: {info.get('game_time', 0):.1f}s")
                print(f"    Press R to Restart or Q to Quit...")

                # Keep showing overlay until user presses R or Q
                waiting_for_input = True
                should_reset = False
                while waiting_for_input:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                env.close()
                                return
                            elif event.key == pygame.K_r:
                                print("Restarting episode...")
                                waiting_for_input = False
                                should_reset = True
                                break
                    env.render()
                    clock.tick(60)

                if should_reset:
                    # Reset the episode
                    obs, info = env.reset(seed=args.seed + episode)

                    # Reset buffers for PPO if needed
                    if agent_type == "PPO" and hasattr(agent, 'reset_buffers'):
                        agent.reset_buffers()

                    done = False
                    ep_reward = 0.0
                    ep_length = 0
                    placement_step = 0
                    print(f"\nEpisode {episode + 1}/{args.episodes} (Restarted)")
                    print("-" * 60)
                    continue
                break

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Reward: {ep_reward:.1f}")
        print(f"  Length: {ep_length} steps")
        print(f"  Victory: {info.get('victory', False)}")
        print(f"  Base HP: {info.get('base_hp', 0)}")
        print(f"  Soldiers Deployed: {info.get('stats', {}).get('soldiers_deployed', 0)}")
        print(f"  Wights Killed: {info.get('stats', {}).get('wights_killed', 0)}")
        print(
            f"  NKs Defeated: {info.get('stats', {}).get('nk_kills', 0)}/"
            f"{len(game.NK_SCHEDULE_TIMES)}"
        )
        print(f"  Game Time: {info.get('game_time', 0):.1f}s")
        print("-" * 60)

        episode += 1

        # Wait a bit before next episode
        if episode < args.episodes:
            time.sleep(2)

    env.close()
    print("\nVisualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained RL agent (Q-Learning or PPO)"
    )
    parser.add_argument(
        "--model-path", type=str,
                       default="trained_models/q_learning/q_agent_final.pkl",
        help="Path to trained model (.pkl for Q-Learning, .pth for PPO)"
    )
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to visualize")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return

    visualize(args)


if __name__ == "__main__":
    main()
