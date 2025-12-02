"""Visualize a trained Q-Learning agent playing the Tower Defense game."""
import argparse
import numpy as np
import pygame
import time
from pathlib import Path

from environment.td_warrior_gym_env import TowerDefenseWarriorEnv
from agents.approx_q_agent import ApproximateQAgent


def visualize(args):
    """Run the agent and visualize its gameplay."""
    # Create environment with human rendering
    env = TowerDefenseWarriorEnv(render_mode="human", fast_mode=False)

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

            # Agent selects action
            action = agent.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            # Print placement info
            if info['phase'] == 'placement':
                placement_step += 1
                grid_x, grid_y = action
                world_x = int((grid_x / env.GRID_SIZE) * 1280)
                world_y = int((grid_y / env.GRID_SIZE) * 800)
                print(f"  Placement {placement_step}: Grid({grid_x}, {grid_y}) -> World({world_x}, {world_y})")

            # Render
            env.render()

            # Update observation
            obs = next_obs

            # Control frame rate
            clock.tick(60)

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Reward: {ep_reward:.1f}")
        print(f"  Length: {ep_length} steps")
        print(f"  Victory: {info.get('victory', False)}")
        print(f"  Base HP: {info.get('base_hp', 0)}/{200}")
        print(f"  Soldiers Alive: {info.get('soldiers_alive', 0)}")
        print(f"  Wights Killed: {info.get('stats', {}).get('wights_killed', 0)}")
        print("-" * 60)

        episode += 1

        # Wait a bit before next episode
        if episode < args.episodes:
            time.sleep(2)

    env.close()
    print("\nVisualization complete!")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained Q-Learning agent")
    parser.add_argument("--model-path", type=str,
                       default="trained_models/q_learning/q_agent_final.pkl",
                       help="Path to trained model (.pkl)")
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
