"""Visualize the A* baseline agent playing the Tower Defense game."""
import argparse
import numpy as np
import pygame
import time
from pathlib import Path

from environment import td_pyastar as game
from environment.td_warrior_gym_env import TowerDefenseWarriorEnv
from agents.astar_baseline_agent import AStarBaselineAgent


def visualize(args):
    """Run the A* baseline agent and visualize its gameplay."""
    # Create environment with human rendering and agent type for window title
    env = TowerDefenseWarriorEnv(render_mode="human", fast_mode=False, agent_type="A* Baseline")

    # Create agent
    print("Initializing A* baseline agent...")
    agent = AStarBaselineAgent(
        observation_space=env.observation_space,
        action_space=env.action_space
    )

    print(f"\n{'='*60}")
    print("A* BASELINE AGENT VISUALIZATION")
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

        print(f"Episode {episode + 1}/{args.episodes}")
        print(f"Placement phase: Place 6 soldiers and 2 heroes")

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
                        break
                    elif event.key == pygame.K_q:
                        env.close()
                        return

            if paused:
                # Still render when paused
                env.render()
                clock.tick(60)
                continue

            # Agent selects action
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

            # Track placement phase with A* reasoning
            if info.get('phase') == 'placement':
                placement_step += 1
                if placement_step <= 8:  # 6 soldiers + 2 heroes = 8 units
                    # Action format: [unit_type, grid_x, grid_y]
                    unit_type, grid_x, grid_y = action
                    world_x = int((grid_x / env.GRID_SIZE) * 1280)
                    world_y = int((grid_y / env.GRID_SIZE) * 800)
                    unit_name = "Soldier" if unit_type == 0 else "Hero"

                    # Get A* decision reasoning
                    grid = obs["grid"]
                    enemy_positions = []
                    night_king_positions = []
                    for y in range(agent.grid_size):
                        for x in range(agent.grid_size):
                            cell_val = grid[y, x]
                            world_x_cell = (x / agent.grid_size) * 1280
                            world_y_cell = (y / agent.grid_size) * 800
                            if cell_val == 3:
                                enemy_positions.append((world_x_cell, world_y_cell))
                            elif cell_val == 4:
                                night_king_positions.append((world_x_cell, world_y_cell))

                    # Determine reasoning (simplified - can't access private methods)
                    if night_king_positions:
                        reason = "A*: Night King defense (60% base-to-NK)"
                    elif enemy_positions:
                        # Estimate: if multiple enemies, likely choke point
                        if len(enemy_positions) >= 3:
                            reason = f"A*: Choke point (multiple paths converge)"
                        else:
                            reason = "A*: Intercept position (midpoint enemy-base)"
                    else:
                        reason = "A*: Defensive ring (no enemies)"

                    print(f"  Placement {placement_step}: {unit_name} at Grid({grid_x}, {grid_y}) -> World({world_x}, {world_y})")
                    print(f"    {reason}")
                    print(f"    Threats: {len(enemy_positions)} wights, {len(night_king_positions)} Night Kings")
            elif info.get('phase') == 'combat' and placement_step < 10:
                print(f"  Combat phase started! Runtime: {info.get('game_time', 0):.1f}s")
                placement_step = 10  # Mark that we've entered combat

            # Render the environment
            env.render()
            clock.tick(60)  # 60 FPS

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
                    done = False
                    ep_reward = 0.0
                    ep_length = 0
                    placement_step = 0
                    print(f"\nEpisode {episode + 1}/{args.episodes} (Restarted)")
                    print(f"Placement phase: Place 6 soldiers and 2 heroes")
                    continue
                break

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Reward: {ep_reward:.1f}")
        print(f"  Length: {ep_length} steps")
        print(f"  Victory: {info.get('victory', False)}")
        print(f"  Base HP: {info.get('base_hp', 0)}/{info.get('base_hp', 0) + info.get('stats', {}).get('base_hits', 0)}")
        print(f"  Soldiers Deployed: {info.get('stats', {}).get('soldiers_deployed', 0)}")
        print(f"  Wights Killed: {info.get('stats', {}).get('wights_killed', 0)}")
        print(
            f"  NKs Defeated: {info.get('stats', {}).get('nk_kills', 0)}/"
            f"{len(game.NK_SCHEDULE_TIMES)}"
        )
        print(f"  Game Time: {info.get('game_time', 0):.1f}s")

        episode += 1

        if episode < args.episodes:
            print("\n" + "-" * 60)
            input("Press Enter to continue to next episode...")

    env.close()
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize A* baseline agent")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to visualize")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    visualize(args)
