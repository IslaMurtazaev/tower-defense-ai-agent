"""Plot learning curves from training statistics."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_curve(args):
    """Load and plot training statistics."""
    stats_path = Path(args.stats_path)

    if not stats_path.exists():
        print(f"Error: Statistics file not found: {stats_path}")
        return

    data = np.load(str(stats_path))
    episode_rewards = data['episode_rewards']
    episode_lengths = data.get('episode_lengths', None)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2 if episode_lengths is not None else 1, figsize=(12, 5))
    if episode_lengths is None:
        axes = [axes]

    # Plot rewards
    ax = axes[0]
    episodes = np.arange(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode reward')

    # Moving average
    window = min(50, len(episode_rewards) // 10)
    if window > 1:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        moving_episodes = episodes[window-1:]
        ax.plot(moving_episodes, moving_avg, color='red', linewidth=2, label=f'Moving avg ({window})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training: Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot episode lengths if available
    if episode_lengths is not None:
        ax = axes[1]
        ax.plot(episodes, episode_lengths, alpha=0.3, color='green', label='Episode length')

        if window > 1:
            moving_avg_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            ax.plot(moving_episodes, moving_avg_len, color='orange', linewidth=2, label=f'Moving avg ({window})')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Training: Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = args.output if args.output else str(stats_path).replace('.npz', '_curve.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--stats-path", type=str,
                       default="trained_models/q_learning/training_stats.npz",
                       help="Path to training statistics .npz file")
    parser.add_argument("--output", type=str, default=None, help="Output path for plot")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")

    args = parser.parse_args()
    plot_curve(args)


if __name__ == "__main__":
    main()
