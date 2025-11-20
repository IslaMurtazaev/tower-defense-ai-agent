"""
Train an RL agent to play Tower Defense
Uses PPO (Proximal Policy Optimization) from Stable-Baselines3
"""
import os
import sys
import argparse
from datetime import datetime

# Add environment to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from td_gym_env import TowerDefenseEnv


class TowerDefenseWrapper(gym.Wrapper):
    """Wrapper to flatten Dict observation space for easier training"""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Flatten observation space
        grid_size = env.observation_space['grid'].shape[0] * env.observation_space['grid'].shape[1]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=4,
            shape=(grid_size + 3,),  # grid + 3 additional features
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info
    
    def _flatten_obs(self, obs_dict):
        """Flatten dict observation to single array"""
        grid = obs_dict['grid'].flatten()
        soldiers_remaining = obs_dict['soldiers_remaining'][0] / 10.0  # Normalize
        current_wave = obs_dict['current_wave'][0] / 5.0  # Normalize
        castle_hp_ratio = obs_dict['castle_hp_ratio'][0]
        
        return np.concatenate([
            grid.astype(np.float32) / 4.0,  # Normalize grid values
            [soldiers_remaining, current_wave, castle_hp_ratio]
        ])


def make_env(rank, seed=0, fast_mode=True):
    """Create a single environment"""
    def _init():
        env = TowerDefenseEnv(fast_mode=fast_mode)  # Enable fast mode for training
        env = TowerDefenseWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_agent(args):
    """Train the RL agent"""
    
    print("=" * 70)
    print("WINTERFELL TOWER DEFENSE - RL TRAINING")
    print("=" * 70)
    print(f"Algorithm: PPO (Proximal Policy Optimization)")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 70)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "logs"), exist_ok=True)
    
    # Create vectorized environments
    fast_mode_str = "ENABLED (5x speed)" if args.fast_mode else "DISABLED (accurate)"
    print(f"\nCreating {args.n_envs} parallel environments...")
    print(f"Fast mode: {fast_mode_str}")
    
    if args.n_envs == 1:
        env = DummyVecEnv([make_env(0, args.seed, fast_mode=args.fast_mode)])
    else:
        env = SubprocVecEnv([make_env(i, args.seed, fast_mode=args.fast_mode) for i in range(args.n_envs)])
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(0, args.seed + 1000, fast_mode=args.fast_mode)])
    
    print("✓ Environments created")
    
    # Create or load model
    if args.continue_training and os.path.exists(os.path.join(args.save_dir, "best_model.zip")):
        print(f"\nLoading existing model from {args.save_dir}/best_model.zip")
        model = PPO.load(
            os.path.join(args.save_dir, "best_model.zip"),
            env=env,
            tensorboard_log=os.path.join(args.save_dir, "logs")
        )
        print("✓ Model loaded")
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=os.path.join(args.save_dir, "logs")
        )
        print("✓ Model created")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=os.path.join(args.save_dir, "checkpoints"),
        name_prefix="td_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=os.path.join(args.save_dir, "logs"),
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Monitor training progress with TensorBoard:")
    print(f"  tensorboard --logdir {os.path.join(args.save_dir, 'logs')}")
    print("=" * 70 + "\n")
    
    # Train
    try:
        # Try with progress bar, fallback to no progress bar if tqdm/rich not installed
        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=True
            )
        except ImportError:
            print("\n⚠️  Progress bar libraries not installed. Training without progress bar...")
            print("   Install with: pip install tqdm rich")
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=False
            )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save final model
    final_path = os.path.join(args.save_dir, "final_model")
    model.save(final_path)
    print(f"\n✓ Final model saved to {final_path}.zip")
    
    # Clean up
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best model: {os.path.join(args.save_dir, 'best_model.zip')}")
    print(f"Final model: {final_path}.zip")
    print(f"Logs: {os.path.join(args.save_dir, 'logs')}")
    print("\nTo evaluate the trained agent, run:")
    print(f"  python evaluate_agent.py --model {os.path.join(args.save_dir, 'best_model.zip')}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Tower Defense")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=500000,
                       help="Total training timesteps (default: 500000)")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="Number of steps per update (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="Number of epochs (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor (default: 0.99)")
    
    # Saving and evaluation
    parser.add_argument("--save-dir", type=str, default="trained_models",
                       help="Directory to save models (default: trained_models)")
    parser.add_argument("--save-freq", type=int, default=50000,
                       help="Save checkpoint every N steps (default: 50000)")
    parser.add_argument("--eval-freq", type=int, default=25000,
                       help="Evaluate every N steps (default: 25000)")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                       help="Number of evaluation episodes (default: 5)")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--continue-training", action="store_true",
                       help="Continue training from best_model.zip if exists")
    
    # Speed optimizations
    parser.add_argument("--fast-mode", action="store_true", default=True,
                       help="Enable fast simulation mode (5x speed, default: True)")
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false",
                       help="Disable fast mode for accurate simulation")
    
    args = parser.parse_args()
    
    try:
        train_agent(args)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

