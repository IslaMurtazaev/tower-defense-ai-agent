# Complete RL Training Guide

**Comprehensive guide for training AI agents to learn optimal defender placement strategies**

This guide explains how to train an AI agent to learn optimal defender placement strategies using Reinforcement Learning.

## 🎯 Overview

The RL agent learns through trial and error to:

- Choose between Footman and Archer for each placement
- Find optimal positions on the map
- Balance unit composition
- Maximize defense effectiveness

**Algorithm**: PPO (Proximal Policy Optimization)  
**Framework**: Stable-Baselines3 + Gymnasium  
**Training Time**: ~25-30 minutes for 500K steps (8 parallel envs with fast mode) ⚡  
**Speed**: **20-40x faster** than baseline with optimizations!

## 📦 Installation

### Install RL Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `stable-baselines3[extra]` - RL algorithms with all features
- `tensorboard` - Training visualization
- `torch` - Neural network backend
- `gymnasium` - Environment interface
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `rich` - Enhanced terminal output

**⚠️ Troubleshooting Dependencies:**

If you get `ImportError: You must install tqdm and rich`:

```bash
# Option 1: Install missing packages
pip install tqdm rich

# Option 2: Reinstall with extras
pip install stable-baselines3[extra]

# Option 3: Reinstall everything
pip uninstall -y stable-baselines3
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import tqdm; import rich; print('✓ Success!')"
```

## 🚀 Quick Start

### 1. Basic Training

Train an agent with default settings:

```bash
python train_rl_agent.py
```

This will:

- Create 4 parallel environments
- Enable fast mode (5x faster simulation)
- Train for 500,000 timesteps (~50 minutes with fast mode)
- Save checkpoints every 50,000 steps
- Save best model based on evaluation

**⚡ Speed Tip**: Use `--n-envs 8` for 2x faster training!

### 2. Monitor Training Progress

Open TensorBoard in another terminal:

```bash
tensorboard --logdir trained_models/logs
```

Then open http://localhost:6006 in your browser to see:

- Episode rewards over time
- Success rate
- Learning curves
- Policy loss

### 3. Evaluate Trained Agent

After training completes:

```bash
python evaluate_agent.py --model trained_models/best_model.zip --episodes 20
```

### 4. Visualize Agent Playing

Watch the AI play with visualization:

```bash
python evaluate_agent.py --model trained_models/best_model.zip --visualize --episodes 3
```

### 5. Compare with Random Policy

See how much better the AI is than random:

```bash
python evaluate_agent.py --model trained_models/best_model.zip --compare --episodes 20
```

## ⚙️ Training Options

### Custom Training Configuration

```bash
python train_rl_agent.py \
    --total-timesteps 1000000 \
    --n-envs 8 \
    --learning-rate 0.0003 \
    --save-dir my_models \
    --seed 123
```

### Key Parameters

| Parameter           | Default        | Description                           |
| ------------------- | -------------- | ------------------------------------- |
| `--total-timesteps` | 500,000        | Total training steps                  |
| `--n-envs`          | 4              | Parallel environments (more = faster) |
| `--learning-rate`   | 3e-4           | Learning rate                         |
| `--n-steps`         | 2048           | Steps per policy update               |
| `--batch-size`      | 64             | Batch size for training               |
| `--n-epochs`        | 10             | Epochs per update                     |
| `--gamma`           | 0.99           | Discount factor                       |
| `--save-dir`        | trained_models | Save directory                        |
| `--save-freq`       | 50,000         | Checkpoint frequency                  |
| `--eval-freq`       | 25,000         | Evaluation frequency                  |

### Training Presets

**Quick Test (5 minutes):**

```bash
python train_rl_agent.py --total-timesteps 50000 --n-envs 4
```

**Fast Training (30 minutes) - RECOMMENDED:**

```bash
python train_rl_agent.py --total-timesteps 500000 --n-envs 8
```

**Standard Training (50 minutes):**

```bash
python train_rl_agent.py --total-timesteps 500000 --n-envs 4
```

**Intensive Training (2 hours, best results):**

```bash
python train_rl_agent.py --total-timesteps 2000000 --n-envs 8 --learning-rate 0.0001
```

**Maximum Speed (15 minutes, powerful CPUs):**

```bash
python train_rl_agent.py --total-timesteps 500000 --n-envs 16 --n-steps 1024
```

💡 **Fast mode is enabled by default for 5x faster training!**

### Continue Training

Resume from a previously saved model:

```bash
python train_rl_agent.py --continue-training --total-timesteps 1000000
```

## 🧠 How It Works

### Observation Space

The agent observes:

- **32x32 Grid** showing positions of:
  - Castle (value: 1)
  - Footmen (value: 2)
  - Archers (value: 3)
  - Wights (value: 4)
  - Empty (value: 0)
- **Soldiers Remaining**: How many more to place
- **Current Wave**: Which wave (0-5)
- **Castle HP Ratio**: Health percentage (0.0-1.0)

**Total Observation Size**: 1,024 + 3 = 1,027 values

### Action Space

The agent outputs:

- **Soldier Type**: 0 = Footman, 1 = Archer
- **Grid X Position**: 0-31
- **Grid Y Position**: 0-31

**Total Actions**: 2 × 32 × 32 = 2,048 possible actions

### Reward Function

The agent learns from these rewards:

**During Combat:**

- +10 per wight killed
- -50 per soldier killed
- -5 per castle damage point

**At Episode End:**

- +500 for victory
- -200 for defeat
- +50 per wave completed
- +200 × (remaining HP ratio)
- +10 per surviving soldier

**Goal**: Maximize total reward by defending effectively!

### Episode Structure

1. **Placement Phase** (10 actions)
   - Agent places all 20 soldiers
   - Invalid placements penalized (-1 reward)
2. **Combat Phase** (auto-run)
   - Game simulates until natural completion
   - Agent receives rewards based on performance
3. **Episode End** (Victory or Defeat only)
   - Victory: All 400 wights defeated
   - Defeat: Castle destroyed
   - Final reward calculated
   - Environment resets for next episode

## 📊 Expected Results

### Training Progress

| Timesteps | Expected Performance             |
| --------- | -------------------------------- |
| 0-50K     | Random exploration, low rewards  |
| 50K-100K  | Learning basic placement         |
| 100K-200K | Discovering effective strategies |
| 200K-500K | Refining optimal placement       |
| 500K+     | Near-optimal performance         |

### Performance Metrics

**Untrained (Random)**:

- Win Rate: ~0-5%
- Avg Reward: -200 to 0
- Avg Waves: 1-2
- Avg Kills: 20-50

**Well-Trained (500K steps)**:

- Win Rate: ~20-40%
- Avg Reward: 0 to +200
- Avg Waves: 3-4
- Avg Kills: 150-250

**Excellent (1M+ steps)**:

- Win Rate: ~40-60%
- Avg Reward: +100 to +400
- Avg Waves: 4-5
- Avg Kills: 200-300

## 🎓 Learning Strategies

### What the Agent Discovers

Through training, successful agents typically learn to:

1. **Coverage Placement**
   - Spread soldiers across spawn areas
   - Avoid clustering in one location
2. **Unit Composition**

   - Balance between footmen and archers
   - Use archers for early detection
   - Use footmen for high-damage intercepts

3. **Positioning Strategy**

   - Place near enemy spawn points
   - Create overlapping detection zones
   - Protect castle approaches

4. **Spatial Awareness**
   - Avoid placing too close to castle (invalid)
   - Use map edges effectively
   - Consider patrol routes

## 🔧 Troubleshooting

### Training Too Slow

**Problem**: Training taking too long  
**Solutions**:

- Increase `--n-envs` (use more CPU cores)
- Decrease `--total-timesteps` for initial testing
- Use faster hardware (GPU if available)

### Poor Performance

**Problem**: Agent not learning well  
**Solutions**:

- Train longer (500K+ timesteps)
- Adjust learning rate (`--learning-rate 0.0001`)
- Check TensorBoard for learning progress
- Try different seeds (`--seed`)

### Out of Memory

**Problem**: System runs out of RAM  
**Solutions**:

- Reduce `--n-envs`
- Reduce `--n-steps`
- Close other applications

### Model Not Found

**Problem**: Can't load saved model  
**Solutions**:

- Check path is correct
- Ensure training completed and saved
- Look in `trained_models/` directory

## 📈 Advanced Training

### Hyperparameter Tuning

Experiment with different configurations:

```python
# In train_rl_agent.py, you can modify PPO parameters:
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,      # Try: 1e-4, 3e-4, 1e-3
    n_steps=2048,            # Try: 1024, 2048, 4096
    batch_size=64,           # Try: 32, 64, 128
    n_epochs=10,             # Try: 5, 10, 20
    gamma=0.99,              # Try: 0.95, 0.99, 0.999
    clip_range=0.2,          # Try: 0.1, 0.2, 0.3
)
```

### Curriculum Learning

Train progressively on harder difficulties:

1. Start with easier settings (fewer enemies)
2. Train until good performance
3. Increase difficulty
4. Continue training with `--continue-training`

### Custom Rewards

Modify `td_gym_env.py` to experiment with different reward structures:

```python
# Example: Emphasize castle protection
reward -= castle_damage_this_step * 10.0  # Was 5.0

# Example: Reward efficient soldier usage
if self.game.phase == GamePhase.VICTORY:
    soldiers_used = len(self.game.soldiers)
    reward += (10 - soldiers_used) * 20  # Bonus for using fewer
```

## 📁 File Structure

```
tower_defense/
├── train_rl_agent.py           # Training script
├── evaluate_agent.py           # Evaluation script
├── trained_models/             # Saved models directory
│   ├── best_model.zip         # Best performing model
│   ├── final_model.zip        # Final trained model
│   ├── checkpoints/           # Periodic checkpoints
│   └── logs/                  # TensorBoard logs
├── environment/
│   ├── td_gym_env.py          # Gymnasium environment
│   └── td_game_core.py        # Game logic
└── RL_TRAINING_GUIDE.md       # This file
```

## 🎯 Next Steps

After training a good agent:

1. **Analyze Strategies**

   - Visualize placement patterns
   - Study unit composition choices
   - Compare different trained models

2. **Experiment with Variants**

   - Train on different difficulties
   - Try different reward functions
   - Test with different soldier counts

3. **Compare Algorithms**

   - Try A2C instead of PPO
   - Test DQN for discrete actions
   - Experiment with SAC

4. **Deploy Best Strategy**
   - Extract learned placement patterns
   - Convert to heuristic rules
   - Use for human gameplay tips

## 📚 Resources

- **Stable-Baselines3 Docs**: https://stable-baselines3.readthedocs.io/
- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **Gymnasium Docs**: https://gymnasium.farama.org/
- **RL Course**: https://spinningup.openai.com/

---

**Ready to train your AI commander? Let's defend Winterfell! 🏰⚔️**
