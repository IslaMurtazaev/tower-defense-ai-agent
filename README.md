# 🏰 Winterfell Tower Defense - AI Strategy Learning

A tower defense game inspired by the Battle of Winterfell, featuring intelligent soldier AI with A* pathfinding and reinforcement learning for optimal defender placement.

## 🎮 Quick Start

### Play the Game (Human)
```bash
pip install pygame
cd environment
python td_pygame.py
```

**Controls:** Click to place soldiers, SPACE to start, R to reset

### Train an AI Agent
```bash
pip install -r requirements.txt
python train_rl_agent.py
```

**Monitor training:**
```bash
tensorboard --logdir trained_models/logs
```

**Evaluate trained agent:**
```bash
python evaluate_agent.py --model trained_models/best_model.zip --episodes 20
```

## ✨ Key Features

### Intelligent Soldier AI
- **A* Pathfinding**: Soldiers navigate intelligently to enemies
- **Detection Radius**: 400px (footmen), 450px (archers)
- **Dynamic Behavior**: Chase enemies, return home when idle
- **Two Unit Types**:
  - **Footmen**: Melee (50px range, 30 damage)
  - **Archers**: Ranged (200px range, 10 damage)

### Challenging Combat
- **300 total enemies** across 5 waves (25, 40, 60, 75, 100)
- **Fast spawning**: 0.3s intervals (~3 per second)
- **No delays**: Continuous wave progression
- **10 soldiers** to defend with

### RL Training System
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Training Time**: ~25-30 minutes for 500K steps (8 parallel envs)
- **Speed**: 20-40x faster with optimizations
- **Observation**: 32x32 grid + game state
- **Action**: Unit type + grid position

## 📊 Game Stats

| Metric | Value |
|--------|-------|
| Total Enemies | 300 wights |
| Waves | 5 progressive waves |
| Max Soldiers | 10 defenders |
| Castle HP | 100 |
| Map Size | 1280x800 |
| Win Rate (Random) | 0-5% |
| Win Rate (Trained AI) | 20-60% |

## 🤖 RL Training

### Basic Training
```bash
# Standard training (30 min)
python train_rl_agent.py --n-envs 8 --total-timesteps 500000

# Quick test (5 min)
python train_rl_agent.py --n-envs 4 --total-timesteps 50000

# Long training (2 hours)
python train_rl_agent.py --n-envs 8 --total-timesteps 2000000
```

### Monitor Progress
```bash
tensorboard --logdir trained_models/logs
# Open http://localhost:6006
```

Watch for:
- **Episode reward** trending upward
- **Win rate** increasing
- **Policy loss** decreasing

### Evaluate Agent
```bash
# Standard evaluation
python evaluate_agent.py --model trained_models/best_model.zip --episodes 20

# Visual demo (with enhanced UI showing castle HP, unit types, stats)
python evaluate_agent.py --model trained_models/best_model.zip --visualize

# Compare with random
python evaluate_agent.py --model trained_models/best_model.zip --compare
```

**Visualization Features:**
- Castle HP bar with color coding (green/orange/red)
- Clear unit distinction: Blue circles (Footmen) vs Green triangles (Archers)
- Real-time stats overlay (wave, soldiers, kills)
- Unit legend for easy identification

## 📁 Project Structure

```
tower_defense/
├── environment/
│   ├── td_game_core.py      # Core game logic with A* AI
│   ├── td_pygame.py         # Human-playable interface
│   └── td_gym_env.py        # RL environment
├── train_rl_agent.py        # Train AI agents
├── evaluate_agent.py        # Evaluate & visualize
├── test_game.py             # Test suite
├── requirements.txt         # Dependencies
├── README.md                # This file
├── TRAINING_GUIDE.md        # Detailed RL training docs
└── GAME_GUIDE.md           # Complete game mechanics
```

## 🚀 Performance

### Training Speed
- **4 envs + fast mode**: ~50 min for 500K steps
- **8 envs + fast mode**: ~25 min for 500K steps
- **16 envs + fast mode**: ~12 min for 500K steps

### RL Results
- **0-50K steps**: Random exploration (~0% win)
- **100K steps**: Learning basics (~10% win)
- **500K steps**: Good strategies (~30% win)
- **1M+ steps**: Near-optimal (~50% win)

## 🎓 What the AI Learns

Through training, agents discover:
1. **Coverage**: Spread soldiers across spawn points
2. **Composition**: Balance footmen and archers (40-60% mix)
3. **Positioning**: Place near enemy spawns for early intercept
4. **Detection Zones**: Create overlapping coverage areas

## 🛠️ Common Commands

### Installation
```bash
# Basic gameplay
pip install pygame

# RL training (includes tqdm, rich for progress bars)
pip install -r requirements.txt
```

### Testing
```bash
# Test core game
python test_game.py

# Test RL environment
python test_rl_env.py

# Demo AI behavior
python demo_ai.py
```

### Troubleshooting
```bash
# If missing tqdm/rich
pip install tqdm rich

# Or reinstall with extras
pip install stable-baselines3[extra]
```

## 📖 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete RL training guide (PPO, hyperparameters, monitoring)
- **[GAME_GUIDE.md](GAME_GUIDE.md)** - Game mechanics, controls, and strategy
- **[AI_SYSTEM.md](AI_SYSTEM.md)** - Technical details on A* pathfinding (advanced)

## 🎯 Quick Examples

### Train and Evaluate
```bash
# Train
python train_rl_agent.py --n-envs 8

# Watch in TensorBoard (separate terminal)
tensorboard --logdir trained_models/logs

# Evaluate after training
python evaluate_agent.py --model trained_models/best_model.zip --episodes 20

# Watch AI play
python evaluate_agent.py --model trained_models/best_model.zip --visualize
```

### Speed Training
```bash
# Fast training with default fast mode (5x simulation speed)
python train_rl_agent.py --n-envs 16 --total-timesteps 500000
```

## 🏆 Victory Conditions

- **Victory**: Survive all 5 waves (kill all 300 wights)
- **Defeat**: Castle destroyed (HP reaches 0)

## 🔧 Customization

Edit `environment/td_game_core.py` to adjust:
```python
MAX_SOLDIERS = 10           # Number of defenders
WAVE_DEFINITIONS = [...]    # Enemies per wave
spawn_interval = 0.3        # Spawn speed
```

## 📊 Reward Structure

**During Combat:**
- +10 per wight killed
- -5 per castle damage point
- -1 for invalid placement

**Episode End:**
- +500 for victory
- -200 for defeat
- +50 per wave completed
- Bonus for remaining HP

## 👥 Credits

**CS6660 Final Project**  
Authors: Islam Murtazaev, Leonel Mainsah Ngu, Raymond Frimpong Amoateng

## 📝 License

Academic project - CS6660 Intro to AI

---

**"Winter is here. Can your AI defend Winterfell?"** ❄️🏰⚔️🤖

