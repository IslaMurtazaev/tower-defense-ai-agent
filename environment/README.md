# Tower Defense Environment

This directory contains the complete Tower Defense game implementation.

## Files

### `td_game_core.py`

Core game logic with no rendering dependencies. Contains:

- `TowerDefenseGame`: Main game state manager
- `Soldier`: Footman and Archer classes
- `Wight`: Enemy class
- `Castle`: The structure to defend
- Game phases, combat mechanics, and wave management

### `td_pygame.py`

Pygame-based visualization for human players. Run with:

```bash
python td_pygame.py
```

Features:

- Interactive soldier placement
- Real-time combat visualization
- HUD with stats and controls
- Game over screens

### `td_gym_env.py`

Gymnasium environment wrapper for RL agent training. Use with:

```python
from td_gym_env import TowerDefenseEnv
env = TowerDefenseEnv()
```

Features:

- Standard Gymnasium API
- Grid-based observations
- Structured action space
- Reward shaping for RL

### `__init__.py`

Package initialization, exports main classes.

## Quick Start

### Human Play

```bash
python td_pygame.py
```

### RL Training Example

```python
import gymnasium as gym
from td_gym_env import TowerDefenseEnv

env = TowerDefenseEnv()
obs, info = env.reset()

# Your RL training loop here
```

### Testing

From the parent directory:

```bash
python test_game.py
```
