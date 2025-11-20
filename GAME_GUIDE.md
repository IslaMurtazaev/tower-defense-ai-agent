# Winterfell Tower Defense - Game Guide

A simplified tower defense game inspired by the Battle of Winterfell from Game of Thrones.

## Game Overview

Defend Winterfell castle against waves of wights (undead enemies) by strategically placing footmen and archers.

### Game Flow

1. **Placement Phase**: Place up to 20 soldiers (any combination of footmen and archers)
2. **Combat Phase**: Watch your soldiers automatically defend against 5 waves of wights
3. **Victory/Defeat**: Survive all waves or lose when castle HP reaches 0

## Soldier Types

### Footman (Blue Circle)

- **Role**: Mobile melee defender
- **Attack Range**: 50 pixels (close combat)
- **Damage**: 30 per attack
- **Attack Speed**: 1.0 seconds
- **Detection Radius**: 400 pixels
- **Movement Speed**: 60 pixels/second
- **AI Behavior**: Uses A\* pathfinding to chase enemies, returns home when idle
- **Best for**: Intercepting and eliminating threats with high damage

### Archer (Green Triangle)

- **Role**: Mobile ranged attacker
- **Attack Range**: 200 pixels (long range)
- **Damage**: 10 per attack
- **Attack Speed**: 0.8 seconds
- **Detection Radius**: 450 pixels
- **Movement Speed**: 50 pixels/second
- **AI Behavior**: Uses A\* pathfinding to maintain range, returns home when idle
- **Best for**: Supporting with sustained ranged fire

## Enemies

### Wights (Red Squares)

- **HP**: 50
- **Speed**: 30 pixels/second
- **Damage**: 20 to castle
- **Behavior**: Move directly toward castle
- **Waves**: 25, 40, 60, 75, 100 wights per wave (300 total)
- **Spawn Rate**: 0.3 seconds between spawns (~3 per second)
- **Difficulty**: High - each soldier must eliminate ~30 wights
- **No delays**: Continuous waves for intense pressure

## Soldier AI System

### Intelligent Behavior

Soldiers now actively patrol and engage enemies using A\* pathfinding:

1. **Detection Phase**: Constantly scans for enemies within detection radius
2. **Pursuit Phase**: When enemy detected, calculates optimal path using A\* algorithm
3. **Attack Phase**: Stops moving when in attack range, engages target
4. **Return Phase**: Returns to original placement position when no enemies nearby

### Visual Indicators (During Gameplay)

- **Gray dot**: Home position marker
- **Faint circle**: Detection radius
- **Yellow/Red line**: Line to current target (red = in attack range)
- **Yellow circle**: Attack range when engaging

## How to Play

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Play the Game (Human Mode)

```bash
cd environment
python td_pygame.py
```

### Controls

**During Placement Phase:**

- **Left Click** on soldier type panel: Switch between Footman and Archer
- **Left Click** on map: Place selected soldier (max 10)
- **SPACE** or click "START BATTLE": Begin combat phase

**During Combat:**

- Watch the battle unfold automatically!

**Anytime:**

- **R**: Reset game
- **G**: Toggle grid overlay
- **Q**: Quit game

### UI Information

**Placement Phase:**

- Soldier type selector (top-left)
- Soldiers placed counter
- Start Battle button

**Combat Phase:**

- Current wave number
- Soldiers alive
- Wights killed
- Castle HP
- Wave timer

## Strategy Tips

1. **Mix Your Forces**: Combine footmen and archers for balanced defense
2. **Strategic Positioning**: Soldiers will chase enemies within detection radius then return home
3. **Cover Approaches**: Wights spawn from top and sides - position soldiers to intercept multiple paths
4. **Use Movement**: Soldiers now actively hunt enemies - place them where they can reach threats quickly
5. **Detection Zones**: Footmen detect at 400px, archers at 450px - position accordingly
6. **A\* Pathfinding**: Soldiers intelligently navigate to targets and back to their posts

## For RL Agent Training

### Using the Gymnasium Environment

```python
from environment.td_gym_env import TowerDefenseEnv

# Create environment
env = TowerDefenseEnv()

# Reset
observation, info = env.reset(seed=42)

# Take actions
for _ in range(10):  # Place soldiers
    action = env.action_space.sample()  # Or use your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
```

### Observation Space

- **Grid**: 32x32 grid showing game state (0=empty, 1=castle, 2=footman, 3=archer, 4=wight)
- **Soldiers Remaining**: How many soldiers left to place
- **Current Wave**: Current wave number
- **Castle HP Ratio**: Castle health as percentage

### Action Space

- **[soldier_type, grid_x, grid_y]**
  - soldier_type: 0=Footman, 1=Archer
  - grid_x: 0-31
  - grid_y: 0-31

### Reward Structure

- **+10**: Per wight killed
- **-50**: Per soldier lost
- **-5**: Per castle damage point
- **-1**: Invalid soldier placement
- **+500**: Victory bonus
- **-200**: Defeat penalty
- **+50**: Per wave completed

### Running Tests

```bash
python test_game.py
```

## Game Balance

- **Castle HP**: 100
- **Max Soldiers**: 10
- **Total Enemies**: 300 wights across 5 waves (25, 40, 60, 75, 100)
- **Spawn Rate**: 0.3 seconds (3+ wights per second!)
- **Waves**: Continuous, no delay between waves
- **Map Size**: 1280x800 pixels
- **Castle Position**: Bottom-center (640, 700)

## Technical Details

### Architecture

```
tower_defense/
├── environment/
│   ├── td_game_core.py      # Core game logic (rendering-independent)
│   ├── td_pygame.py         # Pygame visualization for humans
│   └── td_gym_env.py        # Gymnasium wrapper for RL training
├── test_game.py             # Test suite
├── requirements.txt         # Dependencies
└── GAME_GUIDE.md           # This file
```

### Files

- **td_game_core.py**: Pure game mechanics - soldiers, wights, combat, waves
- **td_pygame.py**: Human-playable interface with graphics and controls
- **td_gym_env.py**: RL environment following OpenAI Gymnasium standards

## Troubleshooting

### "No module named 'pygame'"

```bash
pip install pygame
```

### "No module named 'gymnasium'"

Only needed for RL training:

```bash
pip install gymnasium numpy
```

### Game runs too fast/slow

The game runs at 60 FPS. If performance is an issue, the FPS can be adjusted in `td_pygame.py`.

## Future Enhancements

Possible additions:

- Different tower types (catapults, fire towers)
- Special abilities or hero units
- More complex enemy AI
- Procedural map generation
- Multiplayer support
- More sophisticated RL reward shaping

---

**"Winter is here. Can you defend Winterfell?"**
