# Tower Defense AI Agent

A reinforcement learning project for training AI agents to play a Game of Thrones-inspired tower defense game.

## Project Overview

This project implements a tower defense game where an AI agent learns to strategically place soldiers to defend Winterfell against waves of wights and Night Kings. The agent uses Approximate Q-Learning to learn optimal placement strategies.

## Features

- **Game Engine**: Full tower defense game with soldiers, wights, Night Kings, and heroes
- **RL Environment**: Gymnasium-compatible environment for training agents
- **Q-Learning Agent**: Approximate Q-Learning implementation with feature-based value approximation
- **Visualization**: Watch trained agents play the game in real-time
- **Training Tools**: Scripts for training, evaluation, and visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tower-defense-ai-agent
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Game

### Play the game manually:
```bash
cd environment
python td_pygame.py
```

### Train an RL agent:
```bash
python train_q_learning.py --episodes 500 --max-steps 2500 --alpha 0.0005 --eps-decay 50000
```

### Evaluate a trained agent:
```bash
python evaluate_q_agent.py --model-path trained_models/q_learning/q_agent_final.pkl --episodes 10
```

### Visualize a trained agent:
```bash
python visualize_agent.py --model-path trained_models/q_learning/q_agent_final.pkl
```

### View training progress with TensorBoard:
```bash
tensorboard --logdir=runs/q_learning --port=6006
```

## Project Structure

```
tower-defense-ai-agent/
├── environment/
│   ├── td_pyastar.py          # Main game engine
│   ├── td_pygame.py           # Pygame visualization
│   ├── td_warrior_gym_env.py # RL environment wrapper
│   └── astar_controller.py   # A* pathfinding controller
├── agents/
│   └── approx_q_agent.py     # Q-Learning agent implementation
├── train_q_learning.py        # Training script
├── evaluate_q_agent.py        # Evaluation script
├── visualize_agent.py          # Visualization script
├── plot_training_curve.py     # Plot training statistics
└── requirements.txt           # Python dependencies
```

## Game Mechanics

- **Soldiers**: Deploy up to 4 soldiers that automatically attack nearby enemies
- **Wights**: Regular enemies that spawn continuously
- **Night Kings**: Boss enemies that spawn at 45s, 120s, 210s, 320s, and 480s
- **Heroes**: Jon Snow and Daenerys deploy automatically to fight Night Kings
- **Victory**: Survive 180 seconds or defeat all 5 Night Kings

## Training Parameters

Recommended training settings:
- `--episodes 500`: Number of training episodes
- `--max-steps 2500`: Maximum steps per episode (allows full game time)
- `--alpha 0.0005`: Learning rate
- `--eps-decay 50000`: Epsilon decay steps
- `--eps-end 0.1`: Final exploration rate

## Contributing

### Pushing Code to Branch

To push changes file by file to the `rl-warrior-implementation` branch:

1. Check current branch:
```bash
git branch --show-current
```

2. Stage and commit files individually:
```bash
git add <file-path>
git commit -m "Description of changes"
git push origin rl-warrior-implementation
```

3. Example workflow:
```bash
# Add and commit environment file
git add environment/td_warrior_gym_env.py
git commit -m "Update RL environment with Night King integration"
git push origin rl-warrior-implementation

# Add and commit agent file
git add agents/approx_q_agent.py
git commit -m "Clean up agent code comments"
git push origin rl-warrior-implementation

# Add and commit training script
git add train_q_learning.py
git commit -m "Update training script"
git push origin rl-warrior-implementation
```

4. Push all changes at once (alternative):
```bash
git add .
git commit -m "Update project files"
git push origin rl-warrior-implementation
```

## License

Academic project for Intro to AI course.
