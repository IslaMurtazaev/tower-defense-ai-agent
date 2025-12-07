# Optimized Defender Deployment in a Tower Defence Game

*"What is dead may never die... but with good AI, it dies much faster."*

**Intro to AI — Final Project**

**Authors:** Islam Murtazaev, Leonel Mainsah Ngu, Raymond Frimpong Amoateng

## Winter Is Coming... and So Are the Reinforcement Learning Agents

Imagine you are standing atop the battlements of Winterfell.

The Night King marches from the far North with an endless horde of wights.

Your resources are limited. Your defenses are few. Your survival depends entirely on where, when, and how you deploy your soldiers.

In Game of Thrones, this challenge ended... poorly for many.

In this project, we ask:

**Can AI do better?**

**Can optimized defender placement — powered by A* planning and reinforcement learning — hold the line against a simulated army of the dead?**

This repository contains our attempt to answer that question.

## Project Overview

We built a Tower Defence environment inspired by the tactical tension of the Battle of Winterfell using Pygame.

This environment serves as the foundation for training AI agents that must:

- Predict enemy movement (like Bran watching from the Weirwood)
- Allocate scarce resources (as Daenerys and Jon failed to do)
- Optimize defender strategy better than any panicked human commander

We've implemented multiple reinforcement learning approaches to tackle this challenge, including Approximate Q-Learning and Proximal Policy Optimization (PPO), along with an A* baseline for comparison.

## Game Features

### Core Mechanics

- **Soldiers**: Deploy up to 6 soldiers that automatically target and eliminate nearby wights. Soldiers deal 0 damage to Night Kings and will be instantly killed if caught in a Night King's sweep attack.

- **Wights**: Ordinary undead enemies that spawn continuously. Soldiers kill wights instantly on contact with a burn animation.

- **Night Kings**: Boss enemies that spawn in 4 scheduled waves. Each Night King:
  - Approaches Winterfell on a radial path
  - Locks into battle when within range (340px)
  - Performs a devastating AOE sweep attack every 1.5 seconds that instantly kills all soldiers in radius
  - Can only be damaged by heroes

### Heroes

- **Jon Snow**: Deploys automatically when a Night King engages. Wields Longclaw and deals 60 damage per strike.

- **Daenerys & Drogon**: Deploys automatically when a Night King engages. Breathes dragon fire with 160 DPS at 300px range.

- Only heroes can damage Night Kings; soldiers are ineffective against them.

### Special Features

- **Bran the Seer**: Provides a raven vision overlay warning 10 seconds before each Night King spawns
- **Auto-deploy**: Soldiers automatically deploy when enemies approach within 520px of the base
- **Dynamic snow**: Snow intensity increases during Night King battles
- **Victory condition**: Defeat all 4 Night Kings to win
- **Defeat condition**: The base falls when its HP reaches 0

## Controls

- **Left Click**: Place a soldier (max 6 active)
- **SPACE**: Pause/Resume game
- **R**: Reset game
- **G**: Toggle grid overlay
- **S**: Toggle snow effects
- **M**: Toggle background music
- **V**: Toggle sound effects
- **Q**: Quit game

## Getting Started

### Prerequisites

- Python 3.10+
- Pygame 2.6.1+
- PyTorch (for PPO agent)
- TensorBoard (for training visualization)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/IslaMurtazaev/tower-defense-ai-agent.git
cd tower-defense-ai-agent
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Game

```bash
cd environment
python td_pygame.py
```

The game runs with optional sound support. If a `sounds/` folder is present in the environment directory, sounds will play. Otherwise, the game runs silently.

## Training AI Agents

### Q-Learning Agent

Train an Approximate Q-Learning agent:

```bash
python train_q_learning.py --episodes 3000 --alpha 0.005 --eps-decay 50000 --eps-end 0.05
```

The agent learns to place soldiers strategically, avoiding Night Kings while defending the base. Key features include:
- Feature-based value approximation with 28 handcrafted features
- NK avoidance and base defense features
- Adaptive learning rate and epsilon decay
- Experience replay and gradient clipping for stability

### PPO Agent

Train a Proximal Policy Optimization agent:

```bash
python train_ppo.py --episodes 500 --learning-rate 1e-3 --save-interval 100
```

The PPO agent uses:
- Policy and value networks with layer normalization
- GAE (Generalized Advantage Estimation) for stable learning
- Reward normalization and gradient clipping
- 24-dimensional feature space

### A* Baseline

Compare against an A* pathfinding baseline:

```bash
python evaluate_astar_agent.py --episodes 50
```

The A* agent uses pathfinding heuristics to identify optimal placement positions based on enemy movement patterns.

### Evaluating Agents

Evaluate a trained Q-Learning agent:

```bash
python evaluate_q_agent.py --model-path trained_models/q_learning/q_agent_final.pkl --episodes 50
```

Evaluate a trained PPO agent:

```bash
python evaluate_ppo_agent.py --model-path trained_models/ppo/ppo_agent_final.pth --episodes 50
```

### Visualizing Trained Agents

Watch a trained agent play:

```bash
python visualize_agent.py --model-path trained_models/q_learning/q_agent_final.pkl --episodes 5
```

Find and visualize only victorious episodes:

```bash
python find_and_visualize_victories.py --model-path trained_models/q_learning/q_agent_final.pkl --num-tests 50
```

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir=runs/
```

Open `http://localhost:6006` in your browser to see:
- Episode rewards over time
- Victory rate progression
- Soldiers killed by NK (should decrease over time)
- Base HP at episode end
- Other training metrics

## Game Statistics

The HUD displays:
- **Soldiers Alive**: Current active soldier count
- **Soldiers Killed**: Total soldiers lost
- **Soldiers Deployed**: Total soldiers placed
- **Wights Killed**: Total wights eliminated
- **Night Kings Defeated**: Progress (e.g., 3/4)
- **Wall Integrity**: Base HP status

## AI Agent Implementations

### Model-Based Planning

- **A*** — calculates optimal enemy paths, identifying critical choke points (like the narrow breach in the castle walls). Used as a baseline for comparison.

### Model-Free Reinforcement Learning

- **Approximate Q-Learning** — feature-based value function approximation with linear function approximation. Learns to avoid Night Kings and defend the base through trial and error.

- **PPO (Proximal Policy Optimization)** — policy gradient method that learns a stochastic policy. More sample-efficient than Q-Learning but requires more computation.

### Hybrid Approaches

- **A* Baseline Agent** — uses A* pathfinding to identify optimal placement positions, serving as a non-learning baseline for comparison.

## Project Structure

```
tower-defense-ai-agent/
├── environment/
│   ├── td_pyastar.py          # Main game engine (headless)
│   ├── td_pygame.py           # Pygame visualization
│   ├── td_warrior_gym_env.py  # Gymnasium RL environment wrapper
│   └── astar_controller.py    # A* pathfinding controller
├── agents/
│   ├── approx_q_agent.py      # Q-Learning agent
│   ├── ppo_agent.py           # PPO agent
│   └── astar_baseline_agent.py # A* baseline agent
├── train_q_learning.py        # Q-Learning training script
├── train_ppo.py               # PPO training script
├── evaluate_q_agent.py        # Q-Learning evaluation
├── evaluate_ppo_agent.py      # PPO evaluation
├── evaluate_astar_agent.py    # A* baseline evaluation
├── visualize_agent.py         # Visualize trained agents
├── find_and_visualize_victories.py # Find and visualize victories
├── compare_algorithms.py       # Compare all agents
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Technical Details

- **Resolution**: 1280x800 pixels
- **Frame Rate**: 60 FPS
- **Enemy Paths**: 24 radial paths from the screen edge to Winterfell
- **Night King Schedule**: 4 waves at scheduled times
- **Base HP**: 100 HP
- **Max Soldiers**: 6 active soldiers
- **Max Heroes**: 2 (Jon Snow and Daenerys)

## Training Tips

After training, you should see the agents learn to:
- Place soldiers away from Night Kings (to avoid sweep attacks)
- Keep soldiers near the base when enemies threaten
- Deploy heroes effectively to eliminate Night Kings

Expected improvements after training:
- Victory rate: 20-40% (up from 5-10%)
- Soldiers killed by NK: 0-3 per episode (down from 6)
- Base HP: 10-50 at episode end (up from 0)

## License

This project is part of an academic course project.

---

*"The Long Night is coming, but so is our AI."*
