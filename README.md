# 🛡️ Optimized Defender Deployment in a Tower Defence Game

*"What is dead may never die… but with good AI, it dies much faster."*

**Intro to AI — Final Project**

**Authors:** Islam Murtazaev, Leonel Mainsah Ngu, Raymond Frimpong Amoateng

## ❄️ Winter Is Coming… and So Are the Reinforcement Learning Agents

Imagine you are standing atop the battlements of Winterfell.

The Night King marches from the far North with an endless horde of wights.

Your resources are limited. Your defenses are few. Your survival depends entirely on where, when, and how you deploy your soldiers.

In Game of Thrones, this challenge ended… poorly for many.

In this project, we ask:

**Can AI do better?**

**Can optimized defender placement — powered by A* planning and reinforcement learning — hold the line against a simulated army of the dead?**

This repository contains our attempt to answer that question.

## 🧠 Project Overview

We built a Tower Defence (TD) environment inspired by the tactical tension of the Battle of Winterfell using Pygame.

This environment serves as the foundation for training AI agents that must:

- **Predict enemy movement** (like Bran watching from the Weirwood),
- **Allocate scarce resources** (as Daenerys and Jon failed to do),
- **And optimize defender strategy** better than any panicked human commander.

## 🎮 Game Features

### 🏰 Core Mechanics

- **Soldiers**: Deploy soldiers that automatically target and eliminate nearby wights. Soldiers deal 0 damage to Night Kings.
- **Wights**: Ordinary undead enemies that spawn continuously. Soldiers kill wights instantly on contact with a burn animation.
- **Night Kings**: Boss enemies that spawn in 5 scheduled waves. Each Night King:
  - Approaches Winterfell on a radial path
  - Locks into battle when within range (340px)
  - Performs a devastating AOE sweep attack every 1.5 seconds that instantly kills all soldiers in radius
  - Can only be damaged by heroes

### ⚔️ Heroes

- **Jon Snow**: Deploys automatically when a Night King engages. Wields Longclaw and deals 60 damage per strike.
- **Daenerys & Drogon**: Deploys automatically when a Night King engages. Breathes dragon fire with 160 DPS at 300px range.
- Only heroes can damage Night Kings; soldiers are ineffective against them.

### 🔮 Special Features

- **Bran the Seer**: Provides a raven vision overlay warning 10 seconds before each Night King spawns
- **Auto-deploy**: Soldiers automatically deploy when enemies approach within 520px of the base
- **Dynamic snow**: Snow intensity increases during Night King battles
- **Victory condition**: Defeat all 5 Night Kings to win
- **Defeat condition**: The base falls when its HP reaches 0

## 🎯 Controls

- **Left Click**: Place a soldier (max 32 active)
- **SPACE**: Pause/Resume game
- **R**: Reset game
- **G**: Toggle grid overlay
- **S**: Toggle snow effects
- **M**: Toggle background music
- **V**: Toggle sound effects
- **Q**: Quit game

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Pygame 2.6.1+

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

The game runs with optional sound support. If a `sounds/` folder is present in the root directory, sounds will play. Otherwise, the game runs silently.

## 📊 Game Statistics

The HUD displays:
- **Soldiers Alive**: Current active soldier count
- **Soldiers Killed**: Total soldiers lost
- **Soldiers Deployed**: Total soldiers placed
- **Wights Killed**: Total wights eliminated
- **Night Kings Defeated**: Progress (e.g., 3/5)
- **Wall Integrity**: Base HP status

## 🤖 Future Work: AI Agent Integration

This environment is designed to be integrated with AI agents. Planned implementations include:

### ⚔️ Model-Based Planning

- **A*** — calculates optimal enemy paths, identifying critical choke points (like the narrow breach in the castle walls).

### 🐺 Model-Free Reinforcement Learning

- **PPO** — the Jon Snow of RL: balanced, powerful, reliable
- **A3C** — the Unsullied: trained in parallel, disciplined under pressure
- **REINFORCE** — the simple but loyal Davos Seaworth of policy gradients

### 🧙 Hybrid Approaches

- **A*** guided policy pretraining
- **A*** as a "tactical advisor" blended with learned behavior (AKA the Tyrion strategy)

## 📁 Project Structure P.S Will keep updating :)

```
tower-defense-ai-agent/
├── environment/
│   └── td_pygame.py      # Main game engine (single-file implementation)
├── sounds/               # Optional sound files (handled gracefully if absent)
│   └── base.wav
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎨 Technical Details

- **Resolution**: 1280x800 pixels
- **Frame Rate**: 60 FPS
- **Enemy Paths**: 24 radial paths from the screen edge to Winterfell
- **Night King Schedule**: 5 waves at 45s, 120s, 210s, 320s, and 480s
- **Base HP**: 80 HP
- **Max Soldiers**: 32 active soldiers

## 📝 License

This project is part of an academic course project.

---

*"The Long Night is coming, but so is our AI."*
