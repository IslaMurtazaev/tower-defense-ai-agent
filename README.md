# 🛡️ Optimized Defender Deployment in a Tower Defence Game

*"What is dead may never die… but with good AI, it dies much faster."*

**Intro to AI — Final Project**

**Authors:** Islam Murtazaev, Leonel Mainsah Ngu, Raymond Frimpong Amoateng

## ❄️ Winter Is Coming… and So Are the Reinforcement Learning Agents

Imagine you are standing atop the battlements of Winterfell.

The Night King marches from the far North with an endless horde of wights.

Your resources are limited. Your towers are few. Your survival depends entirely on where, when, and how you deploy your defenses.

In Game of Thrones, this challenge ended… poorly for many.

In this project, we ask:

**Can AI do better?**

**Can optimized defender placement — powered by A* planning and reinforcement learning — hold the line against a simulated army of the dead?**

This repository contains our attempt to answer that question.

## 🧠 Project Overview

We built a grid-based Tower Defence (TD) environment inspired by the tactical tension of the Battle of Winterfell.

Our AI agents must:

- **Predict enemy movement** (like Bran watching from the Weirwood),
- **Allocate scarce resources** (as Daenerys and Jon failed to do 😬),
- **And optimize defender strategy** better than any panicked human commander.

To accomplish this, we implemented and compared:

### ⚔️ Model-Based Planning

- **A*** — calculates optimal enemy paths, identifying critical choke points (like the narrow breach in the castle walls).

### 🐺 Model-Free Reinforcement Learning

- **PPO** — the Jon Snow of RL: balanced, powerful, reliable
- **A3C** — the Unsullied: trained in parallel, disciplined under pressure
- **REINFORCE** — the simple but loyal Davos Seaworth of policy gradients

### 🧙 Hybrid Approaches

- **A*-guided policy pretraining**
- **A* as a "tactical advisor" blended with learned behavior** (AKA the Tyrion strategy)
