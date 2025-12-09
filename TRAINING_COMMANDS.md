# Training Commands

## Q-Learning Training

### Basic Training (Recommended)
```bash
python train_q_learning.py \
    --episodes 3000 \
    --max-steps 1500 \
    --alpha 0.005 \
    --gamma 0.92 \
    --eps-start 1.0 \
    --eps-end 0.05 \
    --eps-decay 50000 \
    --eps-decay-type linear \
    --fast-mode \
    --fast-multiplier 50 \
    --reduce-action-search \
    --save-interval 200 \
    --save-model q_agent_final.pkl \
    --log-interval 10 \
    --seed 42
```

### Continue Training from Checkpoint
```bash
python train_q_learning.py \
    --episodes 5000 \
    --max-steps 1500 \
    --alpha 0.005 \
    --gamma 0.92 \
    --eps-start 1.0 \
    --eps-end 0.05 \
    --eps-decay 50000 \
    --eps-decay-type linear \
    --fast-mode \
    --fast-multiplier 50 \
    --reduce-action-search \
    --save-interval 200 \
    --save-model q_agent_final.pkl \
    --load-model trained_models/q_learning/q_agent_final.pkl \
    --log-interval 10 \
    --seed 42
```

### Faster Training (Lower Quality)
```bash
python train_q_learning.py \
    --episodes 2000 \
    --alpha 0.01 \
    --eps-decay 30000 \
    --eps-end 0.1 \
    --save-interval 200
```

### Slower, More Stable Training
```bash
python train_q_learning.py \
    --episodes 5000 \
    --alpha 0.003 \
    --eps-decay 80000 \
    --eps-end 0.01 \
    --save-interval 200
```

### Commands Used for Current Checkpoints (Dec 2025)
Reproduce the models saved in `trained_models/*/` that were used for the latest comparison runs (1000 episodes each):

```bash
# Q-Learning (Approximate Q)
python train_q_learning.py \
    --episodes 1000 \
    --max-steps 1500 \
    --save-interval 200 \
    --save-model q_agent_final.pkl \
    --log-interval 10 \
    --fast-mode \
    --fast-multiplier 50 \
    --seed 42

# PPO
python train_ppo.py \
    --episodes 1000 \
    --max-steps 2000 \
    --save-interval 200 \
    --log-interval 5 \
    --fast-mode \
    --fast-multiplier 20 \
    --learning-rate 1e-3 \
    --gamma 0.99 \
    --clip-epsilon 0.2 \
    --reward-scale 20.0 \
    --gae-lambda 0.95 \
    --entropy-coef 0.02
```

---

## PPO Training

### Basic Training (Recommended)
```bash
python train_ppo.py \
    --episodes 500 \
    --max-steps 2000 \
    --learning-rate 1e-3 \
    --gamma 0.99 \
    --clip-epsilon 0.2 \
    --update-frequency 32 \
    --log-interval 5 \
    --save-interval 100 \
    --seed 42 \
    --fast-mode \
    --fast-multiplier 20 \
    --reward-scale 20.0 \
    --gae-lambda 0.95 \
    --entropy-coef 0.02
```

### Continue Training from Checkpoint
```bash
python train_ppo.py \
    --episodes 1000 \
    --max-steps 2000 \
    --learning-rate 1e-3 \
    --gamma 0.99 \
    --clip-epsilon 0.2 \
    --update-frequency 32 \
    --log-interval 5 \
    --save-interval 100 \
    --load-model trained_models/ppo/ppo_agent_final.pth \
    --seed 42 \
    --fast-mode \
    --fast-multiplier 20
```

### Faster PPO Training
```bash
python train_ppo.py \
    --episodes 300 \
    --learning-rate 2e-3 \
    --update-frequency 16 \
    --log-interval 5
```

---

## Quick Reference

### Key Hyperparameters Explained

**Q-Learning:**
- `--alpha 0.005`: Learning rate (higher = faster learning, but may be less stable)
- `--eps-decay 50000`: Steps for epsilon to decay (lower = faster exploration→exploitation transition)
- `--eps-end 0.05`: Final exploration rate (lower = more exploitation)
- `--save-interval 200`: Save checkpoint every N episodes (0 = only final)

**PPO:**
- `--learning-rate 1e-3`: Learning rate for policy and value networks
- `--update-frequency 32`: Update policy every N steps (lower = more frequent updates)
- `--clip-epsilon 0.2`: PPO clipping parameter (standard is 0.2)
- `--reward-scale 20.0`: Divide rewards by this value for stability

---

## Recommended Training Sequence

### Step 1: Train Q-Learning (Start Here)
```bash
python train_q_learning.py \
    --episodes 3000 \
    --alpha 0.005 \
    --eps-decay 50000 \
    --eps-end 0.05 \
    --save-interval 200 \
    --log-interval 10
```

**Expected time:** ~2-4 hours (depending on hardware)
**Check progress:** Monitor TensorBoard or terminal output

### Step 2: Evaluate Results
```bash
python evaluate_q_agent.py \
    --model trained_models/q_learning/q_agent_final.pkl \
    --episodes 50
```

**Check metrics:**
- Victory rate (should be > 20%)
- Soldiers killed by NK (should be < 3)
- Base HP at end (should be > 10)

### Step 3: Continue Training if Needed
If victory rate < 20%, continue training:
```bash
python train_q_learning.py \
    --episodes 5000 \
    --load-model trained_models/q_learning/q_agent_final.pkl \
    --alpha 0.003 \
    --save-interval 200
```

### Step 4: Train PPO (Alternative/Comparison)
```bash
python train_ppo.py \
    --episodes 500 \
    --learning-rate 1e-3 \
    --save-interval 100
```

---

## Monitoring Training

### View TensorBoard
```bash
# Q-Learning
tensorboard --logdir=runs/q_learning

# PPO
tensorboard --logdir=runs/ppo
```

**Key metrics to watch:**
- `Episode/Reward_MA50`: Should trend upward
- `Episode/Victory`: Victory rate over time (should increase)
- `Stats/Soldiers_Killed_by_NK`: Should decrease over time
- `Episode/Base_HP_End`: Should increase over time

---

## Troubleshooting

### If Training is Too Slow
- Increase `--fast-multiplier` (50 → 100)
- Reduce `--episodes` for initial testing
- Use `--reduce-action-search` (already enabled)

### If Agent Not Learning
- Check TensorBoard - are rewards trending upward?
- Try higher learning rate: `--alpha 0.01` (Q-Learning) or `--learning-rate 2e-3` (PPO)
- Check if victory rate is improving over time

### If Training Crashes
- Reduce `--fast-multiplier` (50 → 20)
- Check available memory
- Reduce `--max-steps` if episodes are too long

---

## One-Liner Commands (Copy & Paste)

### Q-Learning - Quick Start
```bash
python train_q_learning.py --episodes 3000 --alpha 0.005 --eps-decay 50000 --eps-end 0.05 --save-interval 200 --log-interval 10
```

### PPO - Quick Start
```bash
python train_ppo.py --episodes 500 --learning-rate 1e-3 --save-interval 100 --log-interval 5
```

### Continue Q-Learning Training
```bash
python train_q_learning.py --episodes 5000 --load-model trained_models/q_learning/q_agent_final.pkl --alpha 0.005 --eps-decay 50000 --eps-end 0.05 --save-interval 200
```

---

## Visualizing Victories After Training

### Find and Visualize Victories (Recommended)
```bash
# Find victories and visualize them automatically
python find_and_visualize_victories.py \
    --model-path trained_models/q_learning/q_agent_final.pkl \
    --num-tests 50
```

### Find Victories Only (Get Seeds)
```bash
# Just find which seeds result in victories
python find_and_visualize_victories.py \
    --model-path trained_models/q_learning/q_agent_final.pkl \
    --find-only \
    --num-tests 100
```

### Visualize Specific Seeds
```bash
# If you know which seeds are victories, visualize them directly
python find_and_visualize_victories.py \
    --model-path trained_models/q_learning/q_agent_final.pkl \
    --visualize-only 42 100 200 300
```

### Standard Visualization (Any Episode)
```bash
# Visualize any episode (may or may not be victory)
python visualize_agent.py \
    --model-path trained_models/q_learning/q_agent_final.pkl \
    --episodes 5 \
    --seed 42
```

**Controls during visualization:**
- `SPACE` - Pause/Resume
- `R` - Reset current episode
- `N` - Next victorious episode (victory script only)
- `Q` - Quit

---

## Expected Results

**Observed in latest 1000-episode runs (Dec 2025 checkpoints):**
- Q-Learning: ~10.2% victory rate (102/1000), mean reward ≈ -4489
- PPO: ~11.5% victory rate (115/1000), mean reward ≈ -1794
