# Game Difficulty Settings

## Current Configuration

### Enemy Waves
- **Wave 1**: 25 wights
- **Wave 2**: 40 wights  
- **Wave 3**: 60 wights
- **Wave 4**: 75 wights
- **Wave 5**: 100 wights
- **Total**: 300 wights

### Spawn Rate
- **Interval**: 0.3 seconds between spawns
- **Rate**: ~3.3 wights per second
- **Wave 1 Duration**: ~7.5 seconds to spawn all
- **Wave 5 Duration**: ~30 seconds to spawn all

### Comparison to Original

| Metric | Original | Current | Multiplier |
|--------|----------|---------|------------|
| Total Enemies | 60 | 300 | **5x** |
| Spawn Interval | 1.0s | 0.3s | **3.3x faster** |
| Pressure | Low | High | **Intense** |

## Challenge Level

### Difficulty Rating: ⚔️⚔️⚔️⚔️ (Hard)

With 10 soldiers against 300 enemies:
- Each soldier must eliminate ~30 wights
- Enemies spawn faster than soldiers can kill initially
- Multiple waves overlap if soldiers don't clear quickly
- Castle will likely take damage

### Expected Outcomes

**With Optimal Placement:**
- ✅ Can complete 4-5 waves
- ⚠️ Castle HP: 20-60 remaining
- ✅ Kill rate: 2-3 wights/second
- ⚠️ Expect intense pressure in waves 4-5

**With Poor Placement:**
- ❌ May only complete 2-3 waves
- ❌ Castle destroyed
- ❌ Soldiers overwhelmed

## Strategy Tips for High Difficulty

### 1. Maximize Coverage
Place soldiers to cover all spawn angles:
- Top spawn: 2-3 soldiers
- Left spawn: 2-3 soldiers  
- Right spawn: 2-3 soldiers
- Center reserve: 1-2 soldiers

### 2. Balance Unit Types
**Recommended mix:**
- 5 Footmen (high damage for kills)
- 5 Archers (range for early detection)

### 3. Layered Defense
```
     [A]       [A]       [A]    ← Archer line (forward detection)
        [F]   [F]   [F]         ← Footman line (interceptors)
              [A][F]             ← Castle defense reserve
            🏰 CASTLE
```

### 4. Detection Zones
- Overlap detection radii (400px/450px)
- Create "kill zones" where multiple soldiers engage
- Position for early interception

### 5. Movement Efficiency
- Place soldiers near likely enemy paths
- Use A* pathfinding advantage for positioning
- Consider patrol distance to threats

## Tuning the Difficulty

Want to adjust? Edit `td_game_core.py`:

```python
# Line ~291-293
WAVE_DEFINITIONS = [25, 40, 60, 75, 100]  # Change these numbers
```

```python
# Line ~301
self.spawn_interval = 0.3  # Increase for easier, decrease for harder
```

### Suggested Presets

**Easy Mode:**
```python
WAVE_DEFINITIONS = [10, 15, 20, 25, 30]  # 100 total
self.spawn_interval = 0.8
```

**Normal Mode:**
```python
WAVE_DEFINITIONS = [15, 25, 35, 45, 55]  # 175 total
self.spawn_interval = 0.5
```

**Hard Mode (Current):**
```python
WAVE_DEFINITIONS = [25, 40, 60, 75, 100]  # 300 total
self.spawn_interval = 0.3
```

**Nightmare Mode:**
```python
WAVE_DEFINITIONS = [50, 80, 120, 150, 200]  # 600 total
self.spawn_interval = 0.2
```

## Performance Notes

At current difficulty:
- **Spawns per second**: ~3.3
- **AI soldiers can handle**: ~2-3 kills per second (with 10 soldiers)
- **Balance**: Slightly enemy-favored, requires good strategy
- **Frame rate**: Stable 60 FPS even with 50+ entities

## RL Training Implications

For reinforcement learning:
- **State space complexity**: High (many enemies)
- **Reward signal**: More frequent (more kills)
- **Episode length**: 60-120 seconds
- **Difficulty**: Good challenge for training
- **Strategy depth**: Placement matters significantly

### Recommended Changes for RL

For faster training iterations:
```python
WAVE_DEFINITIONS = [15, 20, 25, 30, 35]  # 125 total
self.spawn_interval = 0.4
```

This provides:
- Faster episodes (30-60s)
- Clear reward signal
- Still challenging enough to learn strategy
- Less computation per episode

---

**Current difficulty creates intense, strategic gameplay! 🔥**

