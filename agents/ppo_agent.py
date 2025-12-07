"""PPO (Proximal Policy Optimization) agent for Tower Defense."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Tuple, List
from pathlib import Path


class PPOPolicyNetwork(nn.Module):
    """Policy network for PPO that outputs action probabilities."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass: features -> action probabilities."""
        x = self.relu(self.ln1(self.fc1(features)))  # Layer norm before activation
        x = self.relu(self.ln2(self.fc2(x)))  # Layer norm before activation
        logits = self.fc3(x)
        return self.softmax(logits)

    def get_action(self, features: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy distribution."""
        probs = self.forward(features)
        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), log_prob, probs


class PPOAgent:
    """PPO (Proximal Policy Optimization) agent for tower defense."""

    REWARD_SCALE = 20.0  # Scale down raw environment rewards to stabilize PPO updates
    def __init__(
        self,
        observation_space,
        action_space,
        feature_dim: int = 24,  # Added 3 new features for base defense (was 21)
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        hidden_dim: int = 128,
        reward_scale: float = None,
        device: str = None,
        gae_lambda: float = 0.95,  # GAE lambda for advantage estimation
    ):
        self.obs_space = observation_space
        self.action_space = action_space
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.reward_scale = reward_scale if reward_scale is not None else self.REWARD_SCALE
        # Handle device: "auto" means auto-detect, None means auto-detect, otherwise use specified device
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Calculate action space size
        # Action space is MultiDiscrete([2, GRID_SIZE, GRID_SIZE]) = [unit_type, grid_x, grid_y]
        unit_type_dim = int(action_space.nvec[0])  # Should be 2
        grid_size = int(action_space.nvec[1])  # Should be 32
        self.action_dim = unit_type_dim * grid_size * grid_size  # 2 * 32 * 32 = 2048
        self.grid_size = grid_size
        self.unit_type_dim = unit_type_dim

        # Policy network
        self.policy_net = PPOPolicyNetwork(feature_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # IMPROVED: Better value network with layer normalization
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer normalization for stability
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer normalization
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        # Weight initialization
        for m in self.value_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)

        # Reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        self.normalize_rewards = True

        # Training buffers
        self.reset_buffers()

        # Track entropy for monitoring
        self.last_entropy = 0.0

    def reset_buffers(self):
        """Reset training buffers for new episode."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def _features(self, obs: Dict) -> np.ndarray:
        """Extract feature vector from observation (vectorized)."""
        grid = obs["grid"]
        soldiers_remaining = float(obs["soldiers_remaining"][0])
        heroes_remaining = float(obs.get("heroes_remaining", [0])[0])
        base_hp_ratio = float(obs["base_hp_ratio"][0])
        current_wave = float(obs["current_wave"][0])
        night_king_active = float(obs["night_king_active"][0])

        G = grid.shape[0]

        # Aggregate grid features
        base_cells = np.argwhere(grid == 1)
        soldier_cells = np.argwhere(grid == 2)
        enemy_cells = np.argwhere(grid == 3)  # wights
        nk_cells = np.argwhere(grid == 4)

        # Distance to base (from center)
        center_x, center_y = G // 2, G // 2
        dist_to_base_norm = np.sqrt((center_x - G/2)**2 + (center_y - G/2)**2) / np.sqrt(2 * (G - 1)**2)

        # Entity densities
        total_soldiers = float(len(soldier_cells))
        total_enemies = float(len(enemy_cells))
        total_nks = float(len(nk_cells))

        soldier_density = total_soldiers / float(G * G) if G > 0 else 0.0
        enemy_density = total_enemies / float(G * G) if G > 0 else 0.0
        nk_density = total_nks / float(G * G) if G > 0 else 0.0

        # Base defense features - helps agent learn to keep soldiers near base
        BASE_DEFENSE_RADIUS = 200.0  # Soldiers within this distance count as "defending"
        ENEMY_THREAT_RADIUS = 250.0  # Enemies within this distance are "threatening" the base
        SCREEN_W, SCREEN_H = 1280, 800
        BASE_POS = (SCREEN_W // 2, SCREEN_H // 2)

        # Count how many soldiers are close to the base (they're defending it)
        soldiers_near_base = 0
        if len(soldier_cells) > 0:
            for soldier_cell in soldier_cells:
                soldier_world_x = (soldier_cell[1] / G) * SCREEN_W
                soldier_world_y = (soldier_cell[0] / G) * SCREEN_H
                dist_to_base = np.linalg.norm([soldier_world_x - BASE_POS[0], soldier_world_y - BASE_POS[1]])
                if dist_to_base <= BASE_DEFENSE_RADIUS:
                    soldiers_near_base += 1

        # Count how many enemies are close to the base (they're a threat)
        enemies_near_base = 0
        if len(enemy_cells) > 0:
            for enemy_cell in enemy_cells:
                enemy_world_x = (enemy_cell[1] / G) * SCREEN_W
                enemy_world_y = (enemy_cell[0] / G) * SCREEN_H
                dist_to_base = np.linalg.norm([enemy_world_x - BASE_POS[0], enemy_world_y - BASE_POS[1]])
                if dist_to_base <= ENEMY_THREAT_RADIUS:
                    enemies_near_base += 1

        # Normalize these values so they're between 0 and 1
        soldiers_near_base_norm = soldiers_near_base / 6.0  # Divide by max 6 soldiers
        enemies_near_base_norm = min(enemies_near_base / 10.0, 1.0)  # Cap at 1.0
        # This score is high when we have soldiers defending AND enemies are threatening
        base_defense_score = soldiers_near_base_norm * enemies_near_base_norm

        # Build feature vector
        feats = np.zeros(self.feature_dim, dtype=np.float32)
        feats[0] = dist_to_base_norm
        feats[1] = soldiers_remaining / 6.0  # Normalize by 6 (we have 6 soldiers max now)
        feats[2] = base_hp_ratio
        feats[3] = current_wave / 5.0  # Normalize by max waves
        feats[4] = night_king_active
        feats[5] = soldier_density
        feats[6] = enemy_density
        feats[7] = nk_density

        # Additional features for remaining dimensions
        if self.feature_dim > 8:
            feats[8] = heroes_remaining  # Heroes remaining to place

        # Vectorized enemy distance features
        if self.feature_dim > 9:
            if len(enemy_cells) > 0:
                enemy_coords = enemy_cells[:, [1, 0]]  # swap to [x, y]
                center_coords = np.array([center_x, center_y])
                dists = np.linalg.norm(enemy_coords - center_coords, axis=1)
                feats[9] = dists.min() / np.sqrt(2 * (G - 1)**2)
            else:
                feats[9] = 1.0

        # Vectorized Night King features
        if self.feature_dim > 10:
            SCREEN_W, SCREEN_H = 1280, 800
            NK_SWEEP_RADIUS = 95.0
            max_screen_dist = math.hypot(SCREEN_W, SCREEN_H)
            center_world_x = (center_x / G) * SCREEN_W
            center_world_y = (center_y / G) * SCREEN_H

            if len(nk_cells) > 0:
                nk_coords = nk_cells[:, [1, 0]]  # swap to [x, y]
                nk_world = nk_coords.astype(np.float32)
                nk_world[:, 0] = nk_world[:, 0] / G * SCREEN_W
                nk_world[:, 1] = nk_world[:, 1] / G * SCREEN_H
                nk_dists = np.linalg.norm(nk_world - np.array([center_world_x, center_world_y]), axis=1)
                nks_within_sweep = np.sum(nk_dists < NK_SWEEP_RADIUS)
                total_nk_danger = np.sum(1.0 / (nk_dists + 1.0))

                feats[10] = nk_dists.min() / max_screen_dist
                if self.feature_dim > 11:
                    feats[11] = 1.0 if nks_within_sweep > 0 else 0.0
                if self.feature_dim > 12:
                    feats[12] = nk_dists.mean() / max_screen_dist
                if self.feature_dim > 13:
                    feats[13] = min(nks_within_sweep / 4.0, 1.0)
                if self.feature_dim > 14:
                    feats[14] = min(total_nk_danger / 4.0, 1.0)

                if self.feature_dim > 15:
                    safety_score = 1.0 - min(feats[14] if self.feature_dim > 14 else 0.0, 1.0)
                    feats[15] = safety_score

                if self.feature_dim > 16:
                    nk_engagement_risk = 1.0 if (feats[11] > 0.0 if self.feature_dim > 11 else False) and (feats[10] < 0.3 if self.feature_dim > 10 else False) else 0.0
                    feats[16] = nk_engagement_risk
                if self.feature_dim > 17:
                    min_nk_dist = feats[10] if self.feature_dim > 10 else 1.0
                    soldier_nk_separation = min_nk_dist  # Already normalized, prefer high values
                    feats[17] = soldier_nk_separation
            else:
                feats[10:18] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0][:min(8, self.feature_dim - 10)]

        # Vectorized wight proximity and density (starts at index 18 after NK features)
        if self.feature_dim > 18:
            SCREEN_W, SCREEN_H = 1280, 800
            max_screen_dist = math.hypot(SCREEN_W, SCREEN_H)
            center_world_x = (center_x / G) * SCREEN_W
            center_world_y = (center_y / G) * SCREEN_H

            if len(enemy_cells) > 0:
                enemy_world = enemy_cells[:, [1, 0]].astype(np.float32)
                enemy_world[:, 0] = enemy_world[:, 0] / G * SCREEN_W
                enemy_world[:, 1] = enemy_world[:, 1] / G * SCREEN_H
                dists = np.linalg.norm(enemy_world - np.array([center_world_x, center_world_y]), axis=1)
                RANGE_PIXELS = 150.0
                wights_within_range = np.sum(dists < RANGE_PIXELS)

                feats[18] = dists.min() / max_screen_dist
                if self.feature_dim > 19:
                    feats[19] = wights_within_range / max(1, len(enemy_cells))
                if self.feature_dim > 20:
                    x0, x1 = max(0, center_x-2), min(G, center_x+3)
                    y0, y1 = max(0, center_y-2), min(G, center_y+3)
                    local_patch = grid[y0:y1, x0:x1]
                    feats[20] = np.sum(local_patch == 3) / 25.0
            else:
                feats[18:21] = [1.0, 0.0, 0.0][:min(3, self.feature_dim - 18)]

        # Base defense features - these help the agent learn to defend the base
        if self.feature_dim > 21:
            feats[21] = soldiers_near_base_norm  # How many soldiers are defending (0-1)
        if self.feature_dim > 22:
            feats[22] = enemies_near_base_norm  # How many enemies are threatening (0-1)
        if self.feature_dim > 23:
            feats[23] = base_defense_score  # Combined score: high when defending against threats

        return feats

    def _is_valid_placement(self, grid_x: int, grid_y: int, obs: Dict = None) -> bool:
        """Check if a grid position is valid for soldier placement.

        Valid placements must:
        - Be at least BASE_RADIUS + 80 pixels from base center
        - Be within screen bounds (50 < x < SCREEN_W-50, 50 < y < SCREEN_H-150)
        - Be at least NK_SWEEP_RADIUS + 50 pixels from any Night King (if obs provided)
        """
        # Convert grid to world coordinates
        SCREEN_W, SCREEN_H = 1280, 800
        BASE_POS = (SCREEN_W // 2, SCREEN_H // 2)  # (640, 400)
        BASE_RADIUS = 56
        MIN_DIST_FROM_BASE = BASE_RADIUS + 80  # 136 pixels
        NK_SWEEP_RADIUS = 95  # From game constants
        MIN_DIST_FROM_NK = NK_SWEEP_RADIUS + 50  # 145 pixels - INCREASED safe distance for better NK avoidance

        world_x = (grid_x / self.grid_size) * SCREEN_W
        world_y = (grid_y / self.grid_size) * SCREEN_H

        # Check screen bounds
        if not (50 < world_x < SCREEN_W - 50 and 50 < world_y < SCREEN_H - 150):
            return False

        # Check distance from base
        dist_to_base = math.hypot(world_x - BASE_POS[0], world_y - BASE_POS[1])
        if dist_to_base < MIN_DIST_FROM_BASE:
            return False

        # Check distance from Night Kings (if observation provided and NKs are present)
        if obs is not None:
            grid = obs.get("grid")
            if grid is not None:
                # Find all Night King positions in grid
                nk_cells = np.argwhere(grid == 4)  # 4 = Night King in grid
                # Only restrict if NKs are actually visible (during combat phase)
                # During placement phase, NKs aren't spawned yet, so no restriction needed
                if len(nk_cells) > 0:
                    for nk_cell in nk_cells:
                        nk_grid_y, nk_grid_x = nk_cell
                        nk_world_x = (nk_grid_x / self.grid_size) * SCREEN_W
                        nk_world_y = (nk_grid_y / self.grid_size) * SCREEN_H

                        # Check if placement is too close to this NK
                        dist_to_nk = math.hypot(world_x - nk_world_x, world_y - nk_world_y)
                        if dist_to_nk < MIN_DIST_FROM_NK:
                            return False  # Too close to NK - will get killed by sweep

        return True

    def _get_valid_action_mask(self, obs: Dict) -> torch.Tensor:
        """Fully vectorized valid action mask for PPO (1=valid, 0=invalid)."""
        soldiers_remaining = obs.get('soldiers_remaining', [0])[0]
        heroes_remaining = obs.get('heroes_remaining', [0])[0]
        G = self.grid_size
        SCREEN_W, SCREEN_H = 1280, 800
        BASE_POS = np.array([SCREEN_W // 2, SCREEN_H // 2])
        BASE_RADIUS = 56
        MIN_DIST_FROM_BASE = BASE_RADIUS + 80
        MIN_DIST_FROM_NK = 145.0  # INCREASED safe distance from Night King for better avoidance
        MIN_X, MAX_X = 50, SCREEN_W - 50
        MIN_Y, MAX_Y = 50, SCREEN_H - 150

        # Generate all grid coordinates
        grid_coords = np.indices((G, G)).reshape(2, -1).T  # shape: (G*G, 2)
        grid_y, grid_x = grid_coords[:, 0], grid_coords[:, 1]

        # Convert grid coords to world coordinates
        world_x = (grid_x / G) * SCREEN_W
        world_y = (grid_y / G) * SCREEN_H

        # Check screen bounds
        valid_mask = (world_x >= MIN_X) & (world_x <= MAX_X) & \
                     (world_y >= MIN_Y) & (world_y <= MAX_Y)

        # Distance from base
        base_dists = np.sqrt((world_x - BASE_POS[0])**2 + (world_y - BASE_POS[1])**2)
        valid_mask &= (base_dists >= MIN_DIST_FROM_BASE)

        # Night King proximity (only matters for soldiers)
        if obs is not None and 'grid' in obs:
            nk_cells = np.argwhere(obs['grid'] == 4)
            if len(nk_cells) > 0:
                nk_world_x = (nk_cells[:, 1] / G) * SCREEN_W
                nk_world_y = (nk_cells[:, 0] / G) * SCREEN_H
                # Compute distance from each cell to all NKs
                nk_dists = np.sqrt((world_x[:, None] - nk_world_x[None, :])**2 +
                                   (world_y[:, None] - nk_world_y[None, :])**2)
                min_nk_dist = nk_dists.min(axis=1)
                nk_safe = min_nk_dist >= MIN_DIST_FROM_NK
            else:
                nk_safe = np.ones_like(world_x, dtype=bool)
        else:
            nk_safe = np.ones_like(world_x, dtype=bool)

        # Build full action mask
        action_mask = np.zeros(self.action_dim, dtype=bool)

        # Unit type 0 = soldiers, 1 = heroes
        # Soldiers: valid if available, in valid cells, and safe from NKs
        if soldiers_remaining > 0:
            soldier_mask = valid_mask & nk_safe
            action_mask[0*G*G : 1*G*G] = soldier_mask

        # Heroes: valid if available and in valid cells (ignore NKs)
        if heroes_remaining > 0:
            hero_mask = valid_mask
            action_mask[1*G*G : 2*G*G] = hero_mask

        return torch.from_numpy(action_mask)

    def _action_to_grid(self, action_idx: int) -> np.ndarray:
        """Convert action index to [unit_type, grid_x, grid_y]."""
        unit_type = action_idx // (self.grid_size * self.grid_size)
        remainder = action_idx % (self.grid_size * self.grid_size)
        grid_x = remainder % self.grid_size
        grid_y = remainder // self.grid_size
        return np.array([unit_type, grid_x, grid_y], dtype=np.int64)

    def select_action(self, obs: Dict) -> np.ndarray:
        """Select action using current policy with action masking."""
        features = self._features(obs)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            # Get action probabilities from policy
            probs = self.policy_net(features_tensor).squeeze(0)  # Shape: (action_dim,)

            # Guard against numerical issues: replace non-finite probs
            if not torch.isfinite(probs).all():
                # Replace NaNs/inf with zeros
                probs = torch.where(
                    torch.isfinite(probs),
                    probs,
                    torch.zeros_like(probs)
                )
            # Clamp and normalize to ensure valid distribution
            probs = torch.clamp(probs, 1e-8, 1.0)
            probs = probs / probs.sum()

            # Apply action mask: set invalid actions to 0 probability
            valid_mask = self._get_valid_action_mask(obs)
            masked_probs = probs * valid_mask.float()

            # Renormalize to ensure valid probability distribution
            prob_sum = masked_probs.sum()
            if prob_sum > 1e-8:
                masked_probs = masked_probs / prob_sum
            else:
                # Fallback: uniform over valid actions
                valid_count = valid_mask.sum().float()
                if valid_count > 0:
                    masked_probs = valid_mask.float() / valid_count
                else:
                    # Emergency fallback: uniform over all actions
                    masked_probs = torch.ones_like(probs) / len(probs)

            # Final safety check for NaNs
            if not torch.isfinite(masked_probs).all():
                # Last resort: uniform over valid actions
                valid_count = valid_mask.sum().float()
                if valid_count > 0:
                    masked_probs = valid_mask.float() / valid_count
                else:
                    masked_probs = torch.ones_like(probs) / len(probs)

            # Sample from masked distribution
            dist = Categorical(masked_probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx))

            value = self.value_net(features_tensor)

        # Store for training
        self.states.append(features)
        self.actions.append(action_idx)
        self.log_probs.append(log_prob)
        self.values.append(value.item())

        # Convert to grid coordinates
        return self._action_to_grid(action_idx)

    def _update_reward_stats(self, reward: float):
        """Update running statistics for reward normalization."""
        if not self.normalize_rewards:
            return

        self.reward_count += 1
        # Online update of mean and std
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std**2 + delta * delta2) / self.reward_count
        )
        # Prevent division by zero
        self.reward_std = max(self.reward_std, 1e-8)

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        if not self.normalize_rewards or self.reward_count < 10:
            return reward / float(self.reward_scale)

        # Normalize and scale
        normalized = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return normalized / float(self.reward_scale)

    def store_transition(self, reward: float, done: bool):
        """Store scaled and normalized reward and done flag for current transition."""
        # Update reward statistics
        self._update_reward_stats(reward)

        # Normalize reward for stable learning
        normalized_reward = self._normalize_reward(reward)
        self.rewards.append(normalized_reward)
        self.dones.append(done)

    def compute_returns(self) -> List[float]:
        """Compute discounted returns (rewards-to-go)."""
        returns = []
        G = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def compute_advantages_gae(self) -> List[float]:
        """Compute advantages using Generalized Advantage Estimation (GAE).

        GAE reduces variance in advantage estimates compared to simple returns - values.
        Formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        if len(self.rewards) == 0:
            return []

        advantages = []
        gae = 0.0

        # Compute advantages backwards using GAE
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                # Terminal state: next value is 0
                next_value = 0.0
            elif t == len(self.rewards) - 1:
                # Last step but not done: bootstrap with 0 (conservative, common in PPO)
                # In practice, if episode continues, we'd compute next state value, but for simplicity use 0
                next_value = 0.0
            else:
                # Use next state's value
                next_value = self.values[t + 1]

            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]

            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            # If done, don't add future GAE (multiply by 0)
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(self.dones[t])) * gae
            advantages.insert(0, gae)

        return advantages

    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.states) == 0:
            return

        # Compute returns and advantages
        returns = self.compute_returns()
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Convert list of numpy arrays to single numpy array first (faster)
        states_array = np.array(self.states)
        states_tensor = torch.FloatTensor(states_array).to(self.device)
        old_log_probs_tensor = torch.stack(self.log_probs).detach().to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)

        # Compute advantages using GAE (reduces variance)
        advantages_list = self.compute_advantages_gae()
        advantages = torch.FloatTensor(advantages_list).to(self.device)

        # Normalize advantages with numerical stability
        # Use unbiased=False to avoid degenerate std warnings when batch size is 1
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        advantages = advantages - adv_mean
        if adv_std > 1e-6:
            advantages = advantages / adv_std

        predicted_values = self.value_net(states_tensor).squeeze(-1)
        value_loss = nn.MSELoss()(predicted_values, returns_tensor)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()

        # Update policy network
        for _ in range(4):  # Multiple PPO epochs
            # Get current policy probabilities
            probs = self.policy_net(states_tensor)

            # Guard against numerical issues: replace non-finite probs and renormalize
            if not torch.isfinite(probs).all():
                # Replace NaNs/inf with uniform distribution as a fallback
                probs = torch.where(
                    torch.isfinite(probs),
                    probs,
                    torch.zeros_like(probs)
                )
            probs = torch.clamp(probs, 1e-8, 1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()

            # Store entropy for monitoring
            self.last_entropy = entropy.item()

            # Compute ratio and clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

        # Reset buffers
        self.reset_buffers()

    def save(self, filepath: str):
        """Save agent to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'feature_dim': self.feature_dim,
            'action_dim': self.action_dim,
            'grid_size': self.grid_size,
        }, filepath)

    @classmethod
    def load(cls, filepath: str, observation_space, action_space):
        """Load agent from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        agent = cls(
            observation_space=observation_space,
            action_space=action_space,
            feature_dim=checkpoint['feature_dim'],
        )
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        agent.value_net.load_state_dict(checkpoint['value_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        return agent
