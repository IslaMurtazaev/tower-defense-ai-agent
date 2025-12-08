import math

import numpy as np


class ApproximateQAgent:
    """Q-Learning agent using linear function approximation for action selection."""

    def __init__(
        self,
        observation_space,
        action_space,
        feature_dim: int = 28,  # Added 3 new features for base defense (was 25)
        alpha: float = 0.001,
        gamma: float = 0.92,  # Reduced from 0.99 for stability with long episodes and variable terminal rewards
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 100_000,
        epsilon_decay_type: str = "linear",  # "linear" or "exponential"
        reduce_action_search: bool = False,
    ):
        self.obs_space = observation_space
        self.action_space = action_space
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_type = epsilon_decay_type
        self.reduce_action_search = reduce_action_search

        limit = np.sqrt(6.0 / (self.feature_dim + 1))
        self.weights = np.random.uniform(-limit, limit, self.feature_dim).astype(np.float32)
        self.step_count = 0

        self.feature_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.feature_std = np.ones(self.feature_dim, dtype=np.float32)
        self.feature_count = 0
        self.normalize_features = True

        self.initial_alpha = alpha
        self.alpha_decay = 0.99995
        self.alpha_min = alpha * 0.1

        # Cache for valid action mask (speeds up select_action)
        self._cached_valid_mask = None
        self._cached_grid_hash = None

        # Cache for world coordinates (avoid repeated calculations)
        self._cached_world_coords = None
        self._cached_world_coords_G = None

    def _epsilon(self) -> float:
        """Calculate current epsilon value using linear or exponential decay."""
        if self.epsilon_decay_type == "exponential":
            decay_rate = self.step_count / float(self.epsilon_decay_steps)
            eps = float(self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-decay_rate))
        else:
            frac = min(1.0, self.step_count / float(self.epsilon_decay_steps))
            eps = float(self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end))

        min_exploration = 0.05
        return max(eps, min_exploration)

    def get_epsilon(self) -> float:
        """Return current exploration rate."""
        return self._epsilon()

    def _is_valid_placement(self, grid_x: int, grid_y: int, grid_size: int, obs: dict = None, unit_type: int = 0) -> bool:
        """Check if a grid position is valid for unit placement.

        Valid placements must:
        - Be at least BASE_RADIUS + 80 pixels from base center
        - Be within screen bounds (50 < x < SCREEN_W-50, 50 < y < SCREEN_H-150)
        - For soldiers (unit_type=0): Be at least NK_SWEEP_RADIUS + 50 pixels from any Night King
        - For heroes (unit_type=1): Can be placed near NKs (they need to fight them)
        """
        # Convert grid to world coordinates
        SCREEN_W, SCREEN_H = 1280, 800
        BASE_POS = (SCREEN_W // 2, SCREEN_H // 2)  # (640, 400)
        BASE_RADIUS = 56
        MIN_DIST_FROM_BASE = BASE_RADIUS + 80  # 136 pixels
        NK_SWEEP_RADIUS = 95  # From game constants
        MIN_DIST_FROM_NK = NK_SWEEP_RADIUS + 50  # 145 pixels - INCREASED safe distance for better NK avoidance

        world_x = (grid_x / grid_size) * SCREEN_W
        world_y = (grid_y / grid_size) * SCREEN_H

        # Check screen bounds
        if not (50 < world_x < SCREEN_W - 50 and 50 < world_y < SCREEN_H - 150):
            return False

        # Check distance from base
        dist_to_base = math.hypot(world_x - BASE_POS[0], world_y - BASE_POS[1])
        if dist_to_base < MIN_DIST_FROM_BASE:
            return False

        # For soldiers, check distance from Night Kings (only during combat phase)
        # During placement phase, NKs don't exist yet, so skip this check
        if unit_type == 0 and obs is not None:
            # Check if we're in combat phase (NKs only exist during combat)
            combat_phase = obs.get("combat_phase", False)
            if combat_phase:
                # Only check NK distance during combat phase
                grid = obs.get("grid")
                if grid is not None:
                    # Find all Night King positions in grid
                    nk_cells = np.argwhere(grid == 4)  # 4 = Night King in grid
                    # Only restrict if NKs are actually visible (during combat phase)
                    if len(nk_cells) > 0:
                        for nk_cell in nk_cells:
                            nk_grid_y, nk_grid_x = nk_cell
                            nk_world_x = (nk_grid_x / grid_size) * SCREEN_W
                            nk_world_y = (nk_grid_y / grid_size) * SCREEN_H

                            # Check if placement is too close to this NK
                            dist_to_nk = math.hypot(world_x - nk_world_x, world_y - nk_world_y)
                            if dist_to_nk < MIN_DIST_FROM_NK:
                                return False  # Too close to NK - will get killed by sweep
            # If not in combat phase, skip NK check (placement phase - NKs don't exist)

        return True

    def _compute_valid_mask(self, obs: dict) -> np.ndarray:
        """Precompute boolean mask of valid placement positions (vectorized).

        Returns:
            Boolean array of shape [2, G, G] where mask[unit_type, gy, gx] = True
            if position (gx, gy) is valid for unit_type.
            Note: Indexing is [gy, gx] to match np.argwhere output format.
        """
        grid = obs.get("grid")
        if grid is None:
            return np.zeros((2, 32, 32), dtype=bool)

        G = grid.shape[0]
        SCREEN_W, SCREEN_H = 1280, 800
        BASE_POS = np.array([SCREEN_W // 2, SCREEN_H // 2])
        BASE_RADIUS = 56
        MIN_DIST_FROM_BASE = BASE_RADIUS + 80
        MIN_DIST_FROM_NK = 145.0  # INCREASED safe distance from Night King for better avoidance
        MIN_X, MAX_X = 50, SCREEN_W - 50
        MIN_Y, MAX_Y = 50, SCREEN_H - 150

        # Generate all grid coordinates (vectorized)
        grid_coords = np.indices((G, G)).reshape(2, -1).T  # shape: (G*G, 2)
        grid_y, grid_x = grid_coords[:, 0], grid_coords[:, 1]

        # Convert grid coords to world coordinates (vectorized)
        world_x = (grid_x / G) * SCREEN_W
        world_y = (grid_y / G) * SCREEN_H

        # Check screen bounds (vectorized)
        # Note: Use strict inequality < to match place_hero validation (not <=)
        valid_mask = (world_x > MIN_X) & (world_x < MAX_X) & \
                     (world_y > MIN_Y) & (world_y < MAX_Y)

        # Distance from base (vectorized)
        base_dists = np.sqrt((world_x - BASE_POS[0])**2 + (world_y - BASE_POS[1])**2)
        valid_mask &= (base_dists >= MIN_DIST_FROM_BASE)

        # Night King proximity (only matters for soldiers) - vectorized
        # Only check NK distance during combat phase (NKs don't exist during placement)
        combat_phase = obs.get("combat_phase", False) if obs is not None else False
        if combat_phase and obs is not None and 'grid' in obs:
            nk_cells = np.argwhere(obs['grid'] == 4)
            if len(nk_cells) > 0:
                nk_world_x = (nk_cells[:, 1] / G) * SCREEN_W
                nk_world_y = (nk_cells[:, 0] / G) * SCREEN_H
                # Compute distance from each cell to all NKs (vectorized)
                nk_dists = np.sqrt((world_x[:, None] - nk_world_x[None, :])**2 +
                                   (world_y[:, None] - nk_world_y[None, :])**2)
                min_nk_dist = nk_dists.min(axis=1)
                nk_safe = min_nk_dist >= MIN_DIST_FROM_NK
            else:
                nk_safe = np.ones_like(world_x, dtype=bool)
        else:
            # During placement phase, no NKs exist, so all positions are safe
            nk_safe = np.ones_like(world_x, dtype=bool)

        # Reshape to [G, G] for each unit type
        # np.indices creates coordinates where grid_coords[:, 0] = grid_y and grid_coords[:, 1] = grid_x
        # The flattened array is in order: [y0,x0], [y0,x1], ..., [y0,x(G-1)], [y1,x0], ...
        # When reshaped to (G, G), this creates mask_2d[gy, gx] (row=gy, col=gx)
        # np.argwhere returns (row, col) = (gy, gx), which matches our indexing
        valid_mask_2d = valid_mask.reshape(G, G)  # Shape: [gy, gx]
        nk_safe_2d = nk_safe.reshape(G, G)  # Shape: [gy, gx]

        # Build mask for both unit types: [2, G, G] where mask[unit_type, gy, gx]
        # Note: The code uses gy, gx = idx[0], idx[1] from np.argwhere, so mask[unit_type, gy, gx]
        mask = np.zeros((2, G, G), dtype=bool)
        # Unit type 0 = soldiers: valid if in valid cells and safe from NKs
        mask[0] = valid_mask_2d & nk_safe_2d
        # Unit type 1 = heroes: valid if in valid cells (ignore NKs)
        mask[1] = valid_mask_2d

        return mask

    def _get_world_coords(self, G: int) -> tuple:
        """Get cached world coordinates for grid cells.

        Returns:
            Tuple of (world_x, world_y) arrays where world_x[gx, gy] and world_y[gx, gy]
            give world coordinates for grid position (gx, gy)
        """
        # Return cached coords if grid size hasn't changed
        if self._cached_world_coords is not None and self._cached_world_coords_G == G:
            return self._cached_world_coords

        # Compute world coordinates for all grid cells
        SCREEN_W, SCREEN_H = 1280, 800
        gx_arr = np.arange(G, dtype=np.float32)
        gy_arr = np.arange(G, dtype=np.float32)
        # world_x[gx, gy] = world x coordinate for grid cell (gx, gy)
        # Use meshgrid to create proper 2D arrays
        gx_grid, gy_grid = np.meshgrid(gx_arr, gy_arr, indexing='ij')
        world_x = (gx_grid / G) * SCREEN_W  # Shape: (G, G), indexed by [gx, gy]
        world_y = (gy_grid / G) * SCREEN_H  # Shape: (G, G), indexed by [gx, gy]

        self._cached_world_coords = (world_x, world_y)
        self._cached_world_coords_G = G
        return self._cached_world_coords

    def _get_valid_mask(self, obs: dict) -> np.ndarray:
        """Get valid action mask, using cache if grid hasn't changed."""
        # Compute hash of grid to detect changes
        grid = obs.get("grid")
        if grid is None:
            return np.zeros((2, 32, 32), dtype=bool)

        # Simple hash: use grid data hash (fast for caching)
        grid_hash = hash(grid.tobytes())

        # Return cached mask if grid hasn't changed
        if self._cached_valid_mask is not None and self._cached_grid_hash == grid_hash:
            return self._cached_valid_mask

        # Compute new mask and cache it
        self._cached_valid_mask = self._compute_valid_mask(obs)
        self._cached_grid_hash = grid_hash
        return self._cached_valid_mask

    def select_action(self, obs: dict) -> np.ndarray:
        """Select action using epsilon-greedy over all grid cells.

        Only considers valid placement positions (filters out invalid ones).

        Action format: [unit_type, grid_x, grid_y] where unit_type: 0=soldier, 1=hero
        """
        self.step_count += 1
        eps = self._epsilon()

        # Check how many units are remaining
        soldiers_remaining = obs.get('soldiers_remaining', [0])[0]
        heroes_remaining = obs.get('heroes_remaining', [0])[0]

        # Precompute valid positions mask (cached if grid hasn't changed)
        valid_mask = self._get_valid_mask(obs)

        # Epsilon-greedy: explore randomly or exploit best action
        if np.random.rand() < eps:
            # Exploration: sample only from valid positions
            valid_positions = []
            for unit_type in range(2):
                if unit_type == 0 and soldiers_remaining <= 0:
                    continue
                if unit_type == 1 and heroes_remaining <= 0:
                    continue
                # Get all valid positions for this unit type
                # np.argwhere returns (row, col) = (gy, gx)
                valid_indices = np.argwhere(valid_mask[unit_type])
                for idx in valid_indices:
                    gy, gx = int(idx[0]), int(idx[1])
                    valid_positions.append(np.array([unit_type, gx, gy], dtype=np.int64))

            if valid_positions:
                idx = np.random.randint(len(valid_positions))
                return valid_positions[int(idx)]
            else:
                # Fallback: return random action if no valid positions found
                return self.action_space.sample()

        # Exploitation: find best Q-value among valid positions
        # Sample a reasonable number of actions to evaluate (for speed)
        best_q = -1e9
        best_action = None

        # Collect all valid actions first
        valid_actions = []
        for unit_type in range(2):
            # Check if this unit type is available
            if unit_type == 0 and soldiers_remaining <= 0:
                continue
            if unit_type == 1 and heroes_remaining <= 0:
                continue

            # Get valid positions for this unit type
            valid_indices = np.argwhere(valid_mask[unit_type])
            # np.argwhere returns (row, col) = (gy, gx)
            for idx in valid_indices:
                gy, gx = int(idx[0]), int(idx[1])
                valid_actions.append((unit_type, gx, gy))

        if not valid_actions:
            # Fallback: return random action if no valid positions found
            return self.action_space.sample()

        # Sample actions to evaluate (for speed)
        if self.reduce_action_search:
            # Sample up to 50 actions for faster evaluation
            max_samples = min(50, len(valid_actions))
            if max_samples < len(valid_actions):
                sampled_indices = np.random.choice(len(valid_actions), size=max_samples, replace=False)
                actions_to_evaluate = [valid_actions[i] for i in sampled_indices]
            else:
                actions_to_evaluate = valid_actions
        else:
            # Evaluate all valid actions
            actions_to_evaluate = valid_actions

        # Evaluate Q-values for sampled actions
        for unit_type, gx, gy in actions_to_evaluate:
            action = np.array([unit_type, gx, gy], dtype=np.int64)
            q_val = self._q_value(obs, action)
            if q_val > best_q:
                best_q = q_val
                best_action = action

        if best_action is None:
            # Fallback: try to find any valid action
            for unit_type in range(2):
                if unit_type == 0 and soldiers_remaining <= 0:
                    continue
                if unit_type == 1 and heroes_remaining <= 0:
                    continue
                valid_indices = np.argwhere(valid_mask[unit_type])
                if len(valid_indices) > 0:
                    idx = valid_indices[0]
                    gy, gx = int(idx[0]), int(idx[1])
                    return np.array([unit_type, gx, gy], dtype=np.int64)
            # Last resort: return random (shouldn't happen)
            return self.action_space.sample()

        return best_action

    # -------------------------
    # Q-value and features
    # -------------------------
    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics for stable learning."""
        if not self.normalize_features or self.feature_count < 10:
            return feats

        # Use running statistics for normalization
        normalized = (feats - self.feature_mean) / (self.feature_std + 1e-8)
        return normalized

    def _update_feature_stats(self, feats: np.ndarray):
        """Update running statistics for feature normalization."""
        self.feature_count += 1
        # Online update of mean and std
        delta = feats - self.feature_mean
        self.feature_mean += delta / self.feature_count
        delta2 = feats - self.feature_mean
        self.feature_std = np.sqrt(
            ((self.feature_count - 1) * self.feature_std**2 + delta * delta2) / self.feature_count
        )
        # Prevent division by zero
        self.feature_std = np.maximum(self.feature_std, 1e-8)

    def _q_value(self, obs: dict, action: np.ndarray) -> float:
        """Estimate Q-value using linear function approximation: Q(s,a) = w·f(s,a)"""
        feats = self._features(obs, action)

        # Update feature statistics
        self._update_feature_stats(feats)

        # Normalize features for stable learning
        feats_normalized = self._normalize_features(feats)

        return float(np.dot(self.weights, feats_normalized))

    def _features(self, obs: dict, action: np.ndarray) -> np.ndarray:
        """Extract feature vector from game state and proposed action."""
        grid = obs["grid"]  # (G, G)
        soldiers_remaining = float(obs["soldiers_remaining"][0])
        heroes_remaining = float(obs.get("heroes_remaining", [0])[0])
        base_hp_ratio = float(obs["base_hp_ratio"][0])
        current_wave = float(obs["current_wave"][0])
        night_king_active = float(obs["night_king_active"][0])

        G = grid.shape[0]
        # Action is now [unit_type, grid_x, grid_y]
        unit_type = int(action[0])
        gx, gy = int(action[1]), int(action[2])
        # G >= 2 always, so no need for max(1, G-1)
        gx_norm = gx / (G - 1) if G > 1 else 0.0
        gy_norm = gy / (G - 1) if G > 1 else 0.0

        # Distance to base (helps agent place soldiers strategically)
        base_cells = np.argwhere(grid == 1)
        if base_cells.size > 0:
            byx = base_cells.mean(axis=0)
            by, bx = float(byx[0]), float(byx[1])
            dist_to_base = np.sqrt((gx - bx) ** 2 + (gy - by) ** 2)
            dist_to_base_norm = dist_to_base / np.sqrt(2 * (G - 1) ** 2)
        else:
            dist_to_base_norm = 0.0

        # Night King proximity features (consider ALL Night Kings, not just nearest)
        # NK_SWEEP_RADIUS = 95 pixels, so soldiers should stay >95 pixels away
        nk_cells = np.argwhere(grid == 4)  # 4 = Night King in grid
        SCREEN_W, SCREEN_H = 1280, 800
        NK_SWEEP_RADIUS = 95.0  # pixels
        max_screen_dist = math.hypot(SCREEN_W, SCREEN_H)

        # Initialize new NK avoidance features with default safe values
        safety_score = 1.0  # Default: maximum safety (no NKs)
        nk_engagement_risk = 0.0  # Default: no risk
        soldier_nk_separation = 1.0  # Default: maximum separation

        if nk_cells.size > 0:
            # Use cached world coordinates
            world_x_grid, world_y_grid = self._get_world_coords(G)
            action_world_x = world_x_grid[gx, gy]
            action_world_y = world_y_grid[gx, gy]

            # Vectorized NK distance computation
            nk_coords = nk_cells.astype(np.float32)
            nk_world_x = (nk_coords[:, 1] / G) * SCREEN_W
            nk_world_y = (nk_coords[:, 0] / G) * SCREEN_H

            # Vectorized distance calculation
            dx = action_world_x - nk_world_x
            dy = action_world_y - nk_world_y
            nk_distances = np.sqrt(dx * dx + dy * dy)

            # Minimum distance to any Night King
            min_nk_dist_pixels = float(np.min(nk_distances))
            dist_to_nk_norm = min_nk_dist_pixels / max_screen_dist if max_screen_dist > 0 else 1.0

            # Feature 2: Binary - is ANY NK within sweep radius?
            is_near_nk = 1.0 if np.any(nk_distances < NK_SWEEP_RADIUS) else 0.0

            # Feature 3: Average distance to ALL Night Kings (overall safety)
            # Normalize correctly - average of normalized distances
            avg_nk_dist_norm = float(np.mean(nk_distances / max_screen_dist)) if max_screen_dist > 0 else 1.0
            avg_dist_to_nk_norm = avg_nk_dist_norm

            # Feature 4: Number of NKs within sweep radius (danger level)
            nks_within_sweep = int(np.sum(nk_distances < NK_SWEEP_RADIUS))
            nks_nearby_norm = min(nks_within_sweep / 4.0, 1.0)  # Normalize by max NKs (4)

            # Feature 5: Total danger score from all NKs (aggregate threat)
            # Vectorized danger calculation
            total_nk_danger = float(np.sum(1.0 / (nk_distances + 1.0)))
            total_danger_norm = min(total_nk_danger / 4.0, 1.0)  # Normalize

            # Safety score (inverse of danger)
            # For soldiers, we want HIGH safety score (far from NKs)
            safety_score = 1.0 - min(total_danger_norm, 1.0)  # Invert danger to get safety

            # NK engagement risk (binary: 1 if any NK is locked for battle and nearby)
            # This indicates active NK threat that soldiers should avoid
            nk_engagement_risk = 1.0 if (is_near_nk > 0.0 and min_nk_dist_pixels < 200.0) else 0.0

            # Soldier-NK separation preference
            # This encourages soldiers to be placed far from NK engagement zones
            soldier_nk_separation = min(min_nk_dist_pixels / 300.0, 1.0)  # Normalize to 0-1, prefer > 200 pixels
        else:
            dist_to_nk_norm = 1.0  # No NK = safe (max distance)
            is_near_nk = 0.0
            avg_dist_to_nk_norm = 1.0
            nks_nearby_norm = 0.0
            total_danger_norm = 0.0
            # safety_score, nk_engagement_risk, soldier_nk_separation already initialized above

        # Wight proximity features (consider ALL wights, not just nearest)
        wight_cells = np.argwhere(grid == 3)  # 3 = wight in grid
        wight_min_dist_norm = 1.0
        wights_nearby_norm = 0.0
        wight_density_window = 0.0
        WIGHT_PROXIMITY_RADIUS = 120.0  # Separate radius for wights (not NK radius)
        if wight_cells.size > 0:
            # Use cached world coordinates
            world_x_grid, world_y_grid = self._get_world_coords(G)
            action_world_x = world_x_grid[gx, gy]
            action_world_y = world_y_grid[gx, gy]

            # Vectorized wight distance computation
            wight_coords = wight_cells.astype(np.float32)
            wx_world = (wight_coords[:, 1] / G) * SCREEN_W
            wy_world = (wight_coords[:, 0] / G) * SCREEN_H

            # Vectorized distance calculation
            dx = action_world_x - wx_world
            dy = action_world_y - wy_world
            wight_distances = np.sqrt(dx * dx + dy * dy)

            min_wight_dist = float(np.min(wight_distances))
            nearby_wights = int(np.sum(wight_distances < WIGHT_PROXIMITY_RADIUS))
            max_dist = max_screen_dist if max_screen_dist > 0 else 1.0
            wight_min_dist_norm = min_wight_dist / max_dist
            wights_nearby_norm = nearby_wights / max(1, len(wight_cells))

            # Local wight density in 3x3 window
            x0w, x1w = max(0, gx - 1), min(G, gx + 2)
            y0w, y1w = max(0, gy - 1), min(G, gy + 2)
            local_patch_w = grid[y0w:y1w, x0w:x1w]
            wight_density_window = float(np.sum(local_patch_w == 3)) / 9.0

        # Local soldier density (3x3 window) - avoid clustering
        x0, x1 = max(0, gx - 1), min(G, gx + 2)
        y0, y1 = max(0, gy - 1), min(G, gy + 2)
        local_patch = grid[y0:y1, x0:x1]
        local_soldiers = float(np.sum(local_patch == 2))
        total_soldiers = float(np.sum(grid == 2))

        max_cells = 9.0
        local_soldiers_norm = local_soldiers / max_cells
        total_soldiers_norm = total_soldiers / float(G * G) if G > 0 else 0.0

        cell_val = int(grid[gy, gx])
        is_on_base = 1.0 if cell_val == 1 else 0.0
        is_on_soldier = 1.0 if cell_val == 2 else 0.0

        # Base defense features - check if we're placing near base and if enemies are threatening
        BASE_DEFENSE_RADIUS = 200.0  # Soldiers within 200 pixels are "defending"
        ENEMY_THREAT_RADIUS = 250.0  # Enemies within 250 pixels are "threatening"
        max_screen_dist = math.hypot(1280, 800)

        # Check if this placement position is close to the base
        SCREEN_W, SCREEN_H = 1280, 800
        BASE_POS = (SCREEN_W // 2, SCREEN_H // 2)
        world_x = (gx / G) * SCREEN_W
        world_y = (gy / G) * SCREEN_H
        placement_dist_to_base = math.hypot(world_x - BASE_POS[0], world_y - BASE_POS[1])
        is_near_base_for_defense = 1.0 if placement_dist_to_base <= BASE_DEFENSE_RADIUS else 0.0

        # Count how many enemies are close to the base
        enemies_near_base_count = 0
        for enemy_cell in wight_cells:
            enemy_world_x = (enemy_cell[1] / G) * SCREEN_W
            enemy_world_y = (enemy_cell[0] / G) * SCREEN_H
            enemy_dist_to_base = math.hypot(enemy_world_x - BASE_POS[0], enemy_world_y - BASE_POS[1])
            if enemy_dist_to_base <= ENEMY_THREAT_RADIUS:
                enemies_near_base_count += 1

        enemies_near_base_norm = min(enemies_near_base_count / 10.0, 1.0)  # Normalize to 0-1

        # This score is high when we place near base AND enemies are threatening
        # So the agent learns: "place soldiers near base when enemies are close"
        base_defense_score = is_near_base_for_defense * enemies_near_base_norm

        # Build feature vector: position, density, game state, placement flags, unit type
        feats = np.zeros(self.feature_dim, dtype=np.float32)
        i = 0
        # Remove duplicate unit_type - only keep can_attack_nk
        if i < self.feature_dim:
            feats[i] = 1.0 if unit_type == 1 else 0.0; i += 1  # can_attack_nk flag (heroes only)
        if i < self.feature_dim:
            feats[i] = gx_norm; i += 1
        if i < self.feature_dim:
            feats[i] = gy_norm; i += 1
        if i < self.feature_dim:
            feats[i] = dist_to_base_norm; i += 1
        if i < self.feature_dim:
            feats[i] = local_soldiers_norm; i += 1
        if i < self.feature_dim:
            feats[i] = total_soldiers_norm; i += 1
        if i < self.feature_dim:
            feats[i] = soldiers_remaining / 6.0; i += 1  # Normalize by 6 (we have 6 soldiers max now)
        if i < self.feature_dim:
            feats[i] = heroes_remaining / 2.0; i += 1  # Normalized
        if i < self.feature_dim:
            feats[i] = base_hp_ratio; i += 1
        if i < self.feature_dim:
            feats[i] = current_wave / 5.0; i += 1
        if i < self.feature_dim:
            feats[i] = night_king_active; i += 1
        if i < self.feature_dim:
            feats[i] = dist_to_nk_norm; i += 1  # Min distance to ANY NK (0=close/dangerous, 1=far/safe)
        if i < self.feature_dim:
            feats[i] = is_near_nk; i += 1  # Binary: 1 if ANY NK within sweep radius, 0 otherwise
        if i < self.feature_dim:
            feats[i] = avg_dist_to_nk_norm; i += 1  # Average distance to ALL NKs
        if i < self.feature_dim:
            feats[i] = nks_nearby_norm; i += 1  # Number of NKs within sweep radius (normalized)
        if i < self.feature_dim:
            feats[i] = total_danger_norm; i += 1  # Aggregate danger score from ALL NKs
        if i < self.feature_dim:
            feats[i] = safety_score; i += 1  # Safety score (higher = safer from NKs)
        if i < self.feature_dim:
            feats[i] = nk_engagement_risk; i += 1  # NK engagement risk (1 = active threat)
        if i < self.feature_dim:
            feats[i] = soldier_nk_separation; i += 1  # Soldier-NK separation preference
        if i < self.feature_dim:
            feats[i] = is_near_base_for_defense; i += 1  # Is this placement near base? (1=yes, 0=no)
        if i < self.feature_dim:
            feats[i] = enemies_near_base_norm; i += 1  # How many enemies are threatening base? (0-1)
        if i < self.feature_dim:
            feats[i] = base_defense_score; i += 1  # Combined: high when near base AND enemies are close
        if i < self.feature_dim:
            feats[i] = wight_min_dist_norm; i += 1  # Min distance to wights (0=close/good to engage, 1=far)
        if i < self.feature_dim:
            feats[i] = wights_nearby_norm; i += 1  # Fraction of wights within proximity radius
        if i < self.feature_dim:
            feats[i] = wight_density_window; i += 1  # Local wight density (3x3 window)
        if i < self.feature_dim:
            feats[i] = 0.0; i += 1  # is_on_base (disabled)
        if i < self.feature_dim:
            feats[i] = 0.0; i += 1  # is_on_soldier (disabled)
        if i < self.feature_dim:
            feats[i] = 1.0; i += 1  # Bias term

        # Safety: replace NaN/Inf with 0
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # For soldiers, boost the features that help them avoid NKs and defend base
        if unit_type == 0:  # This is a soldier placement
            safety_idx = 14  # Safety from Night Kings
            nk_risk_idx = 15  # Risk of engaging Night Kings
            separation_idx = 16  # Distance from Night Kings
            base_defense_idx = 17  # Is placement near base?
            enemies_near_base_idx = 18  # Are enemies threatening?
            defense_score_idx = 19  # Combined defense score
            # Make these features more important for soldiers
            if safety_idx < len(feats):
                feats[safety_idx] *= 2.0  # Really important: stay safe from NKs
            if nk_risk_idx < len(feats):
                feats[nk_risk_idx] *= 2.0  # Really important: avoid NK engagement
            if separation_idx < len(feats):
                feats[separation_idx] *= 2.0  # Really important: stay far from NKs
            if base_defense_idx < len(feats):
                feats[base_defense_idx] *= 1.5  # Important: defend base
            if enemies_near_base_idx < len(feats):
                feats[enemies_near_base_idx] *= 1.5  # Important: know when base is threatened
            if defense_score_idx < len(feats):
                feats[defense_score_idx] *= 2.0  # Really important: defend when needed

        return feats
    def update(self, obs, action, reward, next_obs, done: bool):
        """Update Q-learning weights using TD error."""
        q_sa = self._q_value(obs, action)

        # Compute target: r + gamma * max_a' Q(s', a') or just r if done
        if done:
            target = reward
        else:
            # Use cached valid positions instead of random sampling (faster and more meaningful)
            best_next_q = -1e9
            soldiers_remaining = next_obs.get('soldiers_remaining', [0])[0]
            heroes_remaining = next_obs.get('heroes_remaining', [0])[0]

            # Get valid positions mask (cached)
            valid_mask = self._get_valid_mask(next_obs)

            # Sample up to 50 valid actions (or all if fewer)
            valid_actions = []
            for unit_type in range(2):
                if unit_type == 0 and soldiers_remaining <= 0:
                    continue
                if unit_type == 1 and heroes_remaining <= 0:
                    continue
                # Get valid positions for this unit type
                valid_indices = np.argwhere(valid_mask[unit_type])
                for idx in valid_indices:
                    gy, gx = int(idx[0]), int(idx[1])
                    valid_actions.append((unit_type, gx, gy))

            # Sample up to 50 valid actions (or all if fewer)
            max_samples = min(50, len(valid_actions))
            if max_samples > 0:
                # Randomly sample from valid actions
                sampled_indices = np.random.choice(len(valid_actions), size=max_samples, replace=False)
                for idx in sampled_indices:
                    unit_type, gx, gy = valid_actions[idx]
                    a_next = np.array([unit_type, gx, gy], dtype=np.int64)
                    q_next = self._q_value(next_obs, a_next)
                    if q_next > best_next_q:
                        best_next_q = q_next

            # If no valid actions found, use 0 as fallback
            if best_next_q == -1e9:
                best_next_q = 0.0
            target = reward + self.gamma * best_next_q

        # Adaptive learning rate
        current_alpha = max(self.alpha_min, self.initial_alpha * (self.alpha_decay ** self.step_count))

        # Q-learning update: w += alpha * (target - Q) * features
        td_error = target - q_sa
        feats = self._features(obs, action)

        # Update feature statistics
        self._update_feature_stats(feats)

        # Normalize features for stable learning
        feats_normalized = self._normalize_features(feats)

        # Gradient clipping
        grad_norm = np.linalg.norm(feats_normalized)
        if grad_norm > 10.0:  # Clip large gradients
            feats_normalized = feats_normalized * (10.0 / grad_norm)

        # Add weight regularization (L2) - helps prevent overfitting
        reg_lambda = 1e-5
        self.weights += current_alpha * (td_error * feats_normalized - reg_lambda * self.weights)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def set_weights(self, w: np.ndarray) -> None:
        self.weights = np.array(w, dtype=np.float32)

    def save(self, filepath: str) -> None:
        """Save agent weights and hyperparameters to file."""
        import pickle
        data = {
            'weights': self.weights,
            'feature_dim': self.feature_dim,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'epsilon_decay_type': getattr(self, 'epsilon_decay_type', 'linear'),  # Backward compatible
            'step_count': self.step_count,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str, observation_space, action_space):
        """Load agent from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        agent = cls(
            observation_space=observation_space,
            action_space=action_space,
            feature_dim=data['feature_dim'],
            alpha=data['alpha'],
            gamma=data['gamma'],
            epsilon_start=data['epsilon_start'],
            epsilon_end=data['epsilon_end'],
            epsilon_decay_steps=data['epsilon_decay_steps'],
            epsilon_decay_type=data.get('epsilon_decay_type', 'linear'),  # Backward compatible
        )
        agent.weights = data['weights']
        agent.step_count = data.get('step_count', 0)
        return agent
