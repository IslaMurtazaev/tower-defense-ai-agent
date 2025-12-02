import numpy as np


class ApproximateQAgent:
    """Q-Learning agent using linear function approximation for action selection."""

    def __init__(
        self,
        observation_space,
        action_space,
        feature_dim: int = 16,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 100_000,
    ):
        self.obs_space = observation_space
        self.action_space = action_space
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.weights = np.zeros(self.feature_dim, dtype=np.float32)
        self.step_count = 0

    def _epsilon(self) -> float:
        """Calculate current epsilon value using linear decay."""
        # Epsilon controls exploration vs exploitation
        # Starts high (explore) and decays to low (exploit)
        frac = min(1.0, self.step_count / float(self.epsilon_decay_steps))
        return float(self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start))

    def get_epsilon(self) -> float:
        """Return current exploration rate."""
        return self._epsilon()

    def select_action(self, obs: dict) -> np.ndarray:
        """Select action using epsilon-greedy over all grid cells.

        Action format: [grid_x, grid_y]
        """
        self.step_count += 1
        eps = self._epsilon()

        # Epsilon-greedy: explore randomly or exploit best action
        if np.random.rand() < eps:
            return self.action_space.sample()

        # Find best action by evaluating all grid positions
        best_q = -1e9
        best_action = None
        grid_size = int(self.action_space.nvec[0])
        for gx in range(grid_size):
            for gy in range(grid_size):
                action = np.array([gx, gy], dtype=np.int64)
                q_val = self._q_value(obs, action)
                if q_val > best_q:
                    best_q = q_val
                    best_action = action

        if best_action is None:
            # Fallback
            return self.action_space.sample()
        return best_action

    # -------------------------
    # Q-value and features
    # -------------------------
    def _q_value(self, obs: dict, action: np.ndarray) -> float:
        """Estimate Q-value using linear function approximation: Q(s,a) = w·f(s,a)"""
        feats = self._features(obs, action)
        return float(np.dot(self.weights, feats))

    def _features(self, obs: dict, action: np.ndarray) -> np.ndarray:
        """Extract feature vector from game state and proposed action."""
        grid = obs["grid"]  # (G, G)
        soldiers_remaining = float(obs["soldiers_remaining"][0])
        base_hp_ratio = float(obs["base_hp_ratio"][0])
        current_wave = float(obs["current_wave"][0])
        night_king_active = float(obs["night_king_active"][0])

        G = grid.shape[0]
        gx, gy = int(action[0]), int(action[1])
        gx_norm = gx / max(1, G - 1)
        gy_norm = gy / max(1, G - 1)

        # Distance to base (helps agent place soldiers strategically)
        base_cells = np.argwhere(grid == 1)
        if base_cells.size > 0:
            byx = base_cells.mean(axis=0)
            by, bx = float(byx[0]), float(byx[1])
            dist_to_base = np.sqrt((gx - bx) ** 2 + (gy - by) ** 2)
            dist_to_base_norm = dist_to_base / np.sqrt(2 * (G - 1) ** 2)
        else:
            dist_to_base_norm = 0.0

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

        # Build feature vector: position, density, game state, placement flags
        feats = np.zeros(self.feature_dim, dtype=np.float32)
        i = 0
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
            feats[i] = soldiers_remaining / 10.0; i += 1
        if i < self.feature_dim:
            feats[i] = base_hp_ratio; i += 1
        if i < self.feature_dim:
            feats[i] = current_wave / 5.0; i += 1
        if i < self.feature_dim:
            feats[i] = night_king_active; i += 1
        if i < self.feature_dim:
            feats[i] = is_on_base; i += 1
        if i < self.feature_dim:
            feats[i] = is_on_soldier; i += 1
        if i < self.feature_dim:
            feats[i] = 1.0; i += 1  # Bias term

        return feats
    def update(self, obs, action, reward, next_obs, done: bool):
        """Update Q-learning weights using TD error."""
        q_sa = self._q_value(obs, action)

        # Compute target: r + gamma * max_a' Q(s', a') or just r if done
        if done:
            target = reward
        else:
            # Find best next action (greedy policy for target)
            best_next_q = -1e9
            grid_size = int(self.action_space.nvec[0])
            for gx in range(grid_size):
                for gy in range(grid_size):
                    a_next = np.array([gx, gy], dtype=np.int64)
                    q_next = self._q_value(next_obs, a_next)
                    if q_next > best_next_q:
                        best_next_q = q_next
            target = reward + self.gamma * best_next_q

        # Q-learning update: w += alpha * (target - Q) * features
        td_error = target - q_sa
        feats = self._features(obs, action)
        self.weights += self.alpha * td_error * feats

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
        )
        agent.weights = data['weights']
        agent.step_count = data.get('step_count', 0)
        return agent
