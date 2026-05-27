"""Toroidal Traversal Generator for temporal state mapping on a 2D torus grid."""
import numpy as np
from src.config import TOROIDAL_N, TOROIDAL_GRID_SIZE, TEMPORAL_STATES, SEED


class ToroidalTraversalGenerator:
    """Maps 168 weekly temporal states onto an N×N toroidal grid.

    Implements a deterministic traversal algorithm with toroidal wrapping
    and collision recovery. The traversal runs for N*N steps; only the
    first 168 steps correspond to real temporal states (Day 0 Hour 0
    through Day 6 Hour 23). Remaining steps are "Phantom States".

    Rules (translated to 0-indexed):
        1. Start at (N//2, 0).
        2. Move up-left: R' = R-1, C' = C-1.
        3. Wrap: if R'<0 → R'=N-1; if C'<0 → C'=N-1.
        4. Double wrap: if both <0, both → N-1.
        5. Collision: if cell occupied → R'=R, C'=C+1.
        6. Right wrap: if C'>=N → C'=0.
    """

    def __init__(self, n: int = TOROIDAL_N):
        self.n = n
        self.grid = np.full((n, n), -1, dtype=int)
        self.collision_grid = np.zeros((n, n), dtype=int)
        self.state_positions = {}  # traversal_index -> (r, c)
        self.position_to_state = {}  # (r, c) -> traversal_index
        self.collision_events = []  # list of (traversal_index, r, c) where rule 5 fired
        self._traverse()

    def _traverse(self):
        """Run the deterministic traversal algorithm for N*N steps."""
        n = self.n
        r, c = n // 2, 0  # Rule 1: start position (0-indexed)

        for step in range(n * n):
            # Place current step
            self.grid[r, c] = step
            self.state_positions[step] = (r, c)
            self.position_to_state[(r, c)] = step

            if step == n * n - 1:
                break

            # Compute candidate next position (Rule 2)
            nr, nc = r - 1, c - 1

            # Rule 3: Toroidal wrapping
            wrapped_r = nr < 0
            wrapped_c = nc < 0
            if wrapped_r:
                nr = n - 1
            if wrapped_c:
                nc = n - 1

            # Rule 4: Double wrap (both < 0) → both = N-1
            # Already handled by Rule 3 individually, but ensure consistency
            if r - 1 < 0 and c - 1 < 0:
                nr, nc = n - 1, n - 1

            # Rule 5: Collision recovery
            if self.grid[nr, nc] != -1:
                self.collision_events.append((step + 1, r, c))
                self.collision_grid[r, c] += 1
                nr, nc = r, c + 1  # Same row, one column right

                # Rule 6: Right wrap
                if nc >= n:
                    nc = 0

            r, c = nr, nc

    def get_temporal_state(self, day_of_week: int, hour: int) -> int:
        """Get the traversal index for a given (day_of_week, hour) pair.

        Args:
            day_of_week: 0-6 (Monday=0, Sunday=6)
            hour: 0-23

        Returns:
            Traversal index (0-167), or -1 if phantom state.
        """
        if not (0 <= day_of_week <= 6) or not (0 <= hour <= 23):
            return -1
        state_idx = day_of_week * 24 + hour
        if state_idx >= TEMPORAL_STATES:
            return -1
        return state_idx

    def get_position(self, day_of_week: int, hour: int) -> tuple:
        """Get the (row, col) grid position for a temporal state.

        Args:
            day_of_week: 0-6
            hour: 0-23

        Returns:
            (row, col) tuple, or None if invalid.
        """
        state_idx = self.get_temporal_state(day_of_week, hour)
        if state_idx == -1:
            return None
        return self.state_positions.get(state_idx)

    def get_toroidal_phase(self, day_of_week: int, hour: int) -> float:
        """Compute ToroidalPhase = traversal_index / 256."""
        state_idx = self.get_temporal_state(day_of_week, hour)
        if state_idx == -1:
            return 0.0
        return state_idx / (self.n * self.n)

    def get_neighbors(self, r: int, c: int) -> list:
        """Get the 4 cardinal neighbors (up, down, left, right) with toroidal wrapping."""
        n = self.n
        return [
            ((r - 1) % n, c),  # up
            ((r + 1) % n, c),  # down
            (r, (c - 1) % n),  # left
            (r, (c + 1) % n),  # right
        ]

    def get_neighborhood_entropy(self, day_of_week: int, hour: int, demand_map: dict) -> float:
        """Compute variance of average demand of adjacent cells (ignoring phantoms).

        Args:
            day_of_week: 0-6
            hour: 0-23
            demand_map: dict mapping (day_of_week, hour) -> average demand

        Returns:
            Variance of neighbor demands (0.0 if no valid neighbors).
        """
        pos = self.get_position(day_of_week, hour)
        if pos is None:
            return 0.0

        r, c = pos
        neighbors = self.get_neighbors(r, c)
        neighbor_demands = []

        for nr, nc in neighbors:
            state_idx = self.grid[nr, nc]
            # Ignore phantom states (index >= 168 or -1)
            if state_idx == -1 or state_idx >= TEMPORAL_STATES:
                continue
            # Convert state_idx back to (dow, hour)
            ndow = state_idx // 24
            nhour = state_idx % 24
            if (ndow, nhour) in demand_map:
                neighbor_demands.append(demand_map[(ndow, nhour)])

        if len(neighbor_demands) < 2:
            return 0.0
        return float(np.var(neighbor_demands))

    def get_collision_frequency(self, day_of_week: int, hour: int) -> int:
        """Count collision recovery events (Rule 5) in the 3x3 vicinity of this state.

        Args:
            day_of_week: 0-6
            hour: 0-23

        Returns:
            Count of collision events in 3x3 neighborhood.
        """
        pos = self.get_position(day_of_week, hour)
        if pos is None:
            return 0

        r, c = pos
        n = self.n
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = (r + dr) % n, (c + dc) % n
                count += self.collision_grid[nr, nc]
        return count

    def get_grid_summary(self) -> dict:
        """Return a summary of the grid state for debugging."""
        real_states = np.sum(self.grid >= 0) - np.sum(self.grid >= TEMPORAL_STATES)
        phantom_states = np.sum(self.grid >= TEMPORAL_STATES)
        total_collisions = len(self.collision_events)
        return {
            "grid_size": self.n * self.n,
            "real_states_placed": min(real_states, TEMPORAL_STATES),
            "phantom_states": phantom_states,
            "total_collision_events": total_collisions,
            "collision_rate": total_collisions / (self.n * self.n),
        }
