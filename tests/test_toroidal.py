"""Unit tests for ToroidalTraversalGenerator."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.toroidal import ToroidalTraversalGenerator
from src.config import TOROIDAL_N, TOROIDAL_GRID_SIZE, TEMPORAL_STATES


class TestToroidalTraversalGenerator:
    """Tests for the ToroidalTraversalGenerator class."""

    @pytest.fixture
    def gen(self):
        """Create a fresh generator for each test."""
        return ToroidalTraversalGenerator(n=TOROIDAL_N)

    def test_grid_shape(self, gen):
        """Grid should be N x N."""
        assert gen.grid.shape == (TOROIDAL_N, TOROIDAL_N)

    def test_grid_size(self, gen):
        """Grid should contain exactly GRID_SIZE cells."""
        assert gen.grid.size == TOROIDAL_GRID_SIZE

    def test_start_position(self, gen):
        """Step 0 should be at (N//2, 0) — Rule 1."""
        pos = gen.state_positions.get(0)
        assert pos is not None
        assert pos == (TOROIDAL_N // 2, 0)

    def test_all_168_states_placed(self, gen):
        """All 168 real temporal states should be placed in the grid."""
        placed = sum(1 for idx in gen.state_positions if idx < TEMPORAL_STATES)
        assert placed == TEMPORAL_STATES

    def test_no_duplicate_positions(self, gen):
        """Each grid cell should be occupied by at most one state."""
        occupied = [(r, c) for r in range(TOROIDAL_N) for c in range(TOROIDAL_N) if gen.grid[r, c] != -1]
        assert len(occupied) == len(set(occupied))

    def test_temporal_state_mapping(self, gen):
        """get_temporal_state should return correct index for valid (dow, hour)."""
        # Day 0, Hour 0 -> index 0
        assert gen.get_temporal_state(0, 0) == 0
        # Day 0, Hour 23 -> index 23
        assert gen.get_temporal_state(0, 23) == 23
        # Day 1, Hour 0 -> index 24
        assert gen.get_temporal_state(1, 0) == 24
        # Day 6, Hour 23 -> index 167
        assert gen.get_temporal_state(6, 23) == 167

    def test_temporal_state_invalid(self, gen):
        """Invalid temporal states should return -1."""
        assert gen.get_temporal_state(7, 0) == -1
        assert gen.get_temporal_state(0, 24) == -1

    def test_position_valid(self, gen):
        """get_position should return a valid (row, col) for all temporal states."""
        for dow in range(7):
            for hour in range(24):
                pos = gen.get_position(dow, hour)
                assert pos is not None
                r, c = pos
                assert 0 <= r < TOROIDAL_N
                assert 0 <= c < TOROIDAL_N

    def test_position_invalid(self, gen):
        """get_position should return None for invalid temporal states."""
        assert gen.get_position(7, 0) is None

    def test_toroidal_phase_range(self, gen):
        """ToroidalPhase should be in [0, 1)."""
        for dow in range(7):
            for hour in range(24):
                phase = gen.get_toroidal_phase(dow, hour)
                assert 0.0 <= phase < 1.0

    def test_toroidal_phase_first_state(self, gen):
        """First state (D0H0) should have phase = 0/256 = 0.0."""
        assert gen.get_toroidal_phase(0, 0) == 0.0

    def test_neighbors_count(self, gen):
        """get_neighbors should always return exactly 4 neighbors."""
        for r in range(TOROIDAL_N):
            for c in range(TOROIDAL_N):
                neighbors = gen.get_neighbors(r, c)
                assert len(neighbors) == 4

    def test_neighbors_toroidal_wrapping(self, gen):
        """Neighbors should wrap around the grid edges."""
        # Top-left corner: (0, 0)
        neighbors = gen.get_neighbors(0, 0)
        rows = [n[0] for n in neighbors]
        cols = [n[1] for n in neighbors]
        # Should include wrapped values
        assert (TOROIDAL_N - 1) in rows  # up wraps to bottom
        assert (TOROIDAL_N - 1) in cols  # left wraps to right

    def test_neighborhood_entropy_with_uniform_demand(self, gen):
        """Entropy should be 0 when all neighbors have same demand."""
        demand_map = {}
        for dow in range(7):
            for hour in range(24):
                demand_map[(dow, hour)] = 0.5  # uniform

        for dow in range(7):
            for hour in range(24):
                entropy = gen.get_neighborhood_entropy(dow, hour, demand_map)
                assert entropy == 0.0

    def test_neighborhood_entropy_with_varying_demand(self, gen):
        """Entropy should be > 0 when neighbors have different demands."""
        demand_map = {}
        for dow in range(7):
            for hour in range(24):
                demand_map[(dow, hour)] = float(dow * 24 + hour) / 168.0

        # At least some states should have non-zero entropy
        entropies = []
        for dow in range(7):
            for hour in range(24):
                entropies.append(gen.get_neighborhood_entropy(dow, hour, demand_map))
        assert any(e > 0 for e in entropies)

    def test_collision_frequency_non_negative(self, gen):
        """Collision frequency should always be >= 0."""
        for dow in range(7):
            for hour in range(24):
                freq = gen.get_collision_frequency(dow, hour)
                assert freq >= 0

    def test_grid_summary_keys(self, gen):
        """Grid summary should contain expected keys."""
        summary = gen.get_grid_summary()
        assert "grid_size" in summary
        assert "real_states_placed" in summary
        assert "phantom_states" in summary
        assert "total_collision_events" in summary
        assert "collision_rate" in summary

    def test_grid_summary_real_states(self, gen):
        """Grid summary should report 168 real states."""
        summary = gen.get_grid_summary()
        assert summary["real_states_placed"] == TEMPORAL_STATES

    def test_grid_summary_grid_size(self, gen):
        """Grid summary should report correct grid size."""
        summary = gen.get_grid_summary()
        assert summary["grid_size"] == TOROIDAL_GRID_SIZE

    def test_unique_positions_for_all_states(self, gen):
        """All 168 temporal states should map to unique grid positions."""
        positions = set()
        for dow in range(7):
            for hour in range(24):
                pos = gen.get_position(dow, hour)
                assert pos not in positions, f"Duplicate position {pos} for ({dow}, {hour})"
                positions.add(pos)

    def test_rule5_collision_occurs(self, gen):
        """The traversal should encounter at least some collisions (Rule 5)."""
        assert len(gen.collision_events) > 0

    def test_deterministic_output(self):
        """Two generators should produce identical grids."""
        gen1 = ToroidalTraversalGenerator(n=16)
        gen2 = ToroidalTraversalGenerator(n=16)
        np.testing.assert_array_equal(gen1.grid, gen2.grid)
        assert gen1.collision_events == gen2.collision_events
