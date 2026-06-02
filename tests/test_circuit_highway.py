"""Tests for the CircuitHighway closed-loop entropy minimisation."""

import numpy as np
import pytest
import healpy as hp

from qkc.circuit_highway import (
    CircuitHighway,
    CircuitNode,
    CircuitEdge,
    shannon_entropy,
    total_variation,
    _two_opt,
    _tv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_map(nside: int = 64, seed: int = 42) -> np.ndarray:
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(npix).astype(np.float32) * 100.0  # μK scale


# ---------------------------------------------------------------------------
# Unit tests — entropy helpers
# ---------------------------------------------------------------------------

class TestEntropyHelpers:
    def test_shannon_entropy_uniform(self):
        # Uniform distribution has maximum entropy
        rng = np.random.default_rng(0)
        uniform = rng.uniform(-300, 300, 1000)
        normal = rng.standard_normal(1000) * 50
        h_uniform = shannon_entropy(uniform)
        h_normal = shannon_entropy(normal)
        assert h_uniform > h_normal

    def test_shannon_entropy_constant(self):
        # Single value — near-zero entropy
        h = shannon_entropy(np.full(100, 5.0))
        assert h < 0.1

    def test_shannon_entropy_short(self):
        assert shannon_entropy(np.array([1.0])) == 0.0

    def test_total_variation_monotone(self):
        # Monotone increasing sequence: TV = |last - first|
        t = np.arange(10.0)
        # closed: TV = Σ|t_{i+1}-t_i| + |t_0 - t_{n-1}| = 9 + 9 = 18
        assert np.isclose(total_variation(t), 18.0)

    def test_total_variation_constant(self):
        assert total_variation(np.ones(50)) == 0.0

    def test_total_variation_closed(self):
        # Alternating ±1 — maximum variation
        t = np.array([1.0, -1.0, 1.0, -1.0])
        assert np.isclose(total_variation(t), 8.0)


# ---------------------------------------------------------------------------
# Unit tests — CircuitHighway construction
# ---------------------------------------------------------------------------

class TestCircuitHighwayBuild:
    def setup_method(self):
        self.hmap = _make_synthetic_map(nside=64)
        self.ch = CircuitHighway(n_rings=3, n_spokes=6, radius_deg=8.0)
        self.ch.build_from_map(self.hmap)

    def test_node_count(self):
        # 1 hub + n_rings * n_spokes
        expected = 1 + 3 * 6
        assert len(self.ch.nodes) == expected

    def test_ring_assignments(self):
        rings = [n.ring for n in self.ch.nodes]
        assert rings[0] == 0
        assert set(rings) == {0, 1, 2, 3}

    def test_edge_kinds(self):
        kinds = {e.kind for e in self.ch.edges}
        assert kinds == {"ring", "spoke"}

    def test_edge_weights_nonnegative(self):
        assert all(e.weight >= 0.0 for e in self.ch.edges)

    def test_adjacency_symmetric(self):
        for u, nbrs in self.ch._adj.items():
            for v in nbrs:
                assert u in self.ch._adj[v], f"Adjacency not symmetric: {u} -> {v}"

    def test_ring_entropy_nonnegative(self):
        for r in range(self.ch.n_rings + 1):
            assert self.ch.ring_entropy(r) >= 0.0

    def test_circuit_entropy_sum(self):
        total = self.ch.circuit_entropy()
        ring_sum = sum(self.ch.ring_entropy(r) for r in range(self.ch.n_rings + 1))
        assert np.isclose(total, ring_sum)


# ---------------------------------------------------------------------------
# Unit tests — minimum-entropy loop
# ---------------------------------------------------------------------------

class TestMinimumEntropyLoop:
    def setup_method(self):
        hmap = _make_synthetic_map(nside=64, seed=7)
        self.ch = CircuitHighway(n_rings=3, n_spokes=6, radius_deg=8.0)
        self.ch.build_from_map(hmap)

    def test_loop_covers_all_nodes(self):
        loop = self.ch.find_minimum_entropy_loop()
        assert sorted(loop) == list(range(len(self.ch.nodes)))

    def test_loop_is_closed(self):
        # No repeated indices
        loop = self.ch._loop
        assert len(loop) == len(set(loop))

    def test_loop_entropy_stored(self):
        self.ch.find_minimum_entropy_loop()
        assert np.isfinite(self.ch.loop_entropy)
        assert self.ch.loop_entropy >= 0.0

    def test_loop_entropy_matches_tv(self):
        loop = self.ch.find_minimum_entropy_loop()
        tv = self.ch.loop_total_variation(loop)
        assert np.isclose(tv, self.ch.loop_entropy)

    def test_2opt_improves_or_maintains(self):
        # Run greedy only, then 2-opt; result must be ≤ greedy cost.
        temps = np.array([n.temperature for n in self.ch.nodes])
        n = len(temps)
        rng = np.random.default_rng(99)
        greedy = list(rng.permutation(n))
        greedy_cost = _tv(greedy, temps)
        improved = _two_opt(greedy, temps)
        assert _tv(improved, temps) <= greedy_cost + 1e-9

    def test_entropy_reduction_ratio_lt_1(self):
        self.ch.find_minimum_entropy_loop()
        ratio = self.ch.entropy_reduction_ratio()
        # Optimised path should be more ordered than random
        assert ratio < 1.0


# ---------------------------------------------------------------------------
# Unit tests — entropy bound and summary
# ---------------------------------------------------------------------------

class TestEntropyBoundAndSummary:
    def test_planck_bound_positive(self):
        bound = CircuitHighway.planck_entropy_bound(10.0)
        assert bound > 0.0

    def test_planck_bound_increases_with_radius(self):
        b5 = CircuitHighway.planck_entropy_bound(5.0)
        b10 = CircuitHighway.planck_entropy_bound(10.0)
        assert b10 > b5

    def test_summary_keys(self):
        hmap = _make_synthetic_map(nside=64, seed=3)
        ch = CircuitHighway(n_rings=2, n_spokes=4).build_from_map(hmap)
        s = ch.summary()
        for key in [
            "n_nodes", "n_edges", "n_rings", "n_spokes",
            "circuit_entropy_nats", "loop_entropy_production_uK",
            "entropy_reduction_ratio", "ring_entropies_nats",
            "planck_bound_nats",
        ]:
            assert key in s, f"Missing key: {key}"

    def test_summary_node_edge_counts(self):
        hmap = _make_synthetic_map(nside=64)
        ch = CircuitHighway(n_rings=4, n_spokes=8).build_from_map(hmap)
        s = ch.summary()
        assert s["n_nodes"] == 1 + 4 * 8
        assert s["n_edges"] > 0
