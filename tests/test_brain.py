"""Tests for the EnneractBrain 9D Hypercube Neural Intelligence System."""

import numpy as np
import pytest
from scipy.special import comb as _comb

from qkc.brain import (
    EnneractBrain, Thought, NeuralField, HypercubeMemory, AxisAttention,
    N_DIM, N_VERTICES, N_EDGES, N_SHELLS, COGNITIVE_AXES, SHELL_ROLES,
    vertex_coords, shell_of, shell_slices, neighbor_matrix,
    edge_list, axis_neighbors, petrie_project,
    cmb_stats_to_stimulus,
)


# ========================================================================= #
#  Topology correctness                                                       #
# ========================================================================= #

class TestEnneractTopology:

    def test_vertex_count_and_shape(self):
        coords = vertex_coords()
        assert coords.shape == (512, 9)
        assert coords.dtype == np.float32

    def test_vertex_binary_values(self):
        coords = vertex_coords()
        assert set(np.unique(coords).tolist()) == {0.0, 1.0}

    def test_shell_sizes_match_binomial(self):
        shells = shell_of()
        for k in range(N_SHELLS):
            expected = int(_comb(9, k, exact=True))
            actual   = int((shells == k).sum())
            assert actual == expected, f"Shell k={k}: expected {expected}, got {actual}"

    def test_shell_sizes_sum_to_512(self):
        assert shell_of().shape == (N_VERTICES,)
        assert int((shell_of() >= 0).sum()) == N_VERTICES

    def test_edge_count(self):
        assert len(edge_list()) == N_EDGES

    def test_edges_undirected_and_sorted(self):
        edges = edge_list()
        assert np.all(edges[:, 0] < edges[:, 1]), "All edges should have u < v"

    def test_each_vertex_has_9_neighbors(self):
        nm = neighbor_matrix()
        assert nm.shape == (N_VERTICES, N_DIM)

    def test_neighbors_differ_by_one_bit(self):
        nm = neighbor_matrix()
        for v in range(0, N_VERTICES, 64):   # sample every 64th
            for nb in nm[v]:
                diff = bin(int(v) ^ int(nb)).count("1")
                assert diff == 1

    def test_axis_neighbors_are_bijections(self):
        for nb in axis_neighbors():
            assert len(nb) == N_VERTICES
            assert len(set(nb.tolist())) == N_VERTICES

    def test_petrie_projection_shape(self):
        x, y = petrie_project()
        assert x.shape == (N_VERTICES,)
        assert y.shape == (N_VERTICES,)

    def test_shell_slices_cover_all_vertices(self):
        all_idx = np.concatenate(shell_slices())
        assert set(all_idx.tolist()) == set(range(N_VERTICES))

    def test_constants(self):
        assert N_DIM     == 9
        assert N_VERTICES == 512
        assert N_EDGES    == 2304
        assert N_SHELLS   == 10
        assert len(COGNITIVE_AXES) == N_DIM
        assert len(SHELL_ROLES)    == N_SHELLS


# ========================================================================= #
#  NeuralField                                                                #
# ========================================================================= #

class TestNeuralField:

    def setup_method(self):
        self.f = NeuralField(rng=np.random.default_rng(0))

    def test_initial_state_shape_and_dtype(self):
        assert self.f.state.shape == (N_VERTICES,)
        assert np.iscomplexobj(self.f.state)

    def test_weight_shape(self):
        assert self.f.W.shape == (N_VERTICES, N_DIM)

    def test_propagation_preserves_shape(self):
        self.f.propagate(n_steps=3)
        assert self.f.state.shape == (N_VERTICES,)

    def test_energy_is_non_negative(self):
        assert self.f.energy() >= 0

    def test_shell_energies_sum_to_total(self):
        total    = self.f.energy()
        shell_e  = self.f.shell_energies()
        assert abs(total - shell_e.sum()) < 1e-8

    def test_shell_energies_length(self):
        assert len(self.f.shell_energies()) == N_SHELLS

    def test_axis_coherence_range(self):
        coh = self.f.axis_coherence()
        assert coh.shape == (N_DIM,)
        assert np.all(coh >= 0) and np.all(coh <= 1.0 + 1e-9)

    def test_hebbian_normalises_weights(self):
        self.f.propagate(2)
        self.f.hebbian_update(lr=0.5)
        norms = np.linalg.norm(self.f.W, axis=1)
        assert np.all(norms <= 1.0 + 1e-6), "Row norms should be ≤ 1"

    def test_resonate_preserves_amplitude_order(self):
        amp_before = np.abs(self.f.state).mean()
        self.f.resonate(n_cycles=2)
        amp_after  = np.abs(self.f.state).mean()
        # Resonance only adjusts phases — amplitude should not explode
        assert amp_after < amp_before * 5

    def test_set_and_get_shell(self):
        target = np.array([0.5 + 0j] * 9)
        self.f.set_shell(1, target)
        result = self.f.get_shell(1)
        np.testing.assert_allclose(np.real(result), np.real(target[:9]), atol=1e-12)


# ========================================================================= #
#  HypercubeMemory                                                            #
# ========================================================================= #

class TestHypercubeMemory:

    def setup_method(self):
        self.mem = HypercubeMemory(capacity=10)

    def test_store_increments_count(self):
        self.mem.store(np.ones(N_VERTICES, dtype=complex), "p1")
        assert self.mem.n_stored == 1

    def test_capacity_evicts_oldest(self):
        for i in range(15):
            self.mem.store(np.random.randn(N_VERTICES).astype(complex), f"p{i}")
        assert self.mem.n_stored == 10
        assert self.mem._labels[0] == "p5"  # first 5 evicted

    def test_nearest_finds_exact_match(self):
        p = np.random.default_rng(7).standard_normal(N_VERTICES).astype(complex)
        self.mem.store(p, "exact")
        _, label, overlap = self.mem.nearest_pattern(p)
        assert label   == "exact"
        assert overlap >  0.999

    def test_recall_returns_correct_shape(self):
        self.mem.store(np.random.randn(N_VERTICES).astype(complex))
        result = self.mem.recall(np.random.randn(N_VERTICES).astype(complex))
        assert result.shape == (N_VERTICES,)

    def test_address_encode_is_probability_distribution(self):
        f     = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        probs = self.mem.address_encode(f)
        assert probs.shape == (N_VERTICES,)
        assert abs(probs.sum() - 1.0) < 1e-8
        assert np.all(probs >= 0)

    def test_address_encode_peaks_at_correct_vertex(self):
        f     = np.ones(N_DIM, dtype=float)       # all-ones → vertex 511
        probs = self.mem.address_encode(f)
        assert int(np.argmax(probs)) == N_VERTICES - 1

    def test_empty_memory_nearest_returns_none(self):
        idx, label, overlap = self.mem.nearest_pattern(np.zeros(N_VERTICES))
        assert label is None


# ========================================================================= #
#  AxisAttention                                                              #
# ========================================================================= #

class TestAxisAttention:

    def setup_method(self):
        self.attn = AxisAttention(d_model=N_DIM, rng=np.random.default_rng(1))

    def test_forward_output_shape(self):
        X   = np.random.randn(N_VERTICES, N_DIM)
        out = self.attn.forward(X)
        assert out.shape == (N_VERTICES, N_DIM)

    def test_axis_importance_length(self):
        X = np.random.randn(N_VERTICES, N_DIM)
        s = self.attn.axis_importance(X)
        assert s.shape == (N_DIM,)
        assert np.all(s >= 0)

    def test_axis_importance_named_keys(self):
        X   = np.random.randn(N_VERTICES, N_DIM)
        d   = self.attn.axis_importance_named(X)
        assert set(d.keys()) == set(COGNITIVE_AXES)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            self.attn.forward(np.zeros((100, N_DIM)))


# ========================================================================= #
#  EnneractBrain integration                                                  #
# ========================================================================= #

class TestEnneractBrain:

    def setup_method(self):
        self.brain = EnneractBrain(seed=0, propagation_steps=3, resonance_cycles=1)

    def test_think_returns_thought(self):
        t = self.brain.think(np.random.rand(N_DIM))
        assert isinstance(t, Thought)

    def test_thought_fields_present(self):
        t = self.brain.think(np.zeros(N_DIM))
        assert t.raw_state.shape      == (N_VERTICES,)
        assert t.shell_energies.shape == (N_SHELLS,)
        assert t.axis_coherence.shape == (N_DIM,)

    def test_dominant_axis_is_valid(self):
        t = self.brain.think(np.ones(N_DIM) * 0.5)
        assert t.dominant_axis in COGNITIVE_AXES

    def test_dominant_shell_is_in_range(self):
        t = self.brain.think(np.random.rand(N_DIM))
        assert 0 <= t.dominant_shell < N_SHELLS

    def test_step_increments(self):
        for _ in range(5):
            self.brain.think(np.random.rand(N_DIM))
        assert self.brain._step == 5

    def test_short_stimulus_padded(self):
        t = self.brain.think([0.5, 0.3])
        assert t is not None

    def test_remember_returns_index(self):
        p   = np.random.randn(N_VERTICES).astype(complex)
        idx = self.brain.remember(p, "test")
        assert isinstance(idx, int) and idx >= 0

    def test_recall_shape(self):
        p = np.random.randn(N_VERTICES).astype(complex)
        self.brain.remember(p, "x")
        r = self.brain.recall(p)
        assert r.shape == (N_VERTICES,)

    def test_remember_9d_input(self):
        idx = self.brain.remember(np.random.rand(N_DIM), "9d_pattern")
        assert isinstance(idx, int)

    def test_encode_no_side_effects(self):
        s0 = self.brain.field.state.copy()
        self.brain.encode(np.random.rand(N_DIM))
        np.testing.assert_array_equal(self.brain.field.state, s0)

    def test_introspect_keys(self):
        self.brain.think(np.random.rand(N_DIM))
        info = self.brain.introspect()
        for key in ("step", "total_energy", "dominant_shell", "dominant_axis",
                    "memory", "axis_importance"):
            assert key in info

    def test_reset_state_drops_energy(self):
        for _ in range(5):
            self.brain.think(np.ones(N_DIM))
        e_before = self.brain.field.energy()
        self.brain.reset_state()
        e_after  = self.brain.field.energy()
        assert e_after < e_before


# ========================================================================= #
#  Planck interface                                                           #
# ========================================================================= #

class TestPlanckInterface:

    def test_stimulus_shape_and_range(self):
        stats  = {"ks_statistic": 0.3, "ks_pvalue": 0.02, "z_score": -3.1, "z_pvalue": 0.001}
        patch  = {"mean": -200.0, "std": 80.0, "skewness": -0.5, "kurtosis": 1.2}
        stim   = cmb_stats_to_stimulus(stats, patch)
        assert stim.shape == (N_DIM,)
        assert np.all(stim >= 0) and np.all(stim <= 1)

    def test_stimulus_with_empty_patch_stats(self):
        stats = {"ks_statistic": 0.1, "ks_pvalue": 0.5, "z_score": 0.0, "z_pvalue": 0.5}
        stim  = cmb_stats_to_stimulus(stats)
        assert stim.shape == (N_DIM,)

    def test_high_significance_drives_high_stimulus(self):
        significant = {"ks_statistic": 0.9, "ks_pvalue": 0.001, "z_score": 4.0, "z_pvalue": 0.0001}
        ordinary    = {"ks_statistic": 0.1, "ks_pvalue": 0.9,   "z_score": 0.1, "z_pvalue": 0.9}
        stim_sig = cmb_stats_to_stimulus(significant)
        stim_ord = cmb_stats_to_stimulus(ordinary)
        # memory axis (1-ks_p) should be higher for significant
        assert stim_sig[1] > stim_ord[1]
