"""
EnneractBrain — 9D Hypercube Neural Intelligence System

Architecture is derived directly from the Petrie projection of the 9D
hypercube (enneract), as encoded in the neurological map:

    • 512 neurons across 10 Hamming shells (k = 0 … 9)
    • 2 304 synaptic connections following hypercube adjacency
    • 9-head axis attention (one head per cognitive dimension)
    • Complex-valued neural states  z = r · e^{iφ}
    • Hopfield associative memory over the 512-vertex address space
    • Hebbian synaptic plasticity + shell-wise normalisation

The 9 cognitive axes map to:
    0  perception     — sensory signal encoding
    1  memory         — episodic & semantic memory
    2  reasoning      — logical inference chains
    3  emotion        — affective weighting
    4  language       — symbolic representation
    5  spatial        — geometric / relational structure
    6  temporal       — sequence & causality
    7  social         — theory-of-mind
    8  metacognition  — self-monitoring & control

Usage
-----
    brain = EnneractBrain()
    thought = brain.think([0.8, 0.2, 0.9, 0.1, 0.6, 0.5, 0.4, 0.3, 0.7])
    brain.remember(thought.raw_state, label="hypothesis_A")
    recalled = brain.recall(thought.raw_state)
    print(brain.introspect())
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .enneract import (
    N_DIM, N_VERTICES, N_SHELLS,
    COGNITIVE_AXES, SHELL_ROLES,
    vertex_coords, shell_slices,
)
from .field import NeuralField
from .memory import HypercubeMemory
from .attention import AxisAttention


@dataclass
class Thought:
    """Structured output produced by one call to EnneractBrain.think()."""
    raw_state:      np.ndarray   # (512,) complex — full field snapshot
    shell_energies: np.ndarray   # (10,)  float   — energy per shell
    axis_coherence: np.ndarray   # (9,)   float   — phase coherence per axis
    dominant_shell: int          # shell index with highest energy
    dominant_axis:  str          # cognitive axis with highest coherence
    memory_match:   str | None   # label of nearest stored memory
    memory_overlap: float        # similarity score in [0, 1]


class EnneractBrain:
    """
    High-dimensional AI brain modelled on the 9D hypercube topology.

    Parameters
    ----------
    memory_capacity   : int   — max patterns stored in associative memory
    propagation_steps : int   — graph message-passing steps per think()
    resonance_cycles  : int   — phase-synchronisation passes per think()
    learning_rate     : float — Hebbian synaptic update rate
    seed              : int   — RNG seed for reproducibility
    """

    def __init__(
        self,
        memory_capacity:   int   = 128,
        propagation_steps: int   = 5,
        resonance_cycles:  int   = 2,
        learning_rate:     float = 0.005,
        seed:              int   = 42,
    ):
        self.propagation_steps = propagation_steps
        self.resonance_cycles  = resonance_cycles
        self.learning_rate     = learning_rate

        self.rng = np.random.default_rng(seed)

        self.field     = NeuralField(rng=self.rng)
        self.memory    = HypercubeMemory(capacity=memory_capacity)
        self.attention = AxisAttention(d_model=N_DIM, rng=self.rng)

        self._shell_idx = shell_slices()
        self._coords    = vertex_coords()   # (512, 9)
        self._history:  list[Thought] = []
        self._step:     int = 0

    # ------------------------------------------------------------------ #
    #  Primary cognitive interface                                         #
    # ------------------------------------------------------------------ #

    def think(
        self,
        stimulus: np.ndarray,
        n_propagation: int | None = None,
    ) -> Thought:
        """
        Process a 9-dimensional stimulus through the hypercube brain.

        Pipeline
        --------
        1. Encode stimulus into shell-0 (core) and shell-1 (primary axes).
        2. Derive (512, 9) feature matrix and apply 9-head axis attention.
        3. Propagate activation through the hypercube graph.
        4. Synchronise shell resonance (phase-locking).
        5. Run one Hebbian synaptic update.
        6. Package and return a Thought.

        Parameters
        ----------
        stimulus     : (9,) array-like, values in [0, 1]
        n_propagation: override default propagation step count
        """
        stimulus = _coerce_stimulus(stimulus)

        self._encode_stimulus(stimulus)
        self._apply_axis_attention()

        steps = n_propagation if n_propagation is not None else self.propagation_steps
        self.field.propagate(n_steps=steps)
        self.field.resonate(n_cycles=self.resonance_cycles)
        self.field.hebbian_update(lr=self.learning_rate)

        thought = self._package_thought()
        self._history.append(thought)
        self._step += 1
        return thought

    def remember(self, pattern: np.ndarray, label: str = None) -> int:
        """
        Store a pattern in associative memory.
        Accepts either a (512,) full activation vector or a (9,) feature
        vector (which is soft-addressed into a 512-vector first).
        Returns the slot index.
        """
        pat = np.asarray(pattern).ravel()
        if len(pat) == N_DIM:
            pat = self.memory.address_encode(pat)
        return self.memory.store(pat, label=label)

    def recall(self, query: np.ndarray, n_steps: int = 20) -> np.ndarray:
        """
        Retrieve the stored memory closest to query via Hopfield dynamics.
        Accepts (512,) or (9,) input (same auto-encoding as remember()).
        Returns (512,) complex convergence state.
        """
        q = np.asarray(query).ravel()
        if len(q) == N_DIM:
            q = self.memory.address_encode(q)
        return self.memory.recall(q, n_steps=n_steps)

    def encode(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Pure read-only encoding: map a 9D stimulus → (512,) activation
        without modifying the brain's internal state or learned weights.
        """
        probe = NeuralField(rng=np.random.default_rng(self._step))
        probe.W    = self.field.W.copy()
        probe.bias = self.field.bias.copy()

        stim = _coerce_stimulus(stimulus)
        for i, v_idx in enumerate(self._shell_idx[1]):
            if i < N_DIM:
                probe.state[v_idx] = complex(float(stim[i]))
        probe.propagate(n_steps=self.propagation_steps)
        return probe.state.copy()

    def reset_state(self) -> None:
        """Reset neural activations to near-zero, preserving learned weights."""
        amp   = self.rng.rayleigh(scale=0.01, size=N_VERTICES)
        phase = self.rng.uniform(0, 2 * np.pi, size=N_VERTICES)
        self.field.state = (amp * np.exp(1j * phase)).astype(complex)

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def introspect(self) -> dict:
        """Return a comprehensive snapshot of brain state and configuration."""
        fs = self.field.summary()
        feat = self._state_to_features()
        axis_imp = self.attention.axis_importance_named(feat)
        return {
            **fs,
            "step":                  self._step,
            "memory":                self.memory.summary(),
            "shell_roles":           dict(enumerate(SHELL_ROLES)),
            "axis_importance":       axis_imp,
            "total_synaptic_weight": float(np.abs(self.field.W).sum()),
            "history_length":        len(self._history),
        }

    # ------------------------------------------------------------------ #
    #  Internal pipeline steps                                             #
    # ------------------------------------------------------------------ #

    def _encode_stimulus(self, stimulus: np.ndarray) -> None:
        """
        Inject stimulus into:
          - Shell 0 (1 core neuron)  : receives the mean stimulus value.
          - Shell 1 (9 axis neurons) : one-to-one mapping with a temporal phase.
        """
        core_idx = self._shell_idx[0]
        self.field.state[core_idx] = complex(float(stimulus.mean()))

        t_phase = 2 * np.pi * self._step / 100.0
        for i, v_idx in enumerate(self._shell_idx[1]):
            if i < N_DIM:
                self.field.state[v_idx] = stimulus[i] * np.exp(1j * t_phase)

    def _apply_axis_attention(self) -> None:
        """
        Project field state to a (512, 9) real feature matrix, run 9-head
        axis attention, and add the attended updates as a gated perturbation.
        """
        X        = self._state_to_features()     # (512, 9)
        attended = self.attention.forward(X)     # (512, 9)
        gate     = 0.3
        feat_mag = np.abs(attended).sum(axis=1) / N_DIM  # (512,)
        phase    = np.angle(self.field.state)             # (512,)
        self.field.state += gate * feat_mag * np.exp(1j * phase)

    def _state_to_features(self) -> np.ndarray:
        """
        Convert 512 complex states → (512, 9) real feature matrix.
        Feature d of neuron v measures how much v's activation projects
        onto cognitive axis d:
            feat[v, d] = |state[v]| · cos(angle(state[v]) − θ_d)
        where θ_d = 2π·d/9 are the Petrie projection angles.
        """
        amp   = np.abs(self.field.state)          # (512,)
        phase = np.angle(self.field.state)         # (512,)
        theta = np.linspace(0, 2 * np.pi, N_DIM, endpoint=False)
        return (amp[:, None] * np.cos(phase[:, None] - theta[None, :])).astype(np.float64)

    def _package_thought(self) -> Thought:
        energies  = self.field.shell_energies()
        coherence = self.field.axis_coherence()
        _, label, overlap = self.memory.nearest_pattern(self.field.state)
        return Thought(
            raw_state      = self.field.state.copy(),
            shell_energies = energies,
            axis_coherence = coherence,
            dominant_shell = int(np.argmax(energies)),
            dominant_axis  = COGNITIVE_AXES[int(np.argmax(coherence))],
            memory_match   = label,
            memory_overlap = float(overlap),
        )


# -------------------------------------------------------------------------- #
#  Helpers                                                                    #
# -------------------------------------------------------------------------- #

def _coerce_stimulus(stimulus) -> np.ndarray:
    """Ensure stimulus is a (9,) float64 array clipped to [0, 1]."""
    s = np.asarray(stimulus, dtype=np.float64).ravel()
    if len(s) < N_DIM:
        s = np.pad(s, (0, N_DIM - len(s)))
    return np.clip(s[:N_DIM], 0.0, 1.0)
