"""
Hypercube Associative Memory

The 9-bit vertex address space (2^9 = 512 slots) provides a natural
content-addressable memory.  Patterns are stored as complex activation
profiles over all 512 neurons.  Retrieval is driven by Hopfield-style
energy minimisation on the binary hypercube.

Addressing uses a soft Boltzmann assignment:
    P(vertex v | feature f) ∝ exp( −β · ||coords[v] − f||² )
so a 9D feature vector in [0,1]^9 maps to a probability distribution
over the 512 vertices, peaking at the nearest vertex in Hamming space.
"""

import numpy as np
from .enneract import N_DIM, N_VERTICES, vertex_coords


class HypercubeMemory:
    """
    Content-addressable memory over the 512-vertex 9D hypercube.

    Stores up to `capacity` complex-valued activation patterns.
    Retrieval uses Hopfield attractor dynamics (real-valued projection).
    """

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._patterns: list[np.ndarray] = []
        self._labels:   list[str]        = []

        # Hopfield weight matrix — built lazily, invalidated on store()
        self._J: np.ndarray | None = None
        self._dirty = True

        self._coords = vertex_coords()   # (512, 9)

    # ------------------------------------------------------------------ #
    #  Storage                                                             #
    # ------------------------------------------------------------------ #

    def store(self, pattern: np.ndarray, label: str = None) -> int:
        """
        Store a (512,) complex pattern.  Returns slot index.
        Oldest pattern is evicted when capacity is reached.
        Pattern is normalised to unit energy before storage.
        """
        pattern = np.asarray(pattern, dtype=complex).ravel()
        if len(pattern) != N_VERTICES:
            raise ValueError(f"Pattern length must be {N_VERTICES}, got {len(pattern)}")

        energy = np.sqrt((np.abs(pattern) ** 2).sum())
        if energy > 1e-12:
            pattern = pattern / energy

        if len(self._patterns) >= self.capacity:
            self._patterns.pop(0)
            self._labels.pop(0)

        self._patterns.append(pattern.copy())
        self._labels.append(label or f"memory_{len(self._patterns)}")
        self._dirty = True
        return len(self._patterns) - 1

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def recall(
        self,
        query: np.ndarray,
        n_steps: int = 20,
        beta: float = 2.0,
    ) -> np.ndarray:
        """
        Converge to the nearest stored attractor using Hopfield dynamics.

        The update rule is:
            h[v]    = Σ_u J[v,u] · Re(state[u])
            state   ← tanh(β · h) · e^{i·angle(state)}
        Phase information from the query is preserved throughout.
        """
        if not self._patterns:
            return np.asarray(query, dtype=complex).copy()
        if self._dirty:
            self._build_hopfield()

        state = np.asarray(query, dtype=complex).copy()
        for _ in range(n_steps):
            h = self._J @ np.real(state)
            amp = np.tanh(beta * h)
            state = amp * np.exp(1j * np.angle(state))
        return state

    def nearest_pattern(self, query: np.ndarray) -> tuple:
        """
        Find the stored pattern with highest inner-product overlap |⟨p|q⟩|.
        Returns (index, label, overlap_score).
        """
        if not self._patterns:
            return None, None, 0.0
        q = np.asarray(query, dtype=complex)
        overlaps = np.array([float(np.abs(np.vdot(p, q))) for p in self._patterns])
        best = int(np.argmax(overlaps))
        return best, self._labels[best], float(overlaps[best])

    # ------------------------------------------------------------------ #
    #  Addressing                                                          #
    # ------------------------------------------------------------------ #

    def address_encode(self, features: np.ndarray, beta: float = 4.0) -> np.ndarray:
        """
        Soft-address encoding: map a 9D feature vector in [0,1]^9 to a
        (512,) probability distribution over hypercube vertices.

            P(v) ∝ exp( −β · ||coords[v] − features||² )

        The resulting distribution peaks at the vertex whose binary
        coordinate is closest in Euclidean (= Hamming) distance to `features`.
        """
        f = np.clip(np.asarray(features, dtype=np.float64).ravel()[:N_DIM], 0, 1)
        diff = self._coords.astype(np.float64) - f[None, :]
        dist2 = (diff ** 2).sum(axis=1)
        logits = -beta * dist2
        logits -= logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_stored(self) -> int:
        return len(self._patterns)

    def summary(self) -> dict:
        return {
            "n_stored": self.n_stored,
            "capacity": self.capacity,
            "labels":   self._labels.copy(),
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_hopfield(self) -> None:
        """Build Hebbian weight matrix  J = (1/N) Σ_μ ξ_μ ξ_μᵀ, J_ii = 0."""
        P = np.stack([np.real(p) for p in self._patterns], axis=0)  # (M, 512)
        self._J = (P.T @ P) / N_VERTICES
        np.fill_diagonal(self._J, 0.0)
        self._dirty = False
