"""
Complex-valued neural field over the 9D hypercube.

Each of the 512 neurons holds a complex amplitude  z = r · e^{iφ}:
  r  (magnitude) → firing rate / activation strength
  φ  (phase)     → temporal coordination / spike timing

Synaptic weights W[v, d] scale the signal arriving into neuron v
along axis d (i.e. from its bit-d neighbor).

Dynamics follow axis-aware message passing:
    h[v]   =  bias[v]  +  Σ_d  W[v,d] · σ(state[nb_d(v)])
    state  ←  state  +  dt · (h − state)

where σ is a complex activation that applies tanh to the magnitude
while preserving the phase angle.
"""

import numpy as np
from .enneract import (
    N_DIM, N_VERTICES, N_SHELLS, COGNITIVE_AXES,
    shell_of, shell_slices, axis_neighbors, vertex_coords,
)


class NeuralField:
    """
    Complex-valued neural field: 512 neurons on the 9D hypercube graph.

    Attributes
    ----------
    state : (512,) complex128
    W     : (512, 9)  float64  — axis-directional synaptic weights
    bias  : (512,)   float64  — per-neuron biases
    """

    def __init__(self, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(42)

        amp   = rng.rayleigh(scale=0.1, size=N_VERTICES)
        phase = rng.uniform(0, 2 * np.pi, size=N_VERTICES)
        self.state: np.ndarray = (amp * np.exp(1j * phase)).astype(complex)

        # Axis-directional synaptic weights
        self.W: np.ndarray = rng.normal(0, 0.1, size=(N_VERTICES, N_DIM))

        # Per-neuron biases
        self.bias: np.ndarray = rng.normal(0, 0.01, size=N_VERTICES)

        self._shells   = shell_of()
        self._shell_idx = shell_slices()
        self._ax_nb    = axis_neighbors()
        self._coords   = vertex_coords()

    # ------------------------------------------------------------------ #
    #  Core dynamics                                                       #
    # ------------------------------------------------------------------ #

    def propagate(self, n_steps: int = 1, dt: float = 0.1) -> "NeuralField":
        """
        Run n_steps of axis-aware message-passing.

        Each neuron receives weighted signals from its 9 bit-flip neighbors:
            h[v] = bias[v] + Σ_d  W[v,d] · σ(state[nb_d(v)])
            state ← state + dt · (h − state)
        Shell-wise amplitude normalisation prevents divergence.
        """
        ax_nb = self._ax_nb
        W = self.W
        bias = self.bias

        for _ in range(n_steps):
            h = bias.astype(complex)
            for d in range(N_DIM):
                nb = ax_nb[d]
                h += W[:, d] * _cactivate(self.state[nb])
            self.state += dt * (h - self.state)
            self._normalize_shells()
        return self

    def resonate(self, n_cycles: int = 3, omega: float = 0.5) -> "NeuralField":
        """
        Drive shell-synchronized phase oscillations.

        Nudges each shell's neurons toward their mean phase, creating
        resonant standing waves analogous to gamma-band oscillations.
        """
        for cycle in range(n_cycles):
            for idx in self._shell_idx:
                if len(idx) == 0:
                    continue
                s = self.state[idx]
                mean_phase = np.angle(s.mean())
                target = mean_phase + omega * cycle
                correction = np.exp(1j * 0.1 * (target - np.angle(s)))
                self.state[idx] *= correction
        return self

    def hebbian_update(self, lr: float = 0.001) -> "NeuralField":
        """
        Unsupervised Hebbian synaptic update along each axis:
            ΔW[v, d]  ∝  Re( conj(state[v]) · state[nb_d(v)] )
        Row norms are capped at 1 to prevent weight explosion.
        """
        for d in range(N_DIM):
            nb = self._ax_nb[d]
            dw = np.real(np.conj(self.state) * self.state[nb])
            self.W[:, d] += lr * dw
        # Per-neuron weight normalisation
        norms = np.linalg.norm(self.W, axis=1, keepdims=True).clip(min=1.0)
        self.W /= norms
        return self

    # ------------------------------------------------------------------ #
    #  Shell-level accessors                                               #
    # ------------------------------------------------------------------ #

    def set_shell(self, k: int, values: np.ndarray) -> "NeuralField":
        """Clamp-inject complex values into shell k."""
        idx = self._shell_idx[k]
        n = min(len(idx), len(values))
        self.state[idx[:n]] = values[:n]
        return self

    def get_shell(self, k: int) -> np.ndarray:
        """Return a copy of the complex state for shell k."""
        return self.state[self._shell_idx[k]].copy()

    # ------------------------------------------------------------------ #
    #  Observables                                                         #
    # ------------------------------------------------------------------ #

    def energy(self) -> float:
        """Total field energy  Σ |state[v]|²."""
        return float(np.sum(np.abs(self.state) ** 2))

    def shell_energies(self) -> np.ndarray:
        """(10,) float — energy contribution from each shell."""
        return np.array(
            [float(np.sum(np.abs(self.state[idx]) ** 2)) for idx in self._shell_idx],
            dtype=np.float64,
        )

    def axis_coherence(self) -> np.ndarray:
        """
        (9,) float in [0, 1] — phase-locking between each pair of
        axis-connected neurons (Kuramoto order parameter per axis).
        """
        phases = np.angle(self.state)
        coherence = np.empty(N_DIM, dtype=np.float64)
        for d, nb in enumerate(self._ax_nb):
            delta = phases - phases[nb]
            coherence[d] = float(np.abs(np.mean(np.exp(1j * delta))))
        return coherence

    def activation_map(self) -> np.ndarray:
        """(512,) float — amplitude |state[v]| for each neuron."""
        return np.abs(self.state)

    def summary(self) -> dict:
        energies  = self.shell_energies()
        coherence = self.axis_coherence()
        return {
            "total_energy":    float(energies.sum()),
            "shell_energies":  energies.tolist(),
            "axis_coherence":  {COGNITIVE_AXES[d]: float(coherence[d]) for d in range(N_DIM)},
            "dominant_shell":  int(np.argmax(energies)),
            "dominant_axis":   COGNITIVE_AXES[int(np.argmax(coherence))],
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _normalize_shells(self) -> None:
        """Scale down any shell whose mean amplitude exceeds 1."""
        for idx in self._shell_idx:
            if len(idx) == 0:
                continue
            mean_amp = np.abs(self.state[idx]).mean() + 1e-12
            if mean_amp > 1.0:
                self.state[idx] /= mean_amp


# -------------------------------------------------------------------------- #
#  Module-level activation function                                           #
# -------------------------------------------------------------------------- #

def _cactivate(z: np.ndarray) -> np.ndarray:
    """
    Complex activation: tanh on magnitude, phase preserved.
        σ(z) = tanh(|z|) · z / |z|
    """
    r = np.abs(z)
    safe_r = np.where(r > 1e-12, r, 1.0)
    return z * (np.tanh(r) / safe_r)
