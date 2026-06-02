"""
9-Head Axis Attention over the 9D hypercube.

Each attention head corresponds to one cognitive axis (dimension) of the
enneract.  For axis d and vertex v, the head scores how much information
should flow from v's d-neighbour into v:

    a[v, d]  = tanh( (Q[v] · K[nb_d(v)]) / sqrt(d_model) )
    out[v,d] = a[v, d] · V[nb_d(v)]

All 9 head outputs are concatenated and projected to d_model dimensions.

The `axis_importance` method exposes which cognitive axes are currently
routing the most information — a form of introspective attention audit.
"""

import numpy as np
from .enneract import N_DIM, N_VERTICES, axis_neighbors, COGNITIVE_AXES


class AxisAttention:
    """
    9-head axis attention, one head per cognitive dimension.

    Parameters
    ----------
    d_model : int
        Per-neuron feature dimension.  Must match the feature vectors
        passed to forward().  Defaults to N_DIM = 9.
    """

    def __init__(self, d_model: int = N_DIM, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(42)

        self.d_model = d_model
        scale = 1.0 / np.sqrt(d_model)

        # Query, key, value projection matrices per head: (N_DIM, d_model, d_model)
        self.Wq = rng.normal(0, scale, size=(N_DIM, d_model, d_model))
        self.Wk = rng.normal(0, scale, size=(N_DIM, d_model, d_model))
        self.Wv = rng.normal(0, scale, size=(N_DIM, d_model, d_model))

        # Output projection: concatenated heads → d_model
        self.Wo = rng.normal(0, scale, size=(N_DIM * d_model, d_model))

        self._ax_nb = axis_neighbors()   # tuple of 9 (512,) int32 arrays

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply 9-head axis attention.

        Parameters
        ----------
        X : (512, d_model) float64

        Returns
        -------
        out : (512, d_model) float64 — attended feature matrix
        """
        if X.shape != (N_VERTICES, self.d_model):
            raise ValueError(
                f"Expected ({N_VERTICES}, {self.d_model}), got {X.shape}"
            )

        head_outputs = []
        sq = np.sqrt(self.d_model)

        for d in range(N_DIM):
            nb   = self._ax_nb[d]
            Q    = X          @ self.Wq[d]    # (512, d_model)
            K    = X[nb]      @ self.Wk[d]    # (512, d_model)
            V    = X[nb]      @ self.Wv[d]    # (512, d_model)

            # Scaled dot-product attention score (one scalar per neuron per head)
            score  = np.sum(Q * K, axis=1, keepdims=True) / sq  # (512, 1)
            attn   = np.tanh(score)                               # bounded

            head_outputs.append(attn * V)   # (512, d_model)

        concat = np.concatenate(head_outputs, axis=1)   # (512, 9*d_model)
        return concat @ self.Wo                          # (512, d_model)

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def axis_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Return (9,) float — mean |attention score| per axis.
        Higher value ⇒ that cognitive axis is more actively routing
        information in the current field state.
        """
        if X.shape[1] != self.d_model:
            raise ValueError(f"Feature width mismatch: expected {self.d_model}, got {X.shape[1]}")

        sq = np.sqrt(self.d_model)
        scores = np.empty(N_DIM, dtype=np.float64)
        for d in range(N_DIM):
            nb   = self._ax_nb[d]
            Q    = X     @ self.Wq[d]
            K    = X[nb] @ self.Wk[d]
            scores[d] = float(np.abs(np.sum(Q * K, axis=1)).mean()) / sq
        return scores

    def axis_importance_named(self, X: np.ndarray) -> dict:
        """Return {axis_name: importance} dict."""
        scores = self.axis_importance(X)
        return {COGNITIVE_AXES[d]: float(scores[d]) for d in range(N_DIM)}
