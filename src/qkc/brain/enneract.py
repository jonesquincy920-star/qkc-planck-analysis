"""
9D Hypercube (Enneract) neurological topology — Petrie Projection.

Combinatorial facts:
  Vertices  : 2^9 = 512        → neurons
  Edges     : 9·2^8 = 2304     → synapses
  Faces     : 36·2^7 = 4608
  Cells     : 84·2^6 = 5376
  Shells    : 10  (k = 0 … 9, Hamming weight from the origin vertex)
  Shell k   : C(9,k) neurons  → [1,9,36,84,126,126,84,36,9,1]

The 9 axes map to distinct cognitive domains; the 10 Hamming shells form
a hierarchy from core integration (k=0) to unified awareness (k=9).
"""

import numpy as np
from functools import lru_cache

N_DIM = 9
N_VERTICES = 1 << N_DIM              # 512
N_EDGES = N_DIM * (1 << (N_DIM - 1)) # 2304
N_SHELLS = N_DIM + 1                  # 10

COGNITIVE_AXES = (
    "perception",
    "memory",
    "reasoning",
    "emotion",
    "language",
    "spatial",
    "temporal",
    "social",
    "metacognition",
)

SHELL_ROLES = (
    "core_integration",      # k=0  —   1 neuron
    "primary_axes",          # k=1  —   9 neurons
    "feature_binding",       # k=2  —  36 neurons
    "association",           # k=3  —  84 neurons
    "working_memory",        # k=4  — 126 neurons
    "abstract_reasoning",    # k=5  — 126 neurons
    "metacognition_layer",   # k=6  —  84 neurons
    "self_model",            # k=7  —  36 neurons
    "consciousness_edge",    # k=8  —   9 neurons
    "unified_awareness",     # k=9  —   1 neuron
)


@lru_cache(maxsize=1)
def vertex_coords() -> np.ndarray:
    """Return (512, 9) float32 binary coordinates of all hypercube vertices."""
    idx = np.arange(N_VERTICES, dtype=np.int32)
    return ((idx[:, None] >> np.arange(N_DIM - 1, -1, -1)) & 1).astype(np.float32)


@lru_cache(maxsize=1)
def shell_of() -> np.ndarray:
    """Return (512,) int8 — Hamming weight (shell index) for each vertex."""
    return vertex_coords().sum(axis=1).astype(np.int8)


@lru_cache(maxsize=1)
def shell_slices() -> tuple:
    """Return tuple of 10 int32 arrays, vertex indices per shell."""
    s = shell_of()
    return tuple(np.where(s == k)[0].astype(np.int32) for k in range(N_SHELLS))


@lru_cache(maxsize=1)
def neighbor_matrix() -> np.ndarray:
    """
    Return (512, 9) int32 — for vertex v and axis d,
    neighbor_matrix()[v, d] is the vertex reached by flipping bit d.
    """
    v = np.arange(N_VERTICES, dtype=np.int32)
    d = np.arange(N_DIM, dtype=np.int32)
    return (v[:, None] ^ (1 << d[None, :])).astype(np.int32)


@lru_cache(maxsize=1)
def edge_list() -> np.ndarray:
    """Return (2304, 2) int32 — undirected edges [u, v] with u < v."""
    nm = neighbor_matrix()
    rows, cols = np.where(nm > np.arange(N_VERTICES, dtype=np.int32)[:, None])
    return np.stack([rows.astype(np.int32), nm[rows, cols].astype(np.int32)], axis=1)


@lru_cache(maxsize=1)
def axis_neighbors() -> tuple:
    """
    Return tuple of 9 int32 arrays, each of length 512.
    axis_neighbors()[d][v] is the neighbor of v along axis d.
    """
    nm = neighbor_matrix()
    return tuple(nm[:, d].astype(np.int32) for d in range(N_DIM))


def petrie_project(coords: np.ndarray = None) -> tuple:
    """
    Project N_DIM-D coordinates to 2D via Petrie polygon angles.
    Returns (x, y) arrays of shape (n_vertices,).
    """
    if coords is None:
        coords = vertex_coords()
    theta = np.linspace(0, 2 * np.pi, N_DIM, endpoint=False, dtype=np.float32)
    x = coords @ np.cos(theta)
    y = coords @ np.sin(theta)
    return x, y


def shell_vertex_counts() -> np.ndarray:
    """Return (10,) int array — number of vertices per shell C(9,k)."""
    return np.array([len(idx) for idx in shell_slices()], dtype=np.int32)
