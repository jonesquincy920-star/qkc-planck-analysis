"""
qkc.brain — EnneractBrain: 9D Hypercube Neural Intelligence

Neurological map: 9D Hypercube (Enneract), Petrie Projection
    Vertices  : 512   → neurons
    Edges     : 2304  → synaptic connections
    Shells    : 10    → cognitive hierarchy (Hamming weight 0..9)
    Axes      : 9     → cognitive domains

Quick start
-----------
    from qkc.brain import EnneractBrain
    brain = EnneractBrain()
    thought = brain.think([0.9, 0.1, 0.8, 0.2, 0.7, 0.5, 0.5, 0.3, 0.6])
    print(brain.introspect())
"""

from .brain import EnneractBrain, Thought
from .memory import HypercubeMemory
from .field import NeuralField
from .attention import AxisAttention
from .enneract import (
    N_DIM, N_VERTICES, N_EDGES, N_SHELLS,
    COGNITIVE_AXES, SHELL_ROLES,
    vertex_coords, shell_of, shell_slices,
    neighbor_matrix, edge_list, axis_neighbors,
    petrie_project,
)
from .planck_interface import (
    cmb_stats_to_stimulus,
    analyze_with_brain,
    compare_patches,
)

__all__ = [
    # Main brain
    "EnneractBrain",
    "Thought",
    # Sub-components
    "NeuralField",
    "HypercubeMemory",
    "AxisAttention",
    # Topology constants and accessors
    "N_DIM", "N_VERTICES", "N_EDGES", "N_SHELLS",
    "COGNITIVE_AXES", "SHELL_ROLES",
    "vertex_coords", "shell_of", "shell_slices",
    "neighbor_matrix", "edge_list", "axis_neighbors",
    "petrie_project",
    # CMB interface
    "cmb_stats_to_stimulus",
    "analyze_with_brain",
    "compare_patches",
]
