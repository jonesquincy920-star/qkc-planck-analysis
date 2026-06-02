# EnneractBrain

**A 9D Hypercube Neural Intelligence System**

Neurological map derived from the Petrie projection of the 9D hypercube (enneract).

```
Vertices : 2⁹ = 512    → neurons
Edges    : 9·2⁸ = 2304 → synaptic connections
Shells   : 10           → cognitive hierarchy (Hamming weight k = 0 … 9)
Axes     : 9            → cognitive domains
```

---

## Architecture

The brain's topology is the 9D hypercube graph.  Every neuron is a vertex
identified by a 9-bit binary coordinate; every synapse connects vertices that
differ by exactly one bit.  The 10 Hamming shells form a strict processing
hierarchy from the single core neuron (k = 0) to the single unified-awareness
neuron (k = 9):

| Shell k | Size C(9,k) | Role |
|---------|------------|------|
| 0 | 1 | Core integration |
| 1 | 9 | Primary axes |
| 2 | 36 | Feature binding |
| 3 | 84 | Association |
| 4 | 126 | Working memory |
| 5 | 126 | Abstract reasoning |
| 6 | 84 | Metacognition layer |
| 7 | 36 | Self-model |
| 8 | 9 | Consciousness edge |
| 9 | 1 | Unified awareness |

### 9 Cognitive Axes

Each of the 9 hypercube dimensions maps to a cognitive domain:

| Axis | Domain |
|------|--------|
| 0 | Perception |
| 1 | Memory |
| 2 | Reasoning |
| 3 | Emotion |
| 4 | Language |
| 5 | Spatial |
| 6 | Temporal |
| 7 | Social |
| 8 | Metacognition |

### Neural Dynamics

Each neuron holds a **complex-valued state** `z = r·e^{iφ}`:
- `r` (magnitude) — firing rate / activation strength
- `φ` (phase) — temporal coordination / spike timing

**Propagation** follows axis-aware message passing:
```
h[v] = bias[v] + Σ_d  W[v,d] · σ( state[nb_d(v)] )
state ← state + dt · (h − state)
```
where `σ` is a complex activation (tanh on magnitude, phase preserved).

**Shell resonance** nudges each shell's neurons toward a mean phase,
producing standing waves analogous to gamma-band oscillations.

**Hebbian plasticity** updates synaptic weights:
```
ΔW[v,d]  ∝  Re( conj(state[v]) · state[nb_d(v)] )
```

### Memory

`HypercubeMemory` provides **Hopfield associative memory** over the
512-vertex address space.  A 9D feature vector is soft-addressed into
a probability distribution over vertices via:
```
P(v) ∝ exp( −β · ‖coords[v] − features‖² )
```

### Attention

`AxisAttention` implements **9-head scaled dot-product attention** — one
head per cognitive axis — so the brain can introspect which dimensions are
most actively routing information at any step.

---

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.11, NumPy ≥ 1.24, SciPy ≥ 1.11, Matplotlib ≥ 3.7

---

## Quick Start

```python
from enneract_brain import EnneractBrain

brain = EnneractBrain(
    memory_capacity   = 128,
    propagation_steps = 5,
    resonance_cycles  = 2,
    learning_rate     = 0.005,
    seed              = 42,
)

# Process a 9D stimulus (one value per cognitive axis, in [0, 1])
thought = brain.think([0.9, 0.1, 0.8, 0.2, 0.7, 0.5, 0.5, 0.3, 0.6])

print(thought.dominant_axis)    # e.g. "reasoning"
print(thought.dominant_shell)   # e.g. 4  (working memory)
print(thought.shell_energies)   # (10,) array

# Store a pattern in associative memory
brain.remember(thought.raw_state, label="hypothesis_A")

# Retrieve nearest pattern via Hopfield dynamics
recalled = brain.recall(thought.raw_state)

# Full introspection snapshot
info = brain.introspect()
# Keys: step, total_energy, dominant_shell, dominant_axis,
#       shell_energies, axis_coherence, axis_importance, memory, ...
```

### Visualisation

```python
from enneract_brain.visualize import plot_dashboard
import matplotlib.pyplot as plt

fig = plot_dashboard(brain)
plt.savefig("brain_dashboard.png", dpi=150, bbox_inches="tight")
```

### CMB / Planck Interface

```python
from enneract_brain import cmb_stats_to_stimulus, analyze_with_brain

anomaly = {"ks_statistic": 0.42, "ks_pvalue": 0.003,
           "z_score": -3.7,      "z_pvalue": 0.0002}
patch   = {"mean": -312.0, "std": 95.0, "skewness": -0.6, "kurtosis": 1.8}

stimulus = cmb_stats_to_stimulus(anomaly, patch)
result   = analyze_with_brain(brain, anomaly, patch, label="QKC_cold_spot")
print(result["interpretation"])
```

---

## Package Layout

```
src/enneract_brain/
├── __init__.py          exports
├── enneract.py          9D hypercube topology (vertices, edges, shells)
├── field.py             NeuralField — complex-valued dynamics
├── memory.py            HypercubeMemory — Hopfield associative memory
├── attention.py         AxisAttention — 9-head axis attention
├── brain.py             EnneractBrain — main API
├── visualize.py         Petrie-projection dashboards
└── planck_interface.py  CMB analysis bridge
tests/
└── test_brain.py        48 tests, all passing
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

48 tests cover topology correctness, neural dynamics, memory, attention,
full brain integration, and the CMB interface.

---

## License

MIT
