"""
Circuit highway closed-loop topology for minimising entropy in CMB patches.

The circuit mirrors the QKC geodesic mesh image: n_rings concentric rings of
n_spokes nodes each, plus a central hub, connected by ring (azimuthal) edges
and spoke (radial) edges.  A minimum-entropy Hamiltonian cycle is found via
greedy nearest-neighbour initialisation followed by 2-opt local search, where
"entropy" is operationalised as the total variation of CMB temperatures along
the closed path — the thermodynamic entropy-production proxy for an irreversible
cyclic circuit.

Minimum entropy production ⟺ smoothest possible closed temperature path, i.e.
the circuit visits nodes in an order that minimises unnecessary temperature
jumps, approaching a reversible (Carnot-like) cycle in the limit.
"""

import numpy as np
import healpy as hp
from dataclasses import dataclass
from typing import Optional

from .spot import galactic_to_healpy, QKC_L, QKC_B

# Physical constants
K_B: float = 1.380649e-23      # Boltzmann constant  [J K⁻¹]
H_PLANCK: float = 6.62607015e-34  # Planck constant  [J s]
C_LIGHT: float = 2.99792458e8  # speed of light      [m s⁻¹]
L_PLANCK_SQ: float = 2.612e-70  # Planck area        [m²]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircuitNode:
    """A node in the circuit highway — a single HEALPix sky pixel."""
    index: int
    pixel_id: int
    theta: float        # HEALPix co-latitude [rad]
    phi: float          # HEALPix longitude   [rad]
    temperature: float  # CMB temperature fluctuation [μK]
    ring: int           # 0 = hub, 1..n_rings = outer rings
    slot: int           # azimuthal position within the ring (0..n_spokes-1)


@dataclass(frozen=True)
class CircuitEdge:
    """A highway segment between two circuit nodes."""
    u: int
    v: int
    weight: float  # |ΔT| entropy-production cost [μK]
    kind: str      # 'ring' | 'spoke'


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def shannon_entropy(values: np.ndarray, n_bins: int = 64) -> float:
    """Shannon entropy of a temperature sample (nats)."""
    if len(values) < 2:
        return 0.0
    hist, _ = np.histogram(values, bins=n_bins)
    p = hist.astype(float) + 1e-10
    p /= p.sum()
    return float(-np.dot(p, np.log(p)))


def total_variation(temps: np.ndarray) -> float:
    """
    Total variation of a closed temperature path — entropy-production proxy.

    TV = Σ_i |T_{i+1} - T_i|  (indices cyclic: T_n = T_0)

    Minimising TV drives the circuit toward the thermodynamically reversible
    limit where the closed-loop entropy production ΔS → 0.
    """
    return float(np.sum(np.abs(np.diff(temps, append=temps[0]))))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CircuitHighway:
    """
    Radial mesh closed-loop circuit that minimises entropy production.

    Topology
    --------
    * Hub node  (ring 0) at the QKC target (l=−57°, b=−27°).
    * n_rings outer rings × n_spokes nodes per ring.
    * Ring edges  connect adjacent nodes within the same ring (azimuthal).
    * Spoke edges connect node k in ring r to node k in ring r+1 (radial).

    Optimisation
    ------------
    `find_minimum_entropy_loop` returns the Hamiltonian cycle through all
    nodes that minimises the total variation of CMB temperature (entropy
    production).  Algorithm: greedy nearest-neighbour seed → 2-opt.
    """

    def __init__(
        self,
        n_rings: int = 5,
        n_spokes: int = 8,
        l_deg: float = QKC_L,
        b_deg: float = QKC_B,
        radius_deg: float = 10.0,
    ):
        self.n_rings = n_rings
        self.n_spokes = n_spokes
        self.l_deg = l_deg
        self.b_deg = b_deg
        self.radius_deg = radius_deg

        self.nodes: list[CircuitNode] = []
        self.edges: list[CircuitEdge] = []
        self._adj: dict[int, list[int]] = {}
        self._loop: list[int] = []
        self.loop_entropy: float = float("inf")

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_from_map(self, hmap: np.ndarray) -> "CircuitHighway":
        """
        Populate nodes and edges from a full-sky HEALPix temperature map.

        Ring spacing is uniform in angular radius up to `self.radius_deg`.
        Each outer ring has `n_spokes` nodes placed at equal azimuthal
        intervals around the centre.
        """
        nside = hp.get_nside(hmap)
        c_theta, c_phi = galactic_to_healpy(self.l_deg, self.b_deg)
        ring_dr = np.radians(self.radius_deg) / self.n_rings

        self.nodes = []
        self.edges = []
        ring_map: dict[int, list[int]] = {}

        # Hub — ring 0
        hub_pix = hp.ang2pix(nside, c_theta, c_phi)
        hub_temp = float(hmap[hub_pix]) if np.isfinite(hmap[hub_pix]) else 0.0
        self.nodes.append(CircuitNode(
            index=0, pixel_id=int(hub_pix),
            theta=float(c_theta), phi=float(c_phi),
            temperature=hub_temp, ring=0, slot=0,
        ))
        ring_map[0] = [0]
        idx = 1

        # Outer rings
        for r in range(1, self.n_rings + 1):
            r_rad = r * ring_dr
            ring_indices: list[int] = []
            for k in range(self.n_spokes):
                az = 2.0 * np.pi * k / self.n_spokes
                # Offset from centre along a small-circle arc
                dt = r_rad * np.cos(az)
                dp = r_rad * np.sin(az) / max(np.sin(c_theta), 1e-9)
                th = float(np.clip(c_theta + dt, 0.0, np.pi))
                ph = float((c_phi + dp) % (2.0 * np.pi))
                pix = hp.ang2pix(nside, th, ph)
                temp = float(hmap[pix]) if np.isfinite(hmap[pix]) else 0.0
                self.nodes.append(CircuitNode(
                    index=idx, pixel_id=int(pix),
                    theta=th, phi=ph, temperature=temp,
                    ring=r, slot=k,
                ))
                ring_indices.append(idx)
                idx += 1
            ring_map[r] = ring_indices

        self._adj = {i: [] for i in range(len(self.nodes))}
        self._add_ring_edges(ring_map)
        self._add_spoke_edges(ring_map)
        return self

    def _add_ring_edges(self, ring_map: dict[int, list[int]]) -> None:
        for r in range(1, self.n_rings + 1):
            ring = ring_map[r]
            m = len(ring)
            for k in range(m):
                u, v = ring[k], ring[(k + 1) % m]
                w = abs(self.nodes[u].temperature - self.nodes[v].temperature)
                self.edges.append(CircuitEdge(u=u, v=v, weight=w, kind="ring"))
                self._adj[u].append(v)
                self._adj[v].append(u)

    def _add_spoke_edges(self, ring_map: dict[int, list[int]]) -> None:
        for r in range(self.n_rings):
            inner = ring_map[r]
            outer = ring_map[r + 1]
            if r == 0:
                # Hub fans out to every node in ring 1
                pairs = [(0, v) for v in outer]
            else:
                pairs = [(inner[k], outer[k]) for k in range(self.n_spokes)]
            for u, v in pairs:
                w = abs(self.nodes[u].temperature - self.nodes[v].temperature)
                self.edges.append(CircuitEdge(u=u, v=v, weight=w, kind="spoke"))
                self._adj[u].append(v)
                self._adj[v].append(u)

    # ------------------------------------------------------------------
    # Entropy metrics
    # ------------------------------------------------------------------

    def ring_entropy(self, r: int) -> float:
        """Shannon entropy of the temperature distribution within ring r."""
        temps = np.array([n.temperature for n in self.nodes if n.ring == r])
        return shannon_entropy(temps)

    def circuit_entropy(self) -> float:
        """Total Shannon entropy summed across all rings."""
        return sum(self.ring_entropy(r) for r in range(self.n_rings + 1))

    def loop_total_variation(self, loop: Optional[list[int]] = None) -> float:
        """Entropy production (total variation of T) for a closed loop."""
        if loop is None:
            loop = self._loop
        if not loop:
            return float("inf")
        temps = np.array([self.nodes[i].temperature for i in loop])
        return total_variation(temps)

    def entropy_reduction_ratio(self) -> float:
        """
        Ratio of loop entropy production to a fully random ordering baseline.

        Values < 1 confirm that the optimised loop is more ordered than chance.
        """
        if not self._loop:
            self.find_minimum_entropy_loop()
        rng = np.random.default_rng(0)
        baseline_temps = np.array([n.temperature for n in self.nodes])
        random_tv = np.mean([
            total_variation(baseline_temps[rng.permutation(len(self.nodes))])
            for _ in range(200)
        ])
        return self.loop_entropy / random_tv if random_tv > 0 else float("nan")

    # ------------------------------------------------------------------
    # Minimum-entropy closed loop
    # ------------------------------------------------------------------

    def find_minimum_entropy_loop(self) -> list[int]:
        """
        Find the Hamiltonian cycle that minimises total temperature variation.

        Algorithm
        ---------
        1. Greedy nearest-neighbour seeding: start at hub, always move to the
           unvisited node with the smallest |ΔT| from the current node.
        2. 2-opt local search: repeatedly reverse sub-segments of the path
           until no improvement in TV can be found.

        Returns the loop as an ordered list of node indices (no repeated end).
        """
        n = len(self.nodes)
        if n == 0:
            return []
        temps = np.array([nd.temperature for nd in self.nodes])

        # --- greedy nearest-neighbour ---
        visited = [False] * n
        path = [0]
        visited[0] = True
        for _ in range(n - 1):
            last = path[-1]
            best_j, best_dt = -1, float("inf")
            for j in range(n):
                if not visited[j]:
                    dt = abs(temps[last] - temps[j])
                    if dt < best_dt:
                        best_dt, best_j = dt, j
            path.append(best_j)
            visited[best_j] = True

        # --- 2-opt improvement ---
        path = _two_opt(path, temps)
        self._loop = path
        self.loop_entropy = total_variation(temps[np.array(path)])
        return path

    # ------------------------------------------------------------------
    # Holographic entropy bound
    # ------------------------------------------------------------------

    @staticmethod
    def planck_entropy_bound(radius_deg: float) -> float:
        """
        Bekenstein–Hawking holographic entropy bound S ≤ A / (4 l_P²) [nats].

        Solid angle of the patch is projected onto the Hubble sphere to give
        a physical area, then divided by 4 × Planck area.
        """
        r_hubble = 4.4e26  # m (co-moving Hubble radius)
        omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(radius_deg)))
        area = omega * r_hubble ** 2
        return area / (4.0 * L_PLANCK_SQ)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict of circuit metrics, computing the loop if needed."""
        if not self._loop:
            self.find_minimum_entropy_loop()
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_rings": self.n_rings,
            "n_spokes": self.n_spokes,
            "circuit_entropy_nats": self.circuit_entropy(),
            "loop_entropy_production_uK": self.loop_entropy,
            "entropy_reduction_ratio": self.entropy_reduction_ratio(),
            "ring_entropies_nats": {
                r: self.ring_entropy(r) for r in range(self.n_rings + 1)
            },
            "planck_bound_nats": self.planck_entropy_bound(self.radius_deg),
        }


# ---------------------------------------------------------------------------
# 2-opt TSP solver  —  minimise total variation of temperature
# ---------------------------------------------------------------------------

def _tv(path: list[int], temps: np.ndarray) -> float:
    """Total variation of the closed path (entropy production cost)."""
    t = temps[path]
    return float(np.sum(np.abs(np.diff(t, append=t[0]))))


def _two_opt(path: list[int], temps: np.ndarray) -> list[int]:
    """
    2-opt local search over a Hamiltonian path.

    A 2-opt move reverses segment path[i+1..j], which changes four edge
    costs.  We accept the move whenever the new closed-loop TV is strictly
    smaller than the current best, and restart the scan.  Converges in
    O(n²) passes in the worst case.
    """
    best = list(path)
    n = len(best)
    best_cost = _tv(best, temps)

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip the trivially identical move when i=0, j=n-1
                if i == 0 and j == n - 1:
                    continue
                cand = best[: i + 1] + best[i + 1 : j + 1][::-1] + best[j + 1 :]
                cost = _tv(cand, temps)
                if cost < best_cost - 1e-12:
                    best, best_cost = cand, cost
                    improved = True
                    break
            if improved:
                break
    return best
