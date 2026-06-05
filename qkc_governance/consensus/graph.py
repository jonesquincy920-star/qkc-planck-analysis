"""Lattice communication graph for agent belief exchange.

Defines which governance agents exchange belief messages and the trust
weight assigned to each directed edge.  Higher weight → neighbour's
belief is weighted more heavily during log-opinion pooling.

Topology rationale:
  - Same-role scouts share full trust (same sensors, independent samples)
  - Pipeline-adjacent pairs (stego→analyst, analyst→hunter, hunter→striker)
    carry high trust because they see complementary evidence
  - Guard has moderate trust to all agents; it monitors for meta-attacks
    and its global view warrants influence without dominance
"""

from __future__ import annotations

# {agent_id: [(neighbour_id, trust_weight), ...]}
# Weights are directional: the source agent assigns this weight to the
# receiving agent's reply when pooling incoming beliefs.
LATTICE_EDGES: dict[str, list[tuple[str, float]]] = {
    "A1-SCOUT":   [("A2-SCOUT",   0.90),
                   ("A3-STEGO",   0.60),
                   ("A7-GUARD",   0.65)],

    "A2-SCOUT":   [("A1-SCOUT",   0.90),
                   ("A3-STEGO",   0.60),
                   ("A7-GUARD",   0.65)],

    "A3-STEGO":   [("A1-SCOUT",   0.60),
                   ("A2-SCOUT",   0.60),
                   ("A4-ANALYST", 0.80),
                   ("A7-GUARD",   0.65)],

    "A4-ANALYST": [("A3-STEGO",   0.80),
                   ("A5-HUNTER",  0.85),
                   ("A7-GUARD",   0.65)],

    "A5-HUNTER":  [("A4-ANALYST", 0.85),
                   ("A6-STRIKER", 0.90),
                   ("A7-GUARD",   0.65)],

    "A6-STRIKER": [("A5-HUNTER",  0.90),
                   ("A7-GUARD",   0.65)],

    "A7-GUARD":   [("A1-SCOUT",   0.65),
                   ("A2-SCOUT",   0.65),
                   ("A3-STEGO",   0.65),
                   ("A4-ANALYST", 0.65),
                   ("A5-HUNTER",  0.65),
                   ("A6-STRIKER", 0.65)],
}


def neighbours(agent_id: str) -> list[str]:
    """Return agent IDs that agent_id sends beliefs to."""
    return [n for n, _ in LATTICE_EDGES.get(agent_id, [])]


def trust_weight(from_agent: str, to_agent: str) -> float:
    """Return the trust weight from_agent places on to_agent's beliefs (0 if no edge)."""
    for n, w in LATTICE_EDGES.get(from_agent, []):
        if n == to_agent:
            return w
    return 0.0


def mean_out_weight(agent_id: str) -> float:
    """Mean outbound trust weight — used to weight an agent in the global pool."""
    edges = LATTICE_EDGES.get(agent_id, [])
    if not edges:
        return 0.50
    return sum(w for _, w in edges) / len(edges)
