export const WALK_DEPTH = 12;

export function mcStep(currentNode, targetNode, adj, nodes, nWalks) {
  const neighbors = adj[currentNode];
  if (!neighbors || neighbors.length === 0) return currentNode;
  if (currentNode === targetNode) return currentNode;

  const walksPerNeighbor = Math.max(1, Math.floor(nWalks / neighbors.length));
  let bestNb = neighbors[0];
  let bestScore = -Infinity;

  for (const nb of neighbors) {
    let totalScore = 0;
    for (let w = 0; w < walksPerNeighbor; w++) {
      let pos = nb;
      for (let step = 1; step < WALK_DEPTH; step++) {
        const nexts = adj[pos];
        if (!nexts || nexts.length === 0) break;
        pos = nexts[Math.floor(Math.random() * nexts.length)];
      }
      const dx = nodes[pos].x - nodes[targetNode].x;
      const dy = nodes[pos].y - nodes[targetNode].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      totalScore += 1 / (dist + 1);
    }
    const avgScore = totalScore / walksPerNeighbor;
    if (avgScore > bestScore) {
      bestScore = avgScore;
      bestNb = nb;
    }
  }

  return bestNb;
}

export function randomWalkStep(currentNode, adj) {
  const neighbors = adj[currentNode];
  if (!neighbors || neighbors.length === 0) return currentNode;
  return neighbors[Math.floor(Math.random() * neighbors.length)];
}
