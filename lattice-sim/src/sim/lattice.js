export const CX = 240;
export const CY = 240;
export const W = 480;
export const H = 480;
export const EDGE_THRESHOLD_MAX = 120;
export const EDGE_THRESHOLD_MIN = 5;

export function generateNodes() {
  const nodes = [];

  // Ring 0: core
  nodes.push({ id: 0, x: CX, y: CY, ring: 0 });

  // Ring 1: 4 nodes, radius=50, cardinal directions
  const r1 = 50;
  for (let i = 0; i < 4; i++) {
    const angle = (i * Math.PI / 2) - Math.PI / 2;
    nodes.push({ id: nodes.length, x: CX + r1 * Math.cos(angle), y: CY + r1 * Math.sin(angle), ring: 1 });
  }

  // Ring 2: 8 nodes, radius=100, cardinal+diagonal
  const r2 = 100;
  for (let i = 0; i < 8; i++) {
    const angle = (i * Math.PI / 4) - Math.PI / 2;
    nodes.push({ id: nodes.length, x: CX + r2 * Math.cos(angle), y: CY + r2 * Math.sin(angle), ring: 2 });
  }

  // Ring 3: 12 nodes, radius=155, uniform angular
  const r3 = 155;
  for (let i = 0; i < 12; i++) {
    const angle = (i * 2 * Math.PI / 12) - Math.PI / 2;
    nodes.push({ id: nodes.length, x: CX + r3 * Math.cos(angle), y: CY + r3 * Math.sin(angle), ring: 3 });
  }

  // Ring 4: 16 nodes, radius=205, uniform angular
  const r4 = 205;
  for (let i = 0; i < 16; i++) {
    const angle = (i * 2 * Math.PI / 16) - Math.PI / 2;
    nodes.push({ id: nodes.length, x: CX + r4 * Math.cos(angle), y: CY + r4 * Math.sin(angle), ring: 4 });
  }

  // Ring 5: 8 anchor nodes at fixed positions
  const anchors = [
    { x: CX,       y: CY - 235 },
    { x: CX + 235, y: CY       },
    { x: CX,       y: CY + 235 },
    { x: CX - 235, y: CY       },
    { x: CX + 166, y: CY - 166 },
    { x: CX + 166, y: CY + 166 },
    { x: CX - 166, y: CY + 166 },
    { x: CX - 166, y: CY - 166 },
  ];
  for (const { x, y } of anchors) {
    nodes.push({ id: nodes.length, x, y, ring: 5 });
  }

  return nodes;
}

export function generateEdges(nodes) {
  const edges = [];
  const adj = Array.from({ length: nodes.length }, () => []);

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < EDGE_THRESHOLD_MAX && dist > EDGE_THRESHOLD_MIN) {
        edges.push({ a: i, b: j, dist });
        adj[i].push(j);
        adj[j].push(i);
      }
    }
  }

  return { edges, adj };
}

export function euclidDist(nodes, i, j) {
  const dx = nodes[i].x - nodes[j].x;
  const dy = nodes[i].y - nodes[j].y;
  return Math.sqrt(dx * dx + dy * dy);
}

export const NODE_COLORS = {
  0: '#e2e8f0',
  1: '#c4b5fd',
  2: '#818cf8',
  3: '#6366f1',
  4: '#4338ca',
  5: '#312e81',
};

export const NODE_RADII = {
  0: 9,
  1: 5.5,
  2: 4,
  3: 4,
  4: 4,
  5: 3.5,
};

export const EDGE_OPACITY = (dist) => {
  if (dist < 65)  return 0.45;
  if (dist < 105) return 0.25;
  return 0.12;
};
