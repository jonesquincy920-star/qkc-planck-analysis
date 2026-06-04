import React, { useMemo } from 'react';
import { NODE_COLORS, NODE_RADII, EDGE_OPACITY } from '../sim/lattice.js';
import { topPosterior } from '../sim/bayesian.js';

const AGENT_COLORS = {
  SCOUT:   '#00ff9d',
  STEGO:   '#e879f9',
  ANALYST: '#a78bfa',
  HUNTER:  '#00cfff',
  STRIKER: '#f43f5e',
  GUARD:   '#fbbf24',
};

const THREAT_STATUS_COLORS = {
  ACTIVE:     '#f43f5e',
  LOCATED:    '#fbbf24',
  CLASSIFIED: '#a78bfa',
  CONTAINED:  '#ff6b35',
  DESTROYED:  '#555',
};

function threatColor(t) {
  if (t.isStego && t.status !== 'DESTROYED') return '#e879f9';
  return THREAT_STATUS_COLORS[t.status] ?? '#f43f5e';
}

// SVG arc described by strokeDasharray (confidence arc from top)
function BayesArc({ cx, cy, r, confidence, color }) {
  const circ = 2 * Math.PI * r;
  const dash = circ * Math.max(0, Math.min(1, confidence));
  const gap  = circ - dash;
  return (
    <circle
      cx={cx} cy={cy} r={r}
      fill="none"
      stroke={color}
      strokeWidth={2.5}
      strokeDasharray={`${dash} ${gap}`}
      transform={`rotate(-90 ${cx} ${cy})`}
      strokeLinecap="round"
      opacity={0.85}
    />
  );
}

export default function LatticeCanvas({ nodes, edges, display }) {
  const { agents = [], threats = [] } = display;

  const edgeEls = useMemo(() => edges.map((e, i) => {
    const a = nodes[e.a], b = nodes[e.b];
    return (
      <line
        key={i}
        x1={a.x} y1={a.y} x2={b.x} y2={b.y}
        stroke="#818cf8"
        strokeWidth={0.7}
        opacity={EDGE_OPACITY(e.dist)}
      />
    );
  }), [edges, nodes]);

  const nodeEls = useMemo(() => nodes.map(n => (
    <circle
      key={n.id}
      cx={n.x} cy={n.y}
      r={NODE_RADII[n.ring]}
      fill={NODE_COLORS[n.ring]}
      opacity={0.9}
    />
  )), [nodes]);

  return (
    <svg
      width={480} height={480}
      viewBox="0 0 480 480"
      style={{ display: 'block', borderRadius: 8, background: '#090914' }}
    >
      <defs>
        <radialGradient id="bg-grad" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="#4338ca" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#4338ca" stopOpacity="0"    />
        </radialGradient>

        <filter id="glow-agent" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>

        <filter id="glow-threat" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>

      {/* Background gradient */}
      <rect width={480} height={480} fill="url(#bg-grad)" />

      {/* Edges */}
      <g>{edgeEls}</g>

      {/* Nodes */}
      <g>{nodeEls}</g>

      {/* Threats */}
      {threats.map(t => {
        const node = nodes[t.nodeIdx];
        if (!node) return null;
        const col  = threatColor(t);
        const { confidence } = topPosterior(t.posterior);
        const arcR = 15;

        return (
          <g key={t.id} filter="url(#glow-threat)">
            {/* Threat body */}
            <circle cx={node.x} cy={node.y} r={10} fill={col} opacity={t.status === 'DESTROYED' ? 0.3 : 0.85} />
            {/* Bayes confidence arc */}
            {t.status !== 'ACTIVE' && t.status !== 'DESTROYED' && (
              <BayesArc cx={node.x} cy={node.y} r={arcR} confidence={confidence} color={col} />
            )}
            {/* Label */}
            <text
              x={node.x} y={node.y - 18}
              textAnchor="middle"
              fill={col}
              fontSize={9}
              fontFamily="monospace"
              opacity={0.9}
            >
              {t.id}
            </text>
            {/* Claude badge */}
            {t.claudeResult && (
              <circle cx={node.x + 11} cy={node.y - 11} r={3.5} fill="#00cfff" opacity={0.9} />
            )}
            {t.claudePending && (
              <circle cx={node.x + 11} cy={node.y - 11} r={3.5} fill="#fbbf24" opacity={0.7} />
            )}
          </g>
        );
      })}

      {/* Agents */}
      {agents.map(a => {
        const node = nodes[a.nodeIdx];
        if (!node) return null;
        const col = AGENT_COLORS[a.type] ?? '#fff';

        return (
          <g key={a.id} filter="url(#glow-agent)">
            <circle cx={node.x} cy={node.y} r={6} fill={col} opacity={0.92} />
            <text
              x={node.x} y={node.y + 16}
              textAnchor="middle"
              fill={col}
              fontSize={8}
              fontFamily="monospace"
              opacity={0.85}
            >
              {a.id}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
