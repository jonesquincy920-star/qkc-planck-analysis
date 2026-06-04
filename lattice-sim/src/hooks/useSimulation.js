import { useRef, useState, useCallback, useEffect } from 'react';
import { generateNodes, generateEdges, euclidDist } from '../sim/lattice.js';
import { mcStep, randomWalkStep } from '../sim/monteCarlo.js';
import {
  THREAT_TYPES, FEATURES, uniformPrior,
  bayesUpdatePartial, topPosterior, determineTrueType,
} from '../sim/bayesian.js';
import { detectStego } from '../sim/stego.js';
import { classifyThreat } from '../api/claude.js';

// ─── Constants ───────────────────────────────────────────────────────────────
const TICK_MS          = 950;
const SPAWN_MS         = 5500;
const CLAUDE_POLL_MS   = 3000;
const MAX_THREATS      = 3;
const INTERACT_DIST    = 36;
const PROX_NOISE       = 0.10;
const ANALYST_GATE     = 0.55;
const STRIKER_GATE     = 0.42;

const AGENT_DEFS = [
  { id: 'A1', type: 'SCOUT',   startNode: 3,  nWalks: 60,  sensors: ['evasion', 'signal'],       color: '#00ff9d' },
  { id: 'A2', type: 'SCOUT',   startNode: 7,  nWalks: 60,  sensors: ['evasion', 'signal'],       color: '#00ff9d' },
  { id: 'A3', type: 'STEGO',   startNode: 11, nWalks: 50,  sensors: ['entropy', 'deception'],    color: '#e879f9' },
  { id: 'A4', type: 'ANALYST', startNode: 18, nWalks: 40,  sensors: ['deception', 'mutation'],   color: '#a78bfa' },
  { id: 'A5', type: 'HUNTER',  startNode: 22, nWalks: 100, sensors: ['propagation', 'signal'],   color: '#00cfff' },
  { id: 'A6', type: 'STRIKER', startNode: 30, nWalks: 120, sensors: ['evasion', 'mutation'],     color: '#f43f5e' },
  { id: 'A7', type: 'GUARD',   startNode: 0,  nWalks: 30,  sensors: [],                          color: '#fbbf24' },
];

// Which threat statuses each agent type pursues
const AGENT_TARGETS = {
  SCOUT:   ['ACTIVE'],
  STEGO:   ['ACTIVE', 'LOCATED'],
  ANALYST: ['LOCATED'],
  HUNTER:  ['CLASSIFIED'],
  STRIKER: ['CONTAINED'],
  GUARD:   [],
};

let threatCounter = 1;

function makeAgents() {
  return AGENT_DEFS.map(d => ({ ...d, nodeIdx: d.startNode }));
}

function spawnThreat(nodes) {
  const nodeIdx = Math.floor(Math.random() * nodes.length);
  const features = {};
  for (const f of FEATURES) features[f] = Math.random();
  const strength = 0.5 + Math.random() * 0.5;

  const stegoResult = detectStego(features);
  let trueType = determineTrueType(features);
  if (stegoResult.sig && Math.random() > 0.4) trueType = 'STEGO_CHANNEL';

  return {
    id: `T${threatCounter++}`,
    nodeIdx,
    status: 'ACTIVE',
    features,
    strength,
    trueType,
    posterior: uniformPrior(),
    claudeResult: null,
    claudePending: false,
    isStego: trueType === 'STEGO_CHANNEL',
    spawnTime: Date.now(),
    locatedAt: null,
  };
}

function getTargetNode(agent, threats, nodes) {
  const validStatuses = AGENT_TARGETS[agent.type] || [];
  if (validStatuses.length === 0) return null;

  const candidates = threats.filter(t => validStatuses.includes(t.status));
  if (candidates.length === 0) return null;

  let nearest = candidates[0];
  let minD = Infinity;
  for (const t of candidates) {
    const d = euclidDist(nodes, agent.nodeIdx, t.nodeIdx);
    if (d < minD) { minD = d; nearest = t; }
  }
  return nearest.nodeIdx;
}

function snapshot(state) {
  return {
    tick: state.tick,
    running: state.running,
    agents: state.agents.map(a => ({ ...a })),
    threats: state.threats.map(t => ({
      ...t,
      features: { ...t.features },
      posterior: { ...t.posterior },
      claudeResult: t.claudeResult ? { ...t.claudeResult } : null,
    })),
    log: state.log.slice(-120),
  };
}

function addLog(state, msg) {
  const now = new Date();
  const time = `${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}.${String(now.getMilliseconds()).padStart(3, '0')}`;
  state.log.push({ time, msg });
  if (state.log.length > 200) state.log.splice(0, state.log.length - 150);
}

// ─── Hook ─────────────────────────────────────────────────────────────────────
export function useSimulation() {
  const { nodes, edges, adj } = useRef((() => {
    const n = generateNodes();
    const { edges, adj } = generateEdges(n);
    return { nodes: n, edges, adj };
  })()).current;

  const stateRef = useRef({
    tick: 0,
    running: false,
    agents: makeAgents(),
    threats: [],
    log: [],
  });

  const tickTimerRef   = useRef(null);
  const spawnTimerRef  = useRef(null);
  const claudeTimerRef = useRef(null);

  const [display, setDisplay] = useState(() => snapshot(stateRef.current));

  // ── Tick ────────────────────────────────────────────────────────────────────
  const tick = useCallback(() => {
    const s = stateRef.current;
    if (!s.running) return;
    s.tick++;

    // 1. Move agents (one MC step each)
    for (const agent of s.agents) {
      if (agent.type === 'GUARD') continue;
      const targetNode = getTargetNode(agent, s.threats, nodes);
      if (targetNode !== null) {
        agent.nodeIdx = mcStep(agent.nodeIdx, targetNode, adj, nodes, agent.nWalks);
      } else {
        agent.nodeIdx = randomWalkStep(agent.nodeIdx, adj);
      }
    }

    // 2. Agent–threat interactions within INTERACT_DIST
    const toDestroy = [];
    for (const threat of s.threats) {
      if (threat.status === 'DESTROYED') continue;

      for (const agent of s.agents) {
        const d = euclidDist(nodes, agent.nodeIdx, threat.nodeIdx);
        if (d >= INTERACT_DIST) continue;

        // Bayes update using this agent's sensors (if any)
        if (agent.sensors.length > 0) {
          threat.posterior = bayesUpdatePartial(
            threat.posterior, agent.sensors, threat.features, PROX_NOISE
          );
        }

        // STEGO agent: check for stego channel
        if (agent.type === 'STEGO' && threat.status === 'ACTIVE') {
          const noisy = {};
          for (const f of FEATURES) {
            noisy[f] = Math.max(0, Math.min(1, threat.features[f] + (Math.random() - 0.5) * 2 * PROX_NOISE));
          }
          const { sig } = detectStego(noisy);
          if (sig) {
            threat.status = 'LOCATED';
            threat.locatedAt = s.tick;
            addLog(s, `[STEGO] ${agent.id} confirmed stego channel in ${threat.id} → LOCATED`);
          }
        }

        // SCOUT: locate ACTIVE threats
        if (agent.type === 'SCOUT' && threat.status === 'ACTIVE') {
          threat.status = 'LOCATED';
          threat.locatedAt = s.tick;
          addLog(s, `[SCOUT] ${agent.id} located ${threat.id} → LOCATED`);
        }

        // ANALYST: classify LOCATED threats
        if (agent.type === 'ANALYST' && threat.status === 'LOCATED') {
          const { confidence } = topPosterior(threat.posterior);
          const claudeConf = threat.claudeResult?.confidence;
          const effectiveConf = claudeConf !== undefined ? claudeConf : confidence;
          if (effectiveConf > ANALYST_GATE) {
            const { type } = topPosterior(threat.posterior);
            threat.status = 'CLASSIFIED';
            addLog(s, `[ANALYST] ${agent.id} classified ${threat.id} as ${type} (${(effectiveConf * 100).toFixed(0)}%) → CLASSIFIED`);
          }
        }

        // HUNTER: contain CLASSIFIED threats
        if (agent.type === 'HUNTER' && threat.status === 'CLASSIFIED') {
          threat.status = 'CONTAINED';
          addLog(s, `[HUNTER] ${agent.id} contained ${threat.id} → CONTAINED`);
        }

        // STRIKER: destroy CONTAINED threats
        if (agent.type === 'STRIKER' && threat.status === 'CONTAINED') {
          const { confidence } = topPosterior(threat.posterior);
          const claudeConf = threat.claudeResult?.confidence;
          const effectiveConf = claudeConf !== undefined ? claudeConf : confidence;
          if (effectiveConf > STRIKER_GATE) {
            threat.status = 'DESTROYED';
            addLog(s, `[STRIKER] ${agent.id} destroyed ${threat.id} (conf=${(effectiveConf * 100).toFixed(0)}%) → DESTROYED`);
            toDestroy.push(threat.id);
          }
        }
      }
    }

    // Remove DESTROYED threats after a delay (keep for 3 ticks for visual)
    s.threats = s.threats.filter(t => {
      if (t.status !== 'DESTROYED') return true;
      if (!t.destroyedTick) { t.destroyedTick = s.tick; return true; }
      return s.tick - t.destroyedTick < 4;
    });

    setDisplay(snapshot(s));
  }, [nodes, adj]);

  // ── Threat spawner ──────────────────────────────────────────────────────────
  const spawnTick = useCallback(() => {
    const s = stateRef.current;
    if (!s.running) return;
    const active = s.threats.filter(t => t.status !== 'DESTROYED').length;
    if (active < MAX_THREATS) {
      const threat = spawnThreat(nodes);
      s.threats.push(threat);
      addLog(s, `[SPAWN] ${threat.id} (${threat.trueType}) appeared at node ${threat.nodeIdx}`);
      setDisplay(snapshot(s));
    }
  }, [nodes]);

  // ── Claude poller ───────────────────────────────────────────────────────────
  const claudePoll = useCallback(() => {
    const s = stateRef.current;
    if (!s.running) return;
    const pending = s.threats.find(
      t => t.status === 'LOCATED' && !t.claudeResult && !t.claudePending
    );
    if (!pending) return;

    pending.claudePending = true;
    addLog(s, `[CLAUDE] queuing ${pending.id} for API classification`);
    setDisplay(snapshot(s));

    classifyThreat(pending).then(result => {
      const t = stateRef.current.threats.find(x => x.id === pending.id);
      if (!t) return;
      t.claudePending = false;
      if (result) {
        t.claudeResult = result;
        addLog(stateRef.current, `[CLAUDE] ${t.id} → ${result.threat_type} (${(result.confidence * 100).toFixed(0)}%) | ${result.action}`);
      } else {
        addLog(stateRef.current, `[CLAUDE] ${t.id} → API unavailable, Bayes-only`);
      }
      setDisplay(snapshot(stateRef.current));
    });
  }, []);

  // ── Controls ─────────────────────────────────────────────────────────────────
  const start = useCallback(() => {
    const s = stateRef.current;
    if (s.running) return;
    s.running = true;
    addLog(s, '[SIM] Started');
    setDisplay(snapshot(s));
    tickTimerRef.current   = setInterval(tick,       TICK_MS);
    spawnTimerRef.current  = setInterval(spawnTick,  SPAWN_MS);
    claudeTimerRef.current = setInterval(claudePoll, CLAUDE_POLL_MS);
  }, [tick, spawnTick, claudePoll]);

  const stop = useCallback(() => {
    const s = stateRef.current;
    s.running = false;
    clearInterval(tickTimerRef.current);
    clearInterval(spawnTimerRef.current);
    clearInterval(claudeTimerRef.current);
    addLog(s, '[SIM] Paused');
    setDisplay(snapshot(s));
  }, []);

  const reset = useCallback(() => {
    stop();
    const s = stateRef.current;
    s.tick = 0;
    s.agents = makeAgents();
    s.threats = [];
    s.log = [];
    s.running = false;
    addLog(s, '[SIM] Reset');
    setDisplay(snapshot(s));
  }, [stop]);

  useEffect(() => () => {
    clearInterval(tickTimerRef.current);
    clearInterval(spawnTimerRef.current);
    clearInterval(claudeTimerRef.current);
  }, []);

  return { display, nodes, edges, adj, start, stop, reset };
}
