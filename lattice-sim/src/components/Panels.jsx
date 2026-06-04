import React from 'react';
import { topPosterior } from '../sim/bayesian.js';

const AGENT_COLORS = {
  SCOUT:   '#00ff9d',
  STEGO:   '#e879f9',
  ANALYST: '#a78bfa',
  HUNTER:  '#00cfff',
  STRIKER: '#f43f5e',
  GUARD:   '#fbbf24',
};

const STATUS_COLORS = {
  ACTIVE:     '#f43f5e',
  LOCATED:    '#fbbf24',
  CLASSIFIED: '#a78bfa',
  CONTAINED:  '#ff6b35',
  DESTROYED:  '#555',
};

export function AgentPanel({ agents }) {
  return (
    <div className="panel">
      <div className="panel-title">Agents</div>
      <div className="panel-list">
        {agents.map(a => (
          <div key={a.id} className="panel-row">
            <span className="dot" style={{ background: AGENT_COLORS[a.type] }} />
            <span className="label-id">{a.id}</span>
            <span className="label-type">{a.type}</span>
            <span className="label-node" style={{ marginLeft: 'auto', color: '#6366f1', fontSize: 10 }}>
              n{a.nodeIdx}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ThreatPanel({ threats }) {
  const active = threats.filter(t => t.status !== 'DESTROYED');
  return (
    <div className="panel">
      <div className="panel-title">Threats <span className="badge">{active.length}</span></div>
      <div className="panel-list">
        {active.length === 0 && <div className="muted">No active threats</div>}
        {active.map(t => {
          const { type, confidence } = topPosterior(t.posterior);
          const col = t.isStego ? '#e879f9' : (STATUS_COLORS[t.status] ?? '#f43f5e');
          return (
            <div key={t.id} className="threat-row">
              <div className="threat-row-top">
                <span className="dot" style={{ background: col }} />
                <span className="label-id">{t.id}</span>
                <span className="label-status" style={{ color: col }}>{t.status}</span>
                {t.isStego && <span className="stego-badge">STEGO</span>}
                {t.claudeResult && <span className="claude-badge">API</span>}
              </div>
              <div className="threat-row-detail">
                <span style={{ color: '#818cf8', fontSize: 10 }}>{type}</span>
                <span style={{ marginLeft: 6, color: '#6366f1', fontSize: 10 }}>
                  {(confidence * 100).toFixed(0)}%
                </span>
                {t.claudeResult && (
                  <span style={{ marginLeft: 6, color: '#00cfff', fontSize: 10 }}>
                    {t.claudeResult.action}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function LogPanel({ log }) {
  return (
    <div className="panel panel-log">
      <div className="panel-title">Event Log</div>
      <div className="log-scroll" id="log-scroll">
        {log.length === 0 && <div className="muted">No events yet</div>}
        {[...log].reverse().map((e, i) => (
          <div key={i} className="log-row">
            <span className="log-time">{e.time}</span>
            <span className="log-msg">{e.msg}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Controls({ running, onStart, onStop, onReset, tick }) {
  return (
    <div className="controls">
      <button className="btn btn-primary" onClick={running ? onStop : onStart}>
        {running ? '⏸ Pause' : '▶ Start'}
      </button>
      <button className="btn btn-secondary" onClick={onReset}>↺ Reset</button>
      <span className="tick-counter">tick {tick}</span>
    </div>
  );
}
