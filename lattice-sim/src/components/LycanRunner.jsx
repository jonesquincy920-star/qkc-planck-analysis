import React, { useState, useRef, useEffect, useCallback } from 'react';
import { LYCAN_SCENARIOS, SCENARIO_ORDER, computeHealthDrop, easeOutQuart } from '../sim/lycan.js';
import { generateReport } from '../api/claude.js';

const STEP_DELAY_MS   = 900;
const RECOVERY_MS     = 800;
const RECOVERY_FRAMES = 40;

function HealthBar({ health, outcome }) {
  const color = outcome === 'success'
    ? health > 60 ? '#00ff9d' : health > 30 ? '#fbbf24' : '#f43f5e'
    : '#f43f5e';
  return (
    <div className="health-bar-wrap">
      <div className="health-bar-bg">
        <div
          className="health-bar-fill"
          style={{ width: `${Math.max(0, health)}%`, background: color, transition: 'width 0.3s ease' }}
        />
      </div>
      <span className="health-label">{Math.round(health)}%</span>
    </div>
  );
}

function StepTimeline({ steps, currentStep, outcome }) {
  return (
    <div className="step-timeline">
      {steps.map((s, i) => {
        const done    = i < currentStep;
        const active  = i === currentStep;
        const failed  = outcome === 'failure' && s.includes('FAIL');
        return (
          <div key={i} className={`step-chip ${done ? 'done' : ''} ${active ? 'active' : ''} ${failed ? 'failed' : ''}`}>
            {s}
          </div>
        );
      })}
    </div>
  );
}

export default function LycanRunner({ onReportReady }) {
  const [selected, setSelected]     = useState(null);
  const [running, setRunning]        = useState(false);
  const [health, setHealth]          = useState(100);
  const [stepIdx, setStepIdx]        = useState(-1);
  const [done, setDone]              = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [report, setReport]          = useState('');
  const [simLog]                     = useState(() => []);

  const scenario = selected ? LYCAN_SCENARIOS[selected] : null;

  const resetState = useCallback(() => {
    setRunning(false);
    setHealth(100);
    setStepIdx(-1);
    setDone(false);
    setReport('');
    simLog.length = 0;
  }, [simLog]);

  const handleSelect = useCallback((id) => {
    setSelected(id);
    resetState();
  }, [resetState]);

  const runScenario = useCallback(() => {
    if (!scenario || running) return;
    resetState();
    setRunning(true);

    const drop      = computeHealthDrop(scenario);
    const steps     = scenario.steps;
    const outcome   = scenario.outcome;
    const startTime = Date.now();
    let   idx       = 0;
    let   currentH  = 100;

    function log(msg) {
      const t = ((Date.now() - startTime) / 1000).toFixed(1);
      simLog.push({ time: `+${t}s`, msg });
    }

    log(`Scenario started: ${scenario.name}`);

    const interval = setInterval(() => {
      if (idx >= steps.length) {
        clearInterval(interval);

        if (outcome === 'success') {
          // Ease-out quart recovery over RECOVERY_MS
          let frame = 0;
          const startH = currentH;
          const recovery = setInterval(() => {
            frame++;
            const t     = frame / RECOVERY_FRAMES;
            const eased = easeOutQuart(t);
            const h     = startH + (100 - startH) * eased;
            setHealth(h);
            if (frame >= RECOVERY_FRAMES) {
              clearInterval(recovery);
              setHealth(100);
              log('System health fully recovered');
              setDone(true);
              setRunning(false);
            }
          }, RECOVERY_MS / RECOVERY_FRAMES);
        } else {
          const floor = 10 + Math.random() * 10;
          setHealth(floor);
          currentH = floor;
          log(`Scenario ended: FAILURE — residual threat at ${floor.toFixed(0)}%`);
          setDone(true);
          setRunning(false);
        }
        return;
      }

      currentH -= drop;
      const h = Math.max(0, currentH);
      setHealth(h);
      setStepIdx(idx);
      log(`Step [${steps[idx]}] — health ${h.toFixed(0)}%`);
      idx++;
    }, STEP_DELAY_MS);
  }, [scenario, running, resetState, simLog]);

  const handleReport = useCallback(async () => {
    if (!scenario || !done) return;
    setReportLoading(true);
    const text = await generateReport({ scenario, outcome: scenario.outcome, log: simLog });
    setReport(text);
    setReportLoading(false);
    if (onReportReady) onReportReady(text);
  }, [scenario, done, simLog, onReportReady]);

  return (
    <div className="lycan-root">
      {/* Scenario grid */}
      <div className="scenario-grid">
        {SCENARIO_ORDER.map(id => {
          const sc = LYCAN_SCENARIOS[id];
          return (
            <button
              key={id}
              className={`scenario-card ${selected === id ? 'selected' : ''} ${sc.outcome}`}
              onClick={() => handleSelect(id)}
            >
              <div className="sc-name">{sc.name}</div>
              <div className={`sc-outcome ${sc.outcome}`}>{sc.outcome.toUpperCase()}</div>
              <div className="sc-steps">{sc.steps.length} steps</div>
            </button>
          );
        })}
      </div>

      {/* Active scenario */}
      {scenario && (
        <div className="scenario-detail">
          <div className="sd-header">
            <span className="sd-title">{scenario.name}</span>
            <span className={`sd-badge ${scenario.outcome}`}>{scenario.outcome}</span>
          </div>
          <p className="sd-desc">{scenario.description}</p>

          <HealthBar health={health} outcome={scenario.outcome} />
          <StepTimeline steps={scenario.steps} currentStep={stepIdx} outcome={scenario.outcome} />

          <div className="sd-controls">
            <button
              className="btn btn-primary"
              onClick={runScenario}
              disabled={running}
            >
              {running ? 'Running…' : done ? '↺ Run Again' : '▶ Run Scenario'}
            </button>

            {done && (
              <button
                className="btn btn-secondary"
                onClick={handleReport}
                disabled={reportLoading}
              >
                {reportLoading ? 'Generating…' : '📄 Generate Report'}
              </button>
            )}
          </div>

          {report && (
            <div className="report-box">
              <pre className="report-text">{report}</pre>
            </div>
          )}
        </div>
      )}

      {!scenario && (
        <div className="muted" style={{ marginTop: 24, textAlign: 'center' }}>
          Select a scenario above to begin
        </div>
      )}
    </div>
  );
}
