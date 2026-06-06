import React, { useState, useCallback } from 'react';
import LatticeCanvas from './components/LatticeCanvas.jsx';
import { AgentPanel, ThreatPanel, LogPanel, Controls } from './components/Panels.jsx';
import LycanRunner from './components/LycanRunner.jsx';
import { useSimulation } from './hooks/useSimulation.js';

const TABS = ['Simulation', 'LYCAN Scenarios', 'Reports'];

function ReportsTab({ reports }) {
  if (reports.length === 0) {
    return (
      <div className="report-placeholder">
        <p className="muted">No reports yet. Run a LYCAN scenario and click "Generate Report".</p>
      </div>
    );
  }
  return (
    <div className="reports-list">
      {reports.map((r, i) => (
        <div key={i} className="report-box">
          <div className="report-header">Report #{i + 1}</div>
          <pre className="report-text">{r}</pre>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [tab, setTab]       = useState(0);
  const [reports, setReports] = useState([]);

  const { display, nodes, edges, start, stop, reset } = useSimulation();

  const handleReportReady = useCallback((text) => {
    setReports(prev => [text, ...prev]);
  }, []);

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="app-title">QKC Lattice Threat Analysis</div>
        <nav className="tab-bar">
          {TABS.map((t, i) => (
            <button
              key={t}
              className={`tab-btn ${tab === i ? 'active' : ''}`}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      {/* Simulation Tab */}
      {tab === 0 && (
        <div className="sim-layout">
          <div className="sim-left">
            <LatticeCanvas nodes={nodes} edges={edges} display={display} />
            <Controls
              running={display.running}
              onStart={start}
              onStop={stop}
              onReset={reset}
              tick={display.tick}
            />
          </div>
          <div className="sim-right">
            <AgentPanel agents={display.agents} />
            <ThreatPanel threats={display.threats} />
            <LogPanel log={display.log} />
          </div>
        </div>
      )}

      {/* LYCAN Tab */}
      {tab === 1 && (
        <div className="lycan-layout">
          <LycanRunner onReportReady={handleReportReady} />
        </div>
      )}

      {/* Reports Tab */}
      {tab === 2 && (
        <div className="lycan-layout">
          <ReportsTab reports={reports} />
        </div>
      )}
    </div>
  );
}
