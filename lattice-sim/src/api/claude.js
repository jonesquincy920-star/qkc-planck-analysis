const API_KEY = import.meta.env.VITE_ANTHROPIC_API_KEY;
const API_URL = 'https://api.anthropic.com/v1/messages';
const MODEL = 'claude-sonnet-4-20250514';

function headers() {
  return {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY,
    'anthropic-version': '2023-06-01',
    'anthropic-dangerous-allow-browser': 'true',
  };
}

export async function classifyThreat(threat) {
  if (!API_KEY) return null;

  const featStr = Object.entries(threat.features)
    .map(([k, v]) => `${k}=${v.toFixed(3)}`)
    .join(', ');

  const prompt = `You are an AI threat classification system. Analyze this threat node in a lattice security grid.

Threat ID: ${threat.id}
Features: ${featStr}
Bayesian posterior: ${Object.entries(threat.posterior).map(([k, v]) => `${k}=${(v * 100).toFixed(1)}%`).join(', ')}
Strength: ${threat.strength?.toFixed(3) ?? 'unknown'}
Status: ${threat.status}

Respond ONLY with JSON matching this schema exactly (no extra text or markdown):
{"threat_type":"<ROGUE_AI|DECEPTION_NODE|INJECT_AGENT|ALIGNMENT_BREACH|GOAL_DRIFT|STEGO_CHANNEL>","confidence":<0-1>,"reasoning":"<one sentence>","priority":"<LOW|MEDIUM|HIGH|CRITICAL>","action":"<MONITOR|CONTAIN|DESTROY|ISOLATE>"}`;

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ model: MODEL, max_tokens: 1000, messages: [{ role: 'user', content: prompt }] }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const text = data.content[0].text.trim();
    const match = text.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);
  } catch {
    // fallback to Bayes-only
  }
  return null;
}

export async function generateReport(scenarioData) {
  if (!API_KEY) {
    return '**Report generation unavailable** — set `VITE_ANTHROPIC_API_KEY` in `.env` and restart the dev server.';
  }

  const { scenario, outcome, log } = scenarioData;
  const logText = log.slice(0, 40).map(e => `[${e.time}] ${e.msg}`).join('\n');

  const prompt = `Generate a concise cybersecurity incident report.

Scenario: ${scenario.name}
Description: ${scenario.description}
Outcome: ${outcome.toUpperCase()}
Steps executed: ${scenario.steps.join(' → ')}
Event log:
${logText}

Use exactly these section headings (no extra preamble):
## Executive Summary
## Attack & Defense Timeline
## Vulnerability Analysis
## Performance Analysis
## Recommendations
Use "- " prefix for each actionable recommendation item.`;

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ model: MODEL, max_tokens: 2000, messages: [{ role: 'user', content: prompt }] }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.content[0].text;
  } catch (err) {
    return `**Report generation failed:** ${err.message}`;
  }
}
