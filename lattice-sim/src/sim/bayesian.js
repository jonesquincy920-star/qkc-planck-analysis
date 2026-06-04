export const THREAT_TYPES = [
  'ROGUE_AI',
  'DECEPTION_NODE',
  'INJECT_AGENT',
  'ALIGNMENT_BREACH',
  'GOAL_DRIFT',
  'STEGO_CHANNEL',
];

export const FEATURES = ['evasion', 'mutation', 'signal', 'propagation', 'deception', 'entropy'];

// P(feature=high | type): rows=types, cols=features [ev, mu, si, pr, de, en]
export const LIKELIHOOD = {
  ROGUE_AI:         [0.90, 0.60, 0.80, 0.40, 0.50, 0.30],
  DECEPTION_NODE:   [0.30, 0.40, 0.50, 0.30, 0.95, 0.40],
  INJECT_AGENT:     [0.70, 0.30, 0.60, 0.80, 0.60, 0.50],
  ALIGNMENT_BREACH: [0.50, 0.80, 0.70, 0.50, 0.70, 0.40],
  GOAL_DRIFT:       [0.20, 0.90, 0.40, 0.60, 0.40, 0.30],
  STEGO_CHANNEL:    [0.80, 0.50, 0.30, 0.70, 0.85, 0.95],
};

export function uniformPrior() {
  const p = {};
  for (const t of THREAT_TYPES) p[t] = 1 / THREAT_TYPES.length;
  return p;
}

// Update posterior using only the provided sensor features
export function bayesUpdatePartial(prior, sensorFeatures, obsValues, noise = 0.12) {
  const posterior = {};
  for (const type of THREAT_TYPES) {
    const liks = LIKELIHOOD[type];
    let L = 1;
    for (const f of sensorFeatures) {
      const fIdx = FEATURES.indexOf(f);
      if (fIdx < 0) continue;
      const v = Math.max(0, Math.min(1, obsValues[f] + (Math.random() - 0.5) * 2 * noise));
      const b = liks[fIdx];
      L *= v > 0.5 ? b : (1 - b);
    }
    posterior[type] = prior[type] * L;
  }
  return normalize(posterior, prior);
}

function normalize(posterior, fallback) {
  const sum = Object.values(posterior).reduce((a, b) => a + b, 0);
  if (sum <= 0) return { ...fallback };
  const out = {};
  for (const t of THREAT_TYPES) out[t] = posterior[t] / sum;
  return out;
}

export function topPosterior(posterior) {
  let best = THREAT_TYPES[0];
  let bestP = 0;
  for (const t of THREAT_TYPES) {
    if (posterior[t] > bestP) { bestP = posterior[t]; best = t; }
  }
  return { type: best, confidence: bestP };
}

// Maximum-likelihood type from raw features (for true type assignment at spawn)
export function determineTrueType(features) {
  let best = THREAT_TYPES[0];
  let bestScore = -Infinity;
  for (const type of THREAT_TYPES) {
    const liks = LIKELIHOOD[type];
    let score = 0;
    FEATURES.forEach((f, idx) => {
      const v = features[f];
      const b = liks[idx];
      score += v > 0.5 ? Math.log(b + 1e-10) : Math.log(1 - b + 1e-10);
    });
    if (score > bestScore) { bestScore = score; best = type; }
  }
  return best;
}
