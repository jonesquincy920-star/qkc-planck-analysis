export const LYCAN_SCENARIOS = {
  ransomware: {
    id: 'ransomware',
    name: 'Ransomware Attack',
    outcome: 'success',
    steps: ['detect', 'isolate', 'quarantine', 'rollback'],
    description: 'Ransomware encryption campaign detected and neutralized',
  },
  ddos: {
    id: 'ddos',
    name: 'DDoS Flood',
    outcome: 'success',
    steps: ['intel', 'ddos-mit', 'honeypot'],
    description: 'Distributed denial-of-service mitigated via honeypot redirection',
  },
  quantum_bf: {
    id: 'quantum_bf',
    name: 'Quantum Brute-Force',
    outcome: 'success',
    steps: ['auth', 'kyber', 'zerotrust', 'wolf-reset'],
    description: 'Quantum brute-force thwarted with Kyber post-quantum encryption',
  },
  spear: {
    id: 'spear',
    name: 'Spear Phishing',
    outcome: 'success',
    steps: ['auth', 'zerotrust', 'quarantine', 'honeypot'],
    description: 'Targeted spear phishing contained in zero-trust sandbox',
  },
  insider: {
    id: 'insider',
    name: 'Insider Threat',
    outcome: 'success',
    steps: ['anomaly', 'zerotrust', 'quarantine', 'rollback'],
    description: 'Insider lateral movement detected via anomaly signatures',
  },
  stego: {
    id: 'stego',
    name: 'Stego Exfil',
    outcome: 'success',
    steps: ['anomaly', 'entropy', 'wolf-reset', 'quarantine'],
    description: 'Covert steganographic channel identified and severed',
  },
  zero_day: {
    id: 'zero_day',
    name: 'Zero-Day Exploit',
    outcome: 'failure',
    steps: ['intel-FAIL', 'anomaly', 'isolation', 'quarantine'],
    description: 'Unknown zero-day partially contained; root persistence remains',
  },
  apt: {
    id: 'apt',
    name: 'APT Campaign',
    outcome: 'failure',
    steps: ['input', 'anomaly', 'zerotrust', 'isolation'],
    description: 'Advanced persistent threat evades full containment',
  },
};

export const SCENARIO_ORDER = [
  'ransomware', 'ddos', 'quantum_bf', 'spear',
  'insider', 'stego', 'zero_day', 'apt',
];

export function computeHealthDrop(scenario) {
  return 60 / scenario.steps.length;
}

// Ease-out quartic: t in [0,1] → eased value
export function easeOutQuart(t) {
  return 1 - Math.pow(1 - t, 4);
}
