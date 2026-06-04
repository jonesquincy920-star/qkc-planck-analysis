export function detectStego(features) {
  const { entropy, deception, signal } = features;
  const sig = entropy > 0.70 && deception > 0.60 && signal < 0.40;
  const prob = sig ? entropy * deception * 0.90 : entropy * 0.20;
  return { sig, prob };
}
