"""X-ray pulsar navigation (XNAV) — position estimation from pulse timing.

Physics basis
-------------
A pulsar emits pulses with extraordinary regularity.  A spacecraft at
position r (relative to the solar-system barycenter) observes pulse
arrivals delayed or advanced relative to the barycentric prediction by:

    Δt = -(n̂_p · r) / c

where n̂_p is the unit vector from the barycenter toward the pulsar.
With observations of N ≥ 3 pulsars (non-coplanar), this gives a linear
system for r:

    [n̂_1]              [Δt₁]
    [n̂_2]  · r = −c ·  [Δt₂]
    [...]               [...]

Solved by weighted least squares, giving position in the barycentric
coordinate system.

Timing accuracy: millisecond pulsars achieve TOA residuals of ~100 ns
(equivalent to ~30 m position uncertainty per observation per pulsar).
Over a practical integration period, accumulated timing gives ~1–5 km
(1σ) position accuracy.  NASA NICER demonstrated ~5 km onboard XNAV
(Mitchell et al., 2018, IEEE Aerosp. Conf.).

Pulsar catalogue (subset of millisecond pulsars used for XNAV)
--------------------------------------------------------------
Using J2000 equatorial coordinates, converted to Cartesian unit vectors.

Reference:
  Mitchell, J.W. et al. (2018). "SEXTANT X-Ray Pulsar Navigation
  Demonstration: Additional On-Orbit Results." IEEE Aerosp. Conf.
  DOI:10.1109/AERO.2018.8396375
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence


# ── Pulsar catalogue ──────────────────────────────────────────────────────────
# Fields: name, RA (deg), Dec (deg), period (ms), timing noise σ_TOA (μs)
# Positions are approximate J2000 equatorial; adequate for navigation.

_CATALOGUE_RAW = [
    ("PSR J0437-4715", 69.316,   -47.253,  5.757,  0.10),
    ("PSR J1939+2134", 294.910,  +21.583,  1.558,  0.05),
    ("PSR J0030+0451",   7.606,   +4.858,  4.865,  0.20),
    ("PSR J2124-3358", 321.027,  -33.975,  4.931,  0.30),
    ("PSR B0531+21",    83.633,  +22.015, 33.085, 80.00),   # Crab — noisier
    ("PSR J1713+0747", 258.467,   +7.793,  4.570,  0.03),
    ("PSR B1855+09",   284.580,   +9.725,  5.362,  1.50),
]

C_KM_S  = 299_792.458     # km/s
C_M_S   = C_KM_S * 1e3    # m/s
C_KM_AU = C_KM_S / 1.495_978_707e8  # AU/s


@dataclass(frozen=True)
class Pulsar:
    name: str
    ra_deg: float
    dec_deg: float
    period_ms: float
    sigma_toa_us: float    # 1-σ timing noise per observation (microseconds)

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from barycenter toward pulsar (J2000 equatorial Cartesian)."""
        ra  = np.radians(self.ra_deg)
        dec = np.radians(self.dec_deg)
        return np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ])


CATALOGUE: list[Pulsar] = [
    Pulsar(*row) for row in _CATALOGUE_RAW
]


# ── Forward model ─────────────────────────────────────────────────────────────

def predicted_toa_offset(
    pulsar: Pulsar,
    spacecraft_pos_km: np.ndarray,
) -> float:
    """Time-of-arrival offset (seconds) due to spacecraft displacement.

    Δt = -(n̂_p · r) / c
    Positive Δt → pulse arrives late (spacecraft moved away from pulsar).
    """
    return float(-(pulsar.direction @ spacecraft_pos_km) / C_KM_S)


# ── Simulated observation ─────────────────────────────────────────────────────

def simulate_toa_residuals(
    spacecraft_pos_km: np.ndarray,
    pulsars: Sequence[Pulsar] | None = None,
    seed: int = 0,
) -> tuple[list[Pulsar], np.ndarray]:
    """Simulate noisy TOA residuals for a spacecraft at position r.

    Parameters
    ----------
    spacecraft_pos_km : shape (3,) position in km (barycentric J2000)
    pulsars           : subset of CATALOGUE to use; defaults to all 7
    seed              : RNG seed

    Returns
    -------
    pulsars_used : list[Pulsar]
    residuals_s  : shape (N,) TOA residuals in seconds
    """
    if pulsars is None:
        pulsars = CATALOGUE
    rng = np.random.default_rng(seed)
    residuals = []
    for p in pulsars:
        true_offset = predicted_toa_offset(p, spacecraft_pos_km)
        noise_s = rng.standard_normal() * p.sigma_toa_us * 1e-6
        residuals.append(true_offset + noise_s)
    return list(pulsars), np.array(residuals)


# ── Position estimator ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class XNAVResult:
    position_km: np.ndarray          # estimated position, shape (3,)
    covariance_km2: np.ndarray       # 3×3 position covariance matrix
    position_error_km: float         # |r_est − r_true| (set by caller if known)
    n_pulsars: int
    condition_number: float          # geometry quality (lower = better)


def estimate_position(
    pulsars: Sequence[Pulsar],
    residuals_s: np.ndarray,
) -> XNAVResult:
    """Weighted least-squares position estimate from TOA residuals.

    Model: Δt_i = -(n̂_i · r) / c   for each pulsar i

    Rearranged: n̂_i · r = -c · Δt_i

    Solved via WLS with weights w_i = 1/σ²_i.

    Parameters
    ----------
    pulsars     : N pulsars (must be ≥ 3, non-coplanar for full 3-D fix)
    residuals_s : N TOA residuals in seconds

    Returns
    -------
    XNAVResult
    """
    N = len(pulsars)
    # Design matrix: rows are pulsar direction vectors
    H = np.array([p.direction for p in pulsars])          # (N, 3)
    y = -C_KM_S * residuals_s                              # (N,)  km

    # Weight matrix
    sigma_km = np.array([p.sigma_toa_us * 1e-6 * C_KM_S for p in pulsars])
    W = np.diag(1.0 / sigma_km**2)

    # WLS: r_hat = (HᵀWH)⁻¹ HᵀW y
    HtW  = H.T @ W          # (3, N)
    HtWH = HtW @ H          # (3, 3)
    cov  = np.linalg.inv(HtWH)
    r_hat = cov @ (HtW @ y)

    cond = float(np.linalg.cond(HtWH))

    return XNAVResult(
        position_km     = r_hat,
        covariance_km2  = cov,
        position_error_km = 0.0,   # caller fills in after comparing to truth
        n_pulsars       = N,
        condition_number = cond,
    )
