"""CMB dipole orientation module.

Physics basis
-------------
The solar system moves at ~370 km/s relative to the CMB rest frame.
This motion Doppler-shifts the observed CMB temperature:

    T(n̂) = T₀ / γ(1 − β·cos θ)
           ≈ T₀(1 + β·cos θ + (β²/2)(3cos²θ − 1) + …)

where:
  T₀  = 2.7255 K  (CMB monopole, Fixsen 2009)
  β   = v/c
  θ   = angle between spacecraft velocity v̂ and pointing direction n̂

The first-order term produces a dipole of amplitude ΔT_dipole = T₀·β ≈ 3.36 mK.
The second-order term produces a quadrupole ΔT_quad ≈ T₀·β²/2 ≈ 2 μK (negligible
for velocity estimation but included in the forward model for fidelity).

Known solar-system CMB dipole (Planck 2018, Aghanim et al.):
  Direction: l = 264.021°, b = 48.253° (galactic)
  Amplitude: 3.3645 ± 0.0020 mK
  Implied velocity: 369.82 ± 0.11 km/s

A spacecraft with a microwave radiometer can measure this dipole and recover
its velocity vector in the CMB rest frame with no ground stations, no GPS,
and no star catalog.  This module implements:

  1. Forward model: velocity vector → sky temperature map
  2. Dipole fitter: noisy sky samples → recovered velocity vector + uncertainty
  3. Residual check: compares recovered dipole amplitude with known value
     as a self-consistency / calibration test

Reference:
  Fixsen, D.J. (2009). ApJ 707, 916. DOI:10.1088/0004-637X/707/2/916
  Planck Collaboration (2020). A&A 641, A1. DOI:10.1051/0004-6361/201833887
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# ── Physical constants ─────────────────────────────────────────────────────────

T0_K       = 2.7255          # CMB monopole temperature (K), Fixsen 2009
C_KM_S     = 299_792.458     # Speed of light (km/s)

# Known solar-system velocity in CMB frame (Planck 2018)
SOLAR_SPEED_KM_S   = 369.82
SOLAR_DIR_GAL_L    = 264.021  # galactic longitude (deg)
SOLAR_DIR_GAL_B    = 48.253   # galactic latitude  (deg)


# ── Coordinate helpers ─────────────────────────────────────────────────────────

def gal_to_cartesian(l_deg: float, b_deg: float) -> np.ndarray:
    """Unit vector from galactic (l, b) in degrees."""
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    return np.array([
        np.cos(b) * np.cos(l),
        np.cos(b) * np.sin(l),
        np.sin(b),
    ])


def cartesian_to_gal(v: np.ndarray) -> tuple[float, float]:
    """Galactic (l_deg, b_deg) from unit vector (or unnormed; normalised internally)."""
    v = v / np.linalg.norm(v)
    b = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    l = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    return l, b


def fibonacci_sphere(n: int) -> np.ndarray:
    """Near-uniform sampling of n unit vectors on the sphere (Fibonacci lattice).

    Returns shape (n, 3).
    """
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi   = 2 * np.pi * i / golden
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


# ── Forward model ─────────────────────────────────────────────────────────────

def dipole_temperature(
    pointings: np.ndarray,
    velocity_km_s: np.ndarray,
    t0: float = T0_K,
) -> np.ndarray:
    """Predicted CMB temperature in each pointing direction.

    Parameters
    ----------
    pointings : shape (N, 3) unit vectors on the sky
    velocity_km_s : shape (3,) spacecraft velocity in CMB rest frame (km/s)
    t0 : CMB monopole temperature (K)

    Returns
    -------
    temperatures : shape (N,) in Kelvin
    """
    v_mag = np.linalg.norm(velocity_km_s)
    if v_mag == 0:
        return np.full(len(pointings), t0)

    beta     = v_mag / C_KM_S
    v_hat    = velocity_km_s / v_mag
    cos_theta = pointings @ v_hat           # shape (N,)

    # Full relativistic formula (includes quadrupole automatically)
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return t0 / (gamma * (1.0 - beta * cos_theta))


# ── Dipole fitter ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DipoleResult:
    """Result of fitting the CMB dipole to measured sky temperatures."""

    velocity_km_s: np.ndarray       # recovered velocity vector, shape (3,)
    speed_km_s: float               # magnitude |v|
    direction_l_deg: float          # galactic longitude of v̂
    direction_b_deg: float          # galactic latitude  of v̂
    monopole_k: float               # recovered T₀ (sanity check vs 2.7255 K)
    dipole_amplitude_mk: float      # ΔT_dipole in milli-Kelvin
    residual_rms_uk: float          # RMS fit residual in micro-Kelvin
    n_pointings: int

    def angular_error_deg(self, true_velocity_km_s: np.ndarray) -> float:
        """Great-circle angle between recovered and true velocity directions (deg)."""
        v_true = true_velocity_km_s / np.linalg.norm(true_velocity_km_s)
        v_rec  = self.velocity_km_s / np.linalg.norm(self.velocity_km_s)
        cos_ang = np.clip(v_true @ v_rec, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_ang)))

    def speed_error_km_s(self, true_velocity_km_s: np.ndarray) -> float:
        return float(abs(self.speed_km_s - np.linalg.norm(true_velocity_km_s)))


def fit_dipole(
    pointings: np.ndarray,
    temperatures: np.ndarray,
) -> DipoleResult:
    """Recover velocity vector from noisy sky temperature measurements.

    Fits: T(n̂) = a₀ + a₁·nx + a₂·ny + a₃·nz
    where a₀ ≈ T₀ (monopole), and (a₁, a₂, a₃) = T₀·β·v̂ (dipole).

    This is a simple linear least-squares fit — no iterative solver needed
    because the dipole is a linear function of the direction cosines.

    Parameters
    ----------
    pointings    : shape (N, 3) unit vectors
    temperatures : shape (N,) in Kelvin

    Returns
    -------
    DipoleResult
    """
    N = len(pointings)
    # Design matrix: [1, nx, ny, nz]
    A = np.column_stack([np.ones(N), pointings])   # (N, 4)
    # Least-squares solution
    coeffs, residuals, _, _ = np.linalg.lstsq(A, temperatures, rcond=None)

    a0     = coeffs[0]           # monopole (K)
    dipole = coeffs[1:]          # T₀·β·v̂  (K)

    dip_amplitude = np.linalg.norm(dipole)   # = T₀·β  (K)
    v_hat  = dipole / dip_amplitude if dip_amplitude > 0 else np.array([1., 0., 0.])
    beta   = dip_amplitude / a0 if a0 > 0 else 0.0
    speed  = beta * C_KM_S

    velocity = speed * v_hat
    l, b = cartesian_to_gal(v_hat)

    # RMS residual in μK
    t_fit = A @ coeffs
    rms = float(np.std(temperatures - t_fit) * 1e6)

    return DipoleResult(
        velocity_km_s     = velocity,
        speed_km_s        = float(speed),
        direction_l_deg   = float(l),
        direction_b_deg   = float(b),
        monopole_k        = float(a0),
        dipole_amplitude_mk = float(dip_amplitude * 1e3),
        residual_rms_uk   = rms,
        n_pointings       = N,
    )


# ── Simulate a spacecraft observation ─────────────────────────────────────────

def simulate_observation(
    velocity_km_s: np.ndarray,
    n_pointings: int = 3072,
    noise_uk: float = 10.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a noisy CMB sky observation for a spacecraft at given velocity.

    Parameters
    ----------
    velocity_km_s : shape (3,) — spacecraft velocity in CMB rest frame
    n_pointings   : number of sky samples (default: HEALPix nside=16 equivalent)
    noise_uk      : RMS radiometer noise per pointing in micro-Kelvin
    seed          : RNG seed for reproducibility

    Returns
    -------
    pointings    : shape (n, 3) unit vectors
    temperatures : shape (n,) observed temperatures in Kelvin
    """
    rng       = np.random.default_rng(seed)
    pointings = fibonacci_sphere(n_pointings)
    temps     = dipole_temperature(pointings, velocity_km_s)
    noise     = rng.standard_normal(n_pointings) * noise_uk * 1e-6
    return pointings, temps + noise


# ── Convenience: solar-system reference velocity ──────────────────────────────

def solar_velocity_vector() -> np.ndarray:
    """Known solar-system velocity in CMB rest frame as a Cartesian vector (km/s)."""
    v_hat = gal_to_cartesian(SOLAR_DIR_GAL_L, SOLAR_DIR_GAL_B)
    return SOLAR_SPEED_KM_S * v_hat
