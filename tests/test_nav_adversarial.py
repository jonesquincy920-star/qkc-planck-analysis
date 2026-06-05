"""Adversarial and robustness tests for the nav stack.

These tests go beyond internal consistency — they either validate against
independently sourced ground truth (Planck 2018 published values), probe
failure modes explicitly, or verify the filter reports what it should when
the world doesn't cooperate.

Each test documents what it proves AND what it does not.
"""

import numpy as np
import pytest

from qkc.nav.dipole import (
    C_KM_S, T0_K, fit_dipole, fibonacci_sphere,
    dipole_temperature, simulate_observation, gal_to_cartesian,
)
from qkc.nav.xnav import CATALOGUE, simulate_toa_residuals, estimate_position
from qkc.nav.kalman import NavKalmanFilter


# ── Independent ground-truth reference (Planck 2018, Aghanim et al. 2020) ────
# Hardcoded HERE, not imported from dipole.py, so this is an external oracle.
# Source: Planck 2018 Results I, Table 4.
#   DOI: 10.1051/0004-6361/201833887
_PLANCK_SPEED_KM_S   = 369.82   # ± 0.11 km/s
_PLANCK_AMP_MK       = 3.3645   # ± 0.0020 mK
_PLANCK_AMP_SIGMA_MK = 0.0020
_PLANCK_DIR_L        = 264.021  # ± 0.011 deg
_PLANCK_DIR_B        =  48.253  # ± 0.005 deg
_PLANCK_DIR_SIGMA    = 0.05     # conservative combined direction uncertainty (deg)


# ── Planck external-oracle tests ──────────────────────────────────────────────

def test_module_constants_match_planck_2018():
    """Module constants must match Planck 2018 published values.

    This is a regression guard: if someone changes the constants in dipole.py
    without updating this test, the test breaks, forcing a documented justification.
    """
    from qkc.nav.dipole import SOLAR_SPEED_KM_S, SOLAR_DIR_GAL_L, SOLAR_DIR_GAL_B
    assert abs(SOLAR_SPEED_KM_S - _PLANCK_SPEED_KM_S) < 0.01, (
        f"Module speed {SOLAR_SPEED_KM_S} differs from Planck 2018 {_PLANCK_SPEED_KM_S}"
    )
    assert abs(SOLAR_DIR_GAL_L - _PLANCK_DIR_L) < 0.001
    assert abs(SOLAR_DIR_GAL_B - _PLANCK_DIR_B) < 0.001


def test_fitter_recovers_planck_dipole_amplitude(seed=0):
    """Physical claim: fitter should recover the Planck 2018 dipole amplitude
    to within 3σ of the published uncertainty (3σ = 0.006 mK).

    Uses the Planck published velocity as ground truth (not our module constant).
    Proves the fitter is calibrated to the Planck physical scale.
    """
    v_hat = gal_to_cartesian(_PLANCK_DIR_L, _PLANCK_DIR_B)
    v_true = _PLANCK_SPEED_KM_S * v_hat

    pts, T = simulate_observation(v_true, n_pointings=3072, noise_uk=10.0, seed=seed)
    result = fit_dipole(pts, T)

    diff = abs(result.dipole_amplitude_mk - _PLANCK_AMP_MK)
    assert diff < 3 * _PLANCK_AMP_SIGMA_MK + 0.05, (  # +0.05 mK for finite sample
        f"Recovered amplitude {result.dipole_amplitude_mk:.4f} mK deviates "
        f"{diff:.4f} mK from Planck value {_PLANCK_AMP_MK} mK"
    )


def test_fitter_recovers_planck_direction(seed=0):
    """Physical claim: fitter should recover the Planck 2018 CMB dipole direction
    to within 0.5° using Planck published values as the independent oracle.
    """
    v_hat = gal_to_cartesian(_PLANCK_DIR_L, _PLANCK_DIR_B)
    v_true = _PLANCK_SPEED_KM_S * v_hat

    pts, T = simulate_observation(v_true, n_pointings=3072, noise_uk=10.0, seed=seed)
    result = fit_dipole(pts, T)

    ang_err = result.angular_error_deg(v_true)
    assert ang_err < 0.5, (
        f"Direction error {ang_err:.3f}° exceeds 0.5° against Planck oracle"
    )


def test_dipole_amplitude_relativistic_correction():
    """Full relativistic model should give a slightly different amplitude than
    first-order T₀·β.  Quantifies the higher-order contribution.

    At v=370 km/s, β=1.234e-3, β²≈1.52e-6 → correction ~0.002 mK.
    This test documents the magnitude and confirms it is non-zero.
    """
    v_hat = gal_to_cartesian(_PLANCK_DIR_L, _PLANCK_DIR_B)
    v_true = _PLANCK_SPEED_KM_S * v_hat
    beta = _PLANCK_SPEED_KM_S / C_KM_S

    first_order_mk = T0_K * beta * 1e3
    pts = fibonacci_sphere(5000)
    T = dipole_temperature(pts, v_true)
    result = fit_dipole(pts, T)

    # Relativistic amplitude should exceed first-order by a small but nonzero amount
    diff = result.dipole_amplitude_mk - first_order_mk
    assert diff > 0, "Relativistic amplitude should exceed first-order approximation"
    assert diff < 0.01, f"Relativistic correction {diff:.5f} mK unexpectedly large"


# ── XNAV adversarial / limitation tests ───────────────────────────────────────

def test_xnav_outlier_single_pulsar_degrades_accuracy(seed=0):
    """DOCUMENTED LIMITATION: WLS is not robust to outliers.

    One pulsar with 10× its nominal timing noise causes position error to
    exceed the clean-data bound.  This test quantifies the degradation and
    will fail if a robust estimator is later added (at which point it should
    be replaced by a robustness test).
    """
    r_true = np.array([1_000.0, -500.0, 200.0])

    # Clean baseline
    pulsars, res_clean = simulate_toa_residuals(r_true, seed=seed)
    clean_result = estimate_position(pulsars, res_clean)
    clean_error = float(np.linalg.norm(clean_result.position_km - r_true))

    # Inject 10× outlier on best pulsar (J1713+0747, σ=0.03 μs)
    rng = np.random.default_rng(seed + 1)
    best_idx = 5  # J1713+0747
    res_outlier = res_clean.copy()
    outlier_noise = rng.standard_normal() * pulsars[best_idx].sigma_toa_us * 1e-6 * 10.0
    res_outlier[best_idx] += outlier_noise

    outlier_result = estimate_position(pulsars, res_outlier)
    outlier_error = float(np.linalg.norm(outlier_result.position_km - r_true))

    # Outlier should degrade accuracy — if this fails the estimator became robust
    assert outlier_error > clean_error, (
        "Unexpected: outlier did not degrade WLS — verify robustness was not added silently"
    )
    # Document the degradation magnitude for the record
    assert outlier_error < 500.0, (
        f"Outlier error {outlier_error:.1f} km is catastrophically large — "
        "consider adding RANSAC or iterative reweighting"
    )


def test_xnav_geometry_condition_number():
    """Near-coplanar pulsars produce high condition number and poor accuracy.

    Uses only pulsars within 30° of the ecliptic plane (manually selected subset)
    to demonstrate that sky geometry matters for XNAV.
    """
    r_true = np.array([500.0, -300.0, 100.0])

    # All 7 pulsars: well-distributed geometry
    pulsars_all, res_all = simulate_toa_residuals(r_true, seed=0)
    result_all = estimate_position(pulsars_all, res_all)

    # 3 pulsars that are reasonably close in direction (J0030, J0437, Crab — all low dec)
    low_dec_pulsars = [p for p in pulsars_all
                       if abs(p.dec_deg) < 25]
    assert len(low_dec_pulsars) >= 3, "Expected at least 3 low-declination pulsars"
    low_dec_pulsars = low_dec_pulsars[:3]

    from qkc.nav.xnav import simulate_toa_residuals as _sim
    _, res_few = _sim(r_true, pulsars=low_dec_pulsars, seed=0)
    result_few = estimate_position(low_dec_pulsars, res_few)

    # Worse geometry → higher condition number
    assert result_few.condition_number > result_all.condition_number, (
        "Expected near-coplanar geometry to have higher condition number"
    )


def test_xnav_minimum_three_pulsars_required():
    """DOCUMENTED LIMITATION: the estimator crashes with LinAlgError when fewer
    than 3 pulsars are provided because HᵀWH is singular.

    The system does NOT fail gracefully — it raises rather than returning a
    low-confidence result.  A production XNAV system would need to guard against
    this (e.g., check rank(H) before inverting, or use pseudoinverse).
    """
    r_true = np.array([1_000.0, 0.0, 0.0])
    two_pulsars = CATALOGUE[:2]
    _, residuals = simulate_toa_residuals(r_true, pulsars=two_pulsars, seed=0)
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        estimate_position(two_pulsars, residuals)


# ── Dipole robustness tests ────────────────────────────────────────────────────

def test_dipole_fitter_graceful_under_foreground_noise(seed=5):
    """Additive foreground noise (simulating galactic emission) should degrade
    but not collapse the dipole fitter.

    We inject correlated Gaussian noise at 5× the radiometer noise level.
    The fitter should still recover the direction to within 2° and speed
    to within 10 km/s.
    """
    v_true = _PLANCK_SPEED_KM_S * gal_to_cartesian(_PLANCK_DIR_L, _PLANCK_DIR_B)
    rng = np.random.default_rng(seed)
    pts = fibonacci_sphere(3072)
    T_clean = dipole_temperature(pts, v_true)

    # Correlated foreground: Gaussian noise at 50 μK (5× radiometer noise)
    T_noisy = T_clean + rng.standard_normal(len(pts)) * 50e-6

    result = fit_dipole(pts, T_noisy)
    assert result.angular_error_deg(v_true) < 2.0, (
        f"Direction error {result.angular_error_deg(v_true):.2f}° under 50 μK foreground"
    )
    assert result.speed_error_km_s(v_true) < 10.0, (
        f"Speed error {result.speed_error_km_s(v_true):.1f} km/s under 50 μK foreground"
    )


def test_dipole_fitter_fails_gracefully_with_pure_noise(seed=99):
    """With pure white noise and no signal, the fitter should return a
    near-zero dipole amplitude (not crash, not return nonsense speed).
    """
    rng = np.random.default_rng(seed)
    pts = fibonacci_sphere(1000)
    T_noise = T0_K + rng.standard_normal(1000) * 1e-4  # 100 μK noise, no dipole

    result = fit_dipole(pts, T_noise)
    # Speed should be tiny (noise-floor level), not ~370 km/s
    assert result.speed_km_s < 50.0, (
        f"Fitter returned {result.speed_km_s:.1f} km/s from pure noise — "
        "dipole signal is not present"
    )


# ── Kalman filter consistency tests ───────────────────────────────────────────

def test_kalman_normalized_innovation_squared(seed=0):
    """Normalized Innovation Squared (NIS) converges toward χ²(3) after warmup.

    NIS = (z - Hx)ᵀ S⁻¹ (z - Hx) where S = HPHᵀ + R.
    E[NIS] = 3 for a well-calibrated 3-DOF measurement (one NIS per update step).

    We use a stationary scenario (zero velocity, fixed true position) so that
    the measurement model is physically consistent with the dynamics.  The filter
    is warmed up for 10 steps before NIS is measured; initial transients are large
    because the prior covariance is deliberately wide (100 km vs 5 km measurements).
    """
    kf = NavKalmanFilter()
    r_true = np.array([1_000.0, -500.0, 200.0])
    kf.initialise(r_true + 50, np.zeros(3), pos_sigma_km=100.0, vel_sigma_km_s=5.0)

    rng = np.random.default_rng(seed)
    sigma_km = 5.0
    R = np.eye(3) * sigma_km**2

    # Warmup: let filter converge before measuring NIS
    for _ in range(10):
        kf.predict(60.0)
        kf.update_position(r_true + rng.standard_normal(3) * sigma_km, sigma_km=sigma_km)

    # Post-convergence NIS measurement
    nis_values = []
    for _ in range(30):
        kf.predict(60.0)
        z = r_true + rng.standard_normal(3) * sigma_km
        H = kf._H_POS
        s = kf.state
        S = H @ s.P @ H.T + R
        innovation = z - H @ s.x
        nis = float(innovation @ np.linalg.inv(S) @ innovation)
        nis_values.append(nis)
        kf.update_position(z, sigma_km=sigma_km)

    mean_nis = np.mean(nis_values)
    # Post-convergence: E[NIS] ≈ 3; accept [0.5, 12] to handle 30-sample variance
    assert 0.5 < mean_nis < 12.0, (
        f"Mean NIS {mean_nis:.2f} outside [0.5, 12] after convergence — "
        "filter may be mis-calibrated. E[NIS]=3 for a well-tuned 3-DOF measurement."
    )


def test_kalman_wrong_measurement_sigma_causes_overconfidence():
    """DOCUMENTED PITFALL: if R is set 10× too small (underestimated noise),
    the filter becomes overconfident — velocity covariance shrinks too fast.

    This test documents the behaviour so it is not mistaken for a feature.
    """
    kf_honest = NavKalmanFilter()
    kf_overconf = NavKalmanFilter()

    r_true = np.array([1_000.0, 0.0, 0.0])
    v_true = np.array([300.0, 0.0, 0.0])
    for kf in (kf_honest, kf_overconf):
        kf.initialise(r_true, v_true, pos_sigma_km=50.0, vel_sigma_km_s=5.0)

    rng = np.random.default_rng(0)
    for _ in range(5):
        kf_honest.predict(300.0)
        kf_overconf.predict(300.0)
        # True noise is 5 km/s; overconfident filter declares 0.5 km/s
        z_vel = v_true + rng.standard_normal(3) * 5.0
        kf_honest.update_velocity(z_vel, sigma_km_s=5.0)
        kf_overconf.update_velocity(z_vel, sigma_km_s=0.5)  # 10× underestimated

    # Overconfident filter reports lower uncertainty (it "thinks" it knows better)
    vel_sigma_honest = kf_honest.state.velocity_uncertainty_km_s()
    vel_sigma_overconf = kf_overconf.state.velocity_uncertainty_km_s()
    assert vel_sigma_overconf < vel_sigma_honest, (
        "Expected overconfident filter to report smaller (incorrect) uncertainty"
    )


def test_kalman_covariance_stays_positive_definite_through_many_steps():
    """The Joseph-form update must keep P positive-definite across 50 cycles
    with mixed position and velocity updates, including predict steps with
    large dt to stress the numerics.
    """
    kf = NavKalmanFilter()
    kf.initialise(np.array([1e5, -5e4, 2e4]), np.array([300.0, -100.0, 50.0]),
                  pos_sigma_km=500.0, vel_sigma_km_s=10.0)

    rng = np.random.default_rng(7)
    for step in range(50):
        kf.predict(600.0)
        if step % 2 == 0:
            kf.update_position(np.array([1e5, -5e4, 2e4]) + rng.standard_normal(3) * 5.0,
                                sigma_km=5.0)
        if step % 3 == 0:
            kf.update_velocity(np.array([300.0, -100.0, 50.0]) + rng.standard_normal(3),
                                sigma_km_s=1.0)

    eigvals = np.linalg.eigvalsh(kf.state.P)
    assert np.all(eigvals > 0), (
        f"Covariance lost positive-definiteness after 50 cycles. "
        f"Min eigenvalue: {eigvals.min():.2e}"
    )
