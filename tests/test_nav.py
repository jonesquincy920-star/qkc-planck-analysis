"""Tests for CMB dipole navigation and XNAV pulsar position estimation.

Each test documents the physical claim it verifies.  Tolerances are
set at roughly 3× the expected 1-σ noise level so the tests are stable
across random seeds while still catching algorithmic failures.
"""

import numpy as np
import pytest

from qkc.nav.dipole import (
    C_KM_S,
    T0_K,
    SOLAR_SPEED_KM_S,
    SOLAR_DIR_GAL_L,
    SOLAR_DIR_GAL_B,
    cartesian_to_gal,
    dipole_temperature,
    fibonacci_sphere,
    fit_dipole,
    gal_to_cartesian,
    simulate_observation,
    solar_velocity_vector,
)
from qkc.nav.xnav import (
    CATALOGUE,
    estimate_position,
    simulate_toa_residuals,
)
from qkc.nav.kalman import NavKalmanFilter


# ── Coordinate helpers ────────────────────────────────────────────────────────

def test_gal_roundtrip():
    """gal_to_cartesian → cartesian_to_gal should recover original coordinates."""
    l_in, b_in = 137.5, -23.8
    v = gal_to_cartesian(l_in, b_in)
    l_out, b_out = cartesian_to_gal(v)
    assert abs(l_out - l_in) < 0.01
    assert abs(b_out - b_in) < 0.01


def test_unit_vector_norm():
    """gal_to_cartesian must return a unit vector."""
    v = gal_to_cartesian(100.0, 45.0)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-12


def test_fibonacci_sphere_unit_norms():
    """All Fibonacci sphere points must be unit vectors."""
    pts = fibonacci_sphere(200)
    norms = np.linalg.norm(pts, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_fibonacci_sphere_coverage():
    """Fibonacci sphere should span both hemispheres."""
    pts = fibonacci_sphere(500)
    assert pts[:, 2].min() < -0.9
    assert pts[:, 2].max() > 0.9


# ── Forward model ─────────────────────────────────────────────────────────────

def test_dipole_monopole_no_motion():
    """A spacecraft at rest (v=0) should see a uniform CMB sky at T₀."""
    pts = fibonacci_sphere(100)
    T = dipole_temperature(pts, np.zeros(3))
    assert np.allclose(T, T0_K, rtol=1e-9)


def test_dipole_amplitude():
    """Dipole amplitude for solar velocity should be ~3.36 mK (ΔT = T₀·β)."""
    v = solar_velocity_vector()
    pts = fibonacci_sphere(3000)
    T = dipole_temperature(pts, v)
    # Temperature variation peak-to-peak ≈ 2 × T₀ × β
    expected_half_amplitude = T0_K * (SOLAR_SPEED_KM_S / C_KM_S)  # ≈ 3.36 mK
    assert abs((T.max() - T.min()) / 2 - expected_half_amplitude) < 0.5e-3


def test_dipole_direction_max_toward_velocity():
    """Hottest sky direction should be close to the velocity direction.

    Tolerance is 2° because a Fibonacci sphere of 3000 points has angular
    resolution ~√(4π/3000) ≈ 3.7°, so the nearest sample can be ~1.8° away
    from the true maximum.  The fitter test verifies sub-0.5° accuracy via
    least-squares; this test only checks the forward model sign.
    """
    v = solar_velocity_vector()
    v_hat = v / np.linalg.norm(v)
    pts = fibonacci_sphere(3000)
    T = dipole_temperature(pts, v)
    hottest = pts[np.argmax(T)]
    cos_ang = np.clip(hottest @ v_hat, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_ang))
    assert angle_deg < 2.0


# ── Dipole fitter ─────────────────────────────────────────────────────────────

def test_fit_recovers_solar_velocity_direction(seed=42):
    """Physical claim: fit_dipole should recover the solar velocity direction
    to within 0.5° from a realistic noisy observation (noise=10 μK, N=3072)."""
    v_true = solar_velocity_vector()
    pts, T_noisy = simulate_observation(v_true, n_pointings=3072,
                                        noise_uk=10.0, seed=seed)
    result = fit_dipole(pts, T_noisy)
    ang_err = result.angular_error_deg(v_true)
    assert ang_err < 0.5, f"Direction error {ang_err:.3f}° exceeds 0.5°"


def test_fit_recovers_solar_speed(seed=42):
    """Physical claim: recovered speed should be within 2 km/s of 369.82 km/s."""
    v_true = solar_velocity_vector()
    pts, T_noisy = simulate_observation(v_true, n_pointings=3072,
                                        noise_uk=10.0, seed=seed)
    result = fit_dipole(pts, T_noisy)
    speed_err = result.speed_error_km_s(v_true)
    assert speed_err < 2.0, f"Speed error {speed_err:.2f} km/s exceeds 2 km/s"


def test_fit_monopole_near_t0(seed=0):
    """Recovered monopole should be within 1 mK of T₀ = 2.7255 K."""
    v = solar_velocity_vector()
    pts, T = simulate_observation(v, noise_uk=10.0, seed=seed)
    result = fit_dipole(pts, T)
    assert abs(result.monopole_k - T0_K) < 1e-3


def test_fit_dipole_amplitude_mk(seed=0):
    """Recovered dipole amplitude should match T₀·β ≈ 3.36 mK within 0.1 mK."""
    v = solar_velocity_vector()
    pts, T = simulate_observation(v, noise_uk=10.0, seed=0)
    result = fit_dipole(pts, T)
    expected = T0_K * SOLAR_SPEED_KM_S / C_KM_S * 1e3   # mK
    assert abs(result.dipole_amplitude_mk - expected) < 0.1


def test_fit_arbitrary_velocity_direction(seed=7):
    """Fitter should work for any velocity direction, not just the solar one."""
    v_true = np.array([200.0, -150.0, 300.0])  # arbitrary km/s
    pts, T = simulate_observation(v_true, n_pointings=3072, noise_uk=10.0, seed=seed)
    result = fit_dipole(pts, T)
    ang_err = result.angular_error_deg(v_true)
    assert ang_err < 1.0


def test_fit_noise_scaling():
    """Higher instrument noise should produce larger fit residuals."""
    v = solar_velocity_vector()
    pts1, T1 = simulate_observation(v, noise_uk=10.0, seed=1)
    pts2, T2 = simulate_observation(v, noise_uk=100.0, seed=1)
    r1 = fit_dipole(pts1, T1)
    r2 = fit_dipole(pts2, T2)
    assert r2.residual_rms_uk > r1.residual_rms_uk


def test_fit_more_pointings_improves_accuracy():
    """More sky samples should give a smaller direction error (central limit)."""
    v = solar_velocity_vector()
    pts_few, T_few = simulate_observation(v, n_pointings=192,  noise_uk=50.0, seed=5)
    pts_many, T_many = simulate_observation(v, n_pointings=3072, noise_uk=50.0, seed=5)
    err_few  = fit_dipole(pts_few,  T_few ).angular_error_deg(v)
    err_many = fit_dipole(pts_many, T_many).angular_error_deg(v)
    assert err_many < err_few


def test_fit_noiseless_perfect_recovery():
    """With zero noise, the fitter should recover velocity to floating-point precision."""
    v_true = np.array([100.0, 200.0, -150.0])
    pts = fibonacci_sphere(1000)
    T = dipole_temperature(pts, v_true)
    result = fit_dipole(pts, T)
    assert result.angular_error_deg(v_true) < 0.001
    assert result.speed_error_km_s(v_true) < 0.01


# ── XNAV pulsar navigation ────────────────────────────────────────────────────

def test_catalogue_has_entries():
    assert len(CATALOGUE) >= 4


def test_pulsar_directions_are_unit_vectors():
    for p in CATALOGUE:
        assert abs(np.linalg.norm(p.direction) - 1.0) < 1e-12


def test_xnav_position_recovery_near_barycenter(seed=0):
    """Physical claim: with 7 pulsars, position near barycenter should be
    recovered to within 30 km (1-σ noise ~5 km, 3-σ ≈ 15 km; being generous)."""
    r_true = np.array([1_000.0, -500.0, 200.0])   # km from barycenter
    pulsars, residuals = simulate_toa_residuals(r_true, seed=seed)
    result = estimate_position(pulsars, residuals)
    error = float(np.linalg.norm(result.position_km - r_true))
    assert error < 30.0, f"Position error {error:.1f} km exceeds 30 km"


def test_xnav_position_deep_space(seed=1):
    """Recovery should also work at large distances (1 AU from barycenter ~1.5e8 km)."""
    r_true = np.array([1.0e8, -5.0e7, 2.0e7])   # ~1 AU
    pulsars, residuals = simulate_toa_residuals(r_true, seed=seed)
    result = estimate_position(pulsars, residuals)
    error = float(np.linalg.norm(result.position_km - r_true))
    # At large distances the angular baseline is the same; noise dominates
    assert error < 200.0, f"Position error {error:.1f} km"


def test_xnav_covariance_positive_definite(seed=0):
    """Covariance matrix must be positive definite (all eigenvalues > 0)."""
    r_true = np.array([500.0, -300.0, 100.0])
    pulsars, residuals = simulate_toa_residuals(r_true, seed=seed)
    result = estimate_position(pulsars, residuals)
    eigvals = np.linalg.eigvalsh(result.covariance_km2)
    assert np.all(eigvals > 0)


def test_xnav_fewer_pulsars_less_accurate(seed=2):
    """3 pulsars should have larger *expected* position error than all 7.

    We compare covariance traces rather than noise-specific errors because any
    single noise realisation can accidentally favour fewer pulsars — the
    statistical claim is about expected error, not one outcome.
    """
    r_true = np.array([800.0, -400.0, 300.0])
    pulsars_all, res_all = simulate_toa_residuals(r_true, seed=seed)
    result_all = estimate_position(pulsars_all, res_all)
    result_few = estimate_position(pulsars_all[:3], res_all[:3])
    # Larger covariance trace ↔ higher expected error
    assert np.trace(result_few.covariance_km2) > np.trace(result_all.covariance_km2)


# ── Kalman filter fusion ──────────────────────────────────────────────────────

def test_kalman_initialise():
    kf = NavKalmanFilter()
    r0 = np.array([1000.0, -500.0, 200.0])
    v0 = np.array([300.0, -100.0, 50.0])
    kf.initialise(r0, v0, pos_sigma_km=100.0, vel_sigma_km_s=5.0)
    assert np.allclose(kf.state.position_km, r0)
    assert np.allclose(kf.state.velocity_km_s, v0)


def test_kalman_predict_propagates_position():
    """After predict(dt), position should advance by v*dt."""
    kf = NavKalmanFilter()
    r0 = np.zeros(3)
    v0 = np.array([1.0, 0.0, 0.0])   # 1 km/s along x
    kf.initialise(r0, v0)
    kf.predict(dt_s=100.0)
    assert abs(kf.state.position_km[0] - 100.0) < 1e-9


def test_kalman_predict_grows_covariance():
    """Prediction without measurement should increase uncertainty."""
    kf = NavKalmanFilter()
    kf.initialise(np.zeros(3), np.zeros(3), pos_sigma_km=10.0)
    P_before = kf.state.P.copy()
    kf.predict(dt_s=3600.0)
    P_after = kf.state.P
    assert np.trace(P_after) > np.trace(P_before)


def test_kalman_update_position_reduces_uncertainty():
    """A good position measurement should reduce position uncertainty."""
    kf = NavKalmanFilter()
    kf.initialise(np.zeros(3), np.zeros(3), pos_sigma_km=500.0)
    sigma_before = kf.state.position_uncertainty_km()
    kf.update_position(np.array([10.0, -5.0, 2.0]), sigma_km=5.0)
    sigma_after = kf.state.position_uncertainty_km()
    assert sigma_after < sigma_before


def test_kalman_update_velocity_reduces_uncertainty():
    """A CMB dipole velocity measurement should reduce velocity uncertainty."""
    kf = NavKalmanFilter()
    kf.initialise(np.zeros(3), np.zeros(3), vel_sigma_km_s=10.0)
    sigma_before = kf.state.velocity_uncertainty_km_s()
    kf.update_velocity(np.array([300.0, -100.0, 50.0]), sigma_km_s=1.0)
    sigma_after = kf.state.velocity_uncertainty_km_s()
    assert sigma_after < sigma_before


def test_kalman_fusion_end_to_end(seed=3):
    """Information-theoretic claim: fusing CMB dipole velocity measurements into
    the Kalman filter must reduce velocity covariance relative to XNAV alone.

    We compare covariance traces (expected squared error) rather than a single
    noise realisation, making the assertion deterministic and physically meaningful.
    """
    r_true = np.array([5_000.0, -2_000.0, 1_000.0])
    v_true = solar_velocity_vector()

    # Realistic initial uncertainties: 50 km position, 2 km/s velocity
    kf_xnav = NavKalmanFilter()
    kf_xnav.initialise(r_true + 20, v_true + 2, pos_sigma_km=50.0, vel_sigma_km_s=2.0)

    kf_fused = NavKalmanFilter()
    kf_fused.initialise(r_true + 20, v_true + 2, pos_sigma_km=50.0, vel_sigma_km_s=2.0)

    rng = np.random.default_rng(seed)
    for _ in range(3):
        pulsars, residuals = simulate_toa_residuals(r_true, seed=int(rng.integers(1000)))
        xnav_result = estimate_position(pulsars, residuals)

        pts, T_noisy = simulate_observation(v_true, noise_uk=10.0,
                                            seed=int(rng.integers(1000)))
        dipole_result = fit_dipole(pts, T_noisy)

        # 600 s step — realistic inter-measurement interval
        kf_xnav.predict(600.0)
        kf_xnav.update_position(xnav_result.position_km, sigma_km=5.0)

        kf_fused.predict(600.0)
        kf_fused.update_position(xnav_result.position_km, sigma_km=5.0)
        kf_fused.update_velocity(dipole_result.velocity_km_s, sigma_km_s=1.0)

    # Velocity covariance trace: adding a CMB dipole measurement must reduce it
    vel_cov_xnav  = np.trace(kf_xnav.state.P[3:, 3:])
    vel_cov_fused = np.trace(kf_fused.state.P[3:, 3:])
    assert vel_cov_fused < vel_cov_xnav, (
        f"Fused vel cov {vel_cov_fused:.4f} not less than XNAV-only {vel_cov_xnav:.4f}"
    )

    # Position covariance should be similar (both receive same XNAV updates)
    pos_cov_xnav  = np.trace(kf_xnav.state.P[:3, :3])
    pos_cov_fused = np.trace(kf_fused.state.P[:3, :3])
    assert pos_cov_fused <= pos_cov_xnav * 1.01, (
        f"Fused pos cov {pos_cov_fused:.4f} unexpectedly larger than XNAV-only {pos_cov_xnav:.4f}"
    )


def test_kalman_innovation_direction(seed=0):
    """Innovation (measurement − prediction) should point from prediction toward truth."""
    kf = NavKalmanFilter()
    r0 = np.array([0.0, 0.0, 0.0])
    kf.initialise(r0, np.zeros(3), pos_sigma_km=1000.0)
    z = np.array([100.0, 0.0, 0.0])   # truth is at +x
    result = kf.update_position(z, sigma_km=5.0)
    # Innovation should have positive x-component
    assert result.innovation_pos_km[0] > 0
