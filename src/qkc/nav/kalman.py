"""Kalman filter fusing XNAV position and CMB dipole velocity measurements.

State and measurement models
-----------------------------
State vector  x = [rx, ry, rz, vx, vy, vz]ᵀ   (km and km/s, barycentric J2000)

Process model (constant velocity between measurements):
    x_{k+1} = F(dt) · x_k + w_k,   w_k ~ N(0, Q)

    F(dt) = [ I₃   dt·I₃ ]
            [ 0₃     I₃  ]

Measurement models:
    XNAV (position):   z_pos = H_pos · x + v_pos,  H_pos = [I₃ | 0₃]
    CMB dipole (velocity): z_vel = H_vel · x + v_vel,  H_vel = [0₃ | I₃]

Both are linear, so a standard (not extended) Kalman filter suffices.
The CMB dipole gives a full 3-D velocity measurement because the dipole
fit recovers all three Cartesian components of the velocity vector.

Noise parameters
-----------------
XNAV position noise R_pos:
  ~5 km (1σ) per observation period → R_pos = 25·I₃  km²

CMB dipole velocity noise R_vel:
  Direction uncertainty ~0.1° → transverse velocity error ≈ |v|·sin(0.1°) ≈ 0.6 km/s
  Amplitude uncertainty ~1 km/s
  R_vel ≈ diag(1, 1, 1)·1²  km²/s²  (approximation; full covariance from fit available)

Process noise Q:
  Accounts for unmodelled accelerations (gravity gradients, solar pressure).
  ~0.001 km/s² per axis → small for interplanetary cruise.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class KalmanState:
    """Kalman filter state estimate."""

    x: np.ndarray        # shape (6,): [rx, ry, rz, vx, vy, vz]
    P: np.ndarray        # shape (6,6): state covariance

    @property
    def position_km(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity_km_s(self) -> np.ndarray:
        return self.x[3:].copy()

    def position_uncertainty_km(self) -> float:
        """1-σ position uncertainty (scalar: RMS of diagonal)."""
        return float(np.sqrt(max(0.0, np.mean(np.diag(self.P)[:3]))))

    def velocity_uncertainty_km_s(self) -> float:
        """1-σ velocity uncertainty (scalar: RMS of diagonal)."""
        return float(np.sqrt(max(0.0, np.mean(np.diag(self.P)[3:]))))


@dataclass
class FusionResult:
    """Single-step fusion result."""

    state: KalmanState
    innovation_pos_km: np.ndarray | None = None
    innovation_vel_km_s: np.ndarray | None = None
    step: int = 0


class NavKalmanFilter:
    """Linear Kalman filter for XNAV + CMB dipole fusion.

    Usage::

        kf = NavKalmanFilter()
        kf.initialise(position_km=r0, velocity_km_s=v0,
                      pos_sigma_km=500.0, vel_sigma_km_s=10.0)

        # propagate dt seconds
        kf.predict(dt_s=3600.0)

        # update with XNAV position measurement
        kf.update_position(z_pos_km=r_meas, sigma_km=5.0)

        # update with CMB dipole velocity measurement
        kf.update_velocity(z_vel_km_s=v_meas, sigma_km_s=1.0)

        print(kf.state.position_km)
    """

    # Measurement matrices
    _H_POS = np.hstack([np.eye(3), np.zeros((3, 3))])   # extracts position
    _H_VEL = np.hstack([np.zeros((3, 3)), np.eye(3)])   # extracts velocity

    def __init__(
        self,
        process_noise_pos_km: float = 0.01,
        process_noise_vel_km_s: float = 0.001,
    ) -> None:
        self._q_pos = process_noise_pos_km
        self._q_vel = process_noise_vel_km_s
        self._state: KalmanState | None = None
        self._step = 0

    def initialise(
        self,
        position_km: np.ndarray,
        velocity_km_s: np.ndarray,
        pos_sigma_km: float = 500.0,
        vel_sigma_km_s: float = 10.0,
    ) -> None:
        x = np.concatenate([position_km, velocity_km_s]).astype(float)
        P = np.diag([pos_sigma_km**2] * 3 + [vel_sigma_km_s**2] * 3)
        self._state = KalmanState(x=x, P=P)
        self._step = 0

    @property
    def state(self) -> KalmanState:
        if self._state is None:
            raise RuntimeError("Filter not initialised — call initialise() first")
        return self._state

    # ── Predict step ──────────────────────────────────────────────────────────

    def predict(self, dt_s: float) -> None:
        """Propagate state forward by dt_s seconds (constant-velocity model)."""
        s = self.state
        F = self._transition_matrix(dt_s)
        Q = self._process_noise(dt_s)
        x_pred = F @ s.x
        P_pred = F @ s.P @ F.T + Q
        self._state = KalmanState(x=x_pred, P=P_pred)

    # ── Update steps ──────────────────────────────────────────────────────────

    def update_position(
        self,
        z_pos_km: np.ndarray,
        sigma_km: float = 5.0,
        R: np.ndarray | None = None,
    ) -> FusionResult:
        """Incorporate an XNAV position measurement.

        Parameters
        ----------
        z_pos_km : shape (3,) measured position in km
        sigma_km : scalar 1-σ isotropic noise (used if R not given)
        R        : optional full 3×3 measurement noise covariance
        """
        R_meas = R if R is not None else np.eye(3) * sigma_km**2
        innovation = z_pos_km - self._H_POS @ self.state.x
        self._state = self._update(self.state, self._H_POS, z_pos_km, R_meas)
        self._step += 1
        return FusionResult(
            state=self._state,
            innovation_pos_km=innovation,
            step=self._step,
        )

    def update_velocity(
        self,
        z_vel_km_s: np.ndarray,
        sigma_km_s: float = 1.0,
        R: np.ndarray | None = None,
    ) -> FusionResult:
        """Incorporate a CMB dipole velocity measurement.

        Parameters
        ----------
        z_vel_km_s : shape (3,) measured velocity in km/s
        sigma_km_s : scalar 1-σ isotropic noise (used if R not given)
        R          : optional full 3×3 measurement noise covariance (km/s)²
        """
        R_meas = R if R is not None else np.eye(3) * sigma_km_s**2
        innovation = z_vel_km_s - self._H_VEL @ self.state.x
        self._state = self._update(self.state, self._H_VEL, z_vel_km_s, R_meas)
        self._step += 1
        return FusionResult(
            state=self._state,
            innovation_vel_km_s=innovation,
            step=self._step,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _update(
        s: KalmanState,
        H: np.ndarray,
        z: np.ndarray,
        R: np.ndarray,
    ) -> KalmanState:
        """Kalman update using the Joseph form for numerical stability.

        Joseph form: P = (I-KH) P (I-KH)ᵀ + K R Kᵀ
        Unlike the simpler (I-KH)P, this is guaranteed positive-semidefinite
        in finite-precision arithmetic, preventing negative covariance entries
        when position noise >> velocity noise.
        """
        S = H @ s.P @ H.T + R
        K = s.P @ H.T @ np.linalg.inv(S)
        x_new = s.x + K @ (z - H @ s.x)
        IKH   = np.eye(6) - K @ H
        P_new = IKH @ s.P @ IKH.T + K @ R @ K.T
        return KalmanState(x=x_new, P=P_new)

    def _transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        """Block-diagonal process noise (independent position and velocity noise)."""
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (self._q_pos ** 2) * dt
        Q[3:, 3:] = np.eye(3) * (self._q_vel ** 2) * dt
        return Q
