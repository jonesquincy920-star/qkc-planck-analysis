"""
Logarithmic spiral geometry on the celestial sphere for QKC analysis.

The spiral is defined in a local tangent frame, then projected onto the
sphere via a rotation matrix so it can be stamped at any (l, b) center.
"""

import numpy as np
import healpy as hp


# QKC spiral parameters (galactic coords)
QKC_L = -57.0   # degrees — spiral center
QKC_B = -27.0   # degrees

# Logarithmic spiral r = r0 * exp(k * theta)
_SPIRAL_R0    = 0.5    # degrees — inner radius
_SPIRAL_K     = 0.22   # growth rate (radians^-1)
_SPIRAL_TURNS = 2.5    # number of turns
_SPIRAL_PTS   = 400    # sampling density


def _log_spiral_offsets(
    r0: float = _SPIRAL_R0,
    k: float = _SPIRAL_K,
    turns: float = _SPIRAL_TURNS,
    n_pts: int = _SPIRAL_PTS,
) -> np.ndarray:
    """
    Return (n_pts, 2) array of (dlon, dlat) offsets in degrees
    for a logarithmic spiral in a flat tangent plane.
    """
    theta = np.linspace(0, 2 * np.pi * turns, n_pts)
    r = r0 * np.exp(k * theta)
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return np.column_stack([dx, dy])


def _rotation_matrix(l_deg: float, b_deg: float) -> np.ndarray:
    """
    3×3 rotation matrix that takes the north-pole unit vector (0,0,1)
    to the direction (l_deg, b_deg) in galactic coordinates.
    Used to rigidly rotate the spiral to any sky position.
    """
    l = np.radians(float(l_deg))
    b = np.radians(float(b_deg))

    # Unit vector for (l, b) in galactic Cartesian frame
    x = float(np.cos(b) * np.cos(l))
    y = float(np.cos(b) * np.sin(l))
    z = float(np.sin(b))
    target = np.array([x, y, z], dtype=float)

    # Rodrigues rotation: north pole -> target
    north = np.array([0.0, 0.0, 1.0])
    axis = np.cross(north, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        # Already at pole
        return np.eye(3) if z > 0 else np.diag([1, 1, -1])
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(north, target), -1, 1))

    # Rodrigues formula
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


def spiral_pixels(
    nside: int,
    l_deg: float = QKC_L,
    b_deg: float = QKC_B,
    r0: float = _SPIRAL_R0,
    k: float = _SPIRAL_K,
    turns: float = _SPIRAL_TURNS,
    n_pts: int = _SPIRAL_PTS,
    beam_deg: float = 0.5,
) -> np.ndarray:
    """
    Return unique HEALPix pixel indices for a logarithmic spiral
    centered at (l_deg, b_deg). Each spiral point is smeared by a
    disc of radius beam_deg to give a 1D path a finite width.
    """
    offsets = _log_spiral_offsets(r0, k, turns, n_pts)
    R = _rotation_matrix(l_deg, b_deg)
    beam_rad = np.radians(beam_deg)
    pixel_set = set()

    for dx, dy in offsets:
        # Local tangent-plane offset -> unit vector (small-angle)
        dx_rad = np.radians(dx)
        dy_rad = np.radians(dy)
        # local Cartesian in tangent frame (north pole coords)
        local = np.array([dx_rad, dy_rad, np.sqrt(max(1 - dx_rad**2 - dy_rad**2, 0))])
        local /= np.linalg.norm(local)
        # Rotate to sky position
        vec = R @ local
        vec /= np.linalg.norm(vec)
        ipix = hp.query_disc(nside, vec, beam_rad, inclusive=False)
        pixel_set.update(ipix.tolist())

    return np.array(sorted(pixel_set), dtype=np.intp)


def rotated_spiral_pixels(
    nside: int,
    l_deg: float,
    b_deg: float,
    roll_deg: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    Same spiral shape stamped at arbitrary (l_deg, b_deg) with an
    optional in-plane rotation roll_deg (degrees) for orientation diversity.
    """
    offsets = _log_spiral_offsets(**{k: v for k, v in kwargs.items()
                                     if k in ('r0', 'k', 'turns', 'n_pts')})
    if roll_deg != 0:
        a = np.radians(roll_deg)
        rot2d = np.array([[np.cos(a), -np.sin(a)],
                          [np.sin(a),  np.cos(a)]])
        offsets = offsets @ rot2d.T

    R = _rotation_matrix(l_deg, b_deg)
    beam_rad = np.radians(kwargs.get('beam_deg', 0.5))
    pixel_set = set()

    for dx, dy in offsets:
        dx_rad, dy_rad = np.radians(dx), np.radians(dy)
        local = np.array([dx_rad, dy_rad, np.sqrt(max(1 - dx_rad**2 - dy_rad**2, 0))])
        local /= np.linalg.norm(local)
        vec = R @ local
        vec /= np.linalg.norm(vec)
        ipix = hp.query_disc(nside, vec, beam_rad, inclusive=False)
        pixel_set.update(ipix.tolist())

    return np.array(sorted(pixel_set), dtype=np.intp)
