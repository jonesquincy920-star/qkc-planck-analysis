"""Extract and characterize the CMB patch around the QKC target coordinates."""

import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u

# QKC target: galactic coordinates
QKC_L = -57.0   # degrees
QKC_B = -27.0   # degrees


def galactic_to_healpy(l_deg: float, b_deg: float) -> tuple[float, float]:
    """Convert galactic (l, b) in degrees to HEALPix (theta, phi) in radians."""
    coord = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    ra = coord.icrs.ra.rad
    dec = coord.icrs.dec.rad
    theta = np.pi / 2 - dec
    phi = ra
    return theta, phi


def extract_patch(
    hmap: np.ndarray,
    l_deg: float = QKC_L,
    b_deg: float = QKC_B,
    radius_deg: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (pixel_indices, pixel_values) for a disc of radius_deg around (l, b).
    Uses galactic coordinates. hmap must be a full-sky HEALPix array.
    """
    nside = hp.get_nside(hmap)
    theta, phi = galactic_to_healpy(l_deg, b_deg)
    vec = hp.ang2vec(theta, phi)
    radius_rad = np.radians(radius_deg)
    ipix = hp.query_disc(nside, vec, radius_rad)
    return ipix, hmap[ipix]


def patch_stats(pixel_values: np.ndarray) -> dict:
    """Basic descriptive statistics for a patch of pixel values."""
    finite = pixel_values[np.isfinite(pixel_values)]
    return {
        "n_pixels": len(finite),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "skewness": float(_skewness(finite)),
        "kurtosis": float(_kurtosis(finite)),
    }


def _skewness(x: np.ndarray) -> float:
    mu = np.mean(x)
    sigma = np.std(x)
    return float(np.mean(((x - mu) / sigma) ** 3)) if sigma > 0 else 0.0


def _kurtosis(x: np.ndarray) -> float:
    mu = np.mean(x)
    sigma = np.std(x)
    return float(np.mean(((x - mu) / sigma) ** 4) - 3) if sigma > 0 else 0.0
