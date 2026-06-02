"""Angular power spectrum, anomaly scoring, and Monte Carlo significance testing."""

import numpy as np
import healpy as hp
from scipy import stats as scipy_stats


def angular_power_spectrum(
    hmap: np.ndarray,
    lmax: int = 200,
    mask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute angular power spectrum Cl up to lmax.
    Optionally apply a binary mask (1=use, 0=exclude).
    Returns (ells, Cl).
    """
    if mask is not None:
        hmap = hmap * mask
    cl = hp.anafast(hmap, lmax=lmax)
    ells = np.arange(len(cl))
    return ells, cl


def camb_theory_cl(lmax: int = 200) -> np.ndarray:
    """Return ΛCDM theory Cl from CAMB for comparison. Returns Cl array indexed by ell."""
    try:
        import camb
        pars = camb.set_params(
            H0=67.4,
            ombh2=0.0224,
            omch2=0.120,
            mnu=0.06,
            omk=0,
            tau=0.054,
            As=2.1e-9,
            ns=0.965,
            lmax=lmax,
        )
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        # totCl[:,0] is TT; convert Dl -> Cl
        dl_tt = powers["total"][:lmax + 1, 0]
        ells = np.arange(lmax + 1)
        cl_theory = np.where(ells > 1, dl_tt * 2 * np.pi / (ells * (ells + 1)), 0.0)
        return cl_theory
    except ImportError:
        return None


def anomaly_score(
    observed_values: np.ndarray,
    reference_values: np.ndarray,
) -> dict:
    """
    Compare observed patch pixel values against a reference distribution.
    Returns KS test statistic, p-value, and z-score of the mean.
    """
    ks_stat, ks_p = scipy_stats.ks_2samp(observed_values, reference_values)
    ref_mean = np.mean(reference_values)
    ref_std = np.std(reference_values)
    z = (np.mean(observed_values) - ref_mean) / (ref_std / np.sqrt(len(observed_values)))
    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "z_score": float(z),
        "z_pvalue": float(2 * (1 - scipy_stats.norm.cdf(abs(z)))),
    }


def monte_carlo_pvalue(
    hmap: np.ndarray,
    observed_stat: float,
    stat_fn,
    n_simulations: int = 1000,
    radius_deg: float = 10.0,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Estimate p-value by placing the same-size disc at n_simulations random locations
    on the full sky and computing stat_fn(pixel_values) each time.
    Returns (p_value, null_distribution).
    """
    rng = np.random.default_rng(seed)
    nside = hp.get_nside(hmap)
    radius_rad = np.radians(radius_deg)
    null_stats = []

    for _ in range(n_simulations):
        vec = _random_unit_vector(rng)
        ipix = hp.query_disc(nside, vec, radius_rad)
        vals = hmap[ipix]
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            null_stats.append(stat_fn(finite))

    null_stats = np.array(null_stats)
    p_value = float(np.mean(null_stats >= observed_stat))
    return p_value, null_stats


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(3)
    return vec / np.linalg.norm(vec)
