"""Visualization utilities: sky maps, patch zoom-ins, power spectra."""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.coordinates import SkyCoord
import astropy.units as u


def plot_mollweide(
    hmap: np.ndarray,
    title: str = "CMB Temperature Map",
    coord: str = "G",
    unit: str = r"$\mu K$",
    mark_qkc: bool = True,
    output_path: str = None,
):
    """Full-sky Mollweide projection. Optionally mark the QKC target location."""
    fig = plt.figure(figsize=(12, 6))
    hp.mollview(
        hmap,
        fig=fig.number,
        title=title,
        coord=[coord],
        unit=unit,
        cmap="RdBu_r",
    )
    hp.graticule()

    if mark_qkc:
        # Mark l=−57°, b=−27° in galactic coords
        hp.projscatter(
            np.radians(90 + 27),   # theta in radians from north pole
            np.radians(360 - 57),  # phi in radians
            marker="x",
            color="yellow",
            s=150,
            linewidths=2,
            label="QKC target",
        )
        plt.legend(loc="lower right")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_patch(
    pixel_indices: np.ndarray,
    pixel_values: np.ndarray,
    nside: int,
    title: str = "QKC Target Patch",
    output_path: str = None,
):
    """Render a disc patch as a gnomonic (flat) projection."""
    patch_map = np.full(hp.nside2npix(nside), hp.UNSEEN)
    patch_map[pixel_indices] = pixel_values

    fig = plt.figure(figsize=(7, 7))
    hp.gnomview(
        patch_map,
        fig=fig.number,
        title=title,
        rot=(360 - 57, -27, 0),  # (l, b, psi) galactic
        coord=["G"],
        reso=3.0,
        cmap="RdBu_r",
        unit=r"$\mu K$",
    )
    hp.graticule()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_power_spectrum(
    ells: np.ndarray,
    cl_observed: np.ndarray,
    cl_theory: np.ndarray = None,
    title: str = "Angular Power Spectrum",
    output_path: str = None,
):
    """Plot Dl = l(l+1)Cl/2π with optional theory overlay."""
    factor = ells * (ells + 1) / (2 * np.pi)
    dl_obs = cl_observed * factor

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ells[2:], dl_obs[2:], label="Observed", color="steelblue", lw=1.5)

    if cl_theory is not None:
        dl_th = cl_theory * factor
        ax.plot(ells[2:], dl_th[2:], label=r"$\Lambda$CDM (CAMB)", color="tomato",
                lw=1.5, linestyle="--")

    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_null_distribution(
    null_stats: np.ndarray,
    observed_stat: float,
    p_value: float,
    stat_label: str = "Statistic",
    output_path: str = None,
):
    """Histogram of Monte Carlo null distribution with observed value marked."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_stats, bins=40, color="steelblue", alpha=0.7, label="Null distribution")
    ax.axvline(observed_stat, color="tomato", lw=2, label=f"Observed (p={p_value:.4f})")
    ax.set_xlabel(stat_label)
    ax.set_ylabel("Count")
    ax.set_title("Monte Carlo Significance Test")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
