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


def plot_circuit_highway(
    circuit,
    highlight_loop: bool = True,
    figsize: tuple = (9, 9),
    output_path: str = None,
):
    """
    Render the circuit highway radial mesh in the style of the QKC geodesic
    image: dark background, glowing hub, colour-coded temperature nodes, and
    the minimum-entropy closed loop highlighted in cyan.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = circuit.nodes
    if not nodes:
        return fig

    c_theta = nodes[0].theta
    c_phi = nodes[0].phi

    def _project(node):
        dt = node.theta - c_theta
        dp = (node.phi - c_phi + np.pi) % (2 * np.pi) - np.pi
        return np.degrees(dp * np.sin(c_theta)), -np.degrees(dt)

    xs = np.array([_project(n)[0] for n in nodes])
    ys = np.array([_project(n)[1] for n in nodes])
    temps = np.array([n.temperature for n in nodes])

    vmin, vmax = temps.min(), temps.max()
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.cool

    # Build loop edge set for highlighting
    loop_edges: set[tuple[int, int]] = set()
    if highlight_loop and circuit._loop:
        loop = circuit._loop
        m = len(loop)
        for k in range(m):
            u, v = loop[k], loop[(k + 1) % m]
            loop_edges.add((u, v))
            loop_edges.add((v, u))

    # Draw edges
    for edge in circuit.edges:
        in_loop = (edge.u, edge.v) in loop_edges
        color = "#00e5ff" if in_loop else "#2a2a4a"
        lw = 1.8 if in_loop else 0.6
        alpha = 0.9 if in_loop else 0.55
        ax.plot(
            [xs[edge.u], xs[edge.v]],
            [ys[edge.u], ys[edge.v]],
            color=color, lw=lw, alpha=alpha, zorder=1,
        )

    # Draw nodes
    for n in nodes:
        s = 90 if n.ring == 0 else max(8, 40 - n.ring * 4)
        ec = "white" if n.ring == 0 else "#aaaacc"
        ax.scatter(
            xs[n.index], ys[n.index],
            c=[n.temperature], cmap=cmap, vmin=vmin, vmax=vmax,
            s=s, zorder=3, edgecolors=ec, linewidths=0.6,
        )

    # Central glow layers
    for r, a in [(5.0, 0.04), (3.0, 0.08), (1.5, 0.15), (0.5, 0.6)]:
        glow = plt.Circle((xs[0], ys[0]), r, color="white", alpha=a, zorder=2)
        ax.add_patch(glow)

    # Colour bar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\delta T\ [\mu\mathrm{K}]$", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    entropy_str = f"{circuit.loop_entropy:.2f}" if np.isfinite(circuit.loop_entropy) else "—"
    ax.set_title(
        f"Circuit Highway Closed Loop  ·  min-entropy path\n"
        f"rings={circuit.n_rings}  spokes={circuit.n_spokes}  "
        f"TV={entropy_str} μK",
        color="white", fontsize=11, pad=10,
    )

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="black")
    return fig


def plot_circuit_significance(
    sig: dict,
    output_path: str = None,
):
    """
    Two-panel significance plot for the circuit highway entropy tests.

    Left:  permutation test (local spatial-arrangement significance).
    Right: sky randomisation test (global sky-location significance).
           Omitted if sky results are absent from sig.
    """
    has_sky = "sky_pvalue" in sig
    ncols = 2 if has_sky else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    def _panel(ax, null_mean, null_std, observed, p_value, z_score, title):
        if null_std < 1e-10:
            null_std = max(abs(null_mean) * 0.05, 1.0)
        # reconstruct approximate null via Gaussian for smooth display
        x = np.linspace(null_mean - 4 * null_std, null_mean + 4 * null_std, 300)
        y = np.exp(-0.5 * ((x - null_mean) / null_std) ** 2)
        ax.fill_between(x, y, alpha=0.25, color="steelblue", label="Null distribution")
        ax.plot(x, y, color="steelblue", lw=1.5)
        ax.axvline(observed, color="tomato", lw=2,
                   label=f"QKC observed\nTV={observed:.1f} μK")
        ax.axvspan(x[0], observed, alpha=0.12, color="tomato")
        sigma = abs(z_score)
        ax.set_title(
            f"{title}\np = {p_value:.3f}  |  z = {z_score:.2f}  ({sigma:.1f}σ)",
            fontsize=10,
        )
        ax.set_xlabel("Min-entropy loop TV [μK]")
        ax.set_ylabel("Density (arb.)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _panel(
        axes[0],
        null_mean=sig["permutation_null_mean"],
        null_std=sig["permutation_null_std"],
        observed=sig["observed_tv"],
        p_value=sig["permutation_pvalue"],
        z_score=sig["permutation_z"],
        title="Permutation test\n(local spatial-arrangement)",
    )

    if has_sky:
        _panel(
            axes[1],
            null_mean=sig["sky_null_mean"],
            null_std=sig["sky_null_std"],
            observed=sig["observed_tv"],
            p_value=sig["sky_pvalue"],
            z_score=sig["sky_z"],
            title="Sky randomisation test\n(global location significance)",
        )

    fig.suptitle("Circuit Highway Closed-Loop Entropy Significance", fontsize=12)
    fig.tight_layout()

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
