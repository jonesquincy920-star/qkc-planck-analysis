"""
Brain-state visualisation using the Petrie projection of the 9D hypercube.

Public API
----------
plot_brain_state(state)   — Petrie-projection scatter/edge plot of field
plot_shell_activity(...)  — bar chart of per-shell energies
plot_axis_coherence(...)  — polar spider chart of cognitive-axis coherence
plot_dashboard(brain)     — composite 3-panel dashboard
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .enneract import (
    N_DIM, N_VERTICES, N_SHELLS,
    shell_of, edge_list, petrie_project,
    COGNITIVE_AXES, SHELL_ROLES,
)

_DARK_BG  = "#050510"
_PANEL_BG = "#0a0a1e"


def plot_brain_state(
    state:   np.ndarray,
    title:   str       = "EnneractBrain — Neural Field (Petrie Projection)",
    figsize: tuple     = (10, 10),
    ax:      plt.Axes  = None,
    cmap:    str       = "plasma",
) -> plt.Figure:
    """
    Render the 512-neuron complex field as a Petrie-projection scatter plot.

    Visual encoding
    ---------------
    Node size        → amplitude  |z|  (firing rate)
    Node colour      → phase  angle(z) / 2π  (spike timing)
    Edge width/alpha → mean amplitude of the two endpoint neurons
    """
    x, y  = petrie_project()
    edges = edge_list()
    amp   = np.abs(state)
    phase = (np.angle(state) / (2 * np.pi) + 0.5) % 1.0  # [0, 1]
    shells = shell_of()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    else:
        fig = ax.get_figure()
    ax.set_facecolor("black")

    # Edges
    edge_amp = (amp[edges[:, 0]] + amp[edges[:, 1]]) * 0.5
    e_norm   = (edge_amp - edge_amp.min()) / (edge_amp.max() - edge_amp.min() + 1e-12)
    segments = np.array([
        [[x[e[0]], y[e[0]]], [x[e[1]], y[e[1]]]]
        for e in edges
    ])
    lc = LineCollection(
        segments,
        linewidths = 0.2 + 1.4 * e_norm,
        alpha      = 0.15 + 0.35 * e_norm,
        colors     = plt.cm.cool(e_norm),
    )
    ax.add_collection(lc)

    # Neurons
    max_amp = amp.max() + 1e-12
    sc = ax.scatter(
        x, y,
        c          = phase,
        s          = 4 + 55 * (amp / max_amp),
        cmap       = cmap,
        vmin       = 0, vmax = 1,
        alpha      = 0.85,
        edgecolors = "none",
        zorder     = 5,
    )

    # Faint concentric shell rings
    for k in range(N_SHELLS):
        mask = shells == k
        if mask.any():
            r = float(np.sqrt(x[mask]**2 + y[mask]**2).mean())
            ax.add_patch(plt.Circle(
                (0, 0), r, fill=False,
                color="white", alpha=0.06, linewidth=0.5,
            ))

    if own_fig:
        cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Phase (×2π)", color="white", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig


def plot_shell_activity(
    shell_energies: np.ndarray,
    ax:      plt.Axes = None,
    figsize: tuple    = (9, 4),
) -> plt.Figure:
    """Bar chart of energy per shell with role labels."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize, facecolor=_DARK_BG)
    else:
        fig = ax.get_figure()
    ax.set_facecolor(_PANEL_BG)

    colors = plt.cm.plasma(np.linspace(0, 1, N_SHELLS))
    ax.bar(range(N_SHELLS), shell_energies, color=colors, edgecolor="none")
    ax.set_xticks(range(N_SHELLS))
    ax.set_xticklabels(
        [f"k={k}\n{SHELL_ROLES[k].replace('_', ' ')}" for k in range(N_SHELLS)],
        rotation=40, ha="right", fontsize=7, color="white",
    )
    ax.set_ylabel("Energy", color="white", fontsize=9)
    ax.set_title("Shell Energies", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

    return fig if own_fig else fig


def plot_axis_coherence(
    axis_coherence: np.ndarray,
    ax:      plt.Axes = None,
    figsize: tuple    = (6, 6),
) -> plt.Figure:
    """Polar spider chart of cognitive-axis phase coherence."""
    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=figsize, facecolor=_DARK_BG)
        ax  = fig.add_subplot(111, polar=True)
    else:
        fig = ax.get_figure()

    angles = np.linspace(0, 2 * np.pi, N_DIM, endpoint=False)
    theta  = np.append(angles, angles[0])
    vals   = np.append(axis_coherence, axis_coherence[0])

    ax.set_facecolor(_PANEL_BG)
    ax.plot(theta, vals, "o-", color="#a040ff", linewidth=2, markersize=4)
    ax.fill(theta, vals, alpha=0.3, color="#6020cc")
    ax.set_thetagrids(np.degrees(angles), COGNITIVE_AXES, color="white", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(color="#334455", linewidth=0.5)
    ax.set_title("Axis Coherence", color="white", pad=18, fontsize=10)
    ax.tick_params(colors="white")

    return fig


def plot_dashboard(brain, title: str = "EnneractBrain — Live Dashboard") -> plt.Figure:
    """
    Full 3-panel dashboard:
      left  — Petrie-projection neural field
      upper-right — shell energy bar chart
      lower-right — axis coherence polar chart + statistics
    """
    fig = plt.figure(figsize=(18, 9), facecolor=_DARK_BG)
    gs  = fig.add_gridspec(
        2, 3,
        width_ratios=[2.2, 1.1, 1.1],
        hspace=0.45, wspace=0.3,
        left=0.04, right=0.97, top=0.93, bottom=0.08,
    )

    ax_petrie = fig.add_subplot(gs[:, 0])
    ax_shells = fig.add_subplot(gs[0, 1])
    ax_polar  = fig.add_subplot(gs[0, 2], polar=True)
    ax_info   = fig.add_subplot(gs[1, 1:])

    # ---- Petrie projection ----
    ax_petrie.set_facecolor("black")
    plot_brain_state(brain.field.state, title="Neural Field", ax=ax_petrie)

    # ---- Shell energies ----
    energies = brain.field.shell_energies()
    ax_shells.set_facecolor(_PANEL_BG)
    colors = plt.cm.plasma(np.linspace(0, 1, N_SHELLS))
    ax_shells.bar(range(N_SHELLS), energies, color=colors, edgecolor="none")
    ax_shells.set_xticks(range(N_SHELLS))
    ax_shells.set_xticklabels([f"k={k}" for k in range(N_SHELLS)], color="white", fontsize=8)
    ax_shells.set_ylabel("Energy", color="white", fontsize=9)
    ax_shells.set_title("Shell Energies", color="white", fontsize=10)
    ax_shells.tick_params(colors="white")
    for sp in ax_shells.spines.values():
        sp.set_edgecolor("#333355")

    # ---- Axis coherence (polar) ----
    coherence = brain.field.axis_coherence()
    angles = np.linspace(0, 2 * np.pi, N_DIM, endpoint=False)
    theta  = np.append(angles, angles[0])
    vals   = np.append(coherence, coherence[0])
    ax_polar.set_facecolor(_PANEL_BG)
    ax_polar.plot(theta, vals, "o-", color="#a040ff", linewidth=2, markersize=4)
    ax_polar.fill(theta, vals, alpha=0.3, color="#6020cc")
    ax_polar.set_thetagrids(np.degrees(angles), COGNITIVE_AXES, color="white", fontsize=7)
    ax_polar.set_ylim(0, 1)
    ax_polar.grid(color="#334455", linewidth=0.5)
    ax_polar.set_title("Axis Coherence", color="white", pad=18, fontsize=10)
    ax_polar.tick_params(colors="white")

    # ---- Info panel ----
    ax_info.set_facecolor(_PANEL_BG)
    ax_info.axis("off")
    info = brain.introspect()
    dom_s = info["dominant_shell"]
    lines = [
        f"  Step            : {info['step']}",
        f"  Total energy    : {info['total_energy']:.5f}",
        f"  Dominant shell  : k={dom_s}  ({SHELL_ROLES[dom_s]})",
        f"  Dominant axis   : {info['dominant_axis']}",
        f"  Memory stored   : {info['memory']['n_stored']} / {info['memory']['capacity']}",
        f"  Synaptic weight : {info['total_synaptic_weight']:.4f}",
    ]
    for i, line in enumerate(lines):
        ax_info.text(
            0.02, 0.88 - i * 0.16, line,
            transform=ax_info.transAxes,
            color="#c8c8ff", fontsize=9, family="monospace",
        )

    fig.suptitle(title, color="white", fontsize=14, y=0.98)
    return fig
