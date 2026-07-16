"""
appendix_closure_figure.py
==========================
Generate the figure for Appendix B of Terzic (2026), illustrating
the relationship between f_closure and d_closure using four
constructed trajectory examples.

The four panels correspond to the 2x2 combinations of
large/small f_closure and large/small d_closure, demonstrating
that these two diagnostics are largely non-redundant and that
the large-f / small-d case is degenerate.

Output
------
figures/appendix_closure_examples.pdf

Usage
-----
    python appendix_closure_figure.py

Requirements
------------
    numpy, matplotlib, scipy, hysteresis_core (same directory)

Author: T. Terzic <tterzic@phy.uniri.hr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from hysteresis_core import compute_areas

Path("figures").mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Construct four illustrative trajectories
# ---------------------------------------------------------------------------

def case_large_f_large_d():
    """
    Large f_closure, large d_closure.
    A nearly straight open path: endpoints far apart, closing segment
    creates most of the enclosed area.
    Points trace a shallow arc from bottom-left to top-right.
    """
    t = np.linspace(0, 1, 10)
    x = t
    y = 0.15 * np.sin(np.pi * t)   # shallow arc, not a loop
    return x, y


def case_small_f_large_d():
    """
    Small f_closure, large d_closure.
    A large well-developed loop, but the source was not observed through
    a complete cycle: endpoints far apart, yet the open path already
    encloses most of the area.
    Points trace ~3/4 of an ellipse.
    """
    theta = np.linspace(0.1 * np.pi, 1.85 * np.pi, 14)
    x = np.cos(theta)
    y = 0.6 * np.sin(theta)
    return x, y


def case_small_f_small_d():
    """
    Small f_closure, small d_closure.
    A complete or nearly complete loop: endpoints close together,
    open path encloses almost all the area. The ideal case.
    Points trace nearly a full ellipse.
    """
    theta = np.linspace(0.05 * np.pi, 1.97 * np.pi, 16)
    x = np.cos(theta)
    y = 0.6 * np.sin(theta)
    return x, y


def case_large_f_small_d():
    """
    Large f_closure, small d_closure.
    Degenerate case: endpoints close together but A_tot ~ 0.
    The open path nearly cancels its own closure contribution
    (figure-eight like structure), so f_closure = |A_closure|/|A_tot|
    is large because A_tot is near zero even though A_closure is small.
    R_can is also small here, so criterion (4) flags this case too.
    """
    # A figure-eight: first half CCW, second half CW, endpoints close
    t1 = np.linspace(0, np.pi, 8)
    t2 = np.linspace(np.pi, 2 * np.pi, 8)
    x1 = 0.6 * np.cos(t1);      y1 =  0.4 * np.sin(t1)
    x2 = 0.6 * np.cos(t2) + 0.05; y2 = -0.4 * np.sin(t2)
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    return x, y


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

cases = [
    (case_large_f_large_d,  "Large $f_{\\rm cl}$, large $d_{\\rm cl}$",
     "Nearly straight open path\n"
     r"$A_{\rm open}$ contaminated by closing segment"),
    (case_small_f_large_d,  "Small $f_{\\rm cl}$, large $d_{\\rm cl}$",
     "Large loop, incomplete cycle\n"
     r"$A_{\rm open}$ reliable; source not fully observed"),
    (case_large_f_small_d,  "Large $f_{\\rm cl}$, small $d_{\\rm cl}$",
     "Degenerate: $A_{\\rm tot}\\approx 0$\n"
     r"$R_{\rm can}$ also small; caught by criterion (4)"),
    (case_small_f_small_d,  "Small $f_{\\rm cl}$, small $d_{\\rm cl}$",
     "Complete loop\n"
     r"$A_{\rm open}$ reliable; ideal case"),
]

fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), constrained_layout=True)

for ax, (fn, title, description) in zip(axes.flatten(), cases):
    x, y = fn()

    # Add small uncertainties so compute_areas gets sx, sy for d_closure
    N = len(x)
    sx = np.full(N, 0.03 * (x.max() - x.min()))
    sy = np.full(N, 0.03 * (y.max() - y.min()))

    r = compute_areas(x, y, sx, sy)

    # --- Plot trajectory ---
    cmap = plt.get_cmap("viridis")
    ax.plot(x, y, "-", color="0.75", lw=1.0, zorder=1)
    sc = ax.scatter(x, y, c=np.arange(N), cmap="viridis",
                    s=40, zorder=3, vmin=0, vmax=N - 1)

    # Mark first and last points
    ax.scatter([x[0]],  [y[0]],  s=90, marker="o", fc="white",
               ec="black", lw=1.5, zorder=4, label="Start")
    ax.scatter([x[-1]], [y[-1]], s=90, marker="s", fc="white",
               ec="black", lw=1.5, zorder=4, label="End")

    # Draw closing segment as dashed line
    ax.plot([x[-1], x[0]], [y[-1], y[0]], "--", color="tomato",
            lw=1.4, zorder=2, label="Closing segment")

    # Centroid
    cx, cy = x.mean(), y.mean()
    ax.scatter([cx], [cy], s=80, marker="+", color="purple",
               linewidths=2, zorder=6, label="Centroid")

    # A_closure triangle: vertices are centroid, P_N, P_1
    # Fill with semi-transparent colour to show the area
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    triangle = Polygon(
        [[cx, cy], [x[-1], y[-1]], [x[0], y[0]]],
        closed=True
    )
    patch = PatchCollection([triangle], alpha=0.25, facecolor="gold",
                             edgecolor="darkorange", linewidth=1.2,
                             zorder=2)
    ax.add_collection(patch)
    # Label the triangle with A_closure value
    # Place label at centroid of triangle
    tx = (cx + x[-1] + x[0]) / 3
    ty = (cy + y[-1] + y[0]) / 3
    ax.text(tx, ty, "$A_{\\rm cl}$", fontsize=7.5,
            ha="center", va="center", color="darkorange",
            fontweight="bold", zorder=7)

    # Arrows on trajectory
    step = max(1, N // 5)
    for i in range(0, N - 1, step):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        ax.annotate("",
                    xy=(x[i] + 0.65 * dx, y[i] + 0.65 * dy),
                    xytext=(x[i] + 0.35 * dx, y[i] + 0.35 * dy),
                    arrowprops=dict(arrowstyle="->", color="k",
                                    lw=0.9, mutation_scale=9),
                    zorder=5)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    # Statistics annotation
    f_cl  = r.f_closure
    d_cl  = r.d_closure
    R_can = r.R_cancel
    A_n   = r.A_norm

    f_str = f"{f_cl:.2f}" if np.isfinite(f_cl) else "undef"
    d_str = f"{d_cl:.1f}" if np.isfinite(d_cl) else "undef"
    R_str = f"{R_can:.2f}" if np.isfinite(R_can) else "undef"
    A_str = f"{A_n:+.3f}" if np.isfinite(A_n) else "undef"

    stats = (
        f"$A_{{\\rm norm}}={A_str}$\n"
        f"$f_{{\\rm cl}}={f_str}$\n"
        f"$d_{{\\rm cl}}={d_str}$\n"
        f"$R_{{\\rm can}}={R_str}$"
    )
    ax.text(0.97, 0.97, stats, transform=ax.transAxes,
            va="top", ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.9))

#    ax.text(0.03, 0.03, description, transform=ax.transAxes,
#            va="bottom", ha="left", fontsize=7.5, color="0.35",
#            style="italic")

# Shared legend
from matplotlib.patches import Patch
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Start"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="End"),
    plt.Line2D([0], [0], color="tomato", ls="--", lw=1.4,
               label="Closing segment"),
    plt.Line2D([0], [0], marker="+", color="purple", markersize=10,
               linewidth=0, markeredgewidth=2, label="Centroid"),
    Patch(facecolor="gold", edgecolor="darkorange", alpha=0.5,
          label=r"$A_{\rm closure}$ triangle"),
]
#fig.legend(handles=handles, loc="lower center", ncol=5,
#           bbox_to_anchor=(0.5, -0.03), fontsize=9, framealpha=0.9)

#fig.suptitle(
#    r"Relationship between $f_{\rm closure}$ and $d_{\rm closure}$: four illustrative cases",
#    fontsize=11
#)

outfile = "results/appendix_closure_examples.pdf"
plt.savefig(outfile, bbox_inches="tight")
print(f"Saved {outfile}")
plt.close(fig)
