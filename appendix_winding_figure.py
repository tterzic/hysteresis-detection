"""
appendix_winding_figure.py
==========================
Generate the figure for Appendix B of Terzic (2026), illustrating
the rare cases in which |A_norm| > 1 due to self-intersecting
(double-winding) trajectories.

Three panels are produced:

  Left:   Explicit double-winding trajectory (N=14) with |A_norm| > 1,
          showing the outer and inner loops, the convex hull, and the
          closing segment.  Point coordinates are printed to stdout.

  Centre: Distribution of |A_norm| for all three null models at
          four sample sizes (K=50 000 realisations each), with a
          dashed line at |A_norm| = 1.

  Right:  Fraction of null realisations with |A_norm| > 1 as
          a function of sample size N, for all three null models.

For the permutation and Fourier nulls, a base trajectory is required.
To avoid sensitivity to the geometry of any single trajectory (which is
particularly important for the permutation null), results are averaged
over N_base=10 trajectories drawn from the toy-model pool at each N.

Output
------
results/winding_number_analysis.pdf

Usage
-----
    python appendix_winding_figure.py [--K 50000] [--quick]

    --K INT     Number of null realisations per base trajectory per null model
                (default 5 000; total realisations per null per N = K x N_base).
    --quick     Use K=2000 for a fast test run.

Requirements
------------
    numpy, matplotlib, scipy
    hysteresis_core.py       -- area computation (same directory)
    generate_candidates.py   -- toy model pool (same directory)
    null_gallery.py          -- single-realisation null generators (same directory)

Author: T. Terzić <tterzic@phy.uniri.hr>
License: MIT
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import sys
sys.path.insert(0, str(Path(__file__).parent))
from hysteresis_core import compute_areas
from generate_candidates import generate_pool
from null_gallery import _null_ar1_once, _null_permutation_once, _null_fourier_once

plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.labelsize" : 11,
    "axes.titlesize" : 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top"      : True,
    "ytick.right"    : True,
    "figure.dpi"     : 150,
})

# Colours for the three null models — consistent with diagnostic figure
NULL_COLOURS = {
    "Permutation": "darkorange",
    "AR(1)"      : "seagreen",
    "Fourier"    : "mediumpurple",
}

NULL_FNS = {
    "Permutation": _null_permutation_once,
    "AR(1)"      : _null_ar1_once,
    "Fourier"    : _null_fourier_once,
}

# Toy model parameters — identical to signal_gallery.py defaults
BASE_KWARGS = dict(
    delta_t=1.0,
    sigma_HR_rise=0.5, sigma_HR_decay=2.0,
    sigma_F_rise=2.0,  sigma_F_decay=0.5,
    sigma_noise=0.05,
    n_sigma_rise=2.0,  n_sigma_decay=2.0,
)


# ---------------------------------------------------------------------------
# Helper: compute |A_norm| with convex hull normalisation
# ---------------------------------------------------------------------------

def anorm_hull(x, y):
    """Return |A_norm| for an open path, or NaN if hull is degenerate."""
    cx, cy = x.mean(), y.mean()
    xt, yt = x - cx, y - cy
    a = 0.5 * (xt[:-1] * yt[1:] - xt[1:] * yt[:-1])
    A_open = np.sum(a)
    try:
        hull = ConvexHull(np.column_stack([xt, yt]))
        return abs(A_open) / hull.volume
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Construct explicit double-winding trajectory
# ---------------------------------------------------------------------------

def make_double_winding():
    """
    Construct a 14-point open path that winds twice CCW around the origin:
    first a 7-point circle of radius 2, then a 7-point circle of radius 1.

    Returns
    -------
    x, y : ndarray, shape (14,)
    """
    theta = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    x = np.concatenate([2.0 * np.cos(theta), 1.0 * np.cos(theta)])
    y = np.concatenate([2.0 * np.sin(theta), 1.0 * np.sin(theta)])
    return x, y


# ---------------------------------------------------------------------------
# Null model simulation
# ---------------------------------------------------------------------------

def get_base_trajectories(N, N_base=10, seed=42):
    """
    Return N_base toy-model trajectories of length N, drawn evenly from
    a pool of 10*N_base trajectories with the given seed.

    Using multiple base trajectories averages out the dependence of the
    null distribution on the specific geometry of any single trajectory,
    giving a more robust estimate of the prevalence of |A_norm| > 1.
    """
    rng = np.random.default_rng(seed)
    kwargs = dict(N=N, **BASE_KWARGS)
    pool = generate_pool(10 * N_base, kwargs, rng)
    indices = np.round(np.linspace(0, len(pool) - 1, N_base)).astype(int)
    return [pool[i] for i in indices]


def simulate_null_anorms(null_name, N, K, N_base=10, seed_null=99):
    """
    Generate K null realisations per base trajectory for the given null
    model and sample size N, averaged over N_base base trajectories.

    Returns the mean fraction of realisations with |A_norm| > 1 and the
    full array of |A_norm| values pooled across all base trajectories.

    Using multiple base trajectories is important because the permutation
    null is sensitive to the specific geometry of the base trajectory:
    self-intersecting base trajectories produce systematically more
    |A_norm| > 1 events than clean single-loop trajectories.
    """
    trajectories = get_base_trajectories(N, N_base=N_base)
    fn = NULL_FNS[null_name]
    all_anorms = []
    for x, y, sx, sy in trajectories:
        rng = np.random.default_rng(seed_null)
        for _ in range(K):
            xs, ys = fn(x, y, rng)
            v = anorm_hull(xs, ys)
            if np.isfinite(v):
                all_anorms.append(v)
    return np.array(all_anorms)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(K=5_000, outdir="results"):

    # --- Double-winding example ---
    x_dw, y_dw = make_double_winding()
    N_dw = len(x_dw)

    r_dw = compute_areas(x_dw, y_dw)
    cx, cy = x_dw.mean(), y_dw.mean()
    xt, yt = x_dw - cx, y_dw - cy
    hull_dw = ConvexHull(np.column_stack([xt, yt]))
    A_hull_dw = hull_dw.volume

    print("=" * 56)
    print("  Double-winding example")
    print("=" * 56)
    print(f"  N = {N_dw}")
    print("\n  Coordinates (x, y):")
    for i, (xi, yi) in enumerate(zip(x_dw, y_dw)):
        print(f"    P{i+1:02d}: ({xi:+.4f}, {yi:+.4f})")
    print(f"\n  A_open    = {r_dw.A_open:+.4f}")
    print(f"  A_closure = {r_dw.A_closure:+.4f}")
    print(f"  A_tot     = {r_dw.A_tot:+.4f}")
    print(f"  A_hull    = {A_hull_dw:.4f}")
    print(f"  |A_norm|  = {abs(r_dw.A_norm):.4f}  (> 1: {abs(r_dw.A_norm) > 1})")
    print(f"  R_cancel  = {r_dw.R_cancel:.4f}")
    print("=" * 56)

    # --- Null distributions at four sample sizes ---
    N_sizes_dist = [10, 15, 20, 30]
    print("\nSimulating null distributions...")
    dist_results = {}   # dist_results[N][null_name] = anorm array
    for N_sim in N_sizes_dist:
        dist_results[N_sim] = {}
        for null_name in NULL_FNS:
            print(f"  N={N_sim:2d}  {null_name:12s}  K={K}...", end=" ", flush=True)
            anorms = simulate_null_anorms(null_name, N_sim, K)
            dist_results[N_sim][null_name] = anorms
            frac = np.mean(anorms > 1.0)
            print(f"frac>1: {frac:.4f}  max: {np.max(anorms):.3f}")

    # --- Fraction > 1 vs N ---
    N_range = [6, 8, 10, 15, 20, 30, 40]
    print("\nComputing fraction > 1 vs N...")
    frac_results = {null_name: [] for null_name in NULL_FNS}
    for N_sim in N_range:
        for null_name in NULL_FNS:
            anorms = simulate_null_anorms(null_name, N_sim, K, seed_null=100)
            f = np.mean(anorms > 1.0)
            frac_results[null_name].append(f)
            print(f"  N={N_sim:2d}  {null_name:12s}  frac>1={f:.4f} ({100*f:.2f}%)")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 4.8), constrained_layout=True)
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

    # --- Panel A: double-winding trajectory (unchanged) ---
    ax1 = fig.add_subplot(gs[0])

    hv = np.append(hull_dw.vertices, hull_dw.vertices[0])
    ax1.fill(xt[hv] + cx, yt[hv] + cy, alpha=0.12, color="steelblue")
#    ax1.plot(np.append(xt[hv] + cx, xt[hv[0]] + cx),
#             np.append(yt[hv] + cy, yt[hv[0]] + cy),
#             "b--", lw=1.0, alpha=0.5, label="Convex hull")

    ax1.plot(x_dw, y_dw, "-", color="0.75", lw=1.0, zorder=2)
    sc = ax1.scatter(x_dw, y_dw, c=np.arange(N_dw), cmap="viridis",
                     s=55, zorder=4, vmin=0, vmax=N_dw - 1)
    plt.colorbar(sc, ax=ax1, label="Time step", shrink=0.75)

    for i, (xi, yi) in enumerate(zip(x_dw, y_dw)):
        ax1.annotate(f"P{i+1}", (xi, yi),
                     textcoords="offset points", xytext=(4, 3),
                     fontsize=7, color="0.3")

    for i in range(0, N_dw - 1, 3):
        dx = x_dw[i + 1] - x_dw[i]
        dy = y_dw[i + 1] - y_dw[i]
        ax1.annotate("",
            xy=(x_dw[i] + 0.65*dx, y_dw[i] + 0.65*dy),
            xytext=(x_dw[i] + 0.35*dx, y_dw[i] + 0.35*dy),
            arrowprops=dict(arrowstyle="->", color="k",
                            lw=1.0, mutation_scale=10),
            zorder=5)

    ax1.plot([x_dw[-1], x_dw[0]], [y_dw[-1], y_dw[0]],
             "r--", lw=1.5, zorder=3, label="Closing segment")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_title(
        f"Double-winding path ($N={N_dw}$)\n"
        f"$|A_{{\\rm open}}|={abs(r_dw.A_open):.2f}$, "
        f"$A_{{\\rm hull}}={A_hull_dw:.2f}$, "
        f"$|A_{{\\rm norm}}|={abs(r_dw.A_norm):.2f}$",
        fontsize=9)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_aspect("equal")

    # --- Panel B: distribution of |A_norm| for all three nulls ---
    ax2 = fig.add_subplot(gs[1])
    bins = np.linspace(0, 2.0, 60)
    linestyles = ["-", "--", ":", "-."]
    for j, N_sim in enumerate(N_sizes_dist):
        for null_name, colour in NULL_COLOURS.items():
            anorms = dist_results[N_sim][null_name]
            ax2.hist(anorms, bins=bins, density=True,
                     histtype="step", color=colour,
                     linestyle=linestyles[j], lw=0.9, alpha=0.85,
                     label=f"{null_name}, $N={N_sim}$" if j == 0 else None)

    ax2.axvline(1.0, color="red", ls="--", lw=1.8,
                label="$|A_{\\rm norm}|=1$")
    ax2.set_xlabel(r"$|A_{\rm norm}|$")
    ax2.set_ylabel("Density")
    ax2.set_title(
        f"Distribution of $|A_{{\\rm norm}}|$\n($K={K:,}$ per null per $N$)",
        fontsize=9)
    ax2.set_xlim(0, 2.0)

    # Two-part legend: null colours + N linestyles
    from matplotlib.lines import Line2D
    legend_null = [Line2D([0], [0], color=c, lw=1.5, label=n)
                   for n, c in NULL_COLOURS.items()]
    legend_N    = [Line2D([0], [0], color="k", lw=1.0,
                          linestyle=linestyles[j], label=f"$N={N_sim}$")
                   for j, N_sim in enumerate(N_sizes_dist)]
    ax2.legend(handles=legend_N + legend_null, fontsize=7, ncol=1)

    # --- Panel C: fraction > 1 vs N for all three nulls ---
    ax3 = fig.add_subplot(gs[2])
    for null_name, colour in NULL_COLOURS.items():
        fracs = frac_results[null_name]
        ax3.plot(N_range, [100 * f for f in fracs],
                 "o-", color=colour, lw=1.5, ms=5, label=null_name)

    ax3.axhline(0, color="0.7", lw=0.8)
    ax3.set_xlabel("Sample size $N$")
    ax3.set_ylabel(r"Fraction with $|A_{\rm norm}|>1$ (%)")
    ax3.set_title(r"Frequency of $|A_{\rm norm}|>1$", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0.001, top=20.)
    ax3.set_yscale("log")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    outpath = Path(outdir) / "winding_number_analysis.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    print(f"\nSaved {outpath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--K", type=int, default=5_000,
        help="Null realisations per model per sample size (default 5 000).")
    parser.add_argument("--outdir", type=str, default="results",
        help="Output directory (default: results).")
    parser.add_argument("--quick", action="store_true",
        help="Quick test run with K=2000.")
    args = parser.parse_args()

    K = 2_000 if args.quick else args.K
    if args.quick:
        print("Running in QUICK mode (K=2000).")

    make_figure(K=K, outdir=args.outdir)
