"""
plot_arms.py
============
Generate Fig. C.1 of Terzic (2026): A_rms vs A_abs for a pool of synthetic
trajectories from the toy flare model, colour-coded by R_cancel.

This script generates K trajectories using the same toy model and parameters
as signal_gallery.py, computes the geometric statistics for each, and plots
A_rms vs A_abs with trajectories grouped into five subsets by R_cancel value.

The two reference lines shown are the theoretical bounds:
  - Upper bound (solid):   A_rms = A_abs        (single non-zero triangle)
  - Lower bound (dashed):  A_rms = A_abs/sqrt(N-1) (all triangles equal magnitude)
All trajectories must fall within the band between these two lines.

Usage
-----
    python plot_arms.py [--K 10000] [--seed 42] [--outdir results]

Output
------
    results/abs_vs_rms.pdf

Dependencies
------------
    numpy, matplotlib
    hysteresis_core.py       -- area computation
    generate_candidates.py   -- flare model and pool generation

Author: T. Terzić <tterzic@phy.uniri.hr>
License: MIT
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from generate_candidates import generate_pool
from hysteresis_core import compute_areas

plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 8,
    "axes.labelsize" : 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top"      : True,
    "ytick.right"    : True,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi"     : 200,
})


def generate_stats(K, seed, kwargs):
    """
    Generate K trajectories and return their geometric statistics.

    A_abs and A_rms are returned normalised by the convex hull area (A_hull)
    so they are dimensionless and directly comparable to A_norm.
    Only trajectories with finite statistics are retained.
    """
    rng = np.random.default_rng(seed)
    print(f"Generating {K} trajectories (seed={seed})...")
    pool = generate_pool(K, kwargs, rng)
    print("Computing statistics...")
    stats = [compute_areas(x, y, sx, sy) for x, y, sx, sy in pool]

    A_abs = np.array([r.A_abs  / r.A_hull for r in stats])
    A_rms = np.array([r.A_rms  / r.A_hull for r in stats])
    R_can = np.array([r.R_cancel          for r in stats])

    valid = np.isfinite(A_abs) & np.isfinite(A_rms) & np.isfinite(R_can)
    print(f"Valid trajectories: {valid.sum()} / {K}")
    return A_abs[valid], A_rms[valid], R_can[valid]


def plot_arms(K=10_000, seed=42, outdir="results",
              N=10, delta_t=1.0,
              sigma_HR_rise=0.5, sigma_HR_decay=2.0,
              sigma_F_rise=2.0,  sigma_F_decay=0.5,
              sigma_noise=0.05,
              n_sigma_rise=2.0,  n_sigma_decay=2.0):
    """
    Generate the A_rms vs A_abs figure and save to outdir/abs_vs_rms.pdf.

    Parameters match generate_candidates.py defaults to ensure the pool is
    drawn from the same distribution as the paper examples.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        N=N, delta_t=delta_t,
        sigma_HR_rise=sigma_HR_rise, sigma_HR_decay=sigma_HR_decay,
        sigma_F_rise=sigma_F_rise,   sigma_F_decay=sigma_F_decay,
        sigma_noise=sigma_noise,
        n_sigma_rise=n_sigma_rise,   n_sigma_decay=n_sigma_decay,
    )

    A_abs, A_rms, R_can = generate_stats(K, seed, kwargs)

    # Five subsets by R_cancel value
    subsets = [
        (R_can <= 0.65,                r'$R_\mathrm{can} \leq 0.65$',           'firebrick'),
        ((R_can>0.65)&(R_can<=0.80),   r'$0.65 < R_\mathrm{can} \leq 0.80$',   'darkorange'),
        ((R_can>0.80)&(R_can<=0.95),   r'$0.80 < R_\mathrm{can} \leq 0.95$',   'goldenrod'),
        ((R_can>0.95)&(R_can<1.0),     r'$0.95 < R_\mathrm{can} < 1$',         'seagreen'),
        (R_can==1.0,                    r'$R_\mathrm{can} = 1$',                'steelblue'),
    ]

    fig, ax = plt.subplots(figsize=(3.46, 3.20), constrained_layout=True)

    for mask, label, colour in subsets:
        ax.scatter(A_abs[mask], A_rms[mask], s=2, alpha=0.3,
                   color=colour, label=f'{label} (N={mask.sum()})', linewidths=0)

    lim = max(A_abs.max(), A_rms.max()) * 1.05

    # Upper bound: A_rms = A_abs (single non-zero triangle contribution)
    ax.plot([0, lim], [0, lim], 'k-', lw=0.8, alpha=0.5,
            label=r'$A_\mathrm{rms} = A_\mathrm{abs}$')

    # Lower bound: A_rms = A_abs / sqrt(N-1) (all N-1 triangles equal magnitude)
    ax.plot([0, lim], [0, lim / np.sqrt(N - 1)], 'k--', lw=0.8, alpha=0.5,
            label=rf'$A_\mathrm{{rms}} = A_\mathrm{{abs}}/\sqrt{{{N-1}}}$')

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r'$A_\mathrm{abs} / A_\mathrm{hull}$')
    ax.set_ylabel(r'$A_\mathrm{rms} / A_\mathrm{hull}$')
    ax.legend(fontsize=5.5, markerscale=4)

    outpath = Path(outdir) / "abs_vs_rms.pdf"
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate A_rms vs A_abs figure (Fig. C.1 of Terzic 2026).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--K",              type=int,   default=10000, help="Number of trajectories.")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--outdir",         type=str,   default="results")
    parser.add_argument("--N",              type=int,   default=10)
    parser.add_argument("--delta_t",        type=float, default=1.0)
    parser.add_argument("--sigma_HR_rise",  type=float, default=0.5)
    parser.add_argument("--sigma_HR_decay", type=float, default=2.0)
    parser.add_argument("--sigma_F_rise",   type=float, default=2.0)
    parser.add_argument("--sigma_F_decay",  type=float, default=0.5)
    parser.add_argument("--sigma_noise",    type=float, default=0.05)
    parser.add_argument("--n_sigma_rise",   type=float, default=2.0)
    parser.add_argument("--n_sigma_decay",  type=float, default=2.0)
    args = parser.parse_args()

    plot_arms(
        K             = args.K,
        seed          = args.seed,
        outdir        = args.outdir,
        N             = args.N,
        delta_t       = args.delta_t,
        sigma_HR_rise = args.sigma_HR_rise,
        sigma_HR_decay= args.sigma_HR_decay,
        sigma_F_rise  = args.sigma_F_rise,
        sigma_F_decay = args.sigma_F_decay,
        sigma_noise   = args.sigma_noise,
        n_sigma_rise  = args.n_sigma_rise,
        n_sigma_decay = args.n_sigma_decay,
    )
