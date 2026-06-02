"""
generate_candidates.py
======================
Generate a PDF of K random trajectories in the (F, HR) plane for visual inspection and selection. Used to select the four representative examples for signal_gallery.py.

Each trajectory is assigned a 1-indexed number. The PDF has pages of per_page panels each (4 rows x 5 columns), with statistics annotated in each panel title.

The pool is generated with the same parameters and seed as signal_gallery.py so that pool indices are directly transferable: if you select trajectory #777 here, passing --idx_a 777 to signal_gallery.py gives the same curve.

Usage
-----
    python generate_candidates.py [--outdir results] [--seed 42] [--K 1000]

Output
------
    results/random_curves.pdf

Dependencies
------------
    numpy, matplotlib
    hysteresis_core.py  -- must be in the same directory

Author: T. Terzić <tterzic@phy.uniri.hr>
License: MIT
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent))
from hysteresis_core import compute_areas

plt.rcParams.update({"font.family": "serif", "font.size": 6})


# ---------------------------------------------------------------------------
# Flare model
# ---------------------------------------------------------------------------

def asymmetric_gaussian(t, t_peak, sigma_rise, sigma_decay, amplitude):
    """
    Asymmetric Gaussian centred on t_peak.

    Returns 1 + amplitude * exp(-0.5 * ((t - t_peak) / sigma)^2) where sigma = sigma_rise for t < t_peak, sigma_decay otherwise.
    The baseline (quiescent) value is 1.
    """
    sigma = np.where(t < t_peak, sigma_rise, sigma_decay)
    return 1.0 + amplitude * np.exp(-0.5 * ((t - t_peak) / sigma) ** 2)


def make_trajectory(
    N, delta_t,
    sigma_HR_rise, sigma_HR_decay,
    sigma_F_rise, sigma_F_decay,
    A_HR=0.8, A_F=0.8,
    sigma_noise=0.05,
    n_sigma_rise=2.0, n_sigma_decay=2.0,
    rng=None,
):
    """
    Generate one (F, HR) trajectory at N randomly spaced times.

    The analysis convention is F on the x-axis and HR on the y-axis, so the trajectory is returned as (x=F, y=HR) with matching uncertainties.
    Measurement uncertainties are fractional:
      sx = sigma_noise * F    (x-axis uncertainty)
      sy = sigma_noise * HR   (y-axis uncertainty)

    Parameters
    ----------
    N : int
    delta_t : float         t_F - t_HR (positive: F peaks after HR)
    sigma_HR_rise/decay : float   HR profile rise/decay widths
    sigma_F_rise/decay : float    F profile rise/decay widths
    A_HR, A_F : float       Fractional flare amplitudes above quiescent=1
    sigma_noise : float     Fractional measurement uncertainty
    n_sigma_rise/decay : float  Sampling window multipliers
    rng : numpy Generator

    Returns
    -------
    x, y   : ndarray  Noise-free F and HR values (x=F, y=HR)
    sx, sy : ndarray  Fractional 1-sigma uncertainties on x and y
    """
    if rng is None:
        rng = np.random.default_rng()
    t_HR, t_F = 0.0, delta_t
    t_start = min(t_HR, t_F) - n_sigma_rise  * max(sigma_HR_rise, sigma_F_rise)
    t_end   = max(t_HR, t_F) + n_sigma_decay * max(sigma_HR_decay, sigma_F_decay)
    t  = np.sort(rng.uniform(t_start, t_end, N))
    hr = asymmetric_gaussian(t, t_HR, sigma_HR_rise, sigma_HR_decay, A_HR)
    f  = asymmetric_gaussian(t, t_F,  sigma_F_rise,  sigma_F_decay,  A_F)
    # x = F (horizontal axis), y = HR (vertical axis)
    x  = f
    y  = hr
    sx = sigma_noise * x   # fractional uncertainty on F
    sy = sigma_noise * y   # fractional uncertainty on HR
    return x, y, sx, sy


def generate_pool(K, kwargs, rng):
    """
    Generate K trajectories.

    Returns a list of tuples (x, y, sx, sy) where x=F, y=HR.
    """
    pool = []
    while len(pool) < K:
        x, y, sx, sy = make_trajectory(**kwargs, rng=rng)
        pool.append((x, y, sx, sy))
    return pool


# ---------------------------------------------------------------------------
# Draw one panel
# ---------------------------------------------------------------------------

def draw_panel(ax, idx, x, y, sx, sy, res):
    """
    Draw one trajectory panel with statistics in the title.
    x = F (x-axis), y = HR (y-axis).
    """
    N = len(x)
    cmap   = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.15, 0.90, N))

    for i in range(N):
        ax.errorbar(x[i], y[i], xerr=sx[i], yerr=sy[i],
                    fmt="none", ecolor=colors[i], elinewidth=0.5,
                    capsize=1.0, capthick=0.5, zorder=2)
    ax.plot(x, y, "-", color="0.78", lw=0.6, zorder=1)
    for i in range(N):
        ax.scatter(x[i], y[i], color=colors[i], s=8, linewidths=0, zorder=3)
    ax.scatter(x[0],  y[0],  marker="o", s=14, color=colors[0],
               edgecolors="k", linewidths=0.4, zorder=4)
    ax.scatter(x[-1], y[-1], marker="s", s=14, color=colors[-1],
               edgecolors="k", linewidths=0.4, zorder=4)

    # Arrow at largest gap
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    i_arr = int(np.argmax(dists))
    ax.annotate("", xy=(x[i_arr+1], y[i_arr+1]),
                xytext=(x[i_arr], y[i_arr]),
                arrowprops=dict(arrowstyle="-|>", color="0.25",
                                lw=0.7, mutation_scale=6))

    f_cl = res.f_closure if np.isfinite(res.f_closure) else float("nan")
    d_cl = res.d_closure if np.isfinite(res.d_closure) else float("nan")
    ax.set_title(
        f"#{idx}  $A_n={res.A_norm:+.2f}$  $R_c={res.R_cancel:.2f}$\n"
        f"$f_c={f_cl:.2f}$  $d_c={d_cl:.1f}$",
        fontsize=5.5)
    ax.set_xlabel("$F$ (arb. units)", fontsize=5.5)
    ax.set_ylabel("$HR$", fontsize=5.5)
    ax.tick_params(labelsize=4.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_candidates(
    K             : int   = 1000,
    N             : int   = 10,
    delta_t       : float = 1.0,
    sigma_HR_rise : float = 0.5,
    sigma_HR_decay: float = 2.0,
    sigma_F_rise  : float = 2.0,
    sigma_F_decay : float = 0.5,
    sigma_noise   : float = 0.05,
    n_sigma_rise  : float = 2.0,
    n_sigma_decay : float = 2.0,
    per_page      : int   = 20,
    seed          : int   = 42,
    outdir        : str   = "results",
):
    """
    Generate K trajectories and write them to a multi-page PDF for visual selection. Pool indices are 1-indexed and match signal_gallery.py when using identical parameters and seed.
    """
    rng = np.random.default_rng(seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        N=N, delta_t=delta_t,
        sigma_HR_rise=sigma_HR_rise, sigma_HR_decay=sigma_HR_decay,
        sigma_F_rise=sigma_F_rise,   sigma_F_decay=sigma_F_decay,
        sigma_noise=sigma_noise,
        n_sigma_rise=n_sigma_rise,   n_sigma_decay=n_sigma_decay,
    )

    print(f"Generating {K} trajectories (seed={seed})...")
    pool = generate_pool(K, kwargs, rng)
    print(f"Done. Computing statistics...")

    # Compute geometric statistics for each trajectory.
    # These are used for panel annotations only; the analysis is not
    # performed here. Use analyse_hysteresis.py on the CSV files instead.
    stats = [compute_areas(x, y, sx, sy) for x, y, sx, sy in pool]
    print(f"Done. Writing PDF...")

    ncols   = 5
    nrows   = per_page // ncols
    n_pages = len(pool) // per_page

    outpath = str(Path(outdir) / "random_curves.pdf")
    with PdfPages(outpath) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(14, 3.2 * nrows),
                                     constrained_layout=True)
            fig.suptitle(
                f"Random trajectories — page {page+1}/{n_pages} "
                f"(curves {page*per_page+1}–{(page+1)*per_page})",
                fontsize=9)
            for k, ax in enumerate(axes.flatten()):
                idx = page * per_page + k + 1   # 1-indexed, matches signal_gallery.py
                x, y, sx, sy = pool[idx - 1]
                draw_panel(ax, idx, x, y, sx, sy, stats[idx - 1])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            if (page + 1) % 10 == 0:
                print(f"  Page {page+1}/{n_pages}")

    print(f"Saved: {outpath}")
    print(f"\nTo use a selected curve in signal_gallery.py, pass its")
    print(f"number as --idx_a / --idx_b / --idx_c / --idx_d.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate PDF of K random trajectories for visual selection. "
            "Pool indices match signal_gallery.py when using the same "
            "parameters and seed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--K",              type=int,   default=1000)
    parser.add_argument("--N",              type=int,   default=10)
    parser.add_argument("--delta_t",        type=float, default=1.0)
    parser.add_argument("--sigma_HR_rise",  type=float, default=0.5)
    parser.add_argument("--sigma_HR_decay", type=float, default=2.0)
    parser.add_argument("--sigma_F_rise",   type=float, default=2.0)
    parser.add_argument("--sigma_F_decay",  type=float, default=0.5)
    parser.add_argument("--sigma_noise",    type=float, default=0.05)
    parser.add_argument("--n_sigma_rise",   type=float, default=2.0)
    parser.add_argument("--n_sigma_decay",  type=float, default=2.0)
    parser.add_argument("--per_page",       type=int,   default=20)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--outdir",         type=str,   default="results")
    args = parser.parse_args()

    generate_candidates(
        K             = args.K,
        N             = args.N,
        delta_t       = args.delta_t,
        sigma_HR_rise = args.sigma_HR_rise,
        sigma_HR_decay= args.sigma_HR_decay,
        sigma_F_rise  = args.sigma_F_rise,
        sigma_F_decay = args.sigma_F_decay,
        sigma_noise   = args.sigma_noise,
        n_sigma_rise  = args.n_sigma_rise,
        n_sigma_decay = args.n_sigma_decay,
        per_page      = args.per_page,
        seed          = args.seed,
        outdir        = args.outdir,
    )
