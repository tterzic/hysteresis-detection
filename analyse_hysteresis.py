#!/usr/bin/env python3
"""
analyse_hysteresis.py
=====================
Command-line tool for applying the hysteresis detection framework to a single trajectory supplied as a CSV file.

Usage
-----
    python analyse_hysteresis.py <input.csv> [options]

Arguments
---------
    input.csv       CSV file with four columns: x observable, y observable, sx uncertainty, sy uncertainty.
                    The first non-comment line is the header row; its four comma-separated values are used as axis labels in figures and output. Lines beginning with '#' are ignored.
                    Set sx=sy=0 for null/synthetic trajectories (no measurement process); MC propagation and d_closure are then skipped.

Options
-------
    --K_mc   INT    Monte Carlo realisations for uncertainty propagation (default: 10000; ignored if sx=sy=0 throughout)
    --K_null INT    Surrogate realisations per null model (default: 10000)
    --seed   INT    Random seed for reproducibility (default: 42)
    --outdir DIR    Directory for all file outputs (default: results)
    --latex         Write a LaTeX table row to <outdir>/<stem>_results.tex
    --no-fig        Suppress figure output

Examples
--------
    python analyse_hysteresis.py results/signal_a.csv --K_null 10000 --latex --outdir results
    python analyse_hysteresis.py mrk421_data.csv --K_null 10000 --latex --outdir results

Output
------
  stdout     : human-readable summary table
  <stem>_trajectory.pdf  : single-panel trajectory figure (always written unless --no-fig is passed)
  <stem>_diagnostics.pdf : 6-panel figure (3 rows x 2 columns): MC distribution; permutation; AR(1); Fourier; combined null distributions; and a summary panel with A_norm; CI; and colour-coded p-values. Always written unless --no-fig is passed.
  <stem>_results.tex     : LaTeX table row (only if --latex is passed)

Author: T. Terzić <tterzic@phy.uniri.hr>
License: MIT
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Imports from hysteresis_core
# ---------------------------------------------------------------------------
try:
    from hysteresis_core import analyse
except ImportError as exc:
    sys.exit(
        f"Cannot import hysteresis_core: {exc}\n"
        "Make sure hysteresis_core.py is in the same directory."
    )


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def read_csv(path: Path):
    """
    Read a CSV file with four columns: x observable, y observable, sx uncertainty, sy uncertainty.

    The first non-comment line is the header row. Its four comma-separated values are used as axis labels in figures and output. All subsequent lines are data rows read by column position.
    Lines beginning with '#' are treated as comments and skipped.

    Returns
    -------
    x, y, sx, sy : ndarray
    x_label, y_label : str   Axis labels from the CSV header row.
    """
    non_comment_rows = []
    with open(path, newline="") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            non_comment_rows.append(stripped)

    if not non_comment_rows:
        raise ValueError(f"No data found in {path}")

    # First non-comment row is the header; extract axis labels
    header = [col.strip() for col in non_comment_rows[0].split(",")]
    x_label  = header[0]
    y_label  = header[1]

    x, y, sx, sy = [], [], [], []
    for row in non_comment_rows[1:]:
        parts = row.split(",")
        x.append(float(parts[0]))
        y.append(float(parts[1]))
        sx.append(float(parts[2]))
        sy.append(float(parts[3]))

    return np.array(x), np.array(y), np.array(sx), np.array(sy), x_label, y_label

# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def fmt_val(v, digits=4):
    """Format a float to `digits` decimal places, or return '---'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    return f"{v:.{digits}f}"


def fmt_pval(p):
    """Format a p-value; show <0.001 when appropriate."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def orientation_str(o) -> str:
    """Return orientation label from HysteresisResult."""
    if o is None or o == "none":
        return "---"
    return o if o in ("CCW", "CW") else "---"


def print_summary(label: str, res, x_label: str = "x", y_label: str = "y"):
    """Print a human-readable summary table to stdout."""
    sep = "-" * 56
    has_mc  = res.mc_ci_low is not None
    has_dcl = res.d_closure is not None and not math.isnan(res.d_closure)

    print()
    print(sep)
    print(f"  analyse_hysteresis  —  {label}")
    print(f"  x: {x_label}    y: {y_label}")
    print(sep)
    print(f"  N (observations)       : {res.N}")
    print()
    print("  Geometric statistics")
    print(f"    A_norm               : {fmt_val(res.A_norm, 4)}"
          f"  ({orientation_str(res.orientation)})")
    print(f"    R_can                : {fmt_val(res.R_cancel, 4)}")
    print(f"    f_cl                 : {fmt_val(res.f_closure, 4)}")
    if has_dcl:
        print(f"    d_cl                 : {fmt_val(res.d_closure, 4)}")
    else:
        print(f"    d_cl                 : ---  (sx=sy=0)")
    print()
    if has_mc:
        print("  MC uncertainty propagation (1-sigma interval)")
        print(f"    A_norm CI            : [{fmt_val(res.mc_ci_low, 4)}, "
              f"{fmt_val(res.mc_ci_high, 4)}]")
        print(f"    MC mean              : {fmt_val(res.mc_mean, 4)}")
        print(f"    MC std               : {fmt_val(res.mc_std, 4)}")
    else:
        print("  MC uncertainty propagation  : skipped (sx=sy=0)")
    print()
    print("  Significance (two-sided empirical p-values)")
    print(f"    p_perm               : {fmt_pval(res.p_value_perm)}")
    print(f"    p_AR                 : {fmt_pval(res.p_value_ar)}")
    print(f"    p_Fourier            : {fmt_pval(res.p_value_fourier)}")
    print(f"    p_full               : {fmt_pval(res.p_value_combined)}")
    print(sep)
    print()


def write_latex(label: str, res, outpath: Path, x_label: str = "x", y_label: str = "y"):
    """
    Write a LaTeX table row (wrapped in a standalone table* environment) suitable for inclusion in tab:worked.

    Columns: Case | N | A_norm | MC 1sigma CI | Orientation | R_can | f_cl | d_cl | p_perm | p_AR | p_Fourier | p_full
    """
    has_mc  = res.mc_ci_low is not None
    has_dcl = res.d_closure is not None and not math.isnan(res.d_closure)

    orient  = orientation_str(res.orientation)
    ci_str  = (f"[{fmt_val(res.mc_ci_low, 2)}, {fmt_val(res.mc_ci_high, 2)}]"
               if has_mc else "---")
    dcl_str = fmt_val(res.d_closure, 2) if has_dcl else "---"

    def pv(p):
        return fmt_pval(p).replace("<", r"$<$")

    row = (
        f"{label} & "
        f"{fmt_val(res.A_norm, 2)} & "
        f"{ci_str} & "
        f"{orient} & "
        f"{fmt_val(res.R_cancel, 2)} & "
        f"{fmt_val(res.f_closure, 2)} & "
        f"{dcl_str} & "
        f"{pv(res.p_value_perm)} & "
        f"{pv(res.p_value_ar)} & "
        f"{pv(res.p_value_fourier)} & "
        f"{pv(res.p_value_combined)} \\\\"
    )

    latex = (
        r"\begin{table*}" + "\n"
        r"\centering" + "\n"
        r"\caption{Hysteresis analysis results. "
        r"Columns: trajectory label; "
        r"normalised signed area $A_\mathrm{norm}$; "
        r"1-sigma Monte Carlo interval on $A_\mathrm{norm}$ "
        r"(omitted when $s_f = s_\mathrm{hr} = 0$); "
        r"loop orientation (CCW/CW); "
        r"cancellation ratio $R_\mathrm{can}$; "
        r"closure fraction $f_\mathrm{cl}$; "
        r"endpoint separation $d_\mathrm{cl}$ (omitted when $s_x = s_y = 0$); "
        r"empirical two-sided $p$-values against the permutation; AR(1); "
        r"Fourier; and combined null ensembles.}" + "\n"
        r"\label{tab:worked}" + "\n"
        r"\begin{tabular}{lcccccccccc}" + "\n"
        r"\hline\hline" + "\n"
        r"Case & $A_\mathrm{norm}$ & MC $1\sigma$ & Orient."
        r" & $R_\mathrm{can}$ & $f_\mathrm{cl}$ & $d_\mathrm{cl}$"
        r" & $p_\mathrm{perm}$ & $p_\mathrm{AR}$"
        r" & $p_\mathrm{Fourier}$ & $p_\mathrm{full}$ \\" + "\n"
        r"\hline" + "\n"
        + row + "\n"
        + r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table*}" + "\n"
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(latex)
    print(f"  LaTeX written to: {outpath}")


def write_trajectory(x, y, sx, sy, label: str, res, outpath: Path, x_label: str = "x", y_label: str = "y"):
    """
    Write a single-panel trajectory figure to outpath (PDF).

    The figure is styled consistently with the signal gallery: points are coloured by time step using the viridis colormap; error bars show the measurement uncertainties; the start point is marked with a circle and the end point with a square; a direction arrow is placed at the largest gap between consecutive points.

    Parameters
    ----------
    x, y : ndarray          Observed coordinates.
    sx, sy : ndarray        1-sigma uncertainties on x and y.
    label : str             Trajectory label used as the panel title.
    res : HysteresisResult  Analysis result; A_norm is annotated in the panel.
    outpath : Path          Output file path (PDF recommended).
    x_label, y_label : str  Axis labels from the CSV header row.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    plt.rcParams.update({
        "font.family"    : "serif",
        "font.size"      : 8,
        "axes.labelsize" : 8,
        "axes.titlesize" : 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top"      : True,
        "ytick.right"    : True,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi"     : 200,
    })

    N      = len(x)
    cmap   = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.15, 0.90, N))
    has_unc = not (np.all(sx == 0) and np.all(sy == 0))

    fig, ax = plt.subplots(figsize=(3.46, 3.20), constrained_layout=True)

    # Error bars and connecting line
    if has_unc:
        for i in range(N):
            ax.errorbar(x[i], y[i], xerr=sx[i], yerr=sy[i],
                        fmt="none", ecolor=colors[i], elinewidth=0.6,
                        capsize=1.5, capthick=0.6, zorder=2)
    ax.plot(x, y, "-", color="0.78", lw=0.7, zorder=1)

    # Points coloured by time step
    for i in range(N):
        ax.scatter(x[i], y[i], color=colors[i], s=12, linewidths=0, zorder=3)

    # Start = circle, end = square
    ax.scatter(x[0],  y[0],  marker="o", s=20, color=colors[0],
               edgecolors="k", linewidths=0.5, zorder=4)
    ax.scatter(x[-1], y[-1], marker="s", s=20, color=colors[-1],
               edgecolors="k", linewidths=0.5, zorder=4)

    # Direction arrow at the largest gap between consecutive points
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    i_arr = int(np.argmax(dists))
    ax.annotate("", xy=(x[i_arr+1], y[i_arr+1]),
                xytext=(x[i_arr], y[i_arr]),
                arrowprops=dict(arrowstyle="-|>", color="0.25",
                                lw=1.0, mutation_scale=10))

    # Annotate A_norm in the panel
    ax.text(0.97, 0.05, rf"$A_\mathrm{{norm}} = {res.A_norm:+.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Shared colourbar for time step
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=N))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.05, pad=0.02, aspect=30)
    cbar.set_label("Time step", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    #ax.set_title(label, fontsize=7, pad=3)
    ax.set_xlabel(x_label, fontsize=7)
    ax.set_ylabel(y_label, fontsize=7)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Trajectory figure written to: {outpath}")


def write_figure(label: str, res, outpath: Path, x_label: str = "x", y_label: str = "y"):
    """
    Write a 6-panel diagnostic figure to outpath (PDF).

    Layout (3 rows x 2 columns):
      Row 1: MC uncertainty  | Permutation null
      Row 2: AR(1) null      | Fourier null
      Row 3: Combined null   | Summary

    The first five panels show histograms of A_norm values from the MC propagation and the three null models, with the observed A_norm marked as a vertical line. Greyed panels indicate skipped distributions (MC: sx=sy=0; Fourier: N < 6).

    The summary panel lists the observed A_norm with its MC 1-sigma CI (if available) and the p-value for each null model alongside a colour-coded bar matching the corresponding histogram.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "font.family"    : "serif",
        "font.size"      : 8,
        "axes.labelsize" : 8,
        "axes.titlesize" : 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top"      : True,
        "ytick.right"    : True,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi"     : 200,
    })

    A_obs  = res.A_norm
    has_mc = res.mc_dist is not None

    # Distribution panels: (distribution_or_None, title, p_value_or_None, colour)
    dist_panels = [
        (res.mc_dist,       "MC uncertainty",   None,                 "steelblue"),
        (res.null_perm,     "Permutation null", res.p_value_perm,     "darkorange"),
        (res.null_ar,       "AR(1) null",       res.p_value_ar,       "seagreen"),
        (res.null_fourier,  "Fourier null",     res.p_value_fourier,  "mediumpurple"),
        (res.null_combined, "Combined null",    res.p_value_combined, "firebrick"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(4.2, 6.3), constrained_layout=True)
    ax_flat = axes.flatten()   # [MC, Perm, AR1, Fourier, Combined, Summary]

    # --- Distribution panels (first 5) ---
    for ax, (dist, title, pval, colour) in zip(ax_flat[:5], dist_panels):
        if dist is None:
            ax.set_title(title, pad=4)
            if title == "MC uncertainty":
                msg = "skipped\n(sx = sy = 0)"
            else:
                msg = "skipped\n(N < 6)"
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="0.5")
            ax.set_xlabel(r"$A_\mathrm{norm}$")
            ax.set_yticks([])
            continue

        ax.hist(dist, bins=50, density=True, color=colour, alpha=0.55, linewidth=0)
        ax.axvline(A_obs, color="black", linewidth=1.2, label=r"$A_\mathrm{obs}$")

        # MC panel: dashed line for MC mean; dotted lines for 1-sigma CI bounds
        if title == "MC uncertainty" and has_mc:
            ax.axvline(res.mc_mean,    color="black", linewidth=0.8, linestyle="--")
            ax.axvline(res.mc_ci_low,  color="black", linewidth=0.7, linestyle=":")
            ax.axvline(res.mc_ci_high, color="black", linewidth=0.7, linestyle=":")

        ax.set_title(title, pad=4)
        ax.set_xlabel(r"$A_\mathrm{norm}$")
        ax.set_ylabel("density" if ax is ax_flat[0] else "")

    # --- Summary panel (6th) ---
    ax_sum = ax_flat[5]
    ax_sum.axis("off")

    lines = []

    # Observed A_norm (2 decimal places, consistent with LaTeX table)
    lines.append((None, None, rf"$A_\mathrm{{norm}} = {res.A_norm:+.2f}$"))
    if has_mc:
        err_up  = res.mc_ci_high - res.mc_mean
        err_dn  = res.mc_mean    - res.mc_ci_low
        lines.append((None, None,
                      rf"$A_\mathrm{{norm}}^\mathrm{{MC}} = "
                      rf"{res.mc_mean:+.2f}^{{+{err_up:.2f}}}_{{-{err_dn:.2f}}}$"))
    lines.append((None, None, ""))   # spacer

    # p-values with colour bars
    null_entries = [
        ("Permutation", res.p_value_perm,    "darkorange"),
        ("AR(1)",        res.p_value_ar,      "seagreen"),
        ("Fourier",      res.p_value_fourier, "mediumpurple"),
        ("Combined",     res.p_value_combined,"firebrick"),
    ]
    for name, pval, colour in null_entries:
        pstr = fmt_pval(pval)
        lines.append((colour, name, rf"$p = {pstr}$" if pstr not in ("---",) else "---"))

    # Draw lines as text with coloured rectangles
    y_start = 0.95
    dy      = 0.13
    for colour, name, text in lines:
        y = y_start - lines.index((colour, name, text)) * dy
        if colour is not None:
            ax_sum.add_patch(mpatches.Rectangle(
                (0.02, y - 0.04), 0.10, 0.08,
                transform=ax_sum.transAxes,
                color=colour, alpha=0.7, clip_on=False
            ))
            ax_sum.text(0.16, y, f"{name}:  {text}",
                        transform=ax_sum.transAxes,
                        va="center", fontsize=7)
        else:
            ax_sum.text(0.02, y, text,
                        transform=ax_sum.transAxes,
                        va="center", fontsize=7,
                        fontweight="bold" if "A_norm" in text else "normal")

    ax_sum.set_title("Summary", pad=4)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure written to: {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_csv", type=Path,
                   help="CSV file with four columns: x observable, y observable, sx uncertainty, sy uncertainty. "
                        "First non-comment line is the header row used as axis labels.")
    p.add_argument("--K_mc",   type=int, default=10000,
                   help="MC realisations for uncertainty propagation (default: 10000)")
    p.add_argument("--K_null", type=int, default=10000,
                   help="Surrogate realisations per null model (default: 10000)")
    p.add_argument("--seed",   type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--outdir", type=Path, default=Path("results"),
                   help="Output directory for figures and LaTeX (default: results)")
    p.add_argument("--latex",  action="store_true",
                   help="Write a LaTeX table row to <outdir>/<stem>_results.tex")
    p.add_argument("--no-fig", action="store_true",
                   help="Suppress figure output")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.input_csv.exists():
        sys.exit(f"File not found: {args.input_csv}")

    rng = np.random.default_rng(args.seed)

    # --- Read data ---
    # x = first observable (horizontal axis), y = second observable (vertical axis)
    x, y, sx, sy, x_label, y_label = read_csv(args.input_csv)
    label = args.input_csv.stem   # e.g. "signal_a"

    print(f"\nReading {args.input_csv}  ({len(x)} points)")
    print(f"  x: {x_label}    y: {y_label}")
    has_unc = not (np.all(sx == 0) and np.all(sy == 0))
    if not has_unc:
        print("  Note: sx=sy=0 throughout -- MC propagation and d_cl skipped.")

    print(f"Running null models ({args.K_null} realisations each) ...", flush=True)

    # --- Analysis ---
    # Pass sx=sy=None when all uncertainties are zero to skip MC propagation.
    res = analyse(
        x, y,
        sx=sx if has_unc else None,
        sy=sy if has_unc else None,
        K_mc=args.K_mc,
        K_null=args.K_null,
        rng=rng,
        verbose=False,
    )

    # --- Output ---
    print_summary(label, res, x_label, y_label)

    if not args.no_fig:
        trajpath = args.outdir / f"{label}_trajectory.pdf"
        write_trajectory(x, y, sx, sy, label, res, trajpath, x_label, y_label)

        figpath = args.outdir / f"{label}_diagnostics.pdf"
        write_figure(label, res, figpath, x_label, y_label)

    if args.latex:
        texpath = args.outdir / f"{label}_results.tex"
        write_latex(label, res, texpath, x_label, y_label)


if __name__ == "__main__":
    main()
