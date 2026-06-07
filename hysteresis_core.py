"""
hysteresis_core.py
==================
Core geometric and statistical routines for hysteresis detection in ordered two-dimensional data.

This module implements the framework described in:
  Terzić, T. (2026), A&A (submitted) "A statistically robust framework for detecting and classifying hysteresis patterns in spectral evolution"

All area statistics follow the sign convention that counter-clockwise loops have positive area (right-hand rule).

Author: T. Terzić <tterzic@phy.uniri.hr>
License: MIT
"""

import warnings
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class HysteresisResult:
    """
    Container for all hysteresis metrics computed for a single dataset.

    Attributes
    ----------
    N : int
        Number of chronologically ordered observations.
    A_open : float
        Signed open-path area (shoelace sum over N-1 segments).
    A_closure : float
        Signed area contribution from the artificial closing segment.
    A_tot : float
        Total signed area (A_open + A_closure).
    A_norm : float
        Normalised open-path area (divided by convex hull area).
    A_hull : float
        Convex hull area of the centroid-centred trajectory. Used as the normalisation
        denominator for A_norm, A_abs, and A_rms.
    A_abs : float
        Absolute incremental area (sum of |a_i|).
    A_rms : float
        Root-mean-square incremental area.
    f_closure : float
        Closure fraction |A_closure| / |A_tot|.
    d_closure : float
        Normalised closure distance (Eq. 8 of the paper).
    R_cancel : float
        Cancellation ratio |A_open| / A_abs.
    orientation : str
        'CCW' (counter-clockwise), 'CW' (clockwise), or 'none'.
    mc_mean : float, optional
        Monte Carlo mean of A_norm over uncertainty realisations.
    mc_std : float, optional
        Monte Carlo standard deviation of A_norm.
    mc_ci_low : float, optional
        Lower bound of the 1-sigma MC interval (15.87th percentile).
    mc_ci_high : float, optional
        Upper bound of the 1-sigma MC interval (84.13th percentile).
    p_positive : float, optional
        Fraction of MC realisations with A_norm > 0 (orientation robustness).
    mc_dist : ndarray, optional
        Full array of K MC A_norm values (finite realisations only). Stored so callers can plot the distribution without re-running the MC.
    p_value_perm : float, optional
        Empirical p-value against the random-permutation null.
    p_value_ar : float, optional
        Empirical p-value against the AR(1) surrogate null.
    p_value_fourier : float, optional
        Empirical p-value against the Fourier surrogate null. None if the Fourier null could not be computed (e.g. N < 6).
    p_value_combined : float, optional
        Combined p-value over all null models that were run.
    null_perm : ndarray, optional
        Full array of K A_norm values from the permutation null.
    null_ar : ndarray, optional
        Full array of K A_norm values from the AR(1) null.
    null_fourier : ndarray, optional
        Full array of K A_norm values from the Fourier null. None if the Fourier null could not be computed (e.g. N < 6).
    null_combined : ndarray, optional
        Concatenation of all null arrays that were run.
    """
    N:           int   = 0
    A_open:      float = np.nan
    A_closure:   float = np.nan
    A_tot:       float = np.nan
    A_hull:      float = np.nan
    A_norm:      float = np.nan
    A_abs:       float = np.nan
    A_rms:       float = np.nan
    f_closure:   float = np.nan
    d_closure:   float = np.nan
    R_cancel:    float = np.nan
    orientation: str   = "none"

    mc_mean:    Optional[float]      = None
    mc_std:     Optional[float]      = None
    mc_ci_low:  Optional[float]      = None
    mc_ci_high: Optional[float]      = None
    p_positive: Optional[float]      = None
    mc_dist:    Optional[np.ndarray] = None

    p_value_perm:     Optional[float] = None
    p_value_ar:       Optional[float] = None
    p_value_fourier:  Optional[float] = None
    p_value_combined: Optional[float] = None

    null_perm:     Optional[np.ndarray] = None
    null_ar:       Optional[np.ndarray] = None
    null_fourier:  Optional[np.ndarray] = None
    null_combined: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 56,
            "  Hysteresis Detection Summary",
            "=" * 56,
            f"  A_open    = {self.A_open:+.4f}",
            f"  A_hull    = {self.A_hull:.4f}",
            f"  A_norm    = {self.A_norm:+.4f}",
            f"  A_abs     = {self.A_abs:.4f}",
            f"  R_cancel  = {self.R_cancel:.3f}",
            f"  f_closure = {self.f_closure:.3f}",
            f"  d_closure = {self.d_closure:.2f}",
            f"  Orientation: {self.orientation}",
        ]
        if self.mc_mean is not None:
            lines += [
                f"  MC mean   = {self.mc_mean:+.4f}  ±  {self.mc_std:.4f}",
                f"  MC 1σ CI  = [{self.mc_ci_low:+.4f}, {self.mc_ci_high:+.4f}]",
                f"  P(A>0)    = {self.p_positive:.3f}",
            ]
        if self.p_value_perm     is not None: lines.append(f"  p (perm)    = {self.p_value_perm:.4f}")
        if self.p_value_ar       is not None: lines.append(f"  p (AR1)     = {self.p_value_ar:.4f}")
        if self.p_value_fourier  is not None: lines.append(f"  p (Fourier) = {self.p_value_fourier:.4f}")
        if self.p_value_combined is not None: lines.append(f"  p (comb)    = {self.p_value_combined:.4f}")
        lines.append("=" * 56)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core area computation
# ---------------------------------------------------------------------------

def _triangle_areas(x, y):
    """
    Compute signed triangle areas for consecutive point pairs, using centroid-centred coordinates.

    Translating to the centroid before applying the shoelace formula makes the result translation- and rotation-invariant (see Sect. 2 of the paper).

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Chronologically ordered coordinates (raw, not pre-centred).

    Returns
    -------
    a : ndarray, shape (N-1,)
        Signed area contributions: a_i = 0.5 * (xt_i * yt_{i+1} - xt_{i+1} * yt_i), where xt_i = x_i - x_bar, yt_i = y_i - y_bar.
    xt, yt : ndarray, shape (N,)
        Centroid-centred coordinates, returned so that compute_areas can reuse them without a second subtraction.
    cx, cy : float
        Centroid coordinates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = x.mean(), y.mean()
    xt = x - cx
    yt = y - cy
    return 0.5 * (xt[:-1] * yt[1:] - xt[1:] * yt[:-1]), xt, yt, cx, cy


def compute_areas(x, y, sx=None, sy=None):
    """
    Compute all area statistics for a single chronologically ordered dataset.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Chronologically ordered observable coordinates.
    sx, sy : array_like, shape (N,), optional
        1-sigma measurement uncertainties on x and y. Required for d_closure; if not provided, d_closure is set to NaN.

    Returns
    -------
    result : HysteresisResult
        All geometric statistics. MC fields and p-values are not populated here; call monte_carlo_uncertainties() and the null_*() functions separately.

    Raises
    ------
    ValueError
        If N < 4.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of the same length.")
    N = len(x)
    if N < 4:
        raise ValueError(f"Need at least 4 points; got {N}. "
                         "Hysteresis requires at least 4 chronologically ordered observations.")

    # Centroid-centred incremental triangle areas (translation- and rotation-invariant; see Sect. 2 of the paper).
    a, xt, yt, _, _ = _triangle_areas(x, y)

    A_open = np.sum(a)
    A_closure = 0.5 * (xt[-1] * yt[0] - xt[0] * yt[-1])
    A_tot = A_open + A_closure   # used only for f_closure below

    # Normalised area: A_open divided by the convex hull area. Using the convex hull makes normalisation translation- and rotation-invariant.
    try:
        A_hull = ConvexHull(np.column_stack([xt, yt])).volume  # .volume returns area in 2-D
        A_norm = A_open / A_hull
    except Exception:
        # Degenerate case: all points are collinear, convex hull has zero area. A_open is also zero in this case, so there is no loop to detect.
        A_hull = np.nan
        A_norm = np.nan
        warnings.warn("All points are collinear or nearly so: convex hull area is zero. "
                      "No loop structure is possible; A_norm set to NaN.",
                      UserWarning, stacklevel=2)

    A_abs = np.sum(np.abs(a))
    A_rms = np.sqrt(np.sum(a**2))

    # Closure fraction: how much of the total area comes from the forced closing segment.
    f_closure = abs(A_closure) / abs(A_tot) if abs(A_tot) > 0 else np.nan

    # Normalised closure distance: endpoint separation measured in units of uncertainty.
    if sx is not None and sy is not None:
        sx = np.asarray(sx, dtype=float)
        sy = np.asarray(sy, dtype=float)
        denom_x = sx[-1]**2 + sx[0]**2
        denom_y = sy[-1]**2 + sy[0]**2
        if denom_x > 0 and denom_y > 0:
            d_closure = np.sqrt((xt[-1] - xt[0])**2 / denom_x + (yt[-1] - yt[0])**2 / denom_y)
        else:
            d_closure = np.nan
    else:
        d_closure = np.nan

    R_cancel = abs(A_open) / A_abs if A_abs > 0 else np.nan

    if   np.isnan(A_norm): orientation = "none"
    elif A_norm > 0:       orientation = "CCW"
    elif A_norm < 0:       orientation = "CW"
    else:                  orientation = "none"

    return HysteresisResult(
        N=N,
        A_open=A_open,
        A_closure=A_closure,
        A_tot=A_tot,
        A_hull=A_hull,
        A_norm=A_norm,
        A_abs=A_abs,
        A_rms=A_rms,
        f_closure=f_closure,
        d_closure=d_closure,
        R_cancel=R_cancel,
        orientation=orientation,
    )


# ---------------------------------------------------------------------------
# Monte Carlo uncertainty propagation
# ---------------------------------------------------------------------------

def monte_carlo_uncertainties(x, y, sx, sy, K=10_000, rng=None):
    """
    Propagate measurement uncertainties into A_norm via Monte Carlo sampling.

    Each of the K realisations independently perturbs every observed point by a Gaussian draw scaled by its uncertainty, then recomputes A_norm. The resulting distribution characterises how sensitive the area measurement is to the noise.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Observed coordinates.
    sx, sy : array_like, shape (N,)
        1-sigma Gaussian measurement uncertainties on x and y.
    K : int
        Number of Monte Carlo realisations (default 10 000).
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    result : HysteresisResult
        Geometric statistics of the observed data, with mc_* fields populated.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)

    base = compute_areas(x, y, sx, sy)

    # Generate all K perturbed realisations at once (K x N matrices), then compute A_norm for each row.
    x_mc = x[None, :] + rng.standard_normal((K, len(x))) * sx[None, :]
    y_mc = y[None, :] + rng.standard_normal((K, len(y))) * sy[None, :]

    A_norm_mc = np.array([compute_areas(x_mc[k], y_mc[k]).A_norm for k in range(K)])
    # Collinear realisations produce A_norm = NaN (no convex hull). These are physically meaningful outcomes — the noise collapsed the loop — so they are counted as A_norm = 0 rather than discarded, ensuring the MC distribution honestly reflects all realisations.
    A_norm_mc = np.where(np.isfinite(A_norm_mc), A_norm_mc, 0.0)

    ci = np.percentile(A_norm_mc, [15.865, 84.135])
    base.mc_mean = float(np.mean(A_norm_mc))
    base.mc_std = float(np.std(A_norm_mc))
    base.mc_ci_low = float(ci[0])
    base.mc_ci_high = float(ci[1])
    base.p_positive = float(np.mean(A_norm_mc > 0))
    base.mc_dist = A_norm_mc  # stored so callers can plot without re-running the MC

    return base


# ---------------------------------------------------------------------------
# Null ensembles
# ---------------------------------------------------------------------------

def null_permutation(x, y, K=10_000, rng=None):
    """
    Generate the random-permutation null distribution of A_norm.

    Tests whether the observed loop area could arise by chance from an unordered set of points with no temporal structure. Each surrogate is a random permutation of the (x, y) pairs (both coordinates permuted together, preserving their point-by-point association).

    Parameters
    ----------
    x, y : array_like, shape (N,)
    K : int
        Number of permutations (default 10 000).
    rng : numpy.random.Generator, optional

    Returns
    -------
    areas : ndarray, shape (K,)
        A_norm for each permuted trajectory.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(x)
    areas = np.empty(K)
    for k in range(K):
        idx = rng.permutation(N)
        areas[k] = compute_areas(x[idx], y[idx]).A_norm
    # Collinear surrogates produce A_norm = NaN; count them as 0 (no loop).
    areas = np.where(np.isfinite(areas), areas, 0.0)
    return areas


def null_ar1(x, y, K=10_000, rng=None):
    """
    Generate an AR(1) surrogate null distribution of A_norm.

    Tests whether the observed loop area could arise from two independent autocorrelated (red-noise) processes with no causal relationship between them.
    AR(1) parameters (mean, variance, lag-1 autocorrelation) are estimated from the observed series by the Yule-Walker method, and K pairs of independent synthetic AR(1) series are generated with those parameters.

    The first point of each surrogate is drawn from the stationary distribution N(mu, std^2) so that it has the correct variance. The innovation variance is sigma^2 = std^2 * (1 - phi^2), which follows from the AR(1) stationarity condition.

    Parameters
    ----------
    x, y : array_like, shape (N,)
    K : int
        Number of surrogate realisations (default 10 000).
    rng : numpy.random.Generator, optional

    Returns
    -------
    areas : ndarray, shape (K,)
        A_norm for each surrogate trajectory.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(x)

    def _fit_ar1(ts):
        """
        Estimate AR(1) parameters from a time series by the Yule-Walker method.

        The lag-1 autocorrelation (phi) is computed on the mean-centred series.
        The guard denom > 0 handles the degenerate case of a constant series, where all centred values are zero and phi is undefined; phi = 0 is returned in that case (white noise with zero variance).

        Returns phi, sigma (innovation std), mu (series mean), std (series std).
        """
        ts_c = ts - ts.mean()
        denom = np.dot(ts_c[:-1], ts_c[:-1])
        phi = np.dot(ts_c[:-1], ts_c[1:]) / denom if denom > 0 else 0.0
        phi = np.clip(phi, -0.99, 0.99)
        sigma = np.std(ts_c) * np.sqrt(1 - phi**2)
        return phi, sigma, ts.mean(), np.std(ts)

    phi_x, sig_x, mu_x, std_x = _fit_ar1(x)
    phi_y, sig_y, mu_y, std_y = _fit_ar1(y)

    areas = np.empty(K)
    for k in range(K):
        xs = np.empty(N)
        ys = np.empty(N)
        # Draw the first point from the stationary distribution N(mu, std^2).
        xs[0] = rng.normal(mu_x, std_x)
        ys[0] = rng.normal(mu_y, std_y)
        eps_x = rng.normal(0, sig_x, N)
        eps_y = rng.normal(0, sig_y, N)
        for i in range(1, N):
            xs[i] = phi_x * (xs[i-1] - mu_x) + mu_x + eps_x[i]
            ys[i] = phi_y * (ys[i-1] - mu_y) + mu_y + eps_y[i]
        areas[k] = compute_areas(xs, ys).A_norm
    # Collinear surrogates produce A_norm = NaN; count them as 0 (no loop).
    areas = np.where(np.isfinite(areas), areas, 0.0)
    return areas


def null_fourier(x, y, K=10_000, rng=None):
    """
    Generate a Fourier phase-randomisation surrogate null distribution of A_norm.

    Tests whether the observed loop area could arise from two processes that have the same power spectrum as x and y but no phase relationship between them.
    The amplitude spectrum of each variable is preserved; only the Fourier phases are independently randomised for x and y (Timmer & König 1995).

    Parameters
    ----------
    x, y : array_like, shape (N,)
    K : int
        Number of surrogate realisations (default 10 000).
    rng : numpy.random.Generator, optional

    Returns
    -------
    areas : ndarray, shape (K,)
        A_norm for each surrogate trajectory.

    Raises
    ------
    ValueError
        If N < 6. For shorter series the real FFT has at most 3 complex coefficients, leaving too few independent phases to produce a meaningful surrogate distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(x)

    if N < 6:
        raise ValueError(
            f"null_fourier: N={N} is too small for Fourier surrogates (minimum N=6). "
            "Collect more data to investigate this null model."
        )

    def _fourier_surrogates(ts, n_surr):
        """
        Generate n_surr Fourier surrogates of a 1-D time series.

        The amplitude spectrum is preserved; phases are drawn uniformly from [0, 2*pi).
        irfft is called with n=N to ensure the output length matches the input exactly, regardless of whether N is even or odd.
        """
        fft = np.fft.rfft(ts)
        amps = np.abs(fft)
        surrogates = np.empty((n_surr, N))
        for k in range(n_surr):
            phases = rng.uniform(0, 2 * np.pi, len(fft))
            phases[0] = 0.0        # DC component must be real
            if N % 2 == 0:
                phases[-1] = 0.0   # Nyquist component must be real for even N
            surrogates[k] = np.fft.irfft(amps * np.exp(1j * phases), n=N)
        return surrogates

    x_surr = _fourier_surrogates(x, K)
    y_surr = _fourier_surrogates(y, K)

    areas = np.array([compute_areas(x_surr[k], y_surr[k]).A_norm for k in range(K)])
    # Collinear surrogates produce A_norm = NaN; count them as 0 (no loop).
    areas = np.where(np.isfinite(areas), areas, 0.0)
    return areas


def empirical_pvalue(obs, null_dist, two_sided=True):
    """
    Compute an empirical p-value from a null distribution.

    The p-value is the fraction of null realisations at least as extreme as the observed statistic. It is floored at 1/K to avoid reporting p=0, which would imply infinite resolving power from a finite sample.

    Parameters
    ----------
    obs : float
        Observed test statistic (A_norm).
    null_dist : array_like
        Array of K null-model A_norm values.
    two_sided : bool
        If True (default), compare |obs| against |null| (direction-agnostic test).

    Returns
    -------
    p : float
        p-value in the range [1/K, 1].
    """
    null_dist = np.asarray(null_dist, dtype=float)
    if two_sided:
        p = float(np.mean(np.abs(null_dist) >= abs(obs)))
    else:
        p = float(np.mean(null_dist >= obs))
    return max(p, 1.0 / len(null_dist))


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyse(x, y, sx=None, sy=None,
            K_mc=10_000, K_null=10_000, null_models=("perm", "ar1", "fourier"),
            rng=None, verbose=True):
    """
    Full hysteresis analysis pipeline.

    Computes all geometric statistics, propagates measurement uncertainties via Monte Carlo (if provided), and constructs null ensembles. Returns all results in a HysteresisResult; no detection decision is made — that is left to the caller.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Chronologically ordered observable coordinates.
    sx, sy : array_like, shape (N,), optional
        1-sigma measurement uncertainties. If None, MC propagation is skipped.
    K_mc : int
        Number of MC realisations for uncertainty propagation (default 10 000).
    K_null : int
        Number of surrogate realisations per null model (default 10 000).
    null_models : tuple of str
        Which null models to run: any subset of ('perm', 'ar1', 'fourier').
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.
    verbose : bool
        If True, print a summary to stdout after the analysis.

    Returns
    -------
    result : HysteresisResult
        All geometric statistics, MC uncertainty propagation, null distributions, and empirical p-values. The null arrays (null_perm, null_ar, null_fourier, null_combined) are stored for plotting without recomputation.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- Step 1: Geometric statistics and MC uncertainty propagation ----
    if sx is not None and sy is not None:
        result = monte_carlo_uncertainties(x, y, sx, sy, K=K_mc, rng=rng)
    else:
        result = compute_areas(x, y)

    # ---- Step 2: Null ensembles and p-values ----
    null_arrays = []

    if "perm" in null_models:
        dist_perm = null_permutation(x, y, K=K_null, rng=rng)
        result.p_value_perm = empirical_pvalue(result.A_norm, dist_perm)
        result.null_perm = dist_perm
        null_arrays.append(dist_perm)

    if "ar1" in null_models:
        dist_ar = null_ar1(x, y, K=K_null, rng=rng)
        result.p_value_ar = empirical_pvalue(result.A_norm, dist_ar)
        result.null_ar = dist_ar
        null_arrays.append(dist_ar)

    if "fourier" in null_models:
        try:
            dist_fourier = null_fourier(x, y, K=K_null, rng=rng)
            result.p_value_fourier = empirical_pvalue(result.A_norm, dist_fourier)
            result.null_fourier = dist_fourier
            null_arrays.append(dist_fourier)
        except ValueError as exc:
            warnings.warn(f"Fourier null skipped: {exc}", UserWarning, stacklevel=2)
            result.p_value_fourier = None
            result.null_fourier = None

    if null_arrays:
        combined = np.concatenate(null_arrays)
        result.p_value_combined = empirical_pvalue(result.A_norm, combined)
        result.null_combined = combined

    if verbose:
        print(result.summary())

    return result
