# hysteresis-detection

A statistically robust framework for detecting and classifying hysteresis patterns in ordered two-dimensional data.

This code accompanies:

> Terzić, T. (2026), *A&A* (submitted)  
> "A statistically robust framework for detecting and classifying hysteresis patterns in astrophysical spectral evolution"

---

## What this code does

Hysteresis occurs when a system traces different paths in a two-parameter space depending on the direction of evolution — producing a closed or partially closed loop. This repository provides tools to:

- **Measure** the signed normalised loop area $A_\mathrm{norm}$ and associated shape diagnostics ($R_\mathrm{can}$, $f_\mathrm{cl}$, $d_\mathrm{cl}$) for any chronologically ordered two-parameter dataset
- **Propagate** measurement uncertainties into $A_\mathrm{norm}$ via Monte Carlo sampling
- **Test significance** against three complementary null models: random permutation, AR(1) surrogates, and Fourier phase-randomisation surrogates
- **Visualise** the trajectory and the full statistical analysis in publication-ready figures

Although developed for blazar spectral hysteresis in the flux–hardness-ratio plane, the framework is applicable to any ordered two-dimensional dataset.

---

## Repository structure

```
hysteresis-detection/
├── hysteresis_core.py        # Core library: geometry, MC propagation, null models
├── analyse_hysteresis.py     # Command-line analysis tool (main entry point)
├── generate_candidates.py    # Generate pool of synthetic trajectories for visual selection
├── signal_gallery.py         # Produce the signal gallery figure (Fig. X of the paper)
├── null_gallery.py           # Produce the null model gallery figure (Fig. X of the paper)
├── examples/                 # Example CSV files (the four paper trajectories)
│   ├── signal_a.csv          # (a) Clean single loop
│   ├── signal_b.csv          # (b) Incomplete path
│   ├── signal_c.csv          # (c) Figure-eight
│   └── signal_d.csv          # (d) Overlapping paths
├── requirements.txt
├── LICENSE
└── README.md
```

### Script hierarchy

The scripts depend on each other as follows:

```
generate_candidates.py          # defines the flare model and pool generation
    └── signal_gallery.py       # imports from generate_candidates; produces paper figures and CSV files
        └── null_gallery.py     # reads CSV from signal_gallery; produces null model figures

hysteresis_core.py              # standalone scientific library; no dependencies on other scripts
    └── analyse_hysteresis.py   # imports from hysteresis_core; analyses any CSV file
```

`hysteresis_core.py` is the only file you need if you want to use the framework programmatically in your own code.

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/tterzic/hysteresis-detection.git
cd hysteresis-detection
pip install -r requirements.txt
```

No package installation is required — all scripts run directly from the repository directory.

---

## Quick start

### Analyse an example trajectory

```bash
python analyse_hysteresis.py examples/signal_a.csv --K_null 10000 --latex --outdir results
```

This will:
1. Read `examples/signal_a.csv`
2. Compute all geometric statistics
3. Propagate measurement uncertainties via Monte Carlo (K = 10 000 realisations)
4. Run three null models (K = 10 000 surrogates each)
5. Print a summary table to stdout
6. Write `results/signal_a_trajectory.pdf` — the trajectory figure
7. Write `results/signal_a_diagnostics.pdf` — the 6-panel diagnostic figure
8. Write `results/signal_a_results.tex` — a LaTeX table row

### Reproduce the paper figures

```bash
# Generate the signal gallery (Fig. X) and CSV files
python signal_gallery.py --outdir results

# Run the analysis on each trajectory
for f in a b c d; do
    python analyse_hysteresis.py results/signal_${f}.csv --K_null 10000 --latex --outdir results
done

# Generate the null model gallery (Fig. X)
python null_gallery.py --signal_csv results/signal_a.csv --outdir results
```

### Use the library in your own code

```python
import numpy as np
from hysteresis_core import analyse

# Your data: chronologically ordered (x, y) pairs with uncertainties
x  = np.array([...])   # observable 1 (e.g. flux)
y  = np.array([...])   # observable 2 (e.g. hardness ratio)
sx = np.array([...])   # 1-sigma uncertainty on x
sy = np.array([...])   # 1-sigma uncertainty on y

rng = np.random.default_rng(42)
result = analyse(x, y, sx=sx, sy=sy, K_mc=10_000, K_null=10_000, rng=rng)

print(result.summary())
print(f"A_norm = {result.A_norm:.3f}  (p_combined = {result.p_value_combined:.3f})")
```

---

## Input CSV format

`analyse_hysteresis.py` accepts any CSV file with the following structure:

```
# Comment lines beginning with # are ignored
# Any number of comment lines is allowed
x_label,y_label,sx_label,sy_label
1.456,1.234,0.073,0.062
1.512,1.301,0.076,0.065
...
```

- The **first non-comment line** is the header row. Its four comma-separated values are used as axis labels in figures and output tables. Column names must not contain commas.
- All subsequent lines are **data rows**, read by column position (column 0 = x, 1 = y, 2 = sx, 3 = sy).
- Set `sx = sy = 0` for synthetic or null trajectories with no measurement process; Monte Carlo propagation and $d_\mathrm{cl}$ are then skipped automatically.

For the flux–hardness-ratio application, the header is `F (A.U.),HR,s_F,s_HR`.

---

## Output

### `analyse_hysteresis.py` output

| File | Description |
|------|-------------|
| stdout | Human-readable summary table (full precision, for verification) |
| `<stem>_trajectory.pdf` | Single-panel trajectory figure |
| `<stem>_diagnostics.pdf` | 6-panel diagnostic figure (3 × 2) |
| `<stem>_results.tex` | LaTeX table row (with `--latex` flag) |

The trajectory figure is styled consistently with the signal gallery: points coloured by time step (viridis), error bars, start marker (circle), end marker (square), direction arrow.

The diagnostic figure layout is:

| MC uncertainty | Permutation null |
|----------------|-----------------|
| AR(1) null     | Fourier null     |
| Combined null  | Summary          |

The summary panel shows $A_\mathrm{norm}$, the MC uncertainty in asymmetric error notation ($A_\mathrm{norm}^\mathrm{MC} = \mu^{+\Delta_+}_{-\Delta_-}$), and the empirical p-value for each null model alongside a colour-coded bar.

The LaTeX output uses 2 decimal places for all geometric statistics and CI bounds. The printed summary uses 4 decimal places for verification.

### p-value precision

p-values are reported to 3 decimal places. With the default `K_null = 10 000` surrogates, the p-value floor is $1/K = 10^{-4}$, so 3 decimal places are fully meaningful. Values below 0.001 are shown as `<0.001`. Using `K_null < 1000` reduces resolution and is not recommended for publication.

---

## Statistical framework

### Geometric statistics

| Statistic | Symbol | Description |
|-----------|--------|-------------|
| Normalised area | $A_\mathrm{norm}$ | Signed open-path area divided by convex hull area. Primary detection statistic. Positive = CCW; negative = CW. |
| Cancellation ratio | $R_\mathrm{can}$ | $\|A_\mathrm{open}\| / A_\mathrm{abs}$. Near 1 = consistent orientation; near 0 = strong self-crossing. |
| Closure fraction | $f_\mathrm{cl}$ | $\|A_\mathrm{closure}\| / \|A_\mathrm{tot}\|$. Near 0 = well-closed path; near 1 = open path. |
| Closure distance | $d_\mathrm{cl}$ | Endpoint separation in units of measurement uncertainty. Near 0 = well-closed; large = open path. |

### Monte Carlo uncertainty propagation

Each of the $K_\mathrm{MC}$ realisations independently perturbs every observed point by a Gaussian draw scaled by its uncertainty and recomputes $A_\mathrm{norm}$. The resulting distribution characterises sensitivity to measurement noise.

The output reports:
- **MC mean** ($\mu$): mean of the MC distribution of $A_\mathrm{norm}$
- **MC 1σ CI**: 15.865th–84.135th percentile interval (Gaussian-equivalent 1σ, valid for non-Gaussian distributions)
- **P(A>0)**: fraction of MC realisations with $A_\mathrm{norm} > 0$ — measures orientation robustness under noise

### Null models

Three null models test different aspects of the data:

| Null model | Tests against | Method |
|------------|---------------|--------|
| Permutation | Unordered point cloud | Random permutation of (x, y) pairs |
| AR(1) | Independent autocorrelated processes | Yule-Walker AR(1) fit; independent surrogates for x and y |
| Fourier | Matched power spectrum, no phase coupling | Phase randomisation preserving amplitude spectrum (Timmer & König 1995) |

The Fourier null requires $N \geq 6$. For shorter series it is skipped and a warning is issued; the combined p-value is then formed from permutation and AR(1) only.

All p-values are two-sided empirical p-values: the fraction of $K_\mathrm{null}$ surrogates with $|A_\mathrm{norm}|$ at least as large as the observed value. The combined p-value uses the concatenated ensemble of all null models that were run ($3K_\mathrm{null}$ values by default).

---

## Synthetic trajectory model

The paper examples use an asymmetric Gaussian flare model in the (F, HR) plane:

$$F(t) = 1 + A_F \exp\!\left(-\frac{(t - t_F)^2}{2\sigma_{F,\pm}^2}\right), \quad HR(t) = 1 + A_{HR} \exp\!\left(-\frac{(t - t_{HR})^2}{2\sigma_{HR,\pm}^2}\right)$$

where $\sigma_{\pm}$ switches between rise and decay widths at the peak, and $\Delta t = t_F - t_{HR}$ is the inter-band time delay. Measurement uncertainties are fractional: $s_F = \sigma_\mathrm{noise} \cdot F$, $s_{HR} = \sigma_\mathrm{noise} \cdot HR$.

`generate_candidates.py` generates a large pool of such trajectories (default: K = 1000) and writes them to a multi-page PDF for visual inspection. The four paper examples were selected from this pool by index.

---

## Reproducing the paper examples

The four paper trajectories are fixed by `seed=42` and pool indices 777, 225, 839, 543. To reproduce them exactly, use the default parameters in `signal_gallery.py`. Changing the seed, pool size, or indices will produce different trajectories — a warning is printed in that case.

---

## Requirements

- Python ≥ 3.9
- numpy
- scipy
- matplotlib

See `requirements.txt` for pinned versions.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Terzic2026,
  author  = {Terzi{\'c}, T.},
  title   = {A statistically robust framework for detecting and classifying hysteresis patterns in astrophysical spectral evolution},
  journal = {A\&A},
  year    = {2026},
  note    = {submitted}
}
```

---

## License

MIT — see `LICENSE`.

## Contact

T. Terzić — tterzic@phy.uniri.hr
