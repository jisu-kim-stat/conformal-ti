# PAC-Calibrated Regression Tolerance Intervals

This repository contains simulation and real-data code for regression tolerance intervals with marginal PAC calibration and conditional PAC diagnostics.

## Methods

The repository compares four methods:

- **Parametric TI**: spline-based model benchmark
- **HCTI**: symmetric standardized residual score
- **HCTI-asym**: asymmetric standardized residual score
- **CQR-TI**: quantile-regression score

The PAC-calibrated methods use the empirical \((C+\lambda_\alpha)\)-quantile of calibration scores, where

```text
lambda_alpha = sqrt(log(2 / alpha) / (2 * n_cal))
```

Default Settings : 
```text
content_level = 0.90
pac_alpha     = 0.05
```

## Structure
```text
ti_project/
├── data/
│   └── real/
│       └── redshift/
│           ├── happy_A
│           └── happy_B
├── scripts/
│   ├── sim/
│   └── real/
│       ├── redshift/
│       └── tsa/
├── results/
│   ├── sim/
│   └── real/
│       ├── redshift/
│       └── tsa/
├── fig/
│   ├── sim/
│   │   ├── uniform/
│   │   └── normal/
│   └── real/
│       ├── redshift/
│       └── tsa/
└── README.md
```

## Simulation
Simulation scripts are in:
```text
scripts/sim/
```

Main scripts:
```
scripts/sim/run_simulation_grid_alt_dgp.R
scripts/sim/make_plot_alt_dgp.R
```

Run from the project root:
```text
Rscript scripts/sim/run_simulation_grid_alt_dgp.R
Rscript scripts/sim/make_plot_alt_dgp.R
```

## Real Data
The repository includes two real-data applications:
1. Reashift Data
2. TSA passenger throughput data

#### Redshift
Run : 
```bash
python scripts/real/redshift/real_redshift_4methods.py
```
Plot :
```bash
python scripts/real/redshift/plot_redshift_4methods.py \
  --summary results/real/redshift/results_redshift_4methods.csv \
  --out_dir fig/real/redshift \
  --content_level 0.90
```

### TSA
Run : 
```bash
python scripts/real/tsa/real_tsa_4methods.py \
  --split_mode random
```
Plot :
```bash
python scripts/real/tsa/plot_tsa_4methods.py \
  --summary results/real/tsa/results_tsa_4methods.csv \
  --intervals results/real/tsa/results_tsa_4methods_intervals.csv \
  --out_dir fig/real/tsa \
  --content_level 0.90 \
  --window 14
```

## Notes
- PAC-calibrated methods use split data.
- Parametric TI is a model-based benchmark.
- Real-data results report empirical content, not true conditional coverage.
- fig/, results/, *.csv, and *.png are generated artifacts.

## References

```text
Guo, Y. and Young, D. S. (2024).
Approximate tolerance intervals for nonparametric regression models.
Journal of Nonparametric Statistics, 36(1), 212--239.

Romano, Y., Patterson, E., and Candès, E. J. (2019).
Conformalized quantile regression.
NeurIPS.

Vovk, V. (2012).
Conditional validity of inductive conformal predictors.
ACML.
```