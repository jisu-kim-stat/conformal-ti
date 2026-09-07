# PAC-Calibrated Regression Tolerance Intervals

This repository contains simulation and real-data code for regression tolerance intervals with marginal PAC calibration and conditional PAC diagnostics.

## Methods

The repository compares five methods:

- **SR-TI**: symmetric standardized residual score
- **ASR-TI**: asymmetric standardized residual score
- **CQR-TI**: conformalized lower- and upper-quantile regression score
- **Parametric-TI**: classical homoscedastic normal-regression TI with
  mean model \(\beta_0+\beta_1\sin(2\pi x)\)
- **GY-TI**: Guo and Young (2024) homoscedastic pointwise TI, using
  Equation (11) and the fast k-factor approximation in Appendix
  Lemma A.1(3)

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

The main simulation suite is the four-DGP design implemented in
`scripts/sim/run_simulation_grid_balanced4.R`:

1. homoscedastic Gaussian errors;
2. unit-variance heavy-tailed \(t_3\) errors;
3. heteroscedastic Gaussian location--scale errors; and
4. smooth \(X\)-dependent endpoint asymmetry.

All methods use \(n_{\rm tr}=n_{\rm cal}\in\{200,500,1000\}\), target
content \(C=0.90\), and confidence level \(1-\alpha=0.95\).  The output
files are written to `results/sim/balanced4/` and are distinguished by the
user-supplied tag.

### CQR base learner

CQR fits the 0.05 and 0.95 conditional quantiles using natural-spline
quantile regression on a fixed covariate support.  For each training split,
the two spline complexities are selected separately by five-fold pinball-loss
cross-validation over df \(\in\{4,6,8,10,12\}\).  The calibration split is
used only to select the conformal score cutoff.

Run the complete four-model study from the project root:

```bash
Rscript scripts/sim/run_simulation_grid_balanced4.R \
  --reps=1000 \
  --cores=4 \
  --tag=full_cv
```

For a faster CQR-only validation on the homoscedastic Gaussian model:

```bash
Rscript scripts/sim/run_simulation_grid_balanced4.R \
  --reps=200 \
  --cores=4 \
  --models=1 \
  --methods=CQR-TI \
  --tag=cqr_cv_pilot
```

The runner also accepts `--ncal=200,500,1000`, `--designs=normal,uniform`,
`--models=1,2,3,4`, and `--methods=` to restrict a run.  The legacy
fixed-B-spline CQR implementation can be reproduced only by explicitly
passing `--cqr_basis_type=legacy_bs`.

Quick validation:

```text
Rscript tests/test_sim_ti_methods.R
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
- Parametric-TI and GY-TI are distinct model-based benchmarks.
- GY-TI assumes homoscedastic errors and is not PAC calibrated.
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
