# PAC-Calibrated Regression Tolerance Intervals

This repository contains simulation and real-data code for regression tolerance intervals with marginal PAC calibration and conditional PAC diagnostics.

## Methods

The paper simulations compare four methods:

- **SR-TI**: symmetric standardized residual score
- **CQR-TI**: conformalized lower- and upper-quantile regression score
- **Parametric-TI**: classical homoscedastic normal-regression TI with a
  fixed cubic B-spline mean basis of 10 spline functions and an intercept
- **GY-TI**: Guo and Young (2024) homoscedastic pointwise TI, using
  Equation (11) and the fast k-factor approximation in Appendix
  Lemma A.1(3)

The PAC-calibrated methods use the empirical \((C+\lambda_\alpha)\)-quantile of calibration scores, where

```text
lambda_alpha = sqrt(log(1 / alpha) / (2 * n_cal))
```

Default Settings : 
```text
content_level = 0.90
pac_alpha     = 0.05
```

## Structure
```text
ti_project/
├── scripts/
│   ├── sim/
├── results/
│   ├── sim/
├── fig/
│   └── sim/
└── README.md
```

## Simulation

The main simulation suite is the five-DGP design implemented in
`scripts/sim/run_simulation_grid_balanced4.R`:

1. homoscedastic Gaussian errors;
2. heteroscedastic heavy-tailed \(t_3\) location--scale errors;
3. heteroscedastic Gaussian location--scale errors;
4. heteroscedastic globally skewed location--scale errors generated from a
   centered, standardized \(\operatorname{Gamma}(4,1)\) innovation; and
5. heteroscedastic two-piece Gaussian errors with \(X\)-dependent tail
   asymmetry (a non-location--scale departure).

All methods use \(n_{\rm tr}=n_{\rm cal}\in\{200,500,1000\}\), target
content \(C=0.90\), and confidence level \(1-\alpha=0.95\).  The output
files are written to `results/sim/balanced4/` and are distinguished by the
user-supplied tag.  Models 2--5 use the common smooth scale
\(\sigma(x)=\sqrt{1+x^2}\).

### CQR base learner

CQR fits the 0.05 and 0.95 conditional quantiles with natural-spline quantile
regression on a fixed covariate support. For each training split, the two
spline complexities are selected separately by pinball-loss cross-validation;
the calibration split is used only to select the conformal score cutoff. An
experimental penalized quantile smoothing spline remains available through
`--cqr_basis_type=cv_rqss`.

Run the complete five-model study from the project root:

```bash
Rscript scripts/sim/run_simulation_grid_balanced4.R \
  --reps=1000 \
  --cores=4 \
  --models=1,2,3,4,5 \
  --ncal=200,500,1000 \
  --designs=normal,uniform \
  --methods=SR-TI,CQR-TI,Parametric-TI,GY-TI \
  --cqr_basis_type=cv_fixed_ns \
  --tag=sr_cqr_full

Rscript scripts/sim/make_plot.R --tag=sr_cqr_full
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
`--models=1,2,3,4,5`, and `--methods=` to restrict a run.  Pass
`--methods=Oracle-CQR-TI` only for a simulation diagnostic that uses known
conditional quantiles; it is not a competing method in the paper.

Quick validation:

```text
Rscript tests/test_sim_ti_methods.R
```

## Notes
- PAC-calibrated methods use split data.
- SR--TI and CQR--TI are the proposed PAC-calibrated procedures in the paper.
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
