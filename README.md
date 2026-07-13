## Simulation

This repository includes simulation code for comparing:

- **HCTI**: Hoeffding-Conformal Tolerance Interval
- **CQR-TI**: CQR-based Tolerance Interval
- **Parametric-TI**: spline-based parametric tolerance interval baseline

### Structure

```text
R/sim/      core simulation functions
R/utils/    plotting and saving utilities
scripts/    runnable scripts
```

Main files:

```text
R/sim/data_generate.R
R/sim/truth_content.R
R/sim/fit_hcti.R
R/sim/fit_cqr.R
R/sim/lambda_hoeffding.R
R/sim/one_replication.R
R/sim/run_one_setting.R
scripts/run_simulation_grid.R
scripts/make_plot.R
```

### Settings

```r
content_level <- 0.90
alpha <- 0.05
n_test <- 1000
n_cal_vec <- c(200, 500, 1000)
models <- 1:6
```

Inside the simulation grid:

```r
n_train <- n_cal
```

### Metrics

The simulation reports:

```text
1. Marginal PAC success
2. PX-good proportion
3. Average interval width
```

For models 1--5, pointwise curves are plotted over the one-dimensional test grid.

For model 6, which is high-dimensional, only summary metrics are reported.

### Run

From the project root:

```bash
Rscript scripts/run_simulation_grid.R
```

Then create plots:

```bash
Rscript scripts/make_plot.R
```

Open plots on macOS:

```bash
open results/sim/models/plots
```

### Outputs

CSV outputs:

```text
results/sim/models/pointwise_success_hcti_cqr_pti_ncal_grid.csv
results/sim/models/marginal_pac_hcti_cqr_pti_ncal_grid.csv
results/sim/models/px_good_proportion_hcti_cqr_pti_ncal_grid.csv
```

Plot outputs:

```text
results/sim/models/plots/
```

Main plot files:

```text
marginal_pac_success_vs_ncal.png
mean_marginal_content_vs_ncal.png
px_good_proportion_vs_ncal.png
average_width_vs_ncal.png
pointwise_pac_success_curve_models1to5.png
mean_conditional_content_curve_models1to5.png
pointwise_width_curve_models1to5.png
```

### Notes

```text
- HCTI and CQR-TI use split data.
- Parametric-TI uses the full fitting sample.
- The Hoeffding/DKW correction can be conservative for small calibration sizes.
- Larger n_cal usually reduces interval width.
- results/, *.csv, and *.png are excluded from version control.
```

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