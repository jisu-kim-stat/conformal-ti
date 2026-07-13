## Simulation (HCTI vs CQR-TI vs Parametric-TI)

This repository includes a simulation framework to compare three tolerance interval methods:

- **HCTI**: Hoeffding-Conformal Tolerance Interval using a mean--variance standardized nonconformity score
- **CQR-TI**: Hoeffding-adjusted tolerance interval based on conformalized quantile regression scores
- **Parametric-TI**: spline-based parametric tolerance interval baseline

The main simulation goal is to evaluate:

1. **Marginal PAC success**
2. **\(P_X\)-good proportion** for the \(P_X\)-averaged conditional PAC criterion
3. **Average interval width**

The target content level is \(C = 0.90\), and the target confidence level is \(1-\alpha = 0.95\).

---

### Setup

All simulation code is under:

- `R/sim/` : core functions for data generation, true content calculation, model fitting, calibration, and replication
- `R/utils/` : plotting and result-saving utilities
- `scripts/` : runnable entry scripts

Key files:

- `R/sim/base_mean.R` : common mean function used in the one-dimensional DGPs
- `R/sim/data_generate.R` : `generate_data(model_id, n)` for simulation DGPs
- `R/sim/truth_content.R` : `content_function(model_id, lower, upper, x)` for true conditional content
- `R/sim/fit_hcti.R` : nuisance fitting functions for HCTI
- `R/sim/fit_cqr.R` : quantile regression fitting functions for CQR-TI
- `R/sim/lambda_hoeffding.R` : Hoeffding/DKW-adjusted score cutoff calculation
- `R/sim/pti_utils.R` : utilities for the parametric tolerance interval baseline
- `R/sim/one_replication.R` : one replication for HCTI, CQR-TI, and Parametric-TI
- `R/sim/run_one_setting.R` : runs \(M\) Monte Carlo replications for one model/sample-size setting
- `R/utils/plotting.R` : plotting functions
- `R/utils/save_results.R` : CSV-saving utilities
- `scripts/run_simulation_grid.R` : runs a grid over DGPs and calibration sizes
- `scripts/make_plot.R` : reads simulation outputs and creates plots

---

### Data-generating processes

The simulation currently uses six DGPs:

1. **Homoscedastic Normal**  
   \[
   Y = \mu(X) + \varepsilon, 
   \qquad \varepsilon \sim N(0,1).
   \]

2. **Heavy-tailed**  
   \[
   Y = \mu(X) + \varepsilon,
   \qquad \varepsilon \sim t_3.
   \]

3. **Heteroscedastic**  
   \[
   Y = \mu(X) + (1+|X|)\varepsilon,
   \qquad \varepsilon \sim N(0,1).
   \]

4. **Asymmetric**  
   \[
   Y = \mu(X) + \varepsilon,
   \qquad \varepsilon \sim \operatorname{Exp}(1)-1.
   \]

5. **Smoothly varying variance**  
   \[
   Y = \mu(X) + \sigma(X)\varepsilon,
   \qquad 
   \sigma(X)^2 = 0.75 + 0.5\sin^2(2\pi X).
   \]

6. **High-dimensional sparse linear model**  
   \[
   X \in \mathbb{R}^{20},
   \qquad
   Y = X^\top \beta + \varepsilon,
   \qquad
   \varepsilon \sim N(0,1),
   \]
   where
   \[
   \beta = (0.5, 0.4, 0.3, 0.2, 0,\ldots,0).
   \]

For models 1--5, the evaluation grid is

```r
x = seq(-2, 2, length.out = n_test)
```

For model 6, the covariates are 20-dimensional, so pointwise curves over a one-dimensional x are not interpreted. Model 6 is summarized numerically using marginal and $P_X$-averaged metrics.

---

### Parameters and notation

- `content` : target content level \(C\), e.g. `0.90`
- `mis = 1 - content` : miscoverage level
- `alpha` : confidence error level, so confidence is \(1-\alpha\)
- `n_train` : training sample size for nuisance model fitting
- `n_cal` : calibration sample size for score calibration
- `n_test` : number of test/evaluation points, typically `1000`
- `M` : number of Monte Carlo replications per `(model_id, n_cal)` setting

The current default simulation uses:

```r
content_level <- 0.90
alpha <- 0.05
n_test <- 1000
n_cal_vec <- c(200, 500, 1000)
models <- 1:6
```

Inside the simulation grid, we typically set:

```r
n_train <- n_cal
```

Thus, for each calibration size, the training sample size is matched to the calibration sample size.

---
### Monte Carlo structure

For each setting `(model_id, n_train, n_cal)`, the simulation repeats the following for `b = 1, ..., M`:

1. Generate training data.
2. Generate calibration data.
3. Fit nuisance models on the training data.
4. Compute calibration scores on the calibration data.
5. Compute the Hoeffding/DKW-adjusted threshold.
6. Construct tolerance intervals over the test grid.
7. Evaluate true conditional content and interval width.

For each test point $x_j$, the simulation estimates

$$
\widehat{P}_D\left[P_{Y \mid X=x_j}\left\{Y \in T(x_j;D)\right\} \ge C\right]
=
\frac{1}{M}\sum_{b=1}^{M}\mathbf{1}\left\{\operatorname{content}^{(b)}(x_j) \ge C\right\}.
$$
---

### Reported metrics

The simulation produces three main summaries.

#### 1. Marginal PAC success

For each replication $b$, marginal content is approximated over the test points:

$$
\operatorname{marginal\_content}^{(b)}
=
\frac{1}{n_{\text{test}}}
\sum_{j=1}^{n_{\text{test}}}
\operatorname{content}^{(b)}(x_j).
$$

The reported marginal PAC success is

$$
\frac{1}{M}
\sum_{b=1}^{M}
\mathbf{1}
\left\{
\operatorname{marginal\_content}^{(b)} \ge C
\right\}.
$$

This estimates

$$
P_D
\left\{
P_{X,Y}\{Y \in T(X;D)\} \ge C
\right\}.
$$

The target is at least $1-\alpha$.

#### 2. $P_X$-good proportion

For each test point $x_j$, define the pointwise PAC success estimate:

$$
\widehat{p}(x_j)
=
\frac{1}{M}
\sum_{b=1}^{M}
\mathbf{1}
\left\{
\operatorname{content}^{(b)}(x_j) \ge C
\right\}.
$$

The $P_X$-good proportion is

$$
\frac{1}{n_{\text{test}}}
\sum_{j=1}^{n_{\text{test}}}
\mathbf{1}
\left\{
\widehat{p}(x_j) \ge 1-\alpha
\right\}.
$$

This estimates

$$
P_X
\left\{
x:
P_D
\left[
P_{Y \mid X=x}\{Y \in T(x;D)\} \ge C
\right]
\ge 1-\alpha
\right\}.
$$

This quantity is used to empirically assess the $P_X$-averaged conditional PAC behavior.

#### 3. Average interval width

For each replication,

$$
\operatorname{average\_width}^{(b)}
=
\frac{1}{n_{\text{test}}}
\sum_{j=1}^{n_{\text{test}}}
|T^{(b)}(x_j)|.
$$

The reported average width is the Monte Carlo mean over $b=1,\ldots,M$.

---

### How to run

From the project root, run:

```bash
Rscript scripts/run_simulation_grid.R
```

Then generate plots:

```bash
Rscript scripts/make_plot.R
```

Open the output folder on macOS:

```bash
open results/sim/models/plots
```

---

### Outputs

The simulation writes three main CSV files.

#### Pointwise results

```text
results/sim/models/pointwise_success_hcti_cqr_pti_ncal_grid.csv
```

Columns:

- `x` : evaluation grid point for models 1--5; test-point index for model 6
- `mean_content` : Monte Carlo mean of true conditional content at `x`
- `pointwise_success` : proportion of replications with `content(x;D) >= content`
- `mean_width` : Monte Carlo mean interval width at `x`
- `na_proportion` : proportion of replications with NA/failed threshold
- `model` : DGP id
- `n_train` : training sample size
- `n_cal` : calibration sample size
- `n_test` : number of test points
- `Method` : `"HCTI"`, `"CQR-TI"`, or `"Parametric-TI"`

#### Marginal results

```text
results/sim/models/marginal_pac_hcti_cqr_pti_ncal_grid.csv
```

Columns:

- `marginal_content_mean` : Monte Carlo mean of marginal content
- `marginal_content_sd` : Monte Carlo standard deviation of marginal content
- `marginal_pac_success` : proportion of replications with marginal content at least `content`
- `average_width_mean` : Monte Carlo mean of average interval width
- `average_width_sd` : Monte Carlo standard deviation of average interval width
- `na_proportion` : average NA/failure proportion
- `model` : DGP id
- `n_train` : training sample size
- `n_cal` : calibration sample size
- `n_test` : number of test points
- `Method` : `"HCTI"`, `"CQR-TI"`, or `"Parametric-TI"`

#### $P_X$-good results

```text
results/sim/models/px_good_proportion_hcti_cqr_pti_ncal_grid.csv
```

Columns:

- `px_good_proportion` : proportion of test covariate values satisfying the pointwise PAC criterion
- `min_pointwise_success` : minimum pointwise PAC success over test points
- `q25_pointwise_success` : first quartile of pointwise PAC success over test points
- `median_pointwise_success` : median pointwise PAC success over test points
- `q75_pointwise_success` : third quartile of pointwise PAC success over test points
- `max_pointwise_success` : maximum pointwise PAC success over test points
- `mean_pointwise_success` : average pointwise PAC success over test points
- `mean_content` : average conditional content over test points
- `mean_width` : average interval width over test points
- `na_proportion` : average NA/failure proportion
- `model` : DGP id
- `n_train` : training sample size
- `n_cal` : calibration sample size
- `n_test` : number of test points
- `Method` : `"HCTI"`, `"CQR-TI"`, or `"Parametric-TI"`

---

### Plots

The script `scripts/make_plot.R` creates the following figures under:

```text
results/sim/models/plots/
```

Main figures:

- `marginal_pac_success_vs_ncal.png`
- `mean_marginal_content_vs_ncal.png`
- `px_good_proportion_vs_ncal.png`
- `average_width_vs_ncal.png`
- `pointwise_pac_success_curve_models1to5.png`
- `mean_conditional_content_curve_models1to5.png`
- `pointwise_width_curve_models1to5.png`

Model 6 is high-dimensional, so pointwise curves are excluded from the main pointwise plots. Its marginal and $P_X$-good summaries are saved separately.

---

### Notes

- HCTI and CQR-TI use split data: training data for nuisance fitting and calibration data for score calibration.
- Parametric-TI is fitted using the full fitting sample `n_train + n_cal`.
- The Hoeffding/DKW correction can be conservative for moderate calibration sizes.
- With $C=0.90$ and $\alpha=0.05$,

$$
\lambda_\alpha
=
\sqrt{
\frac{\log(2/\alpha)}{2n_{\text{cal}}}
}.
$$

Thus, increasing `n_cal` decreases the correction and typically reduces interval width.
- Results under `results/`, `*.csv`, and `*.png` are excluded from version control.

---

## References

- **Guo, Y. and Young, D. S. (2024).**  
  *Approximate tolerance intervals for nonparametric regression models.*  
  Journal of Nonparametric Statistics, **36**(1), 212--239.  
  DOI: https://doi.org/10.1080/10485252.2023.2277260

- **Romano, Y., Patterson, E., and Candès, E. J. (2019).**  
  *Conformalized quantile regression.*  
  Advances in Neural Information Processing Systems, **32**.

- **Vovk, V. (2012).**  
  *Conditional validity of inductive conformal predictors.*  
  Proceedings of the Asian Conference on Machine Learning, **25**, 475--490.