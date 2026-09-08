# Base-R regression tests for the two model-based simulation benchmarks.
# Run from the project root:
#   Rscript tests/test_sim_ti_methods.R

source("R/packages.R")
source("R/sim/base_mean.R")
source("R/sim/data_generate_balanced4.R")
source("R/sim/truth_content_balanced4.R")
source("R/sim/fit_srti.R")
source("R/sim/fit_cqr.R")
source("R/sim/lambda_hoeffding.R")
source("R/sim/pti_utils.R")
source("R/sim/guo_young_ti.R")
source("R/sim/one_replication.R")
source("R/sim/run_one_setting.R")


# 1. Classical Parametric-TI must agree with the corresponding OLS fit.
set.seed(10)
x_classical <- seq(-2, 2, length.out = 80)
y_classical <- 2 + 0.7 * sin(2 * pi * x_classical) +
  stats::rnorm(length(x_classical), sd = 0.8)
x_classical_new <- c(-1.5, 0, 1.25)

classical_fit <- classical_parametric_ti(
  x = x_classical,
  y = y_classical,
  x_new = x_classical_new,
  content = 0.90,
  alpha = 0.05
)
design_train <- classical_parametric_design(x_classical)
design_new <- classical_parametric_design(x_classical_new)
lm_fit <- stats::lm.fit(design_train, y_classical)
lm_prediction <- drop(design_new %*% lm_fit$coefficients)
lm_sigma <- sqrt(sum(lm_fit$residuals^2) / lm_fit$df.residual)

stopifnot(
  max(abs(classical_fit$beta_hat - lm_fit$coefficients)) < 1e-6,
  max(abs(classical_fit$fitted - lm_prediction)) < 1e-6,
  abs(classical_fit$sigma_hat - lm_sigma) < 1e-10,
  classical_fit$nu == lm_fit$df.residual,
  all(classical_fit$interval[, "upper"] >
        classical_fit$interval[, "lower"])
)


# 2. The reconstructed linear smoother must reproduce smooth.spline.
set.seed(11)
x <- stats::runif(60, -2, 2)
y <- base_mean(x) + stats::rnorm(60)
fit <- gy_fit_smoothing_spline(x, y)

stopifnot(
  fit$fit_error < 1e-8,
  is.finite(fit$sigma_hat),
  fit$sigma_hat > 0,
  is.finite(fit$nu),
  fit$nu > 0
)


# 3. The accelerated Equation (14) calculation must agree with a direct
#    noncentral-chi-square calculation and attain gamma.
ell_norm <- 0.25
k_cached <- gy_two_sided_k(
  ell_norm = ell_norm,
  nu = fit$nu,
  content = 0.90,
  gamma = 0.95,
  use_quantile_cache = TRUE
)
k_direct <- gy_two_sided_k(
  ell_norm = ell_norm,
  nu = fit$nu,
  content = 0.90,
  gamma = 0.95,
  use_quantile_cache = FALSE
)
probability_direct <- gy_equation14_probability(
  k = k_cached,
  ell_norm = ell_norm,
  nu = fit$nu,
  content = 0.90,
  use_quantile_cache = FALSE
)

stopifnot(
  abs(k_cached - k_direct) < 1e-5,
  abs(probability_direct - 0.95) < 1e-5
)

# The simulation uses the closed-form approximation in Appendix
# Lemma A.1(3), not Equation (14)'s repeated integration/root solving.
appendix_norm <- c(0.1, 0.25, 0.5)
k_appendix <- gy_appendix_k(
  ell_norm = appendix_norm,
  nu = fit$nu,
  content = 0.90,
  gamma = 0.95
)
k_appendix_reference <- sqrt(
  fit$nu *
    stats::qchisq(0.90, df = 1, ncp = appendix_norm^2) /
    stats::qchisq(0.05, df = fit$nu)
)
stopifnot(max(abs(k_appendix - k_appendix_reference)) < 1e-12)

# Regression test for the integrate() roundoff failure previously observed
# in GY-TI, Model 1, uniform design, replication 1.
k_roundoff_case <- gy_two_sided_k(
  ell_norm = 0.1892904,
  nu = 383.1177,
  content = 0.90,
  gamma = 0.95
)
stopifnot(is.finite(k_roundoff_case), k_roundoff_case > 0)


# 4. Both model-based methods must return finite intervals of the requested
#    length on the same simulation setting.
pti <- one_replication_pti(
  model_id = 1,
  n_train = 30,
  n_cal = 30,
  n_test = 11,
  content = 0.90,
  alpha = 0.05,
  seed = 7,
  design = "uniform"
)
gy <- one_replication_gy(
  model_id = 1,
  n_train = 30,
  n_cal = 30,
  n_test = 11,
  content = 0.90,
  alpha = 0.05,
  seed = 7,
  design = "uniform"
)

stopifnot(
  length(pti$content) == 11L,
  length(gy$content) == 11L,
  all(is.finite(pti$content)),
  all(is.finite(pti$width)),
  all(is.finite(gy$content)),
  all(is.finite(gy$width)),
  all(pti$width > 0),
  all(gy$width > 0)
)

# Large-sample regression test for Classical Parametric-TI.
pti_rep_945 <- one_replication_pti(
  model_id = 1,
  n_train = 200,
  n_cal = 200,
  n_test = 11,
  content = 0.90,
  alpha = 0.05,
  seed = 945,
  design = "uniform"
)

stopifnot(
  all(is.finite(pti_rep_945$content)),
  all(is.finite(pti_rep_945$width)),
  all(pti_rep_945$width > 0)
)


# 5. The complete sequential runner must expose exactly the five paper-facing
#    method labels. n_cal = 200 makes the Hoeffding cutoff feasible.
foreach::registerDoSEQ()
smoke <- run_one_setting(
  model_id = 1,
  n_train = 80,
  n_cal = 200,
  n_test = 9,
  M = 1,
  content = 0.90,
  alpha = 0.05,
  epsilon_grid = c(0, 0.02),
  design = "uniform"
)

expected_methods <- c(
  "SR-TI",
  "ASR-TI",
  "CQR-TI",
  "Parametric-TI",
  "GY-TI"
)

stopifnot(
  setequal(smoke$marginal$Method, expected_methods),
  nrow(smoke$marginal) == 5L,
  nrow(smoke$pointwise) == 9L * 2L * 5L,
  all(is.finite(smoke$marginal$marginal_content_mean))
)

cat("All simulation TI tests passed.\n")
