# Five-DGP simulation suite for the main paper.
# Model 1 is a homoscedastic baseline.  Models 2--4 are heteroscedastic
# location--scale models with an x-invariant standardized residual law.  Model
# 5 additionally has x-dependent residual shape and is not location--scale.

# A smooth common scale for Models 2--5.  Its conditional variance is
# 1 + x^2, which retains substantial heteroscedasticity without the cusp and
# more severe tail extrapolation induced by 1 + |x|.
hetero_scale <- function(x) sqrt(1 + x^2)

generate_data <- function(model_id, n, design = "uniform",
                          heavy_tail_scale = c("original", "unit_variance")) {
  stopifnot(model_id %in% 1:5)
  design <- match.arg(design, c("uniform", "normal"))
  match.arg(heavy_tail_scale)

  x <- if (design == "uniform") runif(n, -2, 2) else rnorm(n)
  base <- base_mean(x)

  if (model_id == 1) {
    y <- base + rnorm(n)
  }

  if (model_id == 2) {
    y <- base + hetero_scale(x) * rt(n, df = 3)
  }

  if (model_id == 3) {
    y <- base + hetero_scale(x) * rnorm(n)
  }

  if (model_id == 4) {
    # Centered, standardized Gamma(4, 1): mean zero, variance one, and
    # moderate global right skewness (skewness one).
    y <- base + hetero_scale(x) * (rgamma(n, shape = 4, rate = 1) - 4) / 2
  }

  if (model_id == 5) {
    # A recentered two-piece Gaussian residual. The ratio of lower and upper
    # tail scales varies with x, so centering and scalar scaling cannot make
    # the conditional residual distribution invariant in x.
    z <- rnorm(n)
    s_minus <- hetero_scale(x)
    s_plus <- s_minus * (1 + 0.8 * plogis(1.5 * x))
    center <- (s_plus - s_minus) / sqrt(2 * pi)
    eps <- ifelse(z < 0, s_minus * z, s_plus * z) - center
    y <- base + eps
  }

  data.frame(x = x, y = y)
}

generate_eval_data <- function(model_id, n, design = "uniform") {
  stopifnot(model_id %in% 1:5)
  design <- match.arg(design, c("uniform", "normal"))
  p <- (seq_len(n) - 0.5) / n
  x <- if (design == "uniform") qunif(p, -2, 2) else qnorm(p)
  data.frame(x = x, y = rep(NA_real_, n))
}

generate_y_given_x <- function(model_id, x,
                                heavy_tail_scale = c("original", "unit_variance")) {
  stopifnot(model_id %in% 1:5)
  match.arg(heavy_tail_scale)
  base <- base_mean(x)
  n <- length(x)

  if (model_id == 1) return(base + rnorm(n))
  if (model_id == 2) return(base + hetero_scale(x) * rt(n, df = 3))
  if (model_id == 3) return(base + hetero_scale(x) * rnorm(n))
  if (model_id == 4) {
    return(base + hetero_scale(x) * (rgamma(n, shape = 4, rate = 1) - 4) / 2)
  }

  z <- rnorm(n)
  s_minus <- hetero_scale(x)
  s_plus <- s_minus * (1 + 0.8 * plogis(1.5 * x))
  center <- (s_plus - s_minus) / sqrt(2 * pi)
  base + ifelse(z < 0, s_minus * z, s_plus * z) - center
}
