# R/sim/lambda_hoeffding.R
# --------------------------------------------------
# Estimate Hoeffding-adjusted lambda
#
# Inputs:
#   mis      : miscoverage level (= 1 - c)
#   alpha    : confidence error (confidence = 1 - alpha)
#   y        : calibration responses
#   pred     : predicted mean on calibration points
#   variance : predicted variance on calibration points
#   n        : calibration sample size
#
# Output:
#   lambda_hat (numeric scalar) or NA if infeasible
# --------------------------------------------------

find_lambda_hat <- function(mis, alpha, y, pred, variance, n) {

  stopifnot(
    mis > 0, mis < 1,
    alpha > 0, alpha < 1,
    length(y) == length(pred),
    length(y) == length(variance)
  )

  # standardized residuals
  z <- abs(y - pred) / sqrt(pmax(variance, 1e-8))

  # empirical CDF of residuals
  z_sorted <- sort(z)
  F_hat <- function(t) mean(z_sorted <= t)

  # Hoeffding correction
  lambda_hoef <- sqrt(log(1 / alpha) / (2 * n))

  target <- 1 - mis + lambda_hoef
  if (target > 1) return(NA_real_)

  # smallest k such that F_hat(k) >= target
  idx <- which(cumsum(rep(1 / n, n)) >= target)[1]
  if (is.na(idx)) return(NA_real_)

  z_sorted[idx]
}
