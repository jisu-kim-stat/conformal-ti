# R/sim/lambda_hoeffding.R

find_score_cutoff <- function(mis, alpha, score) {

  stopifnot(
    mis > 0, mis < 1,
    alpha > 0, alpha < 1,
    all(is.finite(score))
  )

  n <- length(score)

  C <- 1 - mis

  lambda_hoef <- sqrt(log(2 / alpha) / (2 * n))

  target <- C + lambda_hoef

  if (target > 1) {
    return(NA_real_)
  }

  score_sorted <- sort(score)

  idx <- ceiling(n * target)
  idx <- min(max(idx, 1), n)

  score_sorted[idx]
}

# HCTI
find_lambda_hat <- function(mis, alpha, y, pred, variance) {

  stopifnot(
    length(y) == length(pred),
    length(y) == length(variance)
  )

  score <- abs(y - pred) / sqrt(pmax(variance, 1e-8))

  find_score_cutoff(
    mis = mis,
    alpha = alpha,
    score = score
  )
}

find_asym_shape <- function(z, tau = 0.05, eps = 1e-6) {
  stopifnot(
    tau > 0, tau < 0.5,
    all(is.finite(z))
  )

  q_lo <- as.numeric(stats::quantile(z, probs = tau, names = FALSE, type = 8))
  q_hi <- as.numeric(stats::quantile(z, probs = 1 - tau, names = FALSE, type = 8))

  a_minus <- abs(q_lo)
  a_plus  <- q_hi

  a_minus <- max(a_minus, eps)
  a_plus  <- max(a_plus, eps)

  c(a_minus = a_minus, a_plus = a_plus)
}


asym_residual_score <- function(z, a_minus, a_plus) {
  pmax(
    -z / a_minus,
     z / a_plus
  )
}

