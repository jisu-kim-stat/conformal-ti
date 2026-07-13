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
