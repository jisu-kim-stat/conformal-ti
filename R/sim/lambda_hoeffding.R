# R/sim/lambda_hoeffding.R

pac_calibration_index <- function(n, content, alpha,
                                  calibration_rule = c(
                                    "hoeffding",
                                    "exact_binomial"
                                  )) {
  calibration_rule <- match.arg(calibration_rule)

  stopifnot(
    length(n) == 1L, n >= 1L,
    length(content) == 1L, content > 0, content < 1,
    length(alpha) == 1L, alpha > 0, alpha < 1
  )

  if (calibration_rule == "hoeffding") {
    k <- ceiling(
      n * content + sqrt(n * log(1 / alpha) / 2)
    )
  } else {
    # Smallest k such that
    # P{Binomial(n, content) >= k} <= alpha.
    k <- stats::qbinom(1 - alpha, size = n, prob = content) + 1L
  }

  min(as.integer(k), n + 1L)
}


find_score_cutoff <- function(mis, alpha, score,
                              calibration_rule = c(
                                "hoeffding",
                                "exact_binomial"
                              )) {

  calibration_rule <- match.arg(calibration_rule)

  stopifnot(
    mis > 0, mis < 1,
    alpha > 0, alpha < 1,
    all(is.finite(score))
  )

  n <- length(score)

  k <- pac_calibration_index(
    n = n,
    content = 1 - mis,
    alpha = alpha,
    calibration_rule = calibration_rule
  )

  # No finite order statistic can attain the requested distribution-free
  # guarantee. This does not occur in the simulation settings considered here.
  if (k == n + 1L) {
    return(NA_real_)
  }

  score_sorted <- sort(score)

  score_sorted[k]
}

# SR-TI
find_lambda_hat <- function(mis, alpha, y, pred, variance,
                            calibration_rule = c(
                              "hoeffding",
                              "exact_binomial"
                            )) {

  calibration_rule <- match.arg(calibration_rule)

  stopifnot(
    length(y) == length(pred),
    length(y) == length(variance)
  )

  score <- abs(y - pred) / sqrt(pmax(variance, 1e-8))

  find_score_cutoff(
    mis = mis,
    alpha = alpha,
    score = score,
    calibration_rule = calibration_rule
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
