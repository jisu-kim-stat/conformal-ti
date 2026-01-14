## experiments/real/happy/R/hoeffding.R

find_lambda_fast <- function(alpha, delta, z_cal, mu_cal, var_cal) {
  n_cal <- length(z_cal)
  thr <- max(0.0, alpha - sqrt(log(1.0 / delta) / (2.0 * n_cal)))

  sd <- sqrt(pmax(var_cal, 1e-12))
  r <- abs(z_cal - mu_cal) / sd

  q <- 1.0 - thr
  q <- min(max(q, 0.0), 1.0)

  as.numeric(stats::quantile(r[is.finite(r)], probs = q, na.rm = TRUE, type = 8))
}
