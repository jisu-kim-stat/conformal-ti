compute_ti_point <- function(x0, fit_mean, fit_var, lambda_hat) {
  mu0  <- predict_mean(fit_mean, x0)
  v0   <- predict_var(fit_var, x0)
  lower <- mu0 - sqrt(v0) * lambda_hat
  upper <- mu0 + sqrt(v0) * lambda_hat
  list(lower = lower, upper = upper, mu = mu0, var = v0)
}
