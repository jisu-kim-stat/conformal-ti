fit_mean_model <- function(x, y) {
  smooth.spline(x, y, cv = FALSE)
}

fit_var_model <- function(x, y, fit_mean) {
  res <- y - as.numeric(predict(fit_mean, x)$y)
  log_res2 <- log(res^2 + 1e-6)
  df <- data.frame(x = x, log_res2 = log_res2)
  gam::gam(log_res2 ~ s(x), data = df)
}

predict_mean <- function(fit_mean, xnew) {
  as.numeric(predict(fit_mean, xnew)$y)
}

predict_var <- function(fit_var, xnew) {
  as.numeric(exp(predict(fit_var, newdata = data.frame(x = xnew))))
}
