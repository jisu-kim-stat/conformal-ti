# Nuisance mean/variance models shared by SR-TI and ASR-TI.
# The historical filename is retained so existing source() calls keep working.
# One-dimensional implementation.
fit_mean_model <- function(x, y) {
  x <- as.numeric(x)
  y <- as.numeric(y)

  smooth.spline(x = x, y = y, cv = FALSE)
}

fit_var_model <- function(x, y, fit_mean,
                          eps = 1e-4,
                          k = 10,
                          method = "REML") {
  x <- as.numeric(x)
  y <- as.numeric(y)

  res <- y - as.numeric(predict(fit_mean, x)$y)
  res2 <- res^2 + eps

  df <- data.frame(
    xx = x,
    res2 = res2
  )

  mgcv::gam(
    res2 ~ s(xx, k = k),
    data = df,
    family = Gamma(link = "log"),
    method = method
  )
}

predict_mean <- function(fit_mean, xnew) {
  xnew <- as.numeric(xnew)

  as.numeric(predict(fit_mean, xnew)$y)
}

predict_var <- function(fit_var, xnew,
                        min_var = 1e-3,
                        max_var = 1e3) {
  xnew <- as.numeric(xnew)

  vhat <- as.numeric(
    predict(
      fit_var,
      newdata = data.frame(xx = xnew),
      type = "response"
    )
  )

  pmin(pmax(vhat, min_var), max_var)
}


# The high-dimensional functions can remain unused, but the auto wrappers
# should always use the one-dimensional functions for the revised DGPs.

fit_mean_model_auto <- function(x, y, model_id) {
  fit_mean_model(x, y)
}

fit_var_model_auto <- function(x, y, fit_mean, model_id) {
  fit_var_model(x, y, fit_mean)
}

predict_mean_auto <- function(fit_mean, xnew, model_id) {
  predict_mean(fit_mean, xnew)
}

predict_var_auto <- function(fit_var, xnew, model_id) {
  predict_var(fit_var, xnew)
}

check_var_model <- function(fit_var) {
  print(summary(fit_var))
  mgcv::gam.check(fit_var)
}
