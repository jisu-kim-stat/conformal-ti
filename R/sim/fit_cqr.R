fit_quantile_model <- function(x, y, tau, basis_df = 8) {
  df <- data.frame(
    x = x,
    y = y
  )

  quantreg::rq(
    y ~ splines::bs(x, df = basis_df),
    tau = tau,
    data = df
  )
}

predict_quantile <- function(fit_quantile, xnew) {
  as.numeric(
    predict(
      fit_quantile,
      newdata = data.frame(x = xnew)
    )
  )
}

fit_quantile_model_hd <- function(x, y, tau) {
  x <- as.data.frame(x)
  colnames(x) <- paste0("x", seq_len(ncol(x)))

  df <- data.frame(
    y = y,
    x
  )

  quantreg::rq(
    y ~ .,
    tau = tau,
    data = df
  )
}

predict_quantile_hd <- function(fit_quantile, xnew) {
  xnew <- as.data.frame(xnew)
  colnames(xnew) <- paste0("x", seq_len(ncol(xnew)))

  as.numeric(
    predict(
      fit_quantile,
      newdata = xnew
    )
  )
}

fit_quantile_model_auto <- function(x, y, tau, model_id) {
  fit_quantile_model(x, y, tau)
}

predict_quantile_auto <- function(fit_quantile, xnew, model_id) {
  predict_quantile(fit_quantile, xnew)
}