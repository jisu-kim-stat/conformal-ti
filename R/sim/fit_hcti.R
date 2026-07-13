# for low-dimensional data
fit_mean_model <- function(x, y) {
  smooth.spline(x, y, cv = FALSE)
}

fit_var_model <- function(x, y, fit_mean,
                        eps = 1e-4,
                        k = 10,
                        method = 'REML') {
  res <- y - as.numeric(predict(fit_mean, x)$y)
  res2 <- res^2 + eps

  df <- data.frame(x=x, res2 = res2)
  
  mgcv::gam(res2 ~ s(x, k=k), data = df, family = Gamma(link = 'log'), method=method)
}


predict_mean <- function(fit_mean, xnew) {
  as.numeric(predict(fit_mean, xnew)$y)
}

predict_var <- function(fit_var, xnew,
                      min_var = 1e-3, max_var = 1e3) {
  vhat <- as.numeric(
    predict(fit_var, newdata = data.frame(x=xnew), type='response')
  )

  pmin(pmax(vhat, min_var), max_var)
}


# for high-dimensional data
fit_mean_model_hd <- function(x,y) {
  glmnet::cv.glmnet(x = as.matrix(x), y = y, alpha = 1, standardize = TRUE)
}

predict_mean_hd <- function(fit_mean, xnew) {
  as.numeric(predict(fit_mean, newx = as.matrix(xnew), s = 'lambda.min'))
}

fit_var_model_hd <- function(x, y, fit_mean, 
                          min_var = 1e-3, max_var = 20) {
  res <- y - predict_mean_hd(fit_mean, x)
  coef_hat <- as.matrix(coef(fit_mean, s='lambda.min'))
  df_eff <- sum(coef_hat[-1, 1] != 0)

  n <- length(y)
  denom <- max(n - df_eff, 1)

  vhat <- sum(res^2) / denom
  pmin(pmax(vhat, min_var), max_var)
}

predict_var_hd <- function(fit_var, xnew) {
  rep(fit_var, nrow(as.matrix(xnew)))
}


fit_mean_model_auto <- function(x, y, model_id) {
  if (model_id == 6) {
    fit_mean_model_hd(x, y)
  } else {
    fit_mean_model(x, y)
  }
}

fit_var_model_auto <- function(x, y, fit_mean, model_id) {
  if (model_id == 6) {
    fit_var_model_hd(x, y, fit_mean)
  } else {
    fit_var_model(x, y, fit_mean)
  }
}

predict_mean_auto <- function(fit_mean, xnew, model_id) {
  if (model_id == 6) {
    predict_mean_hd(fit_mean, xnew)
  } else {
    predict_mean(fit_mean, xnew)
  }
}

predict_var_auto <- function(fit_var, xnew, model_id) {
  if (model_id == 6) {
    predict_var_hd(fit_var, xnew)
  } else {
    predict_var(fit_var, xnew)
  }
}

check_var_model <- function(fit_var) {
  print(summary(fit_var))
  mgcv::gam.check(fit_var)
}
