suppressPackageStartupMessages({
  library(splines)
  library(MASS)
})

tf  <- function(y) log1p(y)
itf <- function(z) expm1(z)

make_B <- function(x, df, spec = NULL) {
  x <- as.numeric(x)

  if (is.null(spec)) {
    B <- bs(x, df = df, degree = 3, intercept = TRUE)
    spec <- list(
      knots = attr(B, "knots"),
      Boundary.knots = attr(B, "Boundary.knots"),
      degree = attr(B, "degree"),
      intercept = attr(B, "intercept")
    )
    return(list(B = B, spec = spec))
  } else {
    # --- clamp x to boundary knots to avoid ill-conditioning
    lo <- spec$Boundary.knots[1]
    hi <- spec$Boundary.knots[2]
    x <- pmin(pmax(x, lo), hi)

    B <- bs(
      x,
      knots = spec$knots,
      Boundary.knots = spec$Boundary.knots,
      degree = spec$degree,
      intercept = spec$intercept
    )
    return(list(B = B, spec = spec))
  }
}

D2_penalty <- function(p) {
  I <- diag(p)
  D2 <- diff(I, differences = 2)
  crossprod(D2)
}

fit_pspline_gcv <- function(x, y, df, lam_grid = 10^seq(-4, 4, length.out = 80)) {
  x <- as.numeric(drop(x))
  y <- as.numeric(drop(y))

  if (length(x) != length(y)) {
    stop(sprintf("Length mismatch: length(x)=%d, length(y)=%d", length(x), length(y)))
  }

  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]; y <- y[ok]

  n <- length(x)
  if (n < df + 5) stop("Too few finite points after filtering: n=", n, ", df=", df)

  tmp <- make_B(x, df = df, spec = NULL)
  B <- tmp$B
  B_spec <- tmp$spec

  p <- ncol(B)
  BtB <- crossprod(B)
  Bty <- crossprod(B, y)
  P <- D2_penalty(p)

  best <- NULL
  for (lam in lam_grid) {
    M <- BtB + lam * P
    A <- tryCatch(solve(M), error = function(e) ginv(M))
    beta <- A %*% Bty
    yhat <- as.numeric(B %*% beta)
    resid <- y - yhat
    rss <- sum(resid^2)

    df_eff <- sum(diag(A %*% BtB))
    denom <- n - df_eff
    if (!is.finite(rss) || !is.finite(df_eff) || !is.finite(denom) || denom <= 1e-8) next

    gcv <- (n * rss) / (denom^2)
    if (!is.finite(gcv)) next

    if (is.null(best) || gcv < best$gcv) {
      best <- list(
        lam = lam, beta = beta, B = B, B_spec = B_spec,
        A = A, BtB = BtB, df_eff = df_eff, yhat = yhat,
        rss = rss, gcv = gcv
      )
    }
  }
  if (is.null(best)) stop("GCV failed")
  best
}


predict_pspline <- function(x_new, df, beta, B_spec) {
  tmp <- make_B(as.numeric(x_new), df = df, spec = B_spec)
  Bn <- tmp$B
  as.numeric(Bn %*% beta)
}

norm_from_basis <- function(x_new, df, B_spec, A, BtB) {
  tmp <- make_B(as.numeric(x_new), df = df, spec = B_spec)
  B_new <- tmp$B
  M <- A %*% BtB %*% A
  sqrt(rowSums((B_new %*% M) * B_new))
}

k_factor <- function(nu, norm_vals, P, gamma) {
  P <- min(max(P, 1e-12), 1 - 1e-12)
  gamma <- min(max(gamma, 1e-12), 1 - 1e-12)

  # ncp 과도 폭주 방지 (수치 안정 목적)
  ncp <- pmin(norm_vals^2, 1e6)

  num <- qchisq(P, df = 1, ncp = ncp)
  den <- qchisq(1 - gamma, df = nu)

  out <- sqrt(nu * num / den)
  out[!is.finite(out)] <- NA_real_
  out
}

# ---- replacement for your GY block ----
run_gy_1d <- function(train_x, train_y, cal_x, cal_y, test_x, test_y,
                      alpha, delta,
                      df_mean = 12, df_var = 12, df_std = 12) {

  # numeric vector inputs
  train_x <- as.numeric(drop(train_x)); train_y <- as.numeric(drop(train_y))
  cal_x   <- as.numeric(drop(cal_x));   cal_y   <- as.numeric(drop(cal_y))
  test_x  <- as.numeric(drop(test_x));  test_y  <- as.numeric(drop(test_y))

  # -------------------------
  # (GY policy) use ALL = train + cal as training set
  # -------------------------
  x_all <- c(train_x, cal_x)
  y_all <- c(train_y, cal_y)

  ok_all <- is.finite(x_all) & is.finite(y_all)
  x_all <- x_all[ok_all]
  y_all <- y_all[ok_all]

  z_all <- tf(y_all)

  # -------------------------
  # mean on ALL
  # -------------------------
  mean_fit <- fit_pspline_gcv(x_all, z_all, df = df_mean)
  mu_all <- mean_fit$yhat

  # -------------------------
  # var on ALL (residual^2)
  # -------------------------
  res2_all <- (z_all - mu_all)^2
  var_fit <- fit_pspline_gcv(x_all, res2_all, df = df_var)
  var_all <- pmax(var_fit$yhat, 1e-3)

  # -------------------------
  # standardized spline on ALL
  # -------------------------
  y_std <- z_all / sqrt(var_all)
  std_fit <- fit_pspline_gcv(x_all, y_std, df = df_std)

  # variance on standardized scale (use effective df)
  resid_std <- y_std - std_fit$yhat
  rss_std <- sum(resid_std^2)
  df_eff <- std_fit$df_eff
  n_all <- length(x_all)

  nu <- max(5, n_all - df_eff)
  est_var <- rss_std / max(1, (n_all - df_eff))

  # -------------------------
  # test predictions (IMPORTANT: basis must be consistent)
  # 여기서는 "predict_pspline + B_spec 재사용" 버전을 써야 함
  # -------------------------
  mu_test <- predict_pspline(test_x, df_mean, mean_fit$beta, mean_fit$B_spec)
  var_test <- pmax(predict_pspline(test_x, df_var, var_fit$beta, var_fit$B_spec), 1e-3)

  mu_std_test <- predict_pspline(test_x, df_std, std_fit$beta, std_fit$B_spec)

  # k(x_test) via basis on the SAME spec
  norm_test <- norm_from_basis(test_x, df_std, std_fit$B_spec, std_fit$A, std_fit$BtB)
  k_test <- k_factor(nu, norm_test, P = 1 - alpha, gamma = delta)

  # TI on standardized scale
  up_t <- mu_std_test + sqrt(est_var) * k_test
  lo_t <- mu_std_test - sqrt(est_var) * k_test

  # back to z-scale
  up_z <- up_t * sqrt(var_test)
  lo_z <- lo_t * sqrt(var_test)

  # back to y-scale
  up_y <- itf(up_z)
  lo_y <- pmax(itf(lo_z), 0)

  content <- mean(test_y >= lo_y & test_y <= up_y, na.rm = TRUE)
  width   <- mean(up_y - lo_y, na.rm = TRUE)

  list(
    method = "GY(1D mag_r)",
    content = content,
    mean_width = width,
    gy_df_eff = df_eff,
    gy_nu = nu,
    gy_est_var = est_var,
    gy_norm_q = quantile(norm_test, c(0.5, 0.9, 0.99), na.rm = TRUE)
  )
}
